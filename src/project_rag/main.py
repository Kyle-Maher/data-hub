"""
RAG API — FastAPI service that:
  1. Ingests a CSV file into ChromaDB (POST /ingest)
  2. Answers questions using RAG + Ollama  (POST /v1/chat/completions)

The /v1/chat/completions endpoint is OpenAI-compatible so Open WebUI
can use this service as a drop-in backend.
"""

import os
import json
import time
import uuid
import asyncio
import pandas as pd
import httpx
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://ollama:11434")
CHROMA_URL  = os.getenv("CHROMA_URL",  "http://chromadb:8000")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "llama3.2")
COLLECTION  = "project_data"
TOP_K       = 5   # How many chunks to retrieve per query

app = FastAPI(title="RAG API")
chroma = chromadb.HttpClient(host=CHROMA_URL.replace("http://", "").split(":")[0],
                              port=int(CHROMA_URL.split(":")[-1]))


# ── Helpers ────────────────────────────────────────────────────────────────

async def get_embedding(text: str) -> list[float]:
    """Ask Ollama to embed a piece of text."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings",
                              json={"model": EMBED_MODEL, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]


async def pull_model_if_needed(model: str):
    """Pull a model from Ollama registry if it isn't already downloaded."""
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        names = [m["name"] for m in r.json().get("models", [])]
        if not any(model in n for n in names):
            print(f"Pulling model: {model} …")
            await client.post(f"{OLLAMA_URL}/api/pull", json={"name": model})


def get_or_create_collection():
    return chroma.get_or_create_collection(name=COLLECTION)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pull required models on first boot (runs in background)."""
    asyncio.create_task(pull_model_if_needed(EMBED_MODEL))
    asyncio.create_task(pull_model_if_needed(CHAT_MODEL))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file. Each row becomes a searchable chunk in ChromaDB.
    The entire row is converted to a readable string and embedded.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    contents = await file.read()

    # Save to /data for persistence
    path = f"/data/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    df = pd.read_csv(path)
    df.fillna("", inplace=True)

    collection = get_or_create_collection()

    documents, embeddings, ids, metadatas = [], [], [], []

    for i, row in df.iterrows():
        # Convert row to a natural-language-ish string
        text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        embedding = await get_embedding(text)

        documents.append(text)
        embeddings.append(embedding)
        ids.append(f"{file.filename}_{i}")
        metadatas.append({"source": file.filename, "row": i})

    collection.upsert(documents=documents,
                      embeddings=embeddings,
                      ids=ids,
                      metadatas=metadatas)

    return {"ingested": len(documents), "file": file.filename}


@app.post("/ingest/directory")
async def ingest_directory():
    """Ingest all CSVs sitting in the /data volume (useful on startup)."""
    data_dir = "/data"
    results = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path)
            df.fillna("", inplace=True)
            collection = get_or_create_collection()
            documents, embeddings, ids, metadatas = [], [], [], []
            for i, row in df.iterrows():
                text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                embedding = await get_embedding(text)
                documents.append(text)
                embeddings.append(embedding)
                ids.append(f"{fname}_{i}")
                metadatas.append({"source": fname, "row": i})
            collection.upsert(documents=documents, embeddings=embeddings,
                              ids=ids, metadatas=metadatas)
            results[fname] = len(documents)
    return {"ingested_files": results}


# ── OpenAI-compatible chat endpoint (used by Open WebUI) ───────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = CHAT_MODEL
    messages: List[Message]
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    user_query = next((m.content for m in reversed(req.messages)
                       if m.role == "user"), "")

    # 1. Embed the query
    query_embedding = await get_embedding(user_query)

    # 2. Retrieve top-K relevant chunks
    collection = get_or_create_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    context_chunks = results["documents"][0] if results["documents"] else []

    # 3. Build augmented prompt
    context = "\n\n".join(context_chunks) if context_chunks else "No project data found."
    system_prompt = (
        "You are a helpful project assistant. Answer questions using ONLY the "
        "project data provided below. If the answer is not in the data, say so.\n\n"
        f"PROJECT DATA:\n{context}"
    )

    # 4. Call Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}] + \
                      [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        async def stream_response():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat",
                                         json={"model": CHAT_MODEL,
                                               "messages": ollama_messages,
                                               "stream": True}) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        done  = chunk.get("done", False)
                        payload = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": CHAT_MODEL,
                            "choices": [{"delta": {"content": token},
                                         "index": 0,
                                         "finish_reason": "stop" if done else None}]
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        if done:
                            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat",
                                  json={"model": CHAT_MODEL,
                                        "messages": ollama_messages,
                                        "stream": False})
            r.raise_for_status()
            answer = r.json()["message"]["content"]

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": CHAT_MODEL,
            "choices": [{"message": {"role": "assistant", "content": answer},
                         "index": 0, "finish_reason": "stop"}]
        }
