# Local LLM RAG Stack

A fully local RAG pipeline for querying CSV project data via a chat interface.

## Stack
| Service | Purpose | URL |
|---|---|---|
| Ollama | Local LLM + embeddings runtime | http://localhost:11434 |
| ChromaDB | Vector database | http://localhost:8000 |
| RAG API | FastAPI ingestion + query service | http://localhost:8001 |
| Open WebUI | Stakeholder chat interface | http://localhost:3000 |

---

## Quick Start

### 1. Start the stack
```bash
docker compose up -d
```
First boot takes a few minutes — Ollama needs to pull the models.

### 2. Wait for models to download
```bash
docker logs ollama -f
```
Wait until you see the models are ready (llama3.2 + nomic-embed-text).

### 3. Ingest your CSV data

**Option A — Ingest the sample CSV already in /data:**
```bash
curl -X POST http://localhost:8001/ingest/directory
```

**Option B — Upload your own CSV:**
```bash
curl -X POST http://localhost:8001/ingest \
  -F "file=@/path/to/your/projects.csv"
```

### 4. Open the chat UI
Go to http://localhost:3000, create an account, and start asking questions.

In the model selector, choose the **RAG API** model to get context-aware answers,
or choose a standard Ollama model for direct LLM access.

---

## Example Questions to Ask
- "Which projects are delayed?"
- "What is the completion percentage of the Mobile App v2?"
- "Who owns the Data Warehouse Migration project?"
- "Which projects are due before June 2025?"

---

## Changing the LLM Model
Edit `docker-compose.yml` and update the `CHAT_MODEL` env var:
```yaml
- CHAT_MODEL=llama3.1:70b    # Larger, smarter (needs ~40GB RAM or GPU)
- CHAT_MODEL=mistral          # Fast and capable
- CHAT_MODEL=qwen2.5          # Good at structured data
```
Then run `docker compose up -d` to apply.

---

## GPU Support (NVIDIA)
1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the `deploy` block in `docker-compose.yml` under the `ollama` service
3. Run `docker compose up -d`

---

## Updating
```bash
docker compose pull && docker compose up -d
```

## Resetting all data
```bash
docker compose down -v   # Removes all volumes including model downloads
```
