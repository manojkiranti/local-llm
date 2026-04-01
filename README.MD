# RAG Retrieval and Answer Generation Backend

FastAPI backend for document ingestion, embedding, retrieval, and answer generation.

- Supports `.txt`, `.md`, `.pdf`, `.docx` files
- Automatic Preeti-to-Unicode conversion for Nepali documents
- Chunking with configurable size and overlap
- Embeddings via `BAAI/bge-m3` (CPU or GPU)
- Vector storage in Qdrant, metadata in PostgreSQL
- Answer generation using Ollama (Qwen or any OpenAI-compatible LLM)

---

## Features

### Health
- `GET /health` — health check for Qdrant and Ollama

### Chat
- `POST /ask` — embed query, retrieve top-k chunks from Qdrant, send grounded prompt to LLM, return answer + sources
- `GET /search` — semantic search only, returns retrieved chunks

### Document Management
- `POST /documents/files/upload` — upload files to the documents folder
- `GET /documents/files` — list all files in the documents folder
- `DELETE /documents/files/{filename}` — delete a file from the documents folder

### Embedding
- `POST /documents/embed` — trigger embedding pipeline (background task)
- `GET /documents/embed/status/{task_id}` — check embedding task status

### Embedded Documents
- `GET /documents` — list all embedded documents
- `GET /documents/{doc_id}` — get a single embedded document
- `DELETE /documents/{doc_id}` — delete embedded document (removes from Qdrant + Postgres)

### Scraper
- `POST /scraper/scrape` — scrape and download notices
- `GET /scraper/notices` — list downloaded notices

---

## Configuration

All settings are configured via `.env` file. Key settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/nic_notices` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `EMBEDDING_DEVICE` | `cpu` for local, `cuda` for GPU server | `cpu` |
| `LLM_BASE_URL` | Ollama or OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `DOCS_DIR` | Folder for uploaded documents | `./download` |

---

## Run (Development)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Requires PostgreSQL, Qdrant, and Ollama running locally.

---

## Run (Docker - Production)

```bash
docker compose up -d
```

This starts all services: **app**, **PostgreSQL**, **Qdrant**, and **Ollama**.

After first launch, pull the LLM model:

```bash
docker exec ollama ollama pull qwen2.5:latest
```

### GPU Support

On a server with NVIDIA GPU, uncomment the `deploy` section in `docker-compose.yml` for the Ollama service, and set `EMBEDDING_DEVICE=cuda` in `.env`.

Requires `nvidia-container-toolkit` on the host.

---

## Project Structure

```text
app/
├── api/routes/
│   ├── chat.py
│   ├── documents.py
│   ├── health.py
│   └── scraper.py
├── db/
│   ├── postgres.py
│   └── qdrant.py
├── models/
│   ├── database.py
│   └── schemas.py
├── services/
│   ├── embedding.py
│   ├── llm.py
│   └── retrieval.py
├── config.py
└── main.py