# Citation-Aware RAG for Academic Document QA

A lightweight baseline for document question answering over academic PDFs and notes with explicit chunk-level citations.

## Project Structure

```text
citation-aware-rag/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── data/
│   ├── ragpapers/
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── index/
│   │   └── eval/
│   ├── nq/
│   └── pubhealth/
├── notebooks/
├── app/
├── scripts/
├── src/
└── assets/
```

## Features

- PDF and text ingestion with page-aware metadata
- Recursive chunking with overlap
- Deterministic local embeddings for offline development
- Optional FAISS indexing with NumPy fallback
- Retrieval that returns citation metadata for each chunk
- Simple answer generation that preserves source references
- CLI scripts for indexing, querying, and evaluation
- Streamlit app for interactive demos

## Quickstart

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and adjust settings if needed.
3. Put PDFs or `.txt` files under `data/ragpapers/raw/`.
4. Build the index:

```bash
python scripts/build_index.py
```

5. Query the system:

```bash
python scripts/demo_query.py --query "What methodology does the paper use?"
```

6. Run the Web UI:

```bash
source .venv/bin/activate
streamlit run app/app.py
```

The Web UI talks to the FastAPI service at `http://127.0.0.1:8000` by default. Set `RAG_API_BASE_URL` if the API runs elsewhere.

## FastAPI Service

Run the API service:

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start Redis and the indexing worker in separate terminals:

```bash
redis-server
```

```bash
source .venv/bin/activate
python scripts/worker.py
```

Upload a document. The API saves the file, creates a queued indexing task, and returns immediately with `task_id`:

```bash
curl -F "file=@data/ragpapers/raw/1.pdf" http://127.0.0.1:8000/documents/upload
```

Poll the task until it becomes `indexed`:

```bash
curl http://127.0.0.1:8000/tasks/{task_id}
```

Query the indexed documents:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What methodology does the paper use?", "top_k": 5}'
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

Run the Web UI after starting the API, Redis, and worker:

```bash
source .venv/bin/activate
RAG_API_BASE_URL=http://127.0.0.1:8000 streamlit run app/app.py
```

The browser UI supports document upload, document status, delete, reindex, question answering, and citation preview.

Health and runtime configuration endpoints:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/config
```

Errors use a consistent response shape and include the request id:

```json
{
  "error": {
    "code": "invalid_upload",
    "message": "Unsupported file type '.exe'. Allowed types: .md, .pdf, .txt.",
    "request_id": "f6ebc0b9d8b74a39a2f899f4ecf8d541",
    "details": null
  }
}
```

The service persists uploaded document status and index task status in a database. Local development defaults to `sqlite:///data/app.db`; uploaded files default to `data/ragpapers/raw`. PostgreSQL can be enabled with:

```bash
export DATABASE_URL="postgresql+psycopg2://rag_user:rag_password@localhost:5432/citation_rag"
uvicorn src.api.main:app --reload
```

Status endpoints:

```bash
curl http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/tasks
curl http://127.0.0.1:8000/documents/{document_id}
curl http://127.0.0.1:8000/tasks/{task_id}
curl -X POST http://127.0.0.1:8000/tasks/{task_id}/retry
```

Queue settings:

```bash
export REDIS_URL="redis://localhost:6379/0"
export TASK_QUEUE_NAME="rag-indexing"
export RQ_WORKER_CLASS="simple"
export LOG_LEVEL="INFO"
```

## Production-Oriented Configuration

The baseline now exposes provider-style configuration so local development can keep using the deterministic embedding model while production deployments can switch to stronger backends.

```bash
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PROVIDER=faiss
MMR_LAMBDA=0.75
```

Install optional production embedding dependencies before using `EMBEDDING_PROVIDER=sentence-transformers`:

```bash
pip install -r requirements-prod.txt
```

After the model has been downloaded once, offline runs can avoid Hugging Face network checks:

```bash
HF_HUB_OFFLINE=1 python scripts/build_index.py
```

Enable LLM answer generation with OpenAI:

```bash
export OPENAI_API_KEY="your_api_key_here"
export USE_LLM_GENERATION=true
export LLM_PROVIDER=openai
export LLM_MODEL_NAME=gpt-5
python scripts/demo_query.py --query "What retrieval quality problems does Naive RAG suffer from?" --top-k 5
```

If `USE_LLM_GENERATION` is false or `OPENAI_API_KEY` is missing, the system falls back to the local extractive grounded-answer generator.

Answers include citation validation metadata:

- `citation_valid`: whether the answer's citations pass validation.
- `answer_has_citation`: whether at least one `[n]` citation appears.
- `invalid_citation_count`: number of labels that do not exist in the retrieved top-k chunks.
- `refusal`: whether the answer explicitly says there is not enough evidence.
- `unsupported_claim_count`: number of answer claims without citation labels.

## Quality Checks

Run the unit tests:

```bash
python -m pytest
```

Run the retrieval and answer grounding evaluation:

```bash
python scripts/run_eval.py
```

To evaluate retrieval and citation validation without spending LLM API calls:

```bash
USE_LLM_GENERATION=false python scripts/run_eval.py
```

## Notes

- The default embedding backend is `sentence-transformers` for stronger semantic retrieval.
- Set `EMBEDDING_PROVIDER=local` and `EMBEDDING_DIM=256` if you need the deterministic offline fallback.
- If `faiss-cpu` is unavailable, retrieval falls back to a NumPy similarity search.
- Binary assets and sample PDFs are intentionally left as placeholders for your own project materials.
