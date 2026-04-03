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
│   ├── raw/
│   ├── processed/
│   └── eval/
├── faiss_index/
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
3. Put PDFs or `.txt` files under `data/raw/`.
4. Build the index:

```bash
python scripts/build_index.py
```

5. Query the system:

```bash
python scripts/demo_query.py --query "What methodology does the paper use?"
```

6. Run the demo app:

```bash
streamlit run app/app.py
```

## Notes

- The default embedding backend is local and deterministic, so the project works without external APIs.
- If `faiss-cpu` is unavailable, retrieval falls back to a NumPy similarity search.
- Binary assets and sample PDFs are intentionally left as placeholders for your own project materials.
