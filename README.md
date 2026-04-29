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

## Quality Checks

Run the unit tests:

```bash
python -m pytest
```

Run the retrieval and answer grounding evaluation:

```bash
python scripts/run_eval.py
```

## Notes

- The default embedding backend is `sentence-transformers` for stronger semantic retrieval.
- Set `EMBEDDING_PROVIDER=local` and `EMBEDDING_DIM=256` if you need the deterministic offline fallback.
- If `faiss-cpu` is unavailable, retrieval falls back to a NumPy similarity search.
- Binary assets and sample PDFs are intentionally left as placeholders for your own project materials.
