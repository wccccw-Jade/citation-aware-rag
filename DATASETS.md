# Dataset Layout

This project keeps the default API dataset separate from offline benchmark datasets.

## RAG Papers / Default Dataset

The original PDF dataset is now isolated under `data/ragpapers` and remains the
default runtime dataset:

```text
data/ragpapers/raw/          original PDFs and API uploads
data/ragpapers/processed/    processed original documents and chunks
data/ragpapers/index/        default original vector index
data/ragpapers/eval/         original evaluation set and results
```

The FastAPI upload service uses `RAW_DATA_DIR=data/ragpapers/raw` by default through
`src.config.Settings`. Do not export dataset-specific `RAW_DATA_DIR`,
`PROCESSED_DATA_DIR`, or `INDEX_DIR` in the shell that runs the API unless you
intentionally want API uploads to go to that dataset.

Run the original evaluation:

```bash
USE_LLM_GENERATION=false HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_eval.py \
  --eval-path data/ragpapers/eval/qa_eval.json \
  --output-path data/ragpapers/eval/eval_results.csv
```

## Natural Questions

Natural Questions is isolated under `data/nq`:

```text
data/nq/raw/        answer-passage documents only; questions are not indexed
data/nq/processed/  NQ processed documents and chunks
data/nq/index/      NQ vector index
data/nq/eval/       NQ evaluation files and results
data/nq/cache/      Hugging Face cache
```

Prepare and build:

```bash
.venv/bin/python scripts/prepare_nq.py --limit 1000

RAW_DATA_DIR=data/nq/raw PROCESSED_DATA_DIR=data/nq/processed INDEX_DIR=data/nq/index \
EMBEDDING_PROVIDER=local VECTOR_STORE_PROVIDER=numpy USE_FAISS=false \
.venv/bin/python scripts/build_index.py
```

Run a 200-question NQ evaluation:

```bash
RAW_DATA_DIR=data/nq/raw PROCESSED_DATA_DIR=data/nq/processed INDEX_DIR=data/nq/index \
EMBEDDING_PROVIDER=local VECTOR_STORE_PROVIDER=numpy USE_FAISS=false USE_LLM_GENERATION=false \
.venv/bin/python scripts/run_eval.py \
  --eval-path data/nq/eval/qa_eval_200.json \
  --output-path data/nq/eval/eval_results_200.csv
```

## PubHealth

PubHealth is isolated under `data/pubhealth`:

```text
data/pubhealth/raw/        evidence/explanation documents only; claims are not indexed
data/pubhealth/processed/  PubHealth processed documents and chunks
data/pubhealth/index/      PubHealth vector index
data/pubhealth/eval/       PubHealth evaluation files and results
data/pubhealth/cache/      Hugging Face cache
```

Prepare and build:

```bash
.venv/bin/python scripts/prepare_pubhealth.py --split train

RAW_DATA_DIR=data/pubhealth/raw PROCESSED_DATA_DIR=data/pubhealth/processed INDEX_DIR=data/pubhealth/index \
EMBEDDING_PROVIDER=local VECTOR_STORE_PROVIDER=numpy USE_FAISS=false \
.venv/bin/python scripts/build_index.py
```

Run a 200-question PubHealth evaluation:

```bash
RAW_DATA_DIR=data/pubhealth/raw PROCESSED_DATA_DIR=data/pubhealth/processed INDEX_DIR=data/pubhealth/index \
EMBEDDING_PROVIDER=local VECTOR_STORE_PROVIDER=numpy USE_FAISS=false USE_LLM_GENERATION=false \
.venv/bin/python scripts/run_eval.py \
  --eval-path data/pubhealth/eval/qa_eval_200.json \
  --output-path data/pubhealth/eval/eval_results_200.csv
```
