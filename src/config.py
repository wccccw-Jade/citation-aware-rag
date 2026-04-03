from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    raw_data_dir: Path = Field(default=Path(os.getenv("RAW_DATA_DIR", "data/raw")))
    processed_data_dir: Path = Field(default=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")))
    index_dir: Path = Field(default=Path(os.getenv("INDEX_DIR", "faiss_index")))
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "120")))
    top_k: int = Field(default=int(os.getenv("TOP_K", "5")))
    embedding_dim: int = Field(default=int(os.getenv("EMBEDDING_DIM", "256")))
    use_faiss: bool = Field(default=os.getenv("USE_FAISS", "true").lower() == "true")
    retrieval_candidate_pool: int = Field(default=int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "25")))
    dense_weight: float = Field(default=float(os.getenv("DENSE_WEIGHT", "0.45")))
    lexical_weight: float = Field(default=float(os.getenv("LEXICAL_WEIGHT", "0.55")))
    app_title: str = Field(default=os.getenv("APP_TITLE", "Citation-Aware Academic QA"))


def get_settings() -> Settings:
    return Settings()
