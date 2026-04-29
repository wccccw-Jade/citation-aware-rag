from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.config import get_settings
from src.llm import create_llm_provider
from src.pipeline import CitationAwareRAG
from src.prompt import build_citation_aware_prompts
from src.retrieval import Retriever


DEFAULT_DEBUG_QUERY = "What retrieval quality problems does Naive RAG suffer from?"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _format_vector(vector: np.ndarray, preview: int, full: bool) -> str:
    values = vector.tolist() if full else vector[:preview].tolist()
    formatted = ", ".join(f"{value:.6f}" for value in values)
    suffix = "" if full or len(vector) <= preview else f", ... ({len(vector)} dims total)"
    return f"[{formatted}{suffix}]"


def _separator(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def _print_prompt(system_prompt: str, user_prompt: str) -> None:
    _separator("OPENAI SYSTEM PROMPT")
    print(system_prompt)
    _separator("OPENAI USER PROMPT")
    print(user_prompt)


def _iter_chunk_indices(rag: CitationAwareRAG, chunk_ids: Iterable[str]) -> dict[str, int]:
    return {chunk.chunk_id: index for index, chunk in enumerate(rag.chunks) if chunk.chunk_id in chunk_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug one RAG query end to end.")
    parser.add_argument(
        "--query",
        default=os.getenv("DEBUG_QUERY", DEFAULT_DEBUG_QUERY),
        help="Question to ask the system. Defaults to DEBUG_QUERY from .env.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_env_optional_int("DEBUG_TOP_K"),
        help="Number of chunks to retrieve. Defaults to DEBUG_TOP_K from .env or TOP_K.",
    )
    parser.add_argument(
        "--vector-preview",
        type=int,
        default=_env_int("DEBUG_VECTOR_PREVIEW", 12),
        help="Number of vector dimensions to print. Defaults to DEBUG_VECTOR_PREVIEW from .env.",
    )
    parser.add_argument(
        "--full-vectors",
        action="store_true",
        default=_env_bool("DEBUG_FULL_VECTORS", False),
        help="Print full query/chunk vectors. Defaults to DEBUG_FULL_VECTORS from .env.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=_env_bool("DEBUG_NO_LLM", False),
        help="Do not call the LLM; only print retrieval and prompt. Defaults to DEBUG_NO_LLM from .env.",
    )
    args = parser.parse_args()

    settings = get_settings()
    rag = CitationAwareRAG(settings)
    rag.load_index()

    top_k = args.top_k or settings.top_k
    query_embedding = rag.embedder.encode([args.query])
    retriever = Retriever(rag.embedder, rag.vector_store, settings)
    retrieved = retriever.retrieve(args.query, rag.chunks, top_k)
    chunk_index_by_id = _iter_chunk_indices(rag, [item.chunk.chunk_id for item in retrieved])
    embeddings = getattr(rag.vector_store, "embeddings", None)

    _separator("QUERY")
    print(args.query)
    print()
    print(f"embedding_provider: {rag.embedder.provider_name}")
    print(f"embedding_model: {rag.embedder.model_name}")
    print(f"query_vector: {_format_vector(query_embedding[0], args.vector_preview, args.full_vectors)}")

    _separator(f"TOP {len(retrieved)} RETRIEVED CHUNKS")
    for rank, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        chunk_index = chunk_index_by_id.get(chunk.chunk_id)
        print(f"\n--- rank={rank} score={item.score:.6f} index={chunk_index} ---")
        print(f"title: {chunk.title}")
        print(f"source_path: {chunk.source_path}")
        print(f"page_number: {chunk.page_number}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"char_range: {chunk.start_char}-{chunk.end_char}")
        if embeddings is not None and chunk_index is not None:
            print(f"chunk_vector: {_format_vector(embeddings[chunk_index], args.vector_preview, args.full_vectors)}")
        else:
            print("chunk_vector: unavailable")
        print()
        print("chunk_text:")
        print(chunk.text)

    system_prompt, user_prompt = build_citation_aware_prompts(args.query, retrieved)
    _print_prompt(system_prompt, user_prompt)

    _separator("LLM OUTPUT")
    if args.no_llm:
        print("Skipped LLM call because --no-llm was provided.")
        return

    provider = create_llm_provider(settings)
    if provider is None:
        print("LLM provider is not configured. Set USE_LLM_GENERATION=true and OPENAI_API_KEY in .env.")
        return

    try:
        answer = provider.generate(system_prompt, user_prompt)
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        return

    print(answer)


if __name__ == "__main__":
    main()
