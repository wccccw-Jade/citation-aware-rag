from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir


DATASET_NAME = "sentence-transformers/natural-questions"


def _load_split(split: str, limit: int | None, cache_dir: Path) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            ".venv/bin/python -m pip install datasets"
        ) from exc

    ensure_dir(cache_dir)
    split_expr = f"{split}[:{limit}]" if limit else split
    dataset = load_dataset(DATASET_NAME, split=split_expr, cache_dir=str(cache_dir))
    return dataset.to_pandas()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _keywords(answer: str) -> list[str]:
    normalized = _normalize_space(answer)
    if not normalized:
        return []

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", normalized)
    candidates: list[str] = []
    if words:
        candidates.append(" ".join(words[: min(8, len(words))]))
    if len(words) > 8:
        midpoint = max(0, len(words) // 2 - 4)
        candidates.append(" ".join(words[midpoint : midpoint + 8]))
    if len(words) > 16:
        candidates.append(" ".join(words[-8:]))

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate.lower() not in seen:
            unique.append(candidate)
            seen.add(candidate.lower())
    return unique


def _file_name(index: int) -> str:
    return f"nq_{index:06d}.md"


def _write_corpus(df: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(df.itertuples(index=False), start=1):
        record = row._asdict()
        question = _normalize_space(_text(record.get("query")))
        answer = _normalize_space(_text(record.get("answer")))
        if not question or not answer:
            continue

        file_name = _file_name(index)
        path = output_dir / file_name
        path.write_text(
            "\n".join(
                [
                    f"# Natural Questions Item {index}",
                    "",
                    "Answer passage:",
                    answer,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "index": index,
                "file_name": file_name,
                "question": question,
                "answer": answer,
            }
        )
    return rows


def _write_eval(records: list[dict[str, Any]], output_path: Path, eval_limit: int | None) -> Path:
    ensure_dir(output_path.parent)
    selected = records[:eval_limit] if eval_limit else records
    rows = [
        {
            "question": record["question"],
            "gold_source": {
                "file_name": record["file_name"],
                "page_num": 1,
            },
            "gold_keywords": _keywords(record["answer"]),
            "gold_chunk_hint": record["answer"][:240],
            "gold_answer": record["answer"],
        }
        for record in selected
    ]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare sentence-transformers/natural-questions for this RAG project."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/nq/raw"))
    parser.add_argument("--eval-path", type=Path, default=Path("data/nq/eval/qa_eval.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/nq/cache"))
    args = parser.parse_args()

    df = _load_split(args.split, args.limit, args.cache_dir)
    records = _write_corpus(df, args.raw_dir)
    eval_path = _write_eval(records, args.eval_path, args.eval_limit)

    print(f"Loaded dataset: {DATASET_NAME} split={args.split} rows={len(df)}")
    print(f"Wrote corpus: {args.raw_dir} ({len(records)} files)")
    print(f"Wrote eval: {eval_path} ({len(records[:args.eval_limit] if args.eval_limit else records)} rows)")
    print()
    print("Build the NQ index with:")
    print(
        f"RAW_DATA_DIR={args.raw_dir} PROCESSED_DATA_DIR=data/nq/processed "
        f"INDEX_DIR=data/nq/index .venv/bin/python scripts/build_index.py"
    )
    print()
    print("Run evaluation with:")
    print(
        "RAW_DATA_DIR=data/nq/raw PROCESSED_DATA_DIR=data/nq/processed "
        "INDEX_DIR=data/nq/index .venv/bin/python scripts/run_eval.py "
        f"--eval-path {eval_path} --output-path data/nq/eval/eval_results.csv"
    )


if __name__ == "__main__":
    main()
