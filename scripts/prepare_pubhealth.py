from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir


DATASET_NAME = "bigbio/pubhealth"
PARQUET_BASE = "hf://datasets/bigbio/pubhealth@refs/convert/parquet/pubhealth_bigbio_pairs"
SPLITS = ("train", "validation", "test")


def _load_split(split: str, limit: int | None, cache_dir: Path) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            ".venv/bin/python -m pip install datasets"
        ) from exc

    ensure_dir(cache_dir)
    data_file = f"{PARQUET_BASE}/{split}/0000.parquet"
    split_expr = f"train[:{limit}]" if limit else "train"
    dataset = load_dataset(
        "parquet",
        data_files={"train": data_file},
        split=split_expr,
        cache_dir=str(cache_dir),
    )
    return dataset.to_pandas()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _file_name(index: int) -> str:
    return f"pubhealth_{index:06d}.md"


def _write_corpus(df: pd.DataFrame, output_dir: Path) -> list[dict[str, Any]]:
    ensure_dir(output_dir)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(df.itertuples(index=False), start=1):
        payload = row._asdict()
        claim = _text(payload.get("text_1"))
        evidence = _text(payload.get("text_2"))
        label = _text(payload.get("label")).lower()
        document_id = _text(payload.get("document_id"))
        source_id = _text(payload.get("id"))
        if not claim or not evidence or not label:
            continue

        file_name = _file_name(index)
        path = output_dir / file_name
        path.write_text(
            "\n".join(
                [
                    f"# PubHealth Item {index}",
                    "",
                    f"Dataset source id: {source_id}",
                    f"Document id: {document_id}",
                    "",
                    f"Fact-check label: {label}",
                    "",
                    "Evidence and explanation:",
                    evidence,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "index": index,
                "file_name": file_name,
                "source_id": source_id,
                "document_id": document_id,
                "claim": claim,
                "evidence": evidence,
                "label": label,
            }
        )
    return records


def _write_eval(records: list[dict[str, Any]], output_path: Path, eval_limit: int | None) -> Path:
    ensure_dir(output_path.parent)
    selected = records[:eval_limit] if eval_limit else records
    rows = [
        {
            "question": (
                "What is the PubHealth fact-check label for this claim "
                f"(true, false, mixture, or unproven)? Claim: {record['claim']}"
            ),
            "gold_source": {
                "file_name": record["file_name"],
                "page_num": 1,
            },
            "gold_keywords": [
                f"Fact-check label: {record['label']}",
                record["label"],
            ],
            "gold_chunk_hint": record["evidence"][:240],
            "gold_answer": f"{record['label']}. {record['evidence']}",
        }
        for record in selected
    ]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bigbio/pubhealth for this RAG project.")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/pubhealth/raw"))
    parser.add_argument("--eval-path", type=Path, default=Path("data/pubhealth/eval/qa_eval.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/pubhealth/cache"))
    args = parser.parse_args()

    df = _load_split(args.split, args.limit, args.cache_dir)
    records = _write_corpus(df, args.raw_dir)
    eval_path = _write_eval(records, args.eval_path, args.eval_limit)

    print(f"Loaded dataset: {DATASET_NAME} subset=pubhealth_bigbio_pairs split={args.split} rows={len(df)}")
    print(f"Wrote corpus: {args.raw_dir} ({len(records)} files)")
    print(f"Wrote eval: {eval_path} ({len(records[:args.eval_limit] if args.eval_limit else records)} rows)")
    print()
    print("Build the PubHealth index with:")
    print(
        f"RAW_DATA_DIR={args.raw_dir} PROCESSED_DATA_DIR=data/pubhealth/processed "
        f"INDEX_DIR=data/pubhealth/index .venv/bin/python scripts/build_index.py"
    )
    print()
    print("Run evaluation with:")
    print(
        "RAW_DATA_DIR=data/pubhealth/raw PROCESSED_DATA_DIR=data/pubhealth/processed "
        "INDEX_DIR=data/pubhealth/index .venv/bin/python scripts/run_eval.py "
        f"--eval-path {eval_path} --output-path data/pubhealth/eval/eval_results.csv"
    )


if __name__ == "__main__":
    main()
