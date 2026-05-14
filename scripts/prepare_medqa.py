from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir


DATASET_REPO = "GBaker/MedQA-USMLE-4-options"
SPLITS = ("train", "test")
OPTION_LABELS = ("A", "B", "C", "D")


def _load_split(split: str, cache_dir: Path) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with: "
            ".venv/bin/python -m pip install datasets"
        ) from exc

    ensure_dir(cache_dir)
    dataset = load_dataset(DATASET_REPO, cache_dir=str(cache_dir))
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} is not available. Available splits: {available}")
    return dataset[split].to_pandas()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _row_id(row: pd.Series, fallback: int) -> str:
    value = row.get("id")
    if pd.isna(value):
        return f"medqa-{fallback}"
    return _text(value) or f"medqa-{fallback}"


def _question(row: pd.Series) -> str:
    sent1 = _text(row.get("sent1"))
    sent2 = _text(row.get("sent2"))
    if sent1 and sent2:
        return f"{sent1} {sent2}"
    return sent1 or sent2


def _options(row: pd.Series) -> list[str]:
    return [_text(row.get(f"ending{index}")) for index in range(4)]


def _answer_index(row: pd.Series) -> int:
    label = row.get("label")
    if pd.isna(label):
        return -1
    return int(label)


def _answer_text(row: pd.Series) -> str:
    index = _answer_index(row)
    options = _options(row)
    if 0 <= index < len(options):
        return options[index]
    return ""


def _write_corpus(df: pd.DataFrame, split: str, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    path = output_dir / f"medqa_{split}.md"
    lines = [
        f"# MedQA USMLE 4-option {split} corpus",
        "",
        "Each section is a converted multiple-choice QA item. Citations point to this synthetic corpus, not original exam sources.",
        "",
    ]
    for index, row in df.iterrows():
        row_id = _row_id(row, index)
        question = _question(row)
        options = _options(row)
        answer_index = _answer_index(row)
        answer_label = OPTION_LABELS[answer_index] if 0 <= answer_index < len(OPTION_LABELS) else "?"
        answer = _answer_text(row)

        lines.extend(
            [
                f"## Item {row_id}",
                "",
                f"Question: {question}",
                "",
                "Options:",
                *[
                    f"- {OPTION_LABELS[option_index]}. {option}"
                    for option_index, option in enumerate(options)
                ],
                "",
                f"Correct answer: {answer_label}. {answer}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_eval(df: pd.DataFrame, split: str, corpus_path: Path, output_path: Path, limit: int | None) -> Path:
    ensure_dir(output_path.parent)
    rows: list[dict[str, Any]] = []
    selected = df.head(limit) if limit else df
    for _, row in selected.iterrows():
        answer = _answer_text(row)
        answer_index = _answer_index(row)
        answer_label = OPTION_LABELS[answer_index] if 0 <= answer_index < len(OPTION_LABELS) else "?"
        rows.append(
            {
                "question": _question(row),
                "gold_source": {
                    "file_name": corpus_path.name,
                    "page_num": 1,
                },
                "gold_keywords": [answer, f"Correct answer: {answer_label}"],
                "gold_chunk_hint": f"MedQA {split} item with correct answer {answer_label}. {answer}",
                "gold_answer": f"{answer_label}. {answer}",
            }
        )

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GBaker/MedQA-USMLE-4-options for this RAG project.")
    parser.add_argument("--corpus-split", choices=SPLITS, default="train")
    parser.add_argument("--eval-split", choices=SPLITS, default="test")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medqa/raw"))
    parser.add_argument("--eval-path", type=Path, default=Path("data/medqa/eval/qa_eval.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/medqa/cache"))
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--self-contained-eval",
        action="store_true",
        help="Use the eval split as the indexed corpus too. This is useful for demoing citations, but it leaks answers.",
    )
    args = parser.parse_args()

    corpus_split = args.eval_split if args.self_contained_eval else args.corpus_split
    corpus_df = _load_split(corpus_split, args.cache_dir)
    eval_df = corpus_df if args.self_contained_eval else _load_split(args.eval_split, args.cache_dir)

    corpus_path = _write_corpus(corpus_df, corpus_split, args.raw_dir)
    eval_path = _write_eval(eval_df, args.eval_split, corpus_path, args.eval_path, args.eval_limit)

    print(f"Wrote corpus: {corpus_path} ({len(corpus_df)} rows)")
    print(f"Wrote eval: {eval_path} ({len(eval_df.head(args.eval_limit) if args.eval_limit else eval_df)} rows)")
    print()
    print("Build the MedQA index with:")
    print(f"RAW_DATA_DIR={args.raw_dir} PROCESSED_DATA_DIR=data/medqa/processed INDEX_DIR=data/medqa/index .venv/bin/python scripts/build_index.py")
    print()
    print("Run evaluation with the generated eval file by updating scripts/run_eval.py or passing it to a custom runner.")


if __name__ == "__main__":
    main()
