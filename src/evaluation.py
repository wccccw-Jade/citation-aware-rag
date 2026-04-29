from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .pipeline import CitationAwareRAG


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _normalize_file_name(value: str) -> str:
    return Path(value).name.lower()


def _matches_gold_file(citation: dict[str, Any], gold_file: str) -> bool:
    citation_file = _normalize_file_name(str(citation.get("source_path", "")))
    return citation_file == gold_file


def _matches_gold_page(citation: dict[str, Any], gold_file: str, gold_page: int | None) -> bool:
    if not _matches_gold_file(citation, gold_file):
        return False
    if gold_page is None:
        return True
    return citation.get("page_number") == gold_page


def _keyword_hits(text: str, gold_keywords: list[str]) -> int:
    normalized_text = _normalize_text(text)
    return sum(1 for keyword in gold_keywords if _normalize_text(keyword) in normalized_text)


def _answer_grounded(answer: str, gold_answer: str | None, gold_keywords: list[str]) -> bool:
    normalized_answer = _normalize_text(answer)
    if gold_answer and _normalize_text(gold_answer) in normalized_answer:
        return True
    return bool(gold_keywords) and _keyword_hits(answer, gold_keywords) > 0


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "total_questions": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "gold_page_hit_at_5": 0.0,
            "answer_grounded_rate": 0.0,
            "ndcg_at_5": 0.0,
        }

    hit_at_1 = sum(1 for row in rows if row["gold_rank"] == 1) / total
    hit_at_3 = sum(1 for row in rows if row["gold_rank"] and row["gold_rank"] <= 3) / total
    hit_at_5 = sum(1 for row in rows if row["gold_in_top5"]) / total
    mrr = sum(1 / row["gold_rank"] for row in rows if row["gold_rank"]) / total
    gold_page_hit_at_5 = sum(1 for row in rows if row["gold_page_in_top5"]) / total
    answer_grounded_rate = sum(1 for row in rows if row["answer_grounded"]) / total
    ndcg_at_5 = sum(1 / math.log2(row["gold_rank"] + 1) for row in rows if row["gold_rank"]) / total
    return {
        "total_questions": total,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": mrr,
        "gold_page_hit_at_5": gold_page_hit_at_5,
        "answer_grounded_rate": answer_grounded_rate,
        "ndcg_at_5": ndcg_at_5,
    }


def run_evaluation(rag: CitationAwareRAG, eval_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for row in rows:
        result = rag.answer(row["question"], top_k=5)

        gold_source = row["gold_source"]
        gold_file = _normalize_file_name(gold_source["file_name"])
        gold_page = gold_source.get("page_num")
        retrieved_files_top5 = [_normalize_file_name(citation["source_path"]) for citation in result.citations[:5]]
        gold_rank: int | None = None
        gold_page_rank: int | None = None
        for index, citation in enumerate(result.citations[:5], start=1):
            if gold_rank is None and _matches_gold_file(citation, gold_file):
                gold_rank = index
            if gold_page_rank is None and _matches_gold_page(citation, gold_file, gold_page):
                gold_page_rank = index

        gold_keywords = row.get("gold_keywords", [])
        results.append(
            {
                "question": row["question"],
                "gold_file": gold_source["file_name"],
                "gold_page": gold_page,
                "gold_chunk_hint": row.get("gold_chunk_hint", ""),
                "gold_answer": row.get("gold_answer", ""),
                "retrieved_files_top5": " | ".join(retrieved_files_top5),
                "gold_in_top5": gold_rank is not None,
                "gold_rank": gold_rank,
                "gold_page_in_top5": gold_page_rank is not None,
                "gold_page_rank": gold_page_rank,
                "answer_grounded": _answer_grounded(result.answer, row.get("gold_answer"), gold_keywords),
                "keyword_hits": _keyword_hits(result.answer, gold_keywords),
                "keyword_total": len(gold_keywords),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "gold_file",
        "gold_page",
        "gold_chunk_hint",
        "gold_answer",
        "retrieved_files_top5",
        "gold_in_top5",
        "gold_rank",
        "gold_page_in_top5",
        "gold_page_rank",
        "answer_grounded",
        "keyword_hits",
        "keyword_total",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return results
