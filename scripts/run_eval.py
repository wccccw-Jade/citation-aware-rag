from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.evaluation import run_evaluation, summarize_metrics
from src.pipeline import CitationAwareRAG


def main() -> None:
    settings = get_settings()
    rag = CitationAwareRAG(settings)
    rag.load_index()
    output_path = settings.processed_data_dir.parent / "eval" / "eval_results.csv"
    rows = run_evaluation(rag, Path("data/eval/qa_eval.json"), output_path)
    summary = summarize_metrics(rows)
    for index, row in enumerate(rows, start=1):
        print(
            f"q{index} "
            f"gold_file={row['gold_file']!r} "
            f"gold_rank={row['gold_rank']} "
            f"gold_in_top5={row['gold_in_top5']} "
            f"answer_grounded={row['answer_grounded']} "
            f"keyword_hits={row['keyword_hits']}/{row['keyword_total']}"
        )
    print()
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Hit@1: {summary['hit_at_1']:.2f}")
    print(f"Hit@3: {summary['hit_at_3']:.2f}")
    print(f"Hit@5: {summary['hit_at_5']:.2f}")
    print(f"MRR: {summary['mrr']:.2f}")
    print(f"Gold Page Hit@5: {summary['gold_page_hit_at_5']:.2f}")
    print(f"nDCG@5: {summary['ndcg_at_5']:.2f}")
    print(f"Answer Grounded Rate: {summary['answer_grounded_rate']:.2f}")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
