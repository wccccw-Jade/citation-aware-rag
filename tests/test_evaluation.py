from src.evaluation import summarize_metrics


def test_summarize_metrics_includes_extended_retrieval_and_generation_metrics() -> None:
    rows = [
        {"gold_rank": 1, "gold_in_top5": True, "gold_page_in_top5": True, "answer_grounded": True},
        {"gold_rank": 3, "gold_in_top5": True, "gold_page_in_top5": False, "answer_grounded": True},
        {"gold_rank": None, "gold_in_top5": False, "gold_page_in_top5": False, "answer_grounded": False},
    ]

    summary = summarize_metrics(rows)

    assert summary["total_questions"] == 3
    assert summary["hit_at_1"] == 1 / 3
    assert summary["hit_at_5"] == 2 / 3
    assert summary["gold_page_hit_at_5"] == 1 / 3
    assert summary["answer_grounded_rate"] == 2 / 3
    assert summary["ndcg_at_5"] > 0

