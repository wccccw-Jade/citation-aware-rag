from pathlib import Path
import json

import pandas as pd

from scripts.prepare_nq import _write_corpus, _write_eval


def test_prepare_nq_preserves_questions_in_individual_documents(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "query": "who produces the most wool in the world",
                "answer": "Wool Australia is a leading producer of wool and China has high total production.",
            },
            {
                "query": "who sang what in the world's come over you",
                "answer": "Jack Scott recorded What in the World's Come Over You.",
            },
        ]
    )

    records = _write_corpus(df, tmp_path / "raw")
    eval_path = _write_eval(records, tmp_path / "eval" / "qa_eval.json", eval_limit=None)

    first_doc = (tmp_path / "raw" / "nq_000001.md").read_text(encoding="utf-8")
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    assert "Question: who produces the most wool in the world" not in first_doc
    assert "Answer passage:" in first_doc
    assert rows[0]["question"] == "who produces the most wool in the world"
    assert rows[0]["gold_source"]["file_name"] == "nq_000001.md"
    assert rows[1]["gold_source"]["file_name"] == "nq_000002.md"
