from pathlib import Path
import json

import pandas as pd

from scripts.prepare_medqa import _write_corpus, _write_eval


def test_prepare_medqa_writes_project_corpus_and_eval(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "id": "example-1",
                "sent1": "A patient has a clinical finding.",
                "sent2": "What is the most likely diagnosis?",
                "ending0": "Diagnosis A",
                "ending1": "Diagnosis B",
                "ending2": "Diagnosis C",
                "ending3": "Diagnosis D",
                "label": 1,
            }
        ]
    )

    corpus_path = _write_corpus(df, "test", tmp_path / "raw")
    eval_path = _write_eval(df, "test", corpus_path, tmp_path / "eval" / "qa_eval.json", limit=None)

    corpus = corpus_path.read_text(encoding="utf-8")
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    assert "Question: A patient has a clinical finding. What is the most likely diagnosis?" in corpus
    assert "Correct answer: B. Diagnosis B" in corpus
    assert rows[0]["question"] == "A patient has a clinical finding. What is the most likely diagnosis?"
    assert rows[0]["gold_source"]["file_name"] == "medqa_test.md"
    assert rows[0]["gold_answer"] == "B. Diagnosis B"
