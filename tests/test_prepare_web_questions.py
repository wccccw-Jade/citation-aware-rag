from pathlib import Path
import json

import pandas as pd

from scripts.prepare_web_questions import _write_corpus, _write_eval


def test_prepare_web_questions_preserves_question_url_and_answers(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "url": "http://www.freebase.com/view/en/justin_bieber",
                "question": "what is the name of justin bieber brother?",
                "answers": ["Jazmyn Bieber", "Jaxon Bieber"],
            }
        ]
    )

    records = _write_corpus(df, tmp_path / "raw")
    eval_path = _write_eval(records, tmp_path / "eval" / "qa_eval.json", eval_limit=None)

    document = (tmp_path / "raw" / "webq_000001.md").read_text(encoding="utf-8")
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    assert "URL: http://www.freebase.com/view/en/justin_bieber" in document
    assert "Question: what is the name of justin bieber brother?" in document
    assert "- Jazmyn Bieber" in document
    assert "- Jaxon Bieber" in document
    assert rows[0]["question"] == "what is the name of justin bieber brother?"
    assert rows[0]["gold_source"]["file_name"] == "webq_000001.md"
    assert rows[0]["gold_keywords"] == ["Jazmyn Bieber", "Jaxon Bieber"]
