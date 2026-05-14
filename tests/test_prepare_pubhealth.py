from pathlib import Path
import json

import pandas as pd

from scripts.prepare_pubhealth import _write_corpus, _write_eval


def test_prepare_pubhealth_preserves_claim_label_and_evidence(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "id": "0",
                "document_id": "15661",
                "text_1": "A public health claim.",
                "text_2": "Evidence explains why the public health claim is false.",
                "label": "false",
            }
        ]
    )

    records = _write_corpus(df, tmp_path / "raw")
    eval_path = _write_eval(records, tmp_path / "eval" / "qa_eval.json", eval_limit=None)

    document = (tmp_path / "raw" / "pubhealth_000001.md").read_text(encoding="utf-8")
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    assert "Claim: A public health claim." not in document
    assert "Fact-check label: false" in document
    assert "Evidence explains why the public health claim is false." in document
    assert "Claim: A public health claim." in rows[0]["question"]
    assert rows[0]["gold_source"]["file_name"] == "pubhealth_000001.md"
    assert "Fact-check label: false" in rows[0]["gold_keywords"]
