import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from serve.app import create_app  # noqa: E402

GOOD = {"product_group": "credit_card", "issue": "fees_interest", "company": "Citi",
        "amount_in_dispute": 35.0, "contacted_company_first": True, "requested_resolution": "refund"}


class Fake:
    name = "fake"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def extract(self, text):
        self.calls += 1
        label = self.answers.pop(0)
        return label, ([] if label else ["invalid JSON"]), "raw", 0.01


def test_extract_logs_request(tmp_path):
    log = tmp_path / "req.jsonl"
    client = TestClient(create_app(Fake([GOOD]), log))
    r = client.post("/extract", json={"narrative": "a b c", "complaint_id": "1", "received": "2026-01-02"})
    assert r.status_code == 200 and r.json()["valid"] and r.json()["label"] == GOOD
    rec = json.loads(log.read_text())
    assert rec["input_words"] == 3 and rec["received"] == "2026-01-02" and rec["attempts"] == 1
    assert client.get("/health").json() == {"status": "ok", "model": "fake"}


def test_invalid_output_retried_once_then_null(tmp_path):
    fake = Fake([None, None])
    r = TestClient(create_app(fake, tmp_path / "l.jsonl")).post("/extract", json={"narrative": "x"})
    assert r.json() == {"label": None, "valid": False, "notes": ["invalid JSON"], "latency_ms": r.json()["latency_ms"]}
    assert fake.calls == 2


def test_retry_can_recover(tmp_path):
    r = TestClient(create_app(Fake([None, GOOD]), tmp_path / "l.jsonl")).post("/extract", json={"narrative": "x"})
    assert r.json()["valid"]
