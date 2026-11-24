"""Tests for phases 5 and 9 that need no model and no data download."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for d in ("train", "monitor", "label"):
    sys.path.insert(0, str(ROOT / d))
import drift  # noqa: E402
import sft_data  # noqa: E402
from schema import FIELDS  # noqa: E402

LABEL = {"requested_resolution": "refund", "issue": "fees_interest", "product_group": "credit_card",
         "company": "Citi", "amount_in_dispute": 35.0, "contacted_company_first": True}


def test_target_json_fixed_key_order():
    out = sft_data.target_json(LABEL)
    assert list(json.loads(out)) == FIELDS
    assert out.startswith('{"product_group": "credit_card"')


def test_messages_shape():
    m = sft_data.messages("some   complaint\ntext", LABEL)
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[1]["content"] == "some complaint text"
    assert len(sft_data.messages("x")) == 2  # inference: no assistant turn


def test_psi():
    a = {"x": 0.5, "y": 0.5}
    assert drift.psi(a, a) == 0
    assert drift.psi({"x": 0.9, "y": 0.1}, a) > 0.25  # a big shift crosses the retrain band
    assert drift.psi({"x": 0.55, "y": 0.45}, a) < 0.1


def rec(month, product, valid=True, cid=None, words=100):
    label = {**LABEL, "product_group": product} if valid else None
    return {"time": "2026-11-24T00:00:00+00:00", "received": f"{month}-15", "complaint_id": cid,
            "input_words": words, "latency_ms": 1000, "valid": valid, "label": label}


def test_monthly_and_alerts():
    ref = {"product_mix": {"credit_reporting": 0.9, "credit_card": 0.1}, "words_median": 100}
    stable = [rec("2026-01", "credit_reporting", cid=f"a{i}") for i in range(36)] + \
             [rec("2026-01", "credit_card", cid=f"b{i}") for i in range(4)]
    shifted = [rec("2026-02", "credit_card", cid=f"c{i}", words=200) for i in range(38)] + \
              [rec("2026-02", "credit_card", valid=False, cid=f"d{i}") for i in range(2)]
    truth = {r["complaint_id"]: r["label"]["product_group"] for r in stable + shifted if r["valid"]}
    stats = drift.monthly(stable + shifted, truth)
    assert stats["2026-01"]["product_accuracy"] == 1.0 and stats["2026-02"]["valid_rate"] == 0.95
    found = drift.alerts(stats, ref, ref_accuracy=0.8)
    levels = {(m, lvl) for m, lvl, _ in found}
    assert ("2026-02", "RETRAIN") in levels and not any(m == "2026-01" for m, _, _ in found)
    assert any("median length" in msg for m, _, msg in found if m == "2026-02")
