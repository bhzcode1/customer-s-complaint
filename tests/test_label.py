import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "label"))
import prep  # noqa: E402
import schema  # noqa: E402
import teacher  # noqa: E402

GOOD = {"product_group": "debt_collection", "issue": "debt_not_owed", "company": "  Capital   One ",
        "amount_in_dispute": 500, "contacted_company_first": True,
        "requested_resolution": "stop_contact"}


def test_product_groups_match_prep():
    assert set(schema.PRODUCT_GROUPS) == set(prep.GROUPS.values())


def test_validate_accepts_good_label():
    assert schema.validate(GOOD) == []
    assert schema.validate({**GOOD, "amount_in_dispute": None, "requested_resolution": None,
                            "contacted_company_first": None, "company": None}) == []


def test_validate_rejects_bad_labels():
    assert schema.validate([]) == ["not a JSON object"]
    errors = schema.validate({**GOOD, "issue": "made_up", "amount_in_dispute": -5,
                              "company": "", "contacted_company_first": "yes", "extra": 1})
    assert len(errors) == 5
    assert schema.validate({k: v for k, v in GOOD.items() if k != "issue"}) == ["missing issue"]
    assert schema.validate({**GOOD, "amount_in_dispute": True}) != []  # bool is not a number
    for company in ["XXXX", "XXXX XXXX", "CFPB"]:
        assert schema.validate({**GOOD, "company": company}) != []


def test_parse_normalises():
    label, errors = teacher.parse(json.dumps(GOOD))
    assert errors == [] and label["company"] == "Capital One" and label["amount_in_dispute"] == 500.0
    assert teacher.parse("not json")[0] is None
    assert teacher.parse("not json")[1][0].startswith("invalid JSON")


def test_run_retries_and_resumes(tmp_path):
    out = tmp_path / "val.jsonl"
    answers = {"a": [json.dumps(GOOD)], "b": ["{oops", json.dumps(GOOD)], "c": ["{oops", "{oops"]}
    calls = []

    def ask(text):
        calls.append(text)
        return answers[text].pop(0)

    rows = [("1", "a"), ("2", "b"), ("3", "c")]
    assert teacher.run(rows, out, ask, "fake", workers=1) == (2, 1)
    recs = {r["complaint_id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert recs["2"]["attempts"] == 2 and recs["2"]["label"] is not None
    assert recs["3"]["label"] is None and recs["3"]["raw"] == "{oops"
    assert recs["1"]["model"] == "fake" and recs["1"]["prompt_version"] == teacher.PROMPT_VERSION

    # Second run: only the failed row is asked again.
    answers["c"] = [json.dumps(GOOD)]
    calls.clear()
    assert teacher.run(rows, out, ask, "fake", workers=1) == (1, 0)
    assert calls == ["c"]

    # A different model does not reuse the first model's labels.
    answers.update(a=[json.dumps(GOOD)], b=[json.dumps(GOOD)], c=[json.dumps(GOOD)])
    assert teacher.run(rows, out, ask, "other", workers=1) == (3, 0)


def test_request_failure_is_recorded(tmp_path):
    def ask(text):
        raise ConnectionError("server down")

    rec = teacher.label_one("1", "text", ask, "fake")
    assert rec["label"] is None and "server down" in rec["errors"][0] and rec["raw"] is None


import duckdb  # noqa: E402
import gold  # noqa: E402


def test_company_key():
    k = schema.company_key
    assert k("Capital One, N.A.") == k("capital one") == k("Capital One NA") == "capital one"
    assert k("The Bank of New York Mellon Corp.") == "bank of new york mellon"
    assert k("Wells Fargo & Company") == "wells fargo"
    assert k(None) is None and k("Inc.") is None
    assert k("Navient") != k("Nelnet")


def test_score_splits_amount_and_counts_missing():
    g = {"1": {**GOOD, "company": "Capital One"}, "2": {**GOOD, "amount_in_dispute": None}, "3": GOOD}
    p = {"1": {**GOOD, "company": "CAPITAL ONE, N.A.", "issue": "other"},
         "2": {**GOOD, "amount_in_dispute": None}}  # "3" has no valid teacher label
    s = gold.score(p, g)
    assert s["valid_label"] == (2, 3)
    assert s["company"] == (2, 2) and s["issue"] == (1, 2)
    assert s["amount_in_dispute (has amount)"] == (1, 1)
    assert s["amount_in_dispute (no amount)"] == (1, 1)


def test_gold_sample_is_balanced_and_stable(tmp_path):
    path = tmp_path / "val.parquet"
    duckdb.sql(f"""COPY (SELECT i::VARCHAR AS complaint_id, ['a', 'b'][i % 2 + 1] AS product_group,
                   'text ' || i AS narrative FROM range(100) t(i)) TO '{path.as_posix()}' (FORMAT parquet)""")
    rows = gold.sample(path, per_group=5)
    assert len(rows) == 10 and sum(r[1] == "a" for r in rows) == 5
    assert rows == gold.sample(path, per_group=5)


def test_latest_labels_keeps_newest_valid(tmp_path):
    path = tmp_path / "l.jsonl"
    lines = [{"complaint_id": "1", "label": GOOD, "model": "m"},
             {"complaint_id": "1", "label": {**GOOD, "issue": "other"}, "model": "m"},
             {"complaint_id": "1", "label": None, "model": "m"},
             {"complaint_id": "2", "label": GOOD, "model": "x"}]
    path.write_text("\n".join(json.dumps(r) for r in lines))
    assert gold.latest_labels(path, "m") == {"1": {**GOOD, "issue": "other"}}
