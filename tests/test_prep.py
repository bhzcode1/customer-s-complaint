import random
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import prep  # noqa: E402

rng = random.Random(0)
VOCAB = [f"w{i}" for i in range(500)]


def text(n=80):
    """Random words: two calls are never near-duplicates of each other."""
    return " ".join(rng.choices(VOCAB, k=n))


def write_raw(path, rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (complaint_id VARCHAR, date_received DATE, product VARCHAR, narrative VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT parquet)")


def read(out, split):
    return duckdb.sql(f"SELECT * FROM '{(out / f'{split}.parquet').as_posix()}'").fetchall()


def test_near_duplicates_drops_later_copy_only():
    a, b = text(), text()
    words = a.split()
    words[40] = "changed"  # one word edited, like a template with a name swapped in
    edited = " ".join(words)
    assert prep.near_duplicates([("1", a), ("2", b), ("3", edited)]) == {"3"}


def test_build(tmp_path):
    template = text()
    rows = [
        # loose duplicate across splits: only the train (earliest) copy survives
        ("t1", date(2020, 1, 1), "Credit reporting", template),
        ("r1", date(2026, 2, 1), "Credit reporting or other personal consumer reports",
         "  " + template.upper().replace("W1 ", "W1 XXXX ")),
        ("nodate", None, "Mortgage", text()),
        ("v1", date(2025, 7, 1), "Mortgage", text()),   # first day of val
        ("t2", date(2025, 6, 30), "Mortgage", text()),  # last day of train
    ]
    rows += [(f"d{i}", date(2019, 1, i + 1), "Debt collection", text()) for i in range(5)]
    raw, out = tmp_path / "raw.parquet", tmp_path / "out"
    write_raw(raw, rows)

    counts = prep.build(duckdb.connect(), raw, out, per_group={"train": 2, "val": 2}, replay_rows=5)

    train, val, replay = (read(out, s) for s in prep.SPLITS)
    ids = {s: {r[0] for r in rows} for s, rows in zip(prep.SPLITS, (train, val, replay))}
    assert counts["after_loose_dedupe"] == len(rows) - 2  # no-date row and r1 gone
    assert "t1" in ids["train"] and "r1" not in ids["replay"]
    assert "t2" in ids["train"] and ids["val"] == {"v1"}
    assert sum(r[0].startswith("d") for r in train) == 2  # capped per group
    assert {r[3] for r in train} == {"credit_reporting", "mortgage", "debt_collection"}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["replay"])


def test_unmapped_product_raises(tmp_path):
    raw = tmp_path / "raw.parquet"
    write_raw(raw, [("x", date(2020, 1, 1), "Brand new product", text())])
    with pytest.raises(ValueError, match="Brand new product"):
        prep.build(duckdb.connect(), raw, tmp_path / "out")
