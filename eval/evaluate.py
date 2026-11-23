"""Phase 4: score a predictions file, one table for every model and baseline.

Predictions and references are JSONL lines with `complaint_id` and `label` (the teacher's
output format). Reported:
- product_group against the CFPB's own product (true for every row; no labels needed),
- every field against a reference: the gold set (hand labels) and/or the teacher's labels,
- macro-F1 for issue and requested_resolution (accuracy hides weak small classes),
- amount_in_dispute split into "has amount" / "no amount" rows.

Run:
  uv run python eval/evaluate.py data/predictions/rules_val.jsonl
  uv run python eval/evaluate.py data/predictions/rules_val.jsonl --ref data/labels/val.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "label"))
from gold import GOLD, latest_labels, read_jsonl, score  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
F1_FIELDS = ["issue", "requested_resolution"]


def macro_f1(pairs):
    """pairs: (pred, gold). Mean F1 over the classes that occur in gold."""
    classes = {g for _, g in pairs}
    tp, fp, fn = Counter(), Counter(), Counter()
    for p, g in pairs:
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1 = [2 * tp[c] / (2 * tp[c] + fp[c] + fn[c]) for c in classes]
    return sum(f1) / len(f1) if f1 else 0.0


def report(pred, ref):
    """Rows of (metric, value, n) for predictions vs a reference."""
    rows = [(m, ok / n if n else 0.0, n) for m, (ok, n) in score(pred, ref).items()]
    for f in F1_FIELDS:
        pairs = [(pred[c][f], g[f]) for c, g in ref.items() if c in pred]
        rows.append((f"{f} macro-F1", macro_f1(pairs), len(pairs)))
    return rows


def cfpb_products(split):
    sql = f"SELECT complaint_id, product_group FROM '{(PROCESSED / f'{split}.parquet').as_posix()}'"
    return dict(duckdb.sql(sql).fetchall())


def print_rows(title, rows):
    print(f"\n{title}")
    for metric, value, n in rows:
        print(f"  {metric:<36} {value:>7.1%}   n={n:,}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pred", type=Path, help="predictions JSONL")
    p.add_argument("--split", default="val", choices=["val", "replay"])
    p.add_argument("--ref", type=Path, help="reference labels JSONL, e.g. the teacher's data/labels/val.jsonl")
    a = p.parse_args()

    pred = latest_labels(a.pred)
    truth = cfpb_products(a.split)
    pairs = [(pred[c]["product_group"], g) for c, g in truth.items() if c in pred]
    acc = sum(p_ == g for p_, g in pairs) / len(pairs) if pairs else 0.0
    print(f"{a.pred}: {len(pred):,} valid predictions for {len(truth):,} {a.split} rows")
    print_rows("vs CFPB product (all rows)", [("product_group", acc, len(pairs)),
                                              ("product_group macro-F1", macro_f1(pairs), len(pairs))])

    gold = {r["complaint_id"]: r["label"] for r in read_jsonl(GOLD)}
    if gold and a.split == "val":
        print_rows(f"vs gold set ({len(gold)} hand-labelled rows)", report(pred, gold))
    if a.ref:
        print_rows(f"vs {a.ref}", report(pred, latest_labels(a.ref)))


if __name__ == "__main__":
    main()
