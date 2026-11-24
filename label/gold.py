"""Phase 3 gold set: hand-label ~200 val complaints, then score any labels file against them.

Labelling is blind: the teacher's answer is never shown, so it cannot anchor the person
checking. Gold rows are saved after each complaint (data/gold/val_gold.jsonl, kept in git),
so a session can stop and resume.

Run:
  uv run python label/gold.py annotate                     # interactive, resumable
  uv run python label/gold.py score data/labels/val.jsonl  # teacher vs gold, per field
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent))
from schema import FIELDS, ISSUES, RESOLUTIONS, company_key, validate  # noqa: E402
from teacher import MAX_WORDS, truncate  # noqa: E402

ROOT = Path(__file__).parent.parent
VAL = ROOT / "data" / "processed" / "val.parquet"
GOLD = ROOT / "data" / "gold" / "val_gold.jsonl"
PER_GROUP = 20  # 10 groups x 20 = 200 rows


def sample(val=VAL, per_group=PER_GROUP):
    """(complaint_id, product_group, narrative), the same rows every run (md5 order, salted
    so the gold rows are not simply the first rows of every other md5-ordered sample)."""
    return duckdb.sql(f"""
        SELECT complaint_id, product_group, narrative FROM '{Path(val).as_posix()}'
        QUALIFY row_number() OVER (PARTITION BY product_group ORDER BY md5(complaint_id || 'gold')) <= {per_group}
        ORDER BY md5(complaint_id || 'gold')
    """).fetchall()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_labels(path, model=None):
    """complaint_id -> newest valid label in a teacher JSONL (optionally one model only)."""
    out = {}
    for r in read_jsonl(path):
        if r.get("label") is not None and (model is None or r["model"] == model):
            out[r["complaint_id"]] = r["label"]
    return out


# --- scoring -----------------------------------------------------------------

def field_match(field, pred, gold):
    if field == "company":
        return company_key(pred) == company_key(gold)
    if field == "amount_in_dispute":
        return (pred is None and gold is None) or (
            pred is not None and gold is not None and abs(pred - gold) < 0.01)
    return pred == gold


# product_group is defined as the CFPB's filed product (decided 2026-11-24), known for every
# row, so it is scored against the CFPB data directly (eval/evaluate.py), not against labels.
SCORED_FIELDS = [f for f in FIELDS if f != "product_group"]


def score(pred, gold, fields=SCORED_FIELDS):
    """pred, gold: complaint_id -> label. Returns {metric: (correct, total)}.
    Amount is scored separately for 'has amount' and 'no amount' rows (NOTES, phase 1):
    with ~78% nulls, an always-null model would otherwise look good."""
    res = defaultdict(lambda: [0, 0])
    for cid, g in gold.items():
        p = pred.get(cid)
        res["valid_label"][0] += p is not None
        res["valid_label"][1] += 1
        if p is None:
            continue
        for f in fields:
            key = f
            if f == "amount_in_dispute":
                key = f"{f} (has amount)" if g[f] is not None else f"{f} (no amount)"
            res[key][0] += field_match(f, p[f], g[f])
            res[key][1] += 1
    return {k: tuple(v) for k, v in res.items()}


# --- interactive annotation --------------------------------------------------

def ask_choice(prompt, options, allow_null):
    for i, o in enumerate(options, 1):
        print(f"    {i:>2}. {o}")
    while True:
        a = input(f"  {prompt}{' (blank = null)' if allow_null else ''}: ").strip()
        if a == "" and allow_null:
            return None
        if a.isdigit() and 1 <= int(a) <= len(options):
            return options[int(a) - 1]
        if a in options:
            return a
        print("    pick a number from the list")


def ask_amount():
    while True:
        a = input("  amount_in_dispute (number, blank = null): ").strip().replace("$", "").replace(",", "")
        if a == "":
            return None
        try:
            return round(float(a), 2)
        except ValueError:
            print("    a number like 500 or 1234.56")


def ask_bool(prompt):
    while True:
        a = input(f"  {prompt} (y / n / blank = null): ").strip().lower()
        if a in ("", "y", "n"):
            return {"": None, "y": True, "n": False}[a]


def annotate():
    rows = sample()
    done = {r["complaint_id"] for r in read_jsonl(GOLD)}
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(done)} of {len(rows)} done. Enter at any prompt is safe; Ctrl+C stops (progress is saved).")
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    for n, (cid, group, text) in enumerate(todo, len(done) + 1):
        print("\n" + "=" * 100)
        print(f"[{n}/{len(rows)}] complaint {cid}   (product_group from CFPB: {group})\n")
        print(truncate(" ".join(text.split()), MAX_WORDS))  # exactly what the teacher sees
        print()
        label = {
            "product_group": group,  # the CFPB's filed product, by definition (not asked)
            "issue": ask_choice("issue", list(ISSUES), False),
            "company": input("  company (as written, blank = null/redacted): ").strip() or None,
            "amount_in_dispute": ask_amount(),
            "contacted_company_first": ask_bool("contacted_company_first"),
            "requested_resolution": ask_choice("requested_resolution", list(RESOLUTIONS), True),
        }
        errors = validate(label)
        if errors:
            print(f"  not saved, fix: {errors}")
            continue
        note = input("  note (optional, e.g. 'ambiguous issue'): ").strip()
        with open(GOLD, "a", encoding="utf-8") as f:
            f.write(json.dumps({"complaint_id": cid, "label": label, "note": note or None}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annotate")
    s = sub.add_parser("score")
    s.add_argument("labels", type=Path, help="teacher JSONL, e.g. data/labels/val.jsonl")
    s.add_argument("--model", help="only labels from this model")
    a = p.parse_args()

    if a.cmd == "annotate":
        try:
            annotate()
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; progress saved")
        return
    gold = {r["complaint_id"]: r["label"] for r in read_jsonl(GOLD)}
    if not gold:
        sys.exit(f"no gold labels yet in {GOLD}; run: uv run python label/gold.py annotate")
    print(f"{len(gold)} gold rows vs {a.labels}\n")
    for metric, (ok, total) in score(latest_labels(a.labels, a.model), gold).items():
        print(f"  {metric:<36} {ok:>4}/{total:<4} {ok / total if total else 0:>6.1%}")


if __name__ == "__main__":
    main()
