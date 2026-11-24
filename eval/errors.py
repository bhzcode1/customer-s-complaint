"""Phase 6: error analysis of a predictions file against the gold set.

Writes reports/errors_<name>.md: the most common confusions per field and, for each field,
mistakes with the start of the complaint, so they can be read and grouped by cause
(bad reference label, ambiguous text, truncation, model gap).

Run: uv run python eval/errors.py data/predictions/student_pilot_val_gold.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "label"))
from gold import GOLD, field_match, latest_labels, read_jsonl, sample  # noqa: E402

FIELDS = ["issue", "requested_resolution", "company", "amount_in_dispute", "contacted_company_first"]


def confusions(pred, gold, field):
    return Counter((g[field], pred[c][field]) for c, g in gold.items()
                   if c in pred and not field_match(field, pred[c][field], g[field]))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pred", type=Path)
    p.add_argument("--examples", type=int, default=8, help="mistakes shown per field")
    a = p.parse_args()

    pred = latest_labels(a.pred)
    gold_rows = read_jsonl(GOLD)
    gold = {r["complaint_id"]: r["label"] for r in gold_rows}
    notes = {r["complaint_id"]: r.get("note") for r in gold_rows}
    text = {cid: t for cid, _, t in sample()}
    missing = [c for c in gold if c not in pred]

    lines = [f"# Error analysis: {a.pred.name}", "",
             f"{len(gold)} gold rows; {len(missing)} without a valid prediction.", ""]
    for f in FIELDS:
        conf = confusions(pred, gold, f)
        n_wrong = sum(conf.values())
        lines += [f"## {f}: {n_wrong} wrong of {len(gold) - len(missing)}", "",
                  "| gold | predicted | n |", "|---|---|---|"]
        lines += [f"| {g} | {p_} | {n} |" for (g, p_), n in conf.most_common(10)]
        lines += ["", "Examples:", ""]
        shown = 0
        for c, g in gold.items():
            if c in pred and not field_match(f, pred[c][f], g[f]) and shown < a.examples:
                snippet = " ".join(text[c].split())[:300]
                note = f" *(gold note: {notes[c]})*" if notes.get(c) else ""
                lines.append(f"- `{c}` gold **{g[f]}**, predicted **{pred[c][f]}**{note}: {snippet}...")
                shown += 1
        lines.append("")
    out = ROOT / "reports" / f"errors_{a.pred.stem}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
