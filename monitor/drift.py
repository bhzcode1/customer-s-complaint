"""Phase 9: drift monitoring over the service's request log.

Reads data/logs/requests.jsonl (written by serve/app.py), groups requests by the month the
complaint was received, and compares each month with a reference period:
- product mix (the model's product_group) vs the natural mix the model was built for: PSI,
- input length (words) vs reference median and p90,
- valid-output rate, null rate per field,
- product accuracy against the CFPB's own product (known for replay rows; in production it
  would arrive later, so it is a lagging check).

Alerts follow fixed rules (RULES), written down before looking at the replay results.

Run:
  uv run python monitor/drift.py --reference      # once: compute reference.json from raw data
  uv run --group ml python monitor/drift.py       # report -> reports/drift.md + drift.png
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "label"))
sys.path.insert(0, str(ROOT / "data"))
from schema import PRODUCT_GROUPS  # noqa: E402

LOG = ROOT / "data" / "logs" / "requests.jsonl"
REFERENCE = Path(__file__).parent / "reference.json"
REPORTS = ROOT / "reports"
REF_PERIOD = ("2023-01-01", "2025-07-01")  # recent natural traffic before the val/replay periods

# Alert rules, fixed in advance. PSI bands are the usual ones: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
RULES = {
    "psi_product_major": 0.25,     # product mix shifted a lot -> retrain on recent data
    "psi_product_moderate": 0.10,  # watch
    "valid_rate_min": 0.95,        # model output broken -> investigate before anything else
    "length_median_change": 0.25,  # median input length moved >25% -> check truncation and cost
    "accuracy_drop": 0.10,         # product accuracy >10 points below the reference -> retrain
    "min_rows": 30,                # months with fewer requests are reported but never alert
}


def psi(actual, expected, eps=1e-4):
    """Population stability index between two distributions over the same categories."""
    return sum((a - e) * math.log((a + eps) / (e + eps))
               for a, e in ((actual.get(k, 0.0), expected.get(k, 0.0)) for k in set(actual) | set(expected)))


def shares(counter):
    n = sum(counter.values())
    return {k: v / n for k, v in counter.items()} if n else {}


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0


def build_reference():
    """Natural product mix and length of the reference period, from the raw narratives."""
    import prep  # data/prep.py: the same product grouping
    raw = ROOT / "data" / "raw" / "narratives.parquet"
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE g AS SELECT unnest(?::VARCHAR[]) product, unnest(?::VARCHAR[]) grp",
                [list(prep.GROUPS), list(prep.GROUPS.values())])
    mix = dict(con.execute(f"""
        SELECT grp, count(*) FROM '{raw.as_posix()}' r JOIN g USING (product)
        WHERE date_received >= '{REF_PERIOD[0]}' AND date_received < '{REF_PERIOD[1]}' GROUP BY grp""").fetchall())
    q = con.execute(fr"""
        SELECT quantile_cont(len(regexp_split_to_array(trim(narrative), '\s+')), [0.5, 0.9])
        FROM '{raw.as_posix()}' WHERE date_received >= '{REF_PERIOD[0]}' AND date_received < '{REF_PERIOD[1]}'
    """).fetchone()[0]
    ref = {"period": REF_PERIOD, "rows": sum(mix.values()), "product_mix": shares(Counter(mix)),
           "words_median": q[0], "words_p90": q[1]}
    REFERENCE.write_text(json.dumps(ref, indent=2))
    return ref


def monthly(records, truth):
    """Per received-month metrics from request-log records. truth: complaint_id -> CFPB product."""
    by_month = defaultdict(list)
    for r in records:
        by_month[(r.get("received") or r["time"])[:7]].append(r)
    out = {}
    for month, rs in sorted(by_month.items()):
        valid = [r for r in rs if r["valid"]]
        labelled = [r for r in valid if r.get("complaint_id") in truth]
        out[month] = {
            "rows": len(rs),
            "valid_rate": len(valid) / len(rs),
            "product_mix": shares(Counter(r["label"]["product_group"] for r in valid)),
            "true_mix": shares(Counter(truth[r["complaint_id"]] for r in rs if r.get("complaint_id") in truth)),
            "words_median": pct([r["input_words"] for r in rs], 0.5),
            "latency_ms_median": pct([r["latency_ms"] for r in rs], 0.5),
            "null_rate": {f: sum(r["label"][f] is None for r in valid) / max(1, len(valid))
                          for f in ("company", "amount_in_dispute", "contacted_company_first", "requested_resolution")},
            "product_accuracy": (sum(r["label"]["product_group"] == truth[r["complaint_id"]] for r in labelled)
                                 / len(labelled)) if labelled else None,
        }
    return out


def alerts(month_stats, ref, ref_accuracy=None):
    """[(month, level, message)] from RULES."""
    out = []
    for m, s in month_stats.items():
        if s["rows"] < RULES["min_rows"]:
            continue
        p = psi(s["product_mix"], ref["product_mix"])
        s["psi_product"] = p
        if p > RULES["psi_product_major"]:
            out.append((m, "RETRAIN", f"product mix PSI {p:.2f} > {RULES['psi_product_major']}"))
        elif p > RULES["psi_product_moderate"]:
            out.append((m, "watch", f"product mix PSI {p:.2f}"))
        if s["valid_rate"] < RULES["valid_rate_min"]:
            out.append((m, "INVESTIGATE", f"valid output {s['valid_rate']:.1%} < {RULES['valid_rate_min']:.0%}"))
        change = s["words_median"] / ref["words_median"] - 1
        if abs(change) > RULES["length_median_change"]:
            out.append((m, "watch", f"median length {s['words_median']} words ({change:+.0%} vs reference)"))
        if ref_accuracy is not None and s["product_accuracy"] is not None \
                and s["product_accuracy"] < ref_accuracy - RULES["accuracy_drop"]:
            out.append((m, "RETRAIN", f"product accuracy {s['product_accuracy']:.1%} vs {ref_accuracy:.1%} on val"))
    return out


def report(stats, ref, found, ref_accuracy, out_dir=REPORTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    months = list(stats)
    show = sorted(PRODUCT_GROUPS, key=lambda g: -ref["product_mix"].get(g, 0))[:5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for g in show:
        line, = axes[0].plot(months, [stats[m]["product_mix"].get(g, 0) for m in months], marker="o", label=g)
        axes[0].axhline(ref["product_mix"].get(g, 0), color=line.get_color(), ls=":", lw=1)
    axes[0].set_title("Product mix per month (model output); dotted = reference 2023-mid 2025")
    axes[0].set_ylabel("share of requests"); axes[0].legend(fontsize=8)
    axes[1].plot(months, [stats[m].get("psi_product", float("nan")) for m in months], marker="o", color="black")
    axes[1].axhline(RULES["psi_product_major"], color="red", ls="--", label="retrain threshold")
    axes[1].axhline(RULES["psi_product_moderate"], color="orange", ls="--", label="watch threshold")
    axes[1].set_title("Product-mix PSI vs reference"); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out_dir / "drift.png", dpi=120); plt.close(fig)

    lines = ["# Replay drift report", "",
             f"Reference: natural traffic {ref['period'][0]} to {ref['period'][1]} ({ref['rows']:,} complaints), "
             f"median {ref['words_median']:.0f} words." + (f" Product accuracy on val: {ref_accuracy:.1%}." if ref_accuracy else ""),
             "", "![drift](drift.png)", "",
             "| Month | Rows | Valid | PSI product | Product acc. | Median words | Latency p50 (ms) | Null: company / amount / contacted / resolution |",
             "|---|---|---|---|---|---|---|---|"]
    for m, s in stats.items():
        n = s["null_rate"]
        acc = f"{s['product_accuracy']:.1%}" if s["product_accuracy"] is not None else "-"
        lines.append(f"| {m} | {s['rows']} | {s['valid_rate']:.1%} | {s.get('psi_product', float('nan')):.2f} | {acc} | "
                     f"{s['words_median']} | {s['latency_ms_median']} | {n['company']:.0%} / {n['amount_in_dispute']:.0%} / "
                     f"{n['contacted_company_first']:.0%} / {n['requested_resolution']:.0%} |")
    lines += ["", "## Alerts", ""] + ([f"- **{lvl}** {m}: {msg}" for m, lvl, msg in found] or ["- none"])
    lines += ["", "## Rules", "", *[f"- `{k}` = {v}" for k, v in RULES.items()]]
    (out_dir / "drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "drift.md"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", action="store_true", help="(re)build reference.json from raw data")
    p.add_argument("--log", type=Path, default=LOG)
    p.add_argument("--ref-accuracy", type=float, help="product accuracy on val, for the accuracy rule")
    a = p.parse_args()
    if a.reference:
        print(json.dumps(build_reference(), indent=2))
        return
    ref = json.loads(REFERENCE.read_text())
    records = [json.loads(line) for line in open(a.log, encoding="utf-8")]
    truth = dict(duckdb.sql(f"SELECT complaint_id, product_group FROM "
                            f"'{(ROOT / 'data' / 'processed' / 'replay.parquet').as_posix()}'").fetchall())
    stats = monthly(records, truth)
    found = alerts(stats, ref, a.ref_accuracy)
    print(report(stats, ref, found, a.ref_accuracy))
    for m, lvl, msg in found:
        print(f"  {lvl:<12} {m}  {msg}")


if __name__ == "__main__":
    main()
