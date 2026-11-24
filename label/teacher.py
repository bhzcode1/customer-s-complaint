"""Phase 3: label complaints with a large open model (the "teacher").

Talks to any OpenAI-compatible chat server: vLLM on a free cloud GPU for the full run,
llama.cpp / Ollama on the laptop for a smoke test. Output is append-only JSONL, one line
per complaint, so an interrupted run (cloud sessions time out) resumes where it stopped.

Run:
  uv run python label/teacher.py --reparse data/labels/val.jsonl   # after parse() changes; no model
  uv run python label/teacher.py --split val --limit 20 \
      --base-url http://localhost:8000/v1 --model <model-name>
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent))
from schema import ISSUES, JSON_SCHEMA, NOT_COMPANIES, PRODUCT_GROUPS, RESOLUTIONS, normalise, validate  # noqa: E402

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
LABELS = ROOT / "data" / "labels"
PROMPT_VERSION = "v1"   # bump when the prompt or schema changes; stored on every label
MAX_WORDS = 600         # ~5% of rows are longer (p99 ~1,500 tokens); teacher and student see the same cut

bullets = lambda d: "\n".join(f"  - {k}: {v}" for k, v in d.items())
SYSTEM = f"""You extract facts from US consumer financial complaints. Answer with one JSON object, no other text.

The text is redacted: XXXX hides names and numbers, XX/XX/XXXX and XX/XX/year> hide dates.
Amounts look like {{$500.00}}.

Fields:
- product_group: one of {", ".join(PRODUCT_GROUPS)}.
- issue: the main problem, one of:
{bullets(ISSUES)}
- company: the company the complaint is against (e.g. "Capital One", "Mohela"), as written in the text,
  or null if its name is redacted (XXXX) or not given. Never "XXXX" and never the CFPB (the complaint is
  filed with the CFPB, not against it). If several companies, the one the person blames most.
- amount_in_dispute: the sum of money the complaint is mainly about, as a number (500.0), or null if
  no amount is stated. Copy only an amount written in the text; never add amounts up or guess.
- contacted_company_first: true if the person says they contacted the company before complaining
  (called, wrote, disputed with them); false if they say they did not; null if the text does not say.
- requested_resolution: what the person asks for, one of:
{bullets(RESOLUTIONS)}
  or null if they do not ask for anything specific."""


def truncate(text, max_words=MAX_WORDS):
    words = text.split()
    return text if max_words is None or len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def chat(base_url, model, narrative, timeout=300):
    """One request; returns the raw text of the answer."""
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": truncate(narrative)}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "complaint", "schema": JSON_SCHEMA, "strict": True}},
    }
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def repair(label):
    """Fix answers whose meaning is unambiguous; returns notes on what changed.
    A redacted company ("XXXX") or the CFPB means "no company named", which is null by the
    schema's definition. Models write it anyway, and at temperature 0 a retry repeats it
    (14B teacher on val: 229 of 5,000 rows), so it is fixed here instead of re-asked."""
    notes = []
    c = label.get("company") if isinstance(label, dict) else None
    if isinstance(c, str) and (c.replace("X", "").strip() == "" or c.strip().lower() in NOT_COMPANIES):
        label["company"] = None
        notes.append(f"repaired: company {c!r} -> null")
    return notes


def parse(raw):
    """raw answer -> (label or None, notes). Notes are the validation errors when the label is
    None, and the repairs made (if any) when it is valid."""
    try:
        label = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return None, [f"invalid JSON: {e}"]
    repairs = repair(label)
    errors = validate(label)
    return (None, errors) if errors else (normalise(label), repairs)


def reparse(path):
    """Parse the saved raw answers of failed rows again (after parse() changed), appending the
    rows that are now valid. No model calls. Returns the number of rows fixed."""
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    have = {(r["complaint_id"], r["model"], r["prompt_version"]) for r in recs if r["label"] is not None}
    fixed = []
    for r in recs:
        key = (r["complaint_id"], r["model"], r["prompt_version"])
        if r["label"] is None and r["raw"] is not None and key not in have:
            label, notes = parse(r["raw"])
            if label is not None:
                fixed.append({**r, "label": label, "errors": notes, "reparsed": True,
                              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                have.add(key)
    with open(path, "a", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(fixed)


def label_one(complaint_id, narrative, ask, model, tries=2):
    """Ask up to `tries` times; keep the first valid answer, else the last attempt with its errors."""
    for attempt in range(1, tries + 1):
        try:
            raw = ask(narrative)
        except Exception as e:  # network error or server error: record it, retry
            raw, label, errors = None, None, [f"request failed: {e}"]
        else:
            label, errors = parse(raw)
        if label is not None:
            break
    return {"complaint_id": complaint_id, "label": label, "errors": errors, "raw": raw,
            "model": model, "prompt_version": PROMPT_VERSION, "attempts": attempt,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def done_ids(path, model):
    """ids with a valid label from this model and prompt version. Failed rows, and rows from
    another model or an older prompt, are labelled again (readers keep the newest valid line)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["complaint_id"] for r in map(json.loads, f)
                if r["label"] is not None and r["model"] == model and r["prompt_version"] == PROMPT_VERSION}


def run(rows, out, ask, model, workers=8):
    """rows: (complaint_id, narrative). Appends results to `out`; returns (valid, failed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out, model)
    todo = [r for r in rows if r[0] not in skip]
    valid = failed = 0
    start = time.time()
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        for i, rec in enumerate(pool.map(lambda r: label_one(*r, ask, model), todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # a killed session keeps every finished row
            valid += rec["label"] is not None
            failed += rec["label"] is None
            if i % 50 == 0 or i == len(todo):
                rate = i / max(time.time() - start, 1e-6)  # Windows clock can show 0 s elapsed
                print(f"  {i:,}/{len(todo):,}  failed {failed}  {rate:.2f} rows/s", flush=True)
    return valid, failed


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reparse", type=Path, metavar="JSONL",
                   help="only re-parse the saved raw answers of failed rows in this file (no model)")
    p.add_argument("--split", choices=["train", "val"])
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model")
    p.add_argument("--limit", type=int, help="label only the first N rows (smoke test)")
    p.add_argument("--workers", type=int, default=8, help="parallel requests; vLLM batches them")
    p.add_argument("--out", type=Path, help="output file (default data/labels/<split>.jsonl); "
                   "use a separate file for smoke tests")
    a = p.parse_args()
    if a.reparse:
        print(f"{reparse(a.reparse):,} failed rows now valid -> appended to {a.reparse}")
        return
    if not (a.split and a.model):
        p.error("--split and --model are required unless --reparse is given")

    sql = f"SELECT complaint_id, narrative FROM '{(PROCESSED / f'{a.split}.parquet').as_posix()}' ORDER BY complaint_id"
    rows = duckdb.sql(sql + (f" LIMIT {a.limit}" if a.limit else "")).fetchall()
    out = a.out or LABELS / f"{a.split}.jsonl"
    print(f"{len(rows):,} rows, {len(done_ids(out, a.model)):,} already labelled -> {out}")
    valid, failed = run(rows, out, lambda text: chat(a.base_url, a.model, text), a.model, a.workers)
    print(f"done: {valid:,} valid, {failed:,} failed (failed rows are retried on the next run)")


if __name__ == "__main__":
    main()
