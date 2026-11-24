"""Phase 9: replay 2026 complaints through the service as if they were live traffic.

Rows are sent in the order they were received. Every request goes through serve/app.py
(in-process by default, or to a running server with --url), so the request log is written by
the same code path production would use.

On this laptop's CPU a request takes a few seconds, so the default replays a fixed random
sample of --per-month rows from each month (md5 order: same rows every run) instead of all
10,000. Random within a month keeps each month's real product mix, which is what drift
monitoring measures.

Run:
  MODEL_DIR=models/pilot/adapter uv run --group ml python monitor/replay.py --per-month 150
  uv run --group ml python monitor/drift.py
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def sample(per_month=None):
    """(complaint_id, date, narrative) in received order; per_month rows per month if given."""
    limit = f"QUALIFY row_number() OVER (PARTITION BY strftime(date, '%Y-%m') ORDER BY md5(complaint_id)) <= {per_month}" \
        if per_month else ""
    return duckdb.sql(f"""
        SELECT complaint_id, date::VARCHAR, narrative FROM '{(ROOT / 'data' / 'processed' / 'replay.parquet').as_posix()}'
        {limit} ORDER BY date, complaint_id""").fetchall()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-month", type=int, default=150)
    p.add_argument("--url", help="base URL of a running service; default: in-process app")
    a = p.parse_args()

    rows = sample(a.per_month)
    from serve.app import LOG
    done = set()
    if LOG.exists():
        done = {json.loads(line).get("complaint_id") for line in open(LOG, encoding="utf-8")}
    todo = [r for r in rows if r[0] not in done]
    print(f"{len(rows):,} replay rows, {len(rows) - len(todo):,} already logged -> {LOG}", flush=True)

    if a.url:
        def post(body):
            req = urllib.request.Request(f"{a.url.rstrip('/')}/extract", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
    else:
        from fastapi.testclient import TestClient
        from serve.app import app
        client = TestClient(app)
        post = lambda body: client.post("/extract", json=body).json()  # noqa: E731

    start = time.time()
    for i, (cid, date, text) in enumerate(todo, 1):
        post({"narrative": text, "complaint_id": cid, "received": date})
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  {date}  {(time.time() - start) / i:.1f} s/request", flush=True)


if __name__ == "__main__":
    main()
