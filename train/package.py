"""Phase 7: package the student for CPU inference and measure it.

1. Merge the LoRA adapter into the base weights (one self-contained model directory).
2. Compare fp32 with dynamic int8 quantisation of the linear layers (PyTorch, no extra
   runtime): on-disk/in-memory size, seconds per complaint, and agreement on the same rows.
3. Report hardware time per 1,000 complaints, the unit to compare with the teacher.

Result for the 135M pilot (2026-11-24): int8 dynamic quantisation broke the model (0/30 valid
outputs, also with lm_head kept in fp32) and was slower than fp32 on this CPU, so the service
runs fp32. torch.ao dynamic quantisation is also deprecated; for a larger student use a dedicated
runtime (GGUF Q8/Q4 with llama.cpp) and re-check validity before switching.

Run: uv run --group ml python train/package.py --adapter models/pilot/adapter --rows 30
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import duckdb
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "label"))
from gold import GOLD, read_jsonl  # noqa: E402
from predict import Student  # noqa: E402


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--rows", type=int, default=30, help="gold rows timed per variant")
    a = p.parse_args()

    merged = a.adapter.parent / "merged"
    fp32 = Student(a.adapter)
    fp32.model.save_pretrained(merged)
    fp32.tok.save_pretrained(merged)
    print(f"merged model -> {merged}")

    ids = {r["complaint_id"] for r in read_jsonl(GOLD)}
    rows = [t for c, t in duckdb.sql(f"SELECT complaint_id, narrative FROM "
                                     f"'{(ROOT / 'data/processed/val.parquet').as_posix()}' ORDER BY md5(complaint_id)").fetchall()
            if c in ids][:a.rows]
    results = {}
    for name, student in (("fp32", fp32), ("int8", Student(merged, int8=True))):
        student.extract(rows[0])  # warm-up
        t, labels = time.time(), []
        for text in rows:
            labels.append(student.extract(text)[0])
        secs = (time.time() - t) / len(rows)
        results[name] = {"size_mb": round(size_mb(student.model), 1), "seconds_per_complaint": round(secs, 2),
                         "cpu_hours_per_1000": round(secs * 1000 / 3600, 2),
                         "valid": sum(x is not None for x in labels), "labels": labels}
        print(name, {k: v for k, v in results[name].items() if k != "labels"}, flush=True)
    same = sum(x == y for x, y in zip(results["fp32"]["labels"], results["int8"]["labels"]))
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "labels"} for k, v in results.items()}
    summary["int8_identical_outputs"] = f"{same}/{len(rows)}"
    summary["hardware"] = "Intel Core i3-7100U, 2 cores, PyTorch CPU, batch size 1, greedy decoding"
    (a.adapter.parent / "package.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
