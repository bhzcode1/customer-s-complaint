"""Phase 5-6: run the fine-tuned student over a split and write predictions JSONL.

Same output format as the teacher (complaint_id, label, errors, raw, model), so eval/evaluate.py
and label/gold.py score it unchanged. Answers go through the teacher's parse(): the same
repair (redacted company -> null) and the same schema validation.

Run:
  uv run --group ml python train/predict.py --adapter models/pilot/adapter --split val --gold-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "label"))
from sft_data import messages  # noqa: E402
from gold import GOLD, read_jsonl  # noqa: E402
from teacher import parse  # noqa: E402

PREDICTIONS = ROOT / "data" / "predictions"
MAX_NEW_TOKENS = 120  # a full label is ~60-80 tokens


class Student:
    """Base model + LoRA adapter (or a merged model directory), greedy decoding."""

    def __init__(self, path, threads=None, int8=False):
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        path = Path(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        if (path / "adapter_config.json").exists():
            self.model = AutoPeftModelForCausalLM.from_pretrained(path, dtype=torch.float32).merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        self.device = "cuda" if torch.cuda.is_available() and not int8 else "cpu"
        self.model.to(self.device)
        if int8:  # phase 7: dynamic int8 quantisation of the linear layers, CPU only
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
        self.name = str(path)

    @torch.inference_mode()
    def raw(self, narrative):
        prompt = self.tok.apply_chat_template(messages(narrative), tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def extract(self, narrative):
        """-> (label or None, notes, raw text, input tokens, seconds)."""
        t = time.time()
        raw = self.raw(narrative)
        label, notes = parse(raw)
        return label, notes, raw, time.time() - t


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", type=Path, required=True, help="adapter or merged model directory")
    p.add_argument("--split", default="val", choices=["val", "replay"])
    p.add_argument("--gold-only", action="store_true", help="only the 200 gold rows")
    p.add_argument("--limit", type=int)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    rows = duckdb.sql(f"SELECT complaint_id, narrative FROM "
                      f"'{(ROOT / 'data' / 'processed' / f'{a.split}.parquet').as_posix()}' "
                      f"ORDER BY md5(complaint_id)").fetchall()
    if a.gold_only:
        ids = {r["complaint_id"] for r in read_jsonl(GOLD)}
        rows = [r for r in rows if r[0] in ids]
    rows = rows[:a.limit] if a.limit else rows
    out = a.out or PREDICTIONS / f"student_{a.adapter.parent.name}_{a.split}{'_gold' if a.gold_only else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["complaint_id"] for r in read_jsonl(out)} if out.exists() else set()
    student = Student(a.adapter, int8=a.int8)
    print(f"{len(rows):,} rows ({len(done):,} done) -> {out}", flush=True)
    secs = []
    with open(out, "a", encoding="utf-8") as f:
        for i, (cid, text) in enumerate(r for r in rows if r[0] not in done):
            label, notes, raw, s = student.extract(text)
            secs.append(s)
            f.write(json.dumps({"complaint_id": cid, "label": label, "errors": notes, "raw": raw,
                                "model": student.name, "seconds": round(s, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}  {sum(secs) / len(secs):.2f} s/row", flush=True)
    if secs:
        print(f"done: {len(secs)} rows, mean {sum(secs) / len(secs):.2f} s/row")


if __name__ == "__main__":
    main()
