"""Phase 5: LoRA fine-tuning of a small instruct model on teacher labels (CPU or GPU).

Loss is computed on the assistant's JSON only (the instruction and complaint are context, not
something to learn to write). Seed, data, and all settings are saved next to the adapter so a
run can be repeated.

Run (pilot on this laptop's CPU):
  uv run --group ml python train/train_lora.py --model HuggingFaceTB/SmolLM2-360M-Instruct \
      --labels data/labels/val_v1.jsonl --split val --n-train 1500 --out models/pilot
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from sft_data import build, gold_ids  # noqa: E402

ROOT = Path(__file__).parent.parent


def encode(tok, msgs, max_len):
    """input_ids and labels; labels are -100 (ignored) everywhere except the assistant reply.
    If the example is longer than max_len tokens, the complaint is shortened (never the target):
    cutting the end of the token sequence instead would drop the JSON, leave no token to learn
    from, and give a NaN loss (seen in the first benchmark)."""
    words = msgs[1]["content"].split()
    while True:
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(f_ids) <= max_len or len(words) < 20:
            break
        words = words[:int(len(words) * 0.85)]
        msgs = [msgs[0], {**msgs[1], "content": " ".join(words) + " ..."}, msgs[2]]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return torch.tensor(f_ids), torch.tensor(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--labels", type=Path, required=True, help="teacher labels JSONL")
    p.add_argument("--split", default="val", help="split the labels belong to")
    p.add_argument("--n-train", type=int, default=None, help="use the first N examples (md5 order)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=None, help="stop (and save) after this long")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    data = build(a.labels, a.split, exclude=gold_ids())  # gold rows never trained on
    if a.n_train:
        data = data[:a.n_train]
    print(f"{len(data):,} training examples from {a.labels}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW((q for q in model.parameters() if q.requires_grad), lr=a.lr)

    encoded = [encode(tok, msgs, a.max_len) for _, msgs in data]
    total_steps = int(len(encoded) * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup 3%, then linear decay
        opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // a.grad_accum * 0.03))
        * max(0.0, 1 - s / max(1, total_steps // a.grad_accum)))

    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "train_log.jsonl", "w", encoding="utf-8")
    model.train()
    start, running, step = time.time(), [], 0
    order = list(range(len(encoded)))
    for i in range(total_steps):
        if i % len(order) == 0:
            random.shuffle(order)
        ids, labels = encoded[order[i % len(order)]]
        loss = model(input_ids=ids[None], labels=labels[None]).loss
        if not torch.isfinite(loss):  # safety net: never let one bad example poison the weights
            print(f"skipping example {order[i % len(order)]}: loss {loss.item()}", flush=True)
            continue
        (loss / a.grad_accum).backward()
        running.append(loss.item())
        if (i + 1) % a.grad_accum == 0 or i + 1 == total_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            mins = (time.time() - start) / 60
            rec = {"step": step, "examples": i + 1, "loss": sum(running) / max(1, len(running)),
                   "minutes": round(mins, 1)}
            running = []
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 5 == 0:
                print(rec, flush=True)
            if a.max_minutes and mins > a.max_minutes:
                print(f"time budget reached after {i + 1} examples", flush=True)
                break

    model.save_pretrained(a.out / "adapter")
    tok.save_pretrained(a.out / "adapter")
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
           "examples_seen": i + 1, "minutes": round((time.time() - start) / 60, 1),
           "torch": torch.__version__}
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    print("saved", a.out, cfg, flush=True)


if __name__ == "__main__":
    main()
