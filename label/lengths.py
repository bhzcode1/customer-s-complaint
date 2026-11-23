"""Phase 3 length decision: how many tokens are the complaints, and what does a cap cut?

Counts tokens with a small open model's tokenizer (stand-in for the phase 5 student;
modern tokenizers give similar counts on English text) and prints, per split, the token
percentiles and the share of rows a given word cap would truncate.

Run: uv run python label/lengths.py
"""

from pathlib import Path

import duckdb
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
CAPS = [300, 400, 500, 600, 800]  # word caps to compare


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


if __name__ == "__main__":
    tok = Tokenizer.from_pretrained(TOKENIZER)
    print(f"tokenizer: {TOKENIZER}\n")
    print(f"{'split':<8}{'rows':>7}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>7}  tokens/word  "
          + "  ".join(f">{c}w" for c in CAPS))
    for split in ("train", "val", "replay"):
        texts = [r[0] for r in duckdb.sql(
            f"SELECT narrative FROM '{(PROCESSED / f'{split}.parquet').as_posix()}'").fetchall()]
        toks = [len(e.ids) for e in tok.encode_batch(texts)]
        words = [len(t.split()) for t in texts]
        over = "  ".join(f"{sum(w > c for w in words) / len(words):>5.1%}" for c in CAPS)
        print(f"{split:<8}{len(texts):>7,}{pct(toks, .5):>7,}{pct(toks, .9):>7,}{pct(toks, .99):>7,}"
              f"{max(toks):>7,}  {sum(toks) / sum(words):>11.2f}  {over}")
        if split != "replay":
            print(f"{'':<8}total tokens: {sum(toks):,}")
