"""Phase 5: turn teacher labels into supervised fine-tuning examples for the student.

Each example is a chat: a short fixed instruction, the complaint (cut exactly as the teacher saw
it), and the target JSON. The student does not get the teacher's long field definitions: after
fine-tuning they live in its weights, and leaving them out keeps every input ~3x shorter, which
is what makes CPU training and inference affordable.

product_group comes from the CFPB's own product, not the teacher (phase 3 decision).
"""

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "label"))
from gold import GOLD, latest_labels, read_jsonl  # noqa: E402
from schema import FIELDS, normalise, validate  # noqa: E402
from teacher import truncate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
INSTRUCTION = ("Extract the facts of this consumer complaint as JSON with the keys: "
               + ", ".join(FIELDS) + ".")


def target_json(label):
    """Compact JSON in a fixed key order: the exact text the student learns to write."""
    return json.dumps({f: label[f] for f in FIELDS}, ensure_ascii=False, separators=(", ", ": "))


def messages(narrative, label=None):
    """Chat messages for one complaint; with a label, the assistant turn is the target."""
    msgs = [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": truncate(" ".join(narrative.split()))}]
    if label is not None:
        msgs.append({"role": "assistant", "content": target_json(label)})
    return msgs


def cfpb_products(split):
    sql = f"SELECT complaint_id, product_group, narrative FROM '{(PROCESSED / f'{split}.parquet').as_posix()}'"
    return {cid: (g, text) for cid, g, text in duckdb.sql(sql).fetchall()}


def build(labels_path, split, exclude=()):
    """[(complaint_id, messages)] for every row with a valid teacher label, minus `exclude`,
    in a fixed order (md5 of the id, so any prefix is a random but repeatable sample)."""
    teacher = latest_labels(labels_path)
    rows = cfpb_products(split)
    out = []
    for cid in sorted(teacher, key=lambda c: __import__("hashlib").md5(c.encode()).hexdigest()):
        if cid in exclude or cid not in rows:
            continue
        group, text = rows[cid]
        label = normalise({**teacher[cid], "product_group": group})
        assert validate(label) == [], (cid, validate(label))
        out.append((cid, messages(text, label)))
    return out


def gold_ids():
    return {r["complaint_id"] for r in read_jsonl(GOLD)}
