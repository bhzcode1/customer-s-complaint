"""Phase 2: turn the raw narratives into train / val / replay splits.

Steps:
1. Drop rows with no date, merge renamed CFPB products into one group each.
2. Loose dedupe over all rows (same text after lowercasing, removing XXXX and
   collapsing spaces); the earliest copy wins, so a template stays in train.
3. Split by date, sample a fixed number per product group (train, val) or a
   plain random sample (replay keeps the real product mix, like production).
4. Near-duplicate removal (MinHash) on the sampled rows only, then trim to size.

Run: uv run python data/prep.py
"""

from math import ceil
from pathlib import Path
import re

import duckdb
from datasketch import MinHash, MinHashLSH

DATA = Path(__file__).parent
RAW = DATA / "raw" / "narratives.parquet"
OUT = DATA / "processed"

VAL_START = "2025-07-01"     # train: before this date
REPLAY_START = "2026-01-01"  # val: VAL_START up to here; replay: from here on
PER_GROUP = {"train": 2_000, "val": 500}  # rows per product group
REPLAY_ROWS = 10_000
OVERSAMPLE = 1.5   # sample extra so near-duplicate removal does not leave groups short
NEAR_DUP = 0.8     # estimated Jaccard similarity of word 5-grams above this = duplicate
NUM_PERM = 128     # MinHash size: more = more accurate similarity estimate, slower

# CFPB renamed products over the years; each name maps to one stable group.
GROUPS = {
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting": "credit_reporting",
    "Debt collection": "debt_collection",
    "Checking or savings account": "bank_account",
    "Bank account or service": "bank_account",
    "Mortgage": "mortgage",
    # 2017-2023 merged cards and prepaid cards into one product, so they cannot be separated.
    "Credit card": "credit_card",
    "Credit card or prepaid card": "credit_card",
    "Prepaid card": "credit_card",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Money transfers": "money_transfer",
    "Virtual currency": "money_transfer",
    "Student loan": "student_loan",
    "Vehicle loan or lease": "vehicle_loan",
    "Payday loan, title loan, personal loan, or advance loan": "personal_loan",
    "Payday loan, title loan, or personal loan": "personal_loan",
    "Payday loan": "personal_loan",
    "Consumer Loan": "personal_loan",  # 2015-2017 catch-all for installment loans
    "Debt or credit management": "other",
    "Other financial service": "other",
}

SPLITS = ("train", "val", "replay")
LOOSE_KEY = r"hash(trim(regexp_replace(regexp_replace(lower(narrative), 'x{2,}', '', 'g'), '\s+', ' ', 'g')))"


def shingles(text):
    """Set of word 5-grams after the same normalisation as the loose key."""
    words = re.sub(r"x{2,}", "", text.lower()).split()
    return {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def near_duplicates(rows):
    """rows: (id, text) in priority order. Returns ids that repeat an earlier row."""
    lsh = MinHashLSH(threshold=NEAR_DUP, num_perm=NUM_PERM)
    drop = set()
    for cid, text in rows:
        m = MinHash(num_perm=NUM_PERM)
        m.update_batch(s.encode() for s in shingles(text))
        if lsh.query(m):
            drop.add(cid)
        else:
            lsh.insert(cid, m)
    return drop


def build(con, raw, out, per_group=PER_GROUP, replay_rows=REPLAY_ROWS):
    """Write train/val/replay parquet files to `out`; return stage counts."""
    con.execute("SET TimeZone = 'UTC'")  # timestamp -> date must not depend on the laptop's zone
    con.execute(f"CREATE OR REPLACE VIEW src AS SELECT * FROM read_parquet('{Path(raw).as_posix()}')")
    con.execute(
        "CREATE OR REPLACE TEMP TABLE groups AS "
        "SELECT unnest(?::VARCHAR[]) AS product, unnest(?::VARCHAR[]) AS grp",
        [list(GROUPS), list(GROUPS.values())],
    )

    # Steps 1-2. No narrative text in this table, only ids and a hash: keeps the window small.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE keep AS
        SELECT complaint_id, d AS date, s.product, g.grp,
               CASE WHEN d < DATE '{VAL_START}' THEN 'train'
                    WHEN d < DATE '{REPLAY_START}' THEN 'val' ELSE 'replay' END AS split
        FROM (SELECT complaint_id, date_received::DATE AS d, product, {LOOSE_KEY} AS k
              FROM src WHERE date_received IS NOT NULL) s
        LEFT JOIN groups g USING (product)
        QUALIFY row_number() OVER (PARTITION BY k ORDER BY d, complaint_id) = 1
    """)
    unmapped = [r[0] for r in con.execute("SELECT DISTINCT product FROM keep WHERE grp IS NULL").fetchall()]
    if unmapped:
        raise ValueError(f"products missing from GROUPS: {unmapped}")

    # Step 3. md5(id) order = a random but repeatable sample (same rows every run).
    cap = {s: ceil(n * OVERSAMPLE) for s, n in per_group.items()}
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sampled AS
        SELECT k.*, s.narrative FROM (
            SELECT * FROM keep WHERE split = 'train'
            QUALIFY row_number() OVER (PARTITION BY grp ORDER BY md5(complaint_id)) <= {cap['train']}
            UNION ALL
            SELECT * FROM keep WHERE split = 'val'
            QUALIFY row_number() OVER (PARTITION BY grp ORDER BY md5(complaint_id)) <= {cap['val']}
            UNION ALL
            (SELECT * FROM keep WHERE split = 'replay'
             ORDER BY md5(complaint_id) LIMIT {ceil(replay_rows * OVERSAMPLE)})
        ) k JOIN src s USING (complaint_id)
    """)

    # Step 4. Oldest first, so when a pair spans splits the train copy survives (no leakage).
    rows = con.execute("SELECT complaint_id, narrative FROM sampled ORDER BY date, complaint_id").fetchall()
    drop = near_duplicates(rows)
    con.execute("CREATE OR REPLACE TEMP TABLE dropped AS SELECT unnest(?::VARCHAR[]) AS complaint_id",
                [sorted(drop)])
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE final AS
        SELECT * FROM sampled ANTI JOIN dropped USING (complaint_id)
        QUALIFY row_number() OVER (
            PARTITION BY split, CASE WHEN split = 'replay' THEN '' ELSE grp END
            ORDER BY md5(complaint_id)
        ) <= CASE split WHEN 'train' THEN {per_group['train']}
                        WHEN 'val' THEN {per_group['val']} ELSE {replay_rows} END
    """)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        con.execute(f"""
            COPY (SELECT complaint_id, date, product, grp AS product_group, narrative
                  FROM final WHERE split = '{split}' ORDER BY date, complaint_id)
            TO '{(out / f'{split}.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)
        """)

    count = lambda t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    return {"raw": count("src"), "after_loose_dedupe": count("keep"),
            "sampled": len(rows), "near_dups_dropped": len(drop), "final": count("final")}


if __name__ == "__main__":
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")  # 8 GB machine; spill the rest to disk
    con.execute(f"SET temp_directory = '{(DATA / 'tmp').as_posix()}'")
    for stage, n in build(con, RAW, OUT).items():
        print(f"{stage:<20} {n:>10,}")

    print("\nrows per split and product group:")
    table = con.execute("""
        PIVOT (SELECT grp, split FROM final) ON split IN ('train', 'val', 'replay')
        USING count(*) GROUP BY grp ORDER BY grp
    """).fetchall()
    print(f"  {'group':<18}{'train':>8}{'val':>8}{'replay':>8}")
    for grp, *n in table:
        short = " <- short" if n[0] < PER_GROUP["train"] or n[1] < PER_GROUP["val"] else ""
        print(f"  {grp:<18}" + "".join(f"{x:>8,}" for x in n) + short)
    print("\nlength in words (p10, median, p90):")
    for split in SPLITS:
        q = con.execute(fr"""SELECT quantile_cont(len(regexp_split_to_array(trim(narrative), '\s+')),
                             [0.1, 0.5, 0.9]) FROM final WHERE split = '{split}'""").fetchone()[0]
        print(f"  {split:<8} {q}")
