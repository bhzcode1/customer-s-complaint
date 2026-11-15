"""Phase 1 gate: is the Hugging Face CFPB mirror usable for this project?

Step 1 pulls only the rows that have a narrative (4 columns) from Hugging Face
into one local Parquet file. That happens once; phase 2 reuses the file.
Step 2 measures that local file and prints the numbers the gate needs.

Run: uv run python data/check_data.py
"""

from pathlib import Path

import duckdb

SOURCE = "hf://datasets/Mouwiya/cfpb-consumer-complaints/data/*.parquet"
LOCAL = Path(__file__).parent / "raw" / "narratives.parquet"
GATE = 50_000  # fewer unique narratives than this = change dataset

con = duckdb.connect()
# 8 GB machine: cap DuckDB and let it spill big hash tables to disk.
con.execute("SET memory_limit = '3GB'")
con.execute(f"SET temp_directory = '{(LOCAL.parent.parent / 'tmp').as_posix()}'")


def one(sql):
    return con.execute(sql).fetchone()


def show(title, sql):
    print(f"\n{title}")
    for row in con.execute(sql).fetchall():
        print("  ", *row)


# --- Step 1: download once -------------------------------------------------
if not LOCAL.exists():
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    part = LOCAL.with_suffix(".part")  # an interrupted run leaves this, not a fake LOCAL
    print("Counting rows in the mirror (reads file footers only)...")
    print(f"  total rows in mirror: {one(f'SELECT count(*) FROM {SOURCE!r}')[0]:,}")
    print("Pulling rows with a narrative (~1 GB download, takes a while)...")
    con.execute(f"""
        COPY (
            SELECT complaint_id, date_received, product,
                   consumer_complaint_narrative AS narrative
            FROM {SOURCE!r}
            WHERE trim(coalesce(consumer_complaint_narrative, '')) <> ''
        ) TO '{part.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    part.replace(LOCAL)  # only a finished download gets the real name

con.execute(f"CREATE VIEW n AS SELECT * FROM '{LOCAL.as_posix()}'")

# --- Step 2: measure ---------------------------------------------------------
rows = one("SELECT count(*) FROM n")[0]
# hash() instead of the raw text keeps the distinct-count table small in RAM.
exact = one("SELECT count(DISTINCT hash(narrative)) FROM n")[0]
# Loose duplicates: same text after lowercasing, removing XXXX redactions,
# and collapsing whitespace. Catches copy-paste templates with small edits.
loose = one(r"""
    SELECT count(DISTINCT hash(
        regexp_replace(regexp_replace(lower(narrative), 'x{2,}', '', 'g'), '\s+', ' ', 'g')
    )) FROM n
""")[0]
ids = one("SELECT count(DISTINCT complaint_id) FROM n")[0]

print(f"\nrows with a narrative:        {rows:,}")
print(f"distinct complaint_id:        {ids:,}")
print(f"unique narratives (exact):    {exact:,}   duplicate rate {1 - exact / rows:.1%}")
print(f"unique narratives (loose):    {loose:,}   duplicate rate {1 - loose / rows:.1%}")

show("date range (min, max):", "SELECT min(date_received)::DATE, max(date_received)::DATE FROM n")
show("narratives per year:", """
    SELECT year(date_received) AS y, count(*) FROM n GROUP BY y ORDER BY y
""")
show("top products (share of narratives):", """
    SELECT product, count(*) AS c, round(100.0 * c / sum(c) OVER (), 1) || '%'
    FROM n GROUP BY product ORDER BY c DESC LIMIT 10
""")
show("narrative length in words (p10, median, p90):", r"""
    SELECT quantile_cont(w, [0.1, 0.5, 0.9])
    FROM (SELECT len(regexp_split_to_array(trim(narrative), '\s+')) AS w FROM n)
""")

# Can the planned fields be extracted at all? CFPB redacts personal details,
# so check how often amounts and dates survive in the text.
show("share of narratives with (amount, any date, full date, date with only year kept):", r"""
    SELECT
      round(100.0 * avg(regexp_matches(narrative, '\$\s?\d')::INT), 1) || '%',
      round(100.0 * avg(regexp_matches(narrative, '\d{1,2}/\d{1,2}/\d{2,4}|XX/XX/')::INT), 1) || '%',
      round(100.0 * avg(regexp_matches(narrative, '\b\d{1,2}/\d{1,2}/\d{2,4}\b')::INT), 1) || '%',
      round(100.0 * avg(regexp_matches(narrative, 'XX/XX/(19|20)\d\d')::INT), 1) || '%'
    FROM n
""")

# Gate on the stricter (loose) count: phase 2 removes near-duplicates too.
verdict = "PASS" if loose >= GATE else "FAIL - change dataset"
print(f"\nGATE: {loose:,} unique narratives (loose) vs {GATE:,} needed -> {verdict}")
