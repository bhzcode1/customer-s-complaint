# Notes

One short section per phase: what was built, why, terms, how to run, one interview question.

## Phase 1: Setup and data check (gate)

**What was built**
- Git repo + `uv` project (`pyproject.toml`, `uv.lock`, `.python-version` = 3.11).
- `data/check_data.py`: pulls complaint rows that have a narrative from the Hugging Face
  mirror into `data/raw/narratives.parquet` (once), then prints the gate numbers.

**Why it is done this way**
- **DuckDB, not pandas.** The mirror is 67 Parquet files, 1.03 GB compressed, ~16.6M rows.
  pandas would load it all into RAM (8 GB machine). DuckDB streams it, reads only the
  4 columns we ask for (*projection pushdown*), and spills to disk when memory runs out.
- **Download once, measure locally.** Each remote query would re-download the text.
  The local file has only narrative rows, so phase 2 starts from something small.
- **`count(DISTINCT hash(narrative))`** instead of distinct on the text itself: an 8-byte
  hash per row instead of ~1 KB of text. Collision odds at a few million rows are negligible.
- **Two duplicate rates.** *Exact* = byte-identical text. *Loose* = same after lowercasing,
  removing `XXXX` redactions and collapsing spaces, which catches copy-paste templates.
  The gate uses the loose count because phase 2 removes near-duplicates anyway.
- **Field feasibility check.** CFPB redacts personal details (names, dates appear as
  `XX/XX/XXXX`). The script measures how often a dollar amount or an unredacted date
  survives. If dates are almost always redacted, `key_dates` must change in phase 3.
- **Named `check_data.py`, not `inspect.py`.** `inspect` is a Python standard-library
  module; a file with that name shadows it when the script runs from its own folder.

**Terms**
- *Parquet*: columnar file format; reading one column skips the bytes of the others.
- *Projection pushdown*: the engine reads only the columns the query uses.
- *Spill to disk*: when a query's working memory exceeds the limit, write it to temp files.
- *Gate*: a check that decides whether the project continues as planned.

**Run**
```
uv sync
uv run python data/check_data.py
```
(On Windows, set `PYTHONUTF8=1` if the console throws a `UnicodeEncodeError`.)

**Results** (run 2026-11-15)

| Measure | Value |
|---|---|
| Rows in mirror | 16,558,052 |
| Rows with a narrative | 3,815,251 (local file 681 MB) |
| Unique narratives, exact | 2,564,072 (32.8% duplicates) |
| Unique narratives, loose | 2,389,678 (37.4% duplicates) |
| Date range | 2015-03-19 to 2026-06-17; 45,040 rows (1.2%) have no date |
| Busiest years | 2025: 1.22M, 2024: 0.81M, 2023: 0.44M |
| Length (words) | p10 33, median 117, p90 367 |
| Contains a dollar amount (`{$500.00}`) | 22.0% |
| Contains a full date | 0.0% |
| Contains a date with only the year kept (`XX/XX/2024`) | 7.9% |

**Gate: PASS.** 2.39M unique narratives vs 50,000 needed, about 48 times the threshold.

**What this changes in later phases**
- **`key_dates` cannot work as planned.** The CFPB removes the day and month from every
  date. A written-out month is just as rare (0.03%), and only the year survives in 7.9% of rows. Phase 3 must replace
  this field (for example, `years_mentioned`) or drop it. Decide in phase 3.
- **`amount_in_dispute` will be null for about 78% of rows.** Accuracy must be reported
  separately for "has an amount" and "has none", or a model that always answers null looks good.
- **Credit reporting appears under two product names** (43.8% + 21.2% = 65%): the CFPB renamed
  the category. Phase 2 must merge renamed products before balancing.
- **37% of the text is duplicates**, mostly templates. Phase 2 near-duplicate removal is required, not optional.
- **Rows with no date** cannot go into a date-based split; phase 2 drops them.
- The mirror ends in June 2026, two months before the CFPB stopped publishing narratives.

**Interview question**
"Why not just load the CSV into pandas?" — 16.6M rows of free text do not fit in 8 GB RAM;
a columnar engine reads only the needed columns and spills to disk, so the same laptop works.
