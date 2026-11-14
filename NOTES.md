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

**Results** (run 2026-11-14)
_pending_

**Interview question**
"Why not just load the CSV into pandas?" — 16.6M rows of free text do not fit in 8 GB RAM;
a columnar engine reads only the needed columns and spills to disk, so the same laptop works.
