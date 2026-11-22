# Notes

One short section per phase: what was built, why, terms, how to run, one interview question.

## Project overview

**What it is.** A small fine-tuned LLM that reads a consumer complaint written in free text and
returns the key facts as JSON: which product, what went wrong, the amount in dispute, the years
mentioned, whether the company was contacted first, and what the person wants done.

**Why.** Large hosted LLMs can already do this, but they are slow and expensive at millions of
complaints. The project tests whether a small model, trained on labels from a large one, gets
close enough in accuracy at a fraction of the cost and runs on an ordinary laptop. It also
covers what happens after deployment: new complaints look different from old ones, so the model
is monitored for drift.

**Data.** CFPB Consumer Complaint Database (US Consumer Financial Protection Bureau), from a
Hugging Face mirror: 16.6M complaints, 3.8M with a written narrative, 2015 to June 2026.
Personal details are redacted (`XXXX`, `XX/XX/XXXX`).

**How it works, end to end**
1. Check the data is good enough (phase 1).
2. Clean, dedupe, and split it by date into train / val / replay (phase 2).
3. Have a large LLM label the train and val sets with the JSON schema (phase 3).
4. Measure baselines (phase 4), fine-tune a small model with LoRA (phase 5), study its mistakes and retrain (phase 6).
5. Shrink it to run on CPU and serve it as an API (phases 7 and 8).
6. Replay the 2026 complaints through the API as if they were live traffic and watch for drift (phase 9).
7. Write up the results (phase 10).

**Constraints.** One laptop with 8 GB RAM: DuckDB instead of pandas, training on a free cloud GPU,
and inference on CPU. Everything is reproducible: fixed sampling order, pinned dependencies (`uv.lock`).

**Stack.** Python 3.11, `uv`, DuckDB, datasketch (MinHash), pytest. Planned: a LoRA training
library, a quantised runtime, and FastAPI.

**Layout**
```
data/check_data.py   phase 1: download + data check
data/prep.py         phase 2: clean, dedupe, split, sample
data/raw/            narratives.parquet (not in git)
data/processed/      train / val / replay parquet (not in git)
tests/               pytest tests
NOTES.md             this file
```

**Status (2026-11-22).** Phases 1 and 2 are done. Phase 3 is next. The plan for phases 3 to 10 is
at the end of this file.

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

## Phase 2: Data preparation

**What was built**
- `data/prep.py`: raw narratives -> `data/processed/{train,val,replay}.parquet`
  (columns: `complaint_id, date, product, product_group, narrative`).
- `tests/test_prep.py`: 3 tests on a tiny synthetic Parquet file (no download needed).

**Pipeline**
1. Drop rows with no date. Map the 21 CFPB product names to 10 stable groups (`GROUPS`).
   An unknown product name raises an error instead of being silently dropped.
2. *Loose dedupe* over all 3.8M rows: one row per normalised-text hash, **earliest wins**.
3. Split by date: train `< 2025-07-01`, val `2025-07-01 .. 2025-12-31`, replay `>= 2026-01-01`.
4. Sample: train 2,000 and val 500 per group (balanced); replay 10,000 random (real mix).
   1.5x extra is sampled so step 5 does not leave groups short.
5. *Near-duplicate* removal with MinHash on the 52,500 sampled rows, then trim to size.

**Why it is done this way**
- **Split by date, not at random.** The model is used on complaints that arrive *after*
  training. A random split lets the same month's templates and events sit in train and test,
  which makes scores look better than they will be later.
- **Earliest copy wins in both dedupe steps.** If a template appears in 2020 and 2026, the 2020
  copy is kept, so it lands in train and never in val/replay. That is how leakage is avoided.
- **Two-stage dedupe.** Exact hashing is cheap enough for 3.8M rows in DuckDB. MinHash is
  slower (Python), so it runs only on the rows actually used. It removed 750 more (1.4%),
  so the loose hash already catches most templates.
- **Balanced train/val, natural replay.** Balanced so credit reporting (65% of raw) does not
  drown the rest; natural replay because it stands in for real traffic in phase 9.
- **`md5(complaint_id)` ordering for sampling.** Random-looking but identical every run, and
  unlike DuckDB's `hash()` it does not change between DuckDB versions.
- **Prepaid cards merged into credit cards.** 2017-2023 the CFPB filed both under one product,
  so they cannot be split back apart.
- **Trade-off:** sampling before near-dedupe means two near-identical rows that were *not both
  sampled* are never compared. They cannot leak either, because only sampled rows are saved.

**Terms**
- *Leakage*: test data that the model (in effect) saw in training; inflates scores.
- *MinHash*: a short signature per document; two signatures agree in roughly the same share of
  positions as the documents' *Jaccard similarity* (shared 5-word phrases / all phrases).
- *LSH (locality-sensitive hashing)*: buckets signatures so only likely-similar pairs are
  compared, instead of every pair (52,500^2 / 2 = 1.4 billion).
- *Stratified / balanced sampling*: a fixed number per group instead of a plain random draw.

**Run**
```
uv run pytest
uv run python data/prep.py        # ~2.5 min
```
Read 30 random cleaned complaints:
```
uv run python -c "import duckdb; [print(r[0], r[1], r[2][:600], sep='\n', end='\n\n---\n') for r in duckdb.sql(\"SELECT product_group, date, narrative FROM 'data/processed/train.parquet' USING SAMPLE 30\").fetchall()]"
```

**Results** (run 2026-11-15)

| Stage | Rows |
|---|---|
| Raw narratives | 3,815,251 |
| After no-date drop + loose dedupe | 2,358,204 |
| Sampled (1.5x) | 52,500 |
| Near-duplicates dropped | 750 |
| Final | 35,000 = train 20,000 + val 5,000 + replay 10,000 |

Every group reached its train (2,000) and val (500) target. Median length: train 149 words,
val 170, replay 188 (p90 about 450 in all three).

**Findings**
- **The product mix changes completely in 2026.** In the 2026 raw data credit reporting is
  only 3,438 rows (4%), against 1.67M in 2023-2025. Replay therefore holds 225 credit-reporting
  rows and 2,757 debt-collection rows. Not a bug: checked against raw counts. This is a real
  distribution shift, which is exactly what phase 9 replay monitoring should show.
- **Complaints get longer over time** (median 149 -> 188 words). That affects cost and latency
  later: longer input = more tokens.
- **Train covers 2015 to mid-2025.** Older complaints are written differently, so a model
  trained on them may do worse on recent ones. Val (2025 H2) will measure that.
- Not done (left open): no minimum or maximum length filter. Very long complaints (some over
  450 words) cost more to label and may not fit a small model's context; decide in phase 3
  when token counts are known.

**Interview question**
"Why split by date instead of randomly?" — the model will see future complaints. A random split
mixes the same period's templates into train and test and inflates the score; a date split,
plus keeping the earliest copy of any duplicate in train, measures what happens after deployment.

## Roadmap: phases 3 to 10 (planned, not built yet)

Drafted 2026-11-22 from the findings above. The original plan was not saved in this repo, so
this is a proposal: change it as decisions are made. Each phase gets its full notes section
(what was built, why, terms, run, interview question) once it is done.

### Phase 3: Schema and labels
- **Fix the output schema** (one JSON object per complaint). Proposed fields:
  - `product_group`: one of the 10 groups (already known, so it doubles as a sanity check).
  - `issue`: short category, e.g. wrong balance, unauthorised charge, harassment, not on report.
  - `amount_in_dispute`: number or null (null in about 78% of rows).
  - `years_mentioned`: list of years. Replaces `key_dates`, because day and month are redacted.
  - `contacted_company_first`: true/false/null.
  - `requested_resolution`: refund, correction, deletion, stop contact, other, or null.
- **Label with a large hosted LLM** (the "teacher"): train 20k + val 5k. Store the raw response,
  the parsed JSON, and the model version, so labels can be rebuilt and checked.
- **Gold set:** hand-check about 200 val rows. Teacher accuracy on these sets the ceiling the small model can reach.
- **Length decision:** count tokens, then cap or truncate the longest complaints (p90 about 450 words).
- Done when: every row has a JSON label that passes the schema check, and teacher accuracy on the gold set is recorded.

### Phase 4: Baselines and evaluation harness
- One `evaluate.py`: per-field accuracy or F1, JSON-valid rate, and `amount_in_dispute`
  scored separately for "has amount" and "no amount".
- Baselines to beat: (a) regex/rules (dollar amounts, years), (b) the small base model
  with a prompt and no fine-tuning, (c) the teacher itself (upper bound).
- Done when: one table compares all three on val.

### Phase 5: Fine-tuning
- Small open model (about 0.5B to 1.5B parameters), trained with LoRA/QLoRA on the train labels.
- The 8 GB laptop cannot train this. Use a free cloud GPU (Colab or Kaggle T4) and save only the
  LoRA adapter (a few MB).
- Fix the seed and save the config, so a run can be repeated.
- Done when: the fine-tuned model beats baseline (b) on val and the adapter is saved.

### Phase 6: Error analysis and iteration
- Read about 100 val mistakes and group them by cause: bad teacher label, ambiguous text,
  truncation, or a gap in the model.
- Fix the biggest cause (relabel, adjust the schema, change the length cap, more examples for weak groups), then retrain once.
- Check whether older train data hurts: val is 2025 H2, train starts in 2015.
- Done when: each remaining error type is counted and the second run's result is recorded.

### Phase 7: Packaging and inference
- Merge the adapter and quantise it (e.g. GGUF/llama.cpp or ONNX) so it runs on CPU on this laptop.
- Constrain or validate the output against the JSON schema; on failure, retry once, then return nulls.
- Measure latency per complaint and cost per 1,000 complaints, compared with calling the teacher.
- Done when: one command turns a complaint into valid JSON on the laptop, with timings recorded.

### Phase 8: Service
- Small FastAPI endpoint `POST /extract` that logs input length, latency, and output for every request.
- The log format is what phase 9 reads.
- Done when: the endpoint runs locally and a smoke test passes.

### Phase 9: Replay monitoring (distribution shift)
- Send the 10k 2026 replay rows through the service in date order, as if they were live traffic.
- Track per month: product mix, input length, JSON-valid rate, null rates, and accuracy on a
  small labelled sample.
- Expected signal: credit reporting drops from about 65% to 4% and debt collection rises (see the
  phase 2 findings). The monitor has to flag this.
- Done when: a report or chart shows the shift and the rule that would trigger retraining.

### Phase 10: Wrap-up
- Fill in `README.md`: problem, data, results table, how to run, limits.
- Model card: training data period, fields, known weak spots (redacted dates, missing amounts, shift in 2026).
- Clean the repo, pin versions, check that a fresh clone runs end to end.
