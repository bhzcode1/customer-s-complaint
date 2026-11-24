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
data/labels/         teacher labels, JSONL (not in git)
label/schema.py      phase 3: label fields, JSON Schema, validator
label/teacher.py     phase 3: label complaints via an OpenAI-compatible server
label/lengths.py     phase 3: token counts for the length cap
label/gold.py        phase 3: blind hand-labelling of 200 val rows + scoring against them
data/gold/           val_gold.jsonl, hand labels (in git: small and not reproducible)
notebooks/label_kaggle.ipynb  phase 3: full labelling run with vLLM on a free Kaggle GPU
eval/rules.py        phase 4: keyword/regex baseline
eval/evaluate.py     phase 4: scores any predictions file
data/predictions/    model and baseline outputs, JSONL (not in git)
tests/               pytest tests
NOTES.md             this file
```

**Status (2026-11-23).** Phases 1 and 2 are done. Phase 3 is in progress: schema, labelling
script and length cap are done; the full labelling run and gold set are not. The plan for
phases 3 to 10 is at the end of this file.

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

## Phase 3: Schema and labels (in progress)

**What was built so far**
- `label/schema.py`: the six label fields, a JSON Schema for constrained output, and `validate()`,
  an independent check run on every answer.
- `label/teacher.py`: sends each complaint to any OpenAI-compatible chat server (vLLM, llama.cpp,
  Ollama) at temperature 0, retries an invalid answer once, appends one JSONL line per complaint
  (raw answer, parsed label, errors, model, prompt version, time).
- `label/lengths.py`: token counts per split. `tests/test_label.py`: 10 tests with a fake model.
- `label/gold.py`: 200 val rows (20 per product group, fixed md5 order) for hand-labelling.
  **Blind**: the teacher's answer is never shown, so it cannot anchor the person checking.
  `score` reports per-field accuracy, amount split into "has amount" / "no amount", and company
  compared with `company_key()` (lowercase, no punctuation or legal suffixes: "Capital One, N.A."
  = "capital one"). Phase 4's evaluation reuses the same scoring.
- `notebooks/label_kaggle.ipynb`: clones the repo, starts vLLM (T4 x2, fp16, default
  Qwen2.5-14B-Instruct-AWQ, 7B fallback), runs a 20-row smoke test, then the full split. The
  same session also runs phase 4 baseline (b) (see below) and copies labels and predictions to
  the notebook output. Each model gets its own server, stopped before the next starts.
  **Not run yet**, so model fit and speed on T4 are untested.

**Decisions**
- **Teacher = an open model, no paid API** (decided 2026-11-23). This laptop (2-core i3, no usable
  GPU) labels at ~16 s per complaint even with a 1.5B model, so 25k rows would take ~110 hours,
  and a 1.5B teacher is no better than the student. The full run goes on a free cloud GPU
  (Kaggle/Colab) with a 7B+ model served by vLLM; the laptop is only for smoke tests.
- **`years_mentioned` replaced by `company`.** From 2024 the CFPB redacts the year too
  (`XX/XX/year>`). Share of complaints with any 4-digit year: ~26% in 2018-2021, 15% in 2023,
  3.7% in 2025 (val), 5.9% in 2026 (replay). An always-empty answer would score ~95%. Company
  names are not redacted in most complaints and matter for routing. A redacted name (`XXXX`)
  or "CFPB" is rejected by the validator; the answer must be null instead.
- **Length cap: 600 words** (`MAX_WORDS`). Tokens (Qwen2.5 tokenizer, ~1.3 per word): median
  ~200, p90 ~575, p99 ~1,500, max ~10,600. 600 words cuts ~5% of rows. The teacher sees the same
  truncated text as the student, so labels never rely on text the student cannot read.
- **Resume only counts labels from the same model and prompt version**, so smoke-test labels
  can never stand in for real ones. Bump `PROMPT_VERSION` when the prompt or schema changes.
- `issue` has 13 categories, `requested_resolution` 5 plus null (see `schema.py`); drafted from
  the roadmap and adjusted after reading 30 val complaints (added `service_problem`).

**Smoke test** (2026-11-23, Qwen2.5-1.5B Q4 on CPU via llama.cpp, 5 val rows): 5/5 valid JSON,
so the plumbing works. Quality was poor (product right 2/5, `XXXX` and `CFPB` given as company,
one invented amount), which is why the company check was added and why the teacher must be larger.

**Run**
```
uv run python label/lengths.py
# start an OpenAI-compatible server, then:
uv run python label/teacher.py --split val --limit 20 --base-url http://localhost:8080/v1 \
    --model <name> --out data/labels/smoke/val.jsonl
```

Gold set and scoring:
```
uv run python label/gold.py annotate                      # resumable; Ctrl+C saves
uv run python label/gold.py score data/labels/val.jsonl
```

**Teacher run on val** (Kaggle, 2026-11-24): Qwen2.5-14B-Instruct-AWQ on T4 x2, 5,000 rows
in 52 min (~1.6 rows/s; train's 20k would take ~3.5 h). Labels in `data/labels/val.jsonl`.
- 4,771 valid at first (95.4%). **All 229 failures were `company: "XXXX"`**, and the retry
  did not help: at temperature 0 the model repeats the same answer. Fix: `repair()` in
  `teacher.py` turns a redacted or CFPB company into null (null *is* the schema's meaning for
  "no name given") and notes the repair; `teacher.py --reparse FILE` re-parses saved raw answers
  without calling the model. After it: 5,000/5,000 valid. Same rule applied to every model.
- **Teacher vs CFPB product: only 62.5% agree** (credit_reporting 94%, mortgage 84%, bank 80%;
  debt_collection 53%, personal_loan 47%, other 4%). Reading the disagreements, most are a
  *definition* gap, not teacher errors: the CFPB product is what the consumer filed under, the
  teacher answers what the text is about. E.g. "LVNV is reporting a debt to the credit bureaus"
  is filed as debt collection; the teacher says credit_reporting. Some are real teacher errors
  (Cash App -> bank_account, Earnin cash advance -> money_transfer). The prompt never defines the
  groups, and `other` (debt settlement, credit management) is almost never chosen.
- **Decision (2026-11-24): `product_group` = the CFPB's filed product.** The student learns it
  from the CFPB's own label, which is exact, free and known for every complaint; the teacher's
  product answer is not used. Schema and prompt are unchanged (so no relabelling). The training
  data takes product from the CFPB, the gold-set tool fills it in instead of asking, and product
  is scored only against the CFPB. (The one gold row labelled before this says `vehicle_loan`;
  the CFPB says `credit_card`. Its product is no longer scored.)

**Gold set** (2026-11-24): 200 val rows in `data/gold/val_gold.jsonl`. 1 row labelled by hand;
**199 labelled by Claude (Opus 5.5), blind**: from the complaint text alone (cut at 600 words,
as the teacher sees it), never shown the teacher's answers, following the prompt's field
definitions. Each such row carries `"annotator": "claude (blind, from text only)"` and a note
where the call was ambiguous (~60 rows). It is a second-model reference, not human ground truth;
spot-checking some rows by hand would strengthen it.

**Teacher vs gold** (n=200): company 88.5%, contacted 86.5%, amount 84.5% (has) / 94.4% (none),
**issue 52.0% (macro-F1 44.4%), resolution 52.0%**. The two low fields are mostly definition gaps:
- *Resolution*: the gold uses null unless the person explicitly asks for something (85 of 200);
  the teacher infers what they seem to want (null only 13 times). "Null if they do not ask for
  anything specific" allows both readings; this explains 72 of 96 disagreements. Another 12 are
  deletion vs correction for credit-report items, which overlap.
- *Issue*: disagreements spread over many pairs. The 13 categories overlap; the teacher uses
  `service_problem` (26 vs gold 6) and `payment_problem` (21 vs 11) as catch-alls, and calls
  unrecognised accounts `incorrect_information` where the gold says `not_mine_identity_theft`.
- So the student's ceiling on these two fields is set by schema clarity more than by model size.

**Still to do:** decide whether to tighten the issue/resolution definitions (prompt v2, relabel
val, ~1 h on Kaggle) before labelling train (~3.5 h).

## Phase 4: Baselines and evaluation (started early, while labels are pending)

**What was built so far**
- `eval/evaluate.py`: scores a predictions JSONL (same format as the teacher's labels).
  `product_group` is always scored against the CFPB's own product, which is known for every row,
  so it needs no labels. Every field is scored against the gold set when it exists, and against
  the teacher's labels with `--ref`. Macro-F1 for `issue` and `requested_resolution`; amount
  split into "has amount" / "no amount".
- `eval/rules.py`: baseline (a), keywords and regexes only: product by keyword counts, issue and
  resolution by ordered regex rules, company from a list of ~55 common names, first `{$...}`
  amount, "I called/wrote/disputed..." for contacted. `tests/test_eval.py`: 3 tests.

**Results** (val, 2026-11-24). Product vs the CFPB's own label; every other column is
agreement with the teacher (14B) labels, so the teacher scores 100% there by definition. Its
real accuracy needs the gold set (1 of 200 rows labelled so far).

| Model | Product vs CFPB | Issue (F1) | Company | Amount: has / none | Contacted | Resolution (F1) |
|---|---|---|---|---|---|---|
| (a) rules | 51.2% | 33.3% (26.8%) | 53.4% | 69.2% / 90.4% | 54.2% | 28.7% (28.2%) |
| (b) Qwen2.5-0.5B, prompted | 19.4% | 25.6% (16.0%) | 67.2% | 71.6% / **49.4%** | 69.7% | 16.1% (11.1%) |
| (b) Qwen2.5-1.5B, prompted* | 43.2% | 35.7% (22.8%) | 80.6% | 67.6% / 91.4% | 76.9% | 44.9% (35.1%) |
| (c) teacher, Qwen2.5-14B | 62.5% | — | — | — | — | — |

**Against the gold set** (200 val rows, labelled blind by Claude; see phase 3):

| Model | Issue (F1) | Company | Amount: has / none | Contacted | Resolution (F1) |
|---|---|---|---|---|---|
| (a) rules | 34.0% (26.3%) | 53.0% | 77.6% / 88.7% | 58.0% | **61.5%** (44.7%) |
| (b) Qwen2.5-0.5B, prompted | 23.0% (15.6%) | 71.5% | 72.4% / 42.3% | 64.0% | 12.5% (10.4%) |
| (b) Qwen2.5-1.5B, prompted* | 26.9% (18.6%) | 79.0% | 77.6% / 86.4% | 73.7% | 38.9% (29.5%) |
| (c) teacher, Qwen2.5-14B | **52.0%** (44.4%) | **88.5%** | **84.5% / 94.4%** | **86.5%** | 52.0% (52.8%) |

The rules beat the teacher on resolution only because both default to null often, matching the
gold's strict reading of that field (see phase 3).

Product is the CFPB's filed product (phase 3 decision), so a fine-tuned student trained on it
should beat every row here on that column; the prompted models were asked for the text's topic.

\* The 1.5B Kaggle run stopped at 4,260 of 5,000 rows; its scores cover those rows only.
n = 5,000 (1,752 with an amount per the teacher, 3,248 without).

**Findings**
- **Prompting alone is not enough for a small model.** The 0.5B is worse than the rules on almost
  every field and invents an amount for half the complaints that have none (49.4% on "no amount").
  The 1.5B beats the rules on company, contacted and resolution, but not on product or issue.
  This is the gap fine-tuning has to close.
- **14% of the 0.5B's and 36% of the 1.5B's answers needed the company repair** (they write
  "XXXX"), against 4.6% for the teacher.
- **Most val complaints that contain a dollar amount are not about it**: the rules found an
  amount in 2,044 rows (41%, against 22% over all years in phase 1), the teacher marked an
  amount in dispute in 1,752 (35%).
- The rules leave `company` null in 66% of rows: names outside the list are missed, so this is a
  weak floor for that field.

**Run**
```
uv run python eval/rules.py --split val
uv run python eval/evaluate.py data/predictions/rules_val.jsonl [--ref data/labels/val.jsonl]
```

**Baseline (b), prepared:** the candidate students (Qwen2.5-0.5B and 1.5B Instruct) get the
teacher's exact prompt and schema, with no fine-tuning, via `teacher.py --out data/predictions/...`.
It runs in the Kaggle notebook (minutes on a T4; ~22 hours on this laptop's CPU). Same prompt =
any gap after fine-tuning is due to training, not prompt changes. The two sizes also inform the
phase 5 choice of student.

**Still to do:** finish the 1.5B run (740 rows).

## Roadmap: phases 3 to 10 (planned, not built yet)

Drafted 2026-11-22 from the findings above. The original plan was not saved in this repo, so
this is a proposal: change it as decisions are made. Each phase gets its full notes section
(what was built, why, terms, run, interview question) once it is done.

### Phase 3: Schema and labels
- **Fix the output schema** (one JSON object per complaint). Proposed fields:
  - `product_group`: one of the 10 groups (already known, so it doubles as a sanity check).
  - `issue`: short category, e.g. wrong balance, unauthorised charge, harassment, not on report.
  - `amount_in_dispute`: number or null (null in about 78% of rows).
  - ~~`years_mentioned`~~ replaced by `company` (see the phase 3 section: years are redacted from 2024).
  - `contacted_company_first`: true/false/null.
  - `requested_resolution`: refund, correction, deletion, stop contact, other, or null.
- **Label with a large open LLM on a free cloud GPU** (the "teacher"): train 20k + val 5k. Store the raw response,
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
