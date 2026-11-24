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
train/               phase 5-7: SFT data, LoRA training, prediction, packaging
serve/app.py         phase 8: FastAPI service with request log
monitor/             phase 9: replay through the service, drift report, reference mix
reports/             error analyses, drift report and chart
models/              adapters and merged models (not in git)
MODEL_CARD.md        phase 10: model card
eval/rules.py        phase 4: keyword/regex baseline
eval/evaluate.py     phase 4: scores any predictions file
data/predictions/    model and baseline outputs, JSONL (not in git)
tests/               pytest tests
NOTES.md             this file
```

**Status (2026-11-24).** All ten phases are built and run end to end. Phases 5-9 ran as a
**CPU pilot** (135M student on 1,600 teacher-labelled val rows) because this machine has no GPU;
the full-scale steps (teacher labels for train with prompt v2, a 0.5-1.5B student) are ready in
`notebooks/label_kaggle.ipynb` and `train/train_lora.py`. README.md has the summary and results.

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

## Phase 3: Schema and labels (val labelled; train needs the GPU run)

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

**Prompt v2** (decided 2026-11-24: tighten before labelling train). In `schema.py`:
- *Issue*: sharper descriptions plus a precedence order for complaints that fit several
  categories (identity theft > unauthorized/scam > debt not owed > collection tactics > incorrect
  information > merchant dispute > fees > payment > access > servicing > application > service
  > other). `dispute_handling` now means a dispute with a merchant or seller only; an ignored
  credit-report dispute is `incorrect_information`, a denied scam claim `unauthorized_transaction`.
  `service_problem` is a last resort.
- *Resolution*: only what the person explicitly asks for, now or earlier; a dispute or fraud
  claim for money back counts as refund; asking only for an investigation, explanation or
  documents is null; several asks -> refund > deletion > correction > stop_contact > other.
- `PROMPT_VERSION = "v2"`. The gold set was re-checked against the written rules (27 issue and
  31 resolution changes; v1 gold is in git history). Rows are marked "prompt v2 definitions".
- Caveat: the rules mostly write down the conventions the gold annotator (Claude) had used, so the
  gold and the prompt now share one definition. That is the point (a reference must match the
  spec), but it is one annotator's reading; human spot-checks would show how far others agree.
- v1 outputs kept as `data/labels/val_v1.jsonl` and `data/predictions/*_v1.jsonl`.
  **v1 teacher vs v2 gold: issue 50.5%, resolution 46.5%** (the numbers v2 must beat).

**Still to do:** relabel val with v2 on Kaggle (teacher + baselines, same notebook, ~1 h),
score against gold, then label train (~3.5 h).

## Phase 4: Baselines and evaluation

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

## Phase 5: Fine-tuning (CPU pilot)

**Why a pilot.** The plan was a 0.5-1.5B student trained on teacher labels for the 20k train rows,
on a free cloud GPU. This machine has no GPU and no Kaggle credentials, so (decision 2026-11-24)
the pipeline was completed end to end at pilot scale on the CPU; the full run is one notebook away.

**What was built**
- `train/sft_data.py`: chat examples = short fixed instruction + complaint (same 600-word cut as
  the teacher) + target JSON in fixed key order. `product_group` from the CFPB, not the teacher.
  The student gets no field definitions: after training they are in the weights, and leaving
  them out keeps inputs ~3x shorter.
- `train/train_lora.py`: LoRA (r=16 on all attention and MLP projections), loss on the JSON
  tokens only, fixed seed, config and loss log saved with the adapter.
- `train/predict.py`: greedy decoding, answers through the teacher's `parse()` (same repair and
  schema check), output in the teacher's JSONL format.

**Decisions**
- **Model: SmolLM2-135M-Instruct.** Benchmarked 16 examples: ~8-13 s per example on this 2-core
  CPU for 135M; 360M and Qwen2.5-0.5B would be ~3-4x slower (days for a useful amount of data).
- **Data: 1,600 of the 5,000 val rows with 14B teacher labels (prompt v1)**, gold rows excluded.
  Train has no teacher labels yet. Consequences, stated plainly: the test rows (gold) come from
  the same period as the training rows (no time gap, unlike the design), and the student learns
  the v1 definitions of issue and resolution while the gold uses v2.
- **Bug found and fixed:** the first benchmark hit a NaN loss. A 1,252-token example was cut at
  `max_len`, which removed the whole JSON target, so there was nothing to learn from. The complaint
  is now shortened instead of the target, and a non-finite loss is skipped, never applied.
- Settings: 1 epoch, lr 3e-4, effective batch 8 (200 steps), max 768 tokens.

**Run** (2026-11-24): 1,600 examples in 160 min on CPU, loss 1.22 -> ~0.1. Adapter 19 MB.

**Results on the gold set** (200 rows; product vs CFPB):

| Model | Valid | Product | Issue (F1) | Company | Amount has / none | Contacted | Resolution (F1) |
|---|---|---|---|---|---|---|---|
| rules | 100% | 52.0% | 34.0% (24.2) | 53.0% | 77.6% / 88.7% | 58.0% | 61.5% (39.0) |
| Qwen2.5-1.5B prompted (v1)* | 83.5% | 40.7% | 26.9% (17.8) | 79.0% | 77.6% / 86.4% | 73.7% | 38.9% (29.4) |
| **student, SmolLM2-135M pilot** | **99.0%** | 51.5% | 27.3% (18.2) | 79.3% | 81.0% / 90.7% | 75.3% | 26.3% (26.5) |
| teacher, Qwen2.5-14B (v1) | 100% | 60.0% | 50.5% (41.1) | 88.5% | 84.5% / 94.4% | 86.5% | 46.5% (46.2) |

\* incomplete run (167 of 200 gold rows).

## Phase 6: Error analysis (pilot)

`eval/errors.py` writes `reports/errors_<file>.md` (confusions per field, examples with text).
Reading the student's errors gives three causes:
1. **Inherited definitions (largest, resolution).** The student never answers null and uses
   `correction`/`other` heavily, exactly like its v1 teacher labels: 60 of its 146 resolution
   errors are gold-null -> correction/other. Agreement with its *teacher* on the same rows is 57%
   vs 26% with the gold: it copied the teacher, the teacher used the old definition.
2. **Capacity and data (issue, product).** Issue agreement with the teacher is only 42%, so the
   135M model with 1,600 examples did not learn the 13-way issue well. Product errors sit on the
   known ambiguous pairs (personal_loan -> credit_card 10, debt_collection -> credit_reporting 9,
   `other` 14).
3. **Extraction fields are close to the teacher**: company 79% (teacher 88.5%), amount 81% /
   91% (84.5% / 94%), contacted 75% (86.5%), valid JSON 99%.

**Decision: no second CPU run.** The biggest cause can only be fixed by v2 teacher labels, which
need the GPU; retraining on the same v1 labels for another 2.7 h cannot change it. The fix path:
`notebooks/label_kaggle.ipynb` with SPLIT=val then train (prompt v2), then `train_lora.py` on a
GPU with a 0.5-1.5B model and the 20k train labels.

**Interview question**
"Your student scores 26% on resolution but agrees 57% with its teacher. What does that tell you?"
That the student learned what it was taught. The gap to the gold is a label-definition problem
upstream, so the fix is better labels, not a bigger student or more epochs.

## Phase 7: Packaging and inference (pilot)

`train/package.py` merges the adapter into the base weights (`models/pilot/merged`, one
self-contained directory) and times fp32 vs int8 on 30 gold rows.

| Variant | Size | s / complaint | CPU-hours / 1,000 | Valid output |
|---|---|---|---|---|
| fp32 (merged) | 513 MB | 6.4 | 1.8 | 30/30 |
| int8 dynamic (PyTorch) | 237 MB | 9.2 | 2.6 | **0/30** |

- **int8 broke the model** (and was slower on this CPU). Keeping `lm_head` in fp32 did not help:
  the output turned into unrelated JSON or free text. A 135M model has little redundancy, and
  `torch.ao` dynamic quantisation is deprecated. Decision: serve fp32. For the full-size student,
  quantise with a dedicated runtime (GGUF Q8/Q4 via llama.cpp) and check validity before switching.
- **Cost comparison.** Student: 1.8 CPU-hours per 1,000 complaints on a 2-core 2017 laptop CPU, no
  GPU. Teacher: ~0.35 T4 GPU-hours per 1,000 (5,000 rows in 52 min on two T4s), plus a 14B model
  to host. The small model's advantage is where it runs, not raw speed on this hardware; batching
  (one complaint at a time here) and a quantised runtime would cut the student's time a lot.

## Phase 8: Service

`serve/app.py` (FastAPI): `POST /extract` and `GET /health`. One request at a time (the model is
not thread-safe on CPU); invalid output is retried once, then returned as `label: null` with the
reason. Every request is appended to `data/logs/requests.jsonl` with received date, input length,
latency, attempts, validity and the label: the input to phase 9. `tests/test_serve.py` (3 tests,
fake model) covers logging, the retry and the null fallback.

## Phase 9: Replay monitoring (distribution shift)

**What was built**
- `monitor/replay.py`: sends 2026 complaints through `serve/app.py` in received order (in-process,
  or to a running server with `--url`), so the log is written by the production code path.
- `monitor/drift.py`: monthly product mix (PSI vs reference), input length, valid rate, null
  rates, and product accuracy against the CFPB's product (a lagging check: in production that truth
  arrives later). Alert rules are fixed in `RULES` before looking at results. Output:
  `reports/drift.md` and `reports/drift.png`.
- `monitor/reference.json`: the reference is the **natural** traffic of Jan 2023 to Jun 2025
  (1.95M complaints, credit reporting 75%, median 108 words), not train/val, which are balanced on
  purpose and so have no real mix. Stored in git so the monitor runs without the raw file.

**Decisions**
- 100 random complaints per month (600 total), not all 10,000: ~6 s per request on this CPU.
  Random within a month keeps each month's real mix; 100 is above the 30-row minimum per alert.
- Two PSIs: on the model's answers (what production can see immediately) and on the CFPB's true
  product (replay only). The second shows whether a shift is real or the model's own errors.

**Results** (2026-11-24, pilot student; full table in `reports/drift.md`)

| Month | PSI (model) | PSI (truth) | credit_reporting (truth) | Product acc. | Median words | Valid |
|---|---|---|---|---|---|---|
| 2026-01 | 2.27 | 3.74 | 2% | 67.7% | 164 | 99% |
| 2026-03 | 2.09 | 4.30 | 1% | 67.0% | 176 | 100% |
| 2026-06 | 2.05 | 7.93 | 0% | 64.6% | 202 | 99% |

- **The monitor flags the shift every month (RETRAIN, PSI ~2 vs threshold 0.25)**, as expected
  from phase 2: credit reporting falls from 75% of reference traffic to 0-2% in 2026.
- The model-based PSI understates the true shift (it still answers credit_reporting for 9-16% of
  requests, from other groups), but stays far above the threshold, so the alarm is right either way.
- **Complaints keep getting longer**: median 164 -> 223 words (+52% to +106% vs reference), a
  "watch" alert every month: more tokens per request, more of them cut at 600 words.
- Valid output 98-100% in every month; latency ~5.4-6.0 s median, rising with length.
- Product accuracy on replay (57-68%) is higher than on the balanced gold set (51.5%): the natural
  2026 mix is dominated by debt collection and cards, which the student gets right more often. The
  accuracy rule (drop >10 points) did not fire.
- `requested_resolution` is never null (0% every month): the v1-label habit from phase 6, visible
  in production logs without any ground truth.

**Retraining rule** (what would happen in production): a RETRAIN alert two months in a row ->
label a fresh sample of recent complaints with the teacher, add it to training, retrain, and
re-score on the gold set before switching models.

**Interview question**
"Your product-mix PSI is 2.3. Is the model broken or did the world change?" Compare against a
truth signal when one exists (here the CFPB product, PSI 3.7-7.9 with credit reporting at 0-2%):
the world changed. Without truth, look at validity and confidence-like signals first; a broken
model usually shows invalid output or collapsed answers before it shows a mix shift.

## Phase 10: Wrap-up

- `README.md`: problem, data, pipeline, results table, how to run, limits.
- `MODEL_CARD.md`: intended use, training data and period, evaluation, known weaknesses
  (category fields, same-period evaluation, redaction, truncation, 2026 shift, int8), next steps.
- Repo checked from a fresh clone: `uv sync`, `uv sync --group ml`, `uv run pytest` (no data needed).

**What the project shows, in one paragraph.** A 135M model fine-tuned for 2.7 hours on a laptop
CPU produces valid JSON 99% of the time and comes close to a 14B teacher on fields it copies from
the text, but not on judgement fields. The bigger lesson came from the gold set: the teacher's
agreement on issue and resolution (~50%) was limited by how the labels were defined, and a
student faithfully inherits that. Fixing definitions (prompt v2) comes before scaling the model.
Monitoring catches the 2026 shift clearly, and a truth signal (the CFPB product) shows it is the
world that changed, not the model.

**Prepared for the full run:** `train_lora.py` and `predict.py` use a GPU when present
(gradient checkpointing on CUDA so 1.5B fits in fp32 on a T4); `notebooks/train_kaggle.ipynb`
trains the student on `train.jsonl` and scores it on the gold set. **Blocked here:** no GPU and
no Kaggle API token on this machine; with `~/.kaggle/kaggle.json` both notebooks can be pushed and
their outputs pulled from this laptop.

**Next steps (need a GPU):** relabel val and train with prompt v2 on Kaggle; train a 0.5-1.5B
student on the 20k train labels; quantise with GGUF; re-run phases 6-9 with the same scripts.

## Roadmap: phases 3 to 10 (original plan, kept for reference)

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
