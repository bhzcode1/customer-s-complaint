# Model card: complaint-extraction student (CPU pilot)

## Summary

| | |
|---|---|
| Task | Extract six fields as JSON from a US consumer financial complaint narrative |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameters) |
| Method | LoRA (rank 16, all attention and MLP projections), loss on the JSON only; merged for serving |
| Teacher | `Qwen/Qwen2.5-14B-Instruct-AWQ`, prompt v1, temperature 0, schema-constrained output |
| Training data | 1,600 CFPB complaints received Jul-Dec 2025, teacher labels; product from the CFPB |
| Training cost | 160 min on a 2-core laptop CPU (Intel i3-7100U), 1 epoch, seed 0 |
| Size / speed | 513 MB (fp32); ~6 s per complaint on that CPU, batch size 1 |
| Status | **Pilot.** Built to run the whole pipeline end to end without a GPU; not for production use |

## Intended use

Routing and summarising consumer complaints: which product, what went wrong, against which
company, how much money, whether the company was contacted first, and what the person asks for.
Output is always either a schema-valid JSON object or null (the service never returns a
half-valid object). Every field should be treated as a suggestion for a human reviewer, not a
decision about a consumer.

Not intended for: credit, eligibility or legal decisions about individuals; text that is not a
US consumer financial complaint; languages other than English.

## Output fields

| Field | Values |
|---|---|
| `product_group` | 10 groups, defined as the product the consumer would file under with the CFPB |
| `issue` | 13 categories with a precedence order (`label/schema.py`, prompt v2) |
| `company` | name as written in the text, or null when redacted |
| `amount_in_dispute` | number copied from the text, or null |
| `contacted_company_first` | true / false / null |
| `requested_resolution` | refund, deletion, correction, stop_contact, other, or null |

## Evaluation

Gold set: 200 complaints from Jul-Dec 2025, 20 per product group. 199 were labelled blind by
Claude (Opus 5.5) from the text alone, following written field definitions (prompt v2), and one
by hand. It is a second-model reference, not human ground truth. Product is scored against the
CFPB's own product.

| Model | Valid | Product | Issue | Company | Amount has / none | Contacted | Resolution |
|---|---|---|---|---|---|---|---|
| keyword rules | 100% | 52.0% | 34.0% | 53.0% | 77.6% / 88.7% | 58.0% | 61.5% |
| **this model** | **99.0%** | 51.5% | 27.3% | 79.3% | 81.0% / 90.7% | 75.3% | 26.3% |
| teacher (14B) | 100% | 60.0% | 50.5% | 88.5% | 84.5% / 94.4% | 86.5% | 46.5% |

On 600 replayed complaints from Jan-Jun 2026 (natural mix): valid output 98-100% per month,
product accuracy 57-68% per month.

## Known weaknesses

- **Category fields are weak.** Issue (27%) and resolution (26%) are well below the teacher. Two
  causes, measured in `NOTES.md` phase 6: the model learned the teacher's *v1* label definitions
  (it agrees with its teacher 57% on resolution but with the v2 gold only 26%; it never answers
  null for resolution), and 135M parameters with 1,600 examples is too little for a 13-way issue.
- **Same-period evaluation.** Training and gold rows both come from Jul-Dec 2025, so the gold
  scores do not include the time gap the project's design asks for. The 2026 replay does.
- **Redaction.** The CFPB removes names, dates and, since 2024, years. Many amounts and company
  names are redacted, so null is often the correct answer for those fields.
- **Truncation.** Complaints are cut at 600 words (about 5% of rows), and complaints are getting
  longer (2026 median 164-223 words vs 108 in 2023-mid 2025).
- **Distribution shift.** In 2026 credit reporting falls from 75% to 0-2% of complaints; the
  model was trained on a balanced mix and the monitor flags the shift every month.
- **int8 quantisation breaks the model** (0/30 valid outputs with PyTorch dynamic int8); serve fp32.

## Data

CFPB Consumer Complaint Database (public, US government), via a Hugging Face mirror. The CFPB
redacts personal information before publication. Complaints are consumers' own accounts and are
not verified by the CFPB.

## How to improve it

1. Label `val` and `train` with prompt v2 (`notebooks/label_kaggle.ipynb`, free Kaggle GPU, ~4.5 h).
2. Train a 0.5-1.5B student on the 20k train labels on a GPU (`train/train_lora.py --model ...`).
3. Quantise with a dedicated runtime (GGUF Q8/Q4) and check validity before serving.
4. Have a person spot-check the gold set.
