"""Phase 3: the JSON label every complaint gets, and the check that a label is valid.

One object per complaint, six fields. `company` replaced the planned `years_mentioned`:
from 2024 the CFPB redacts years too (XX/XX/year>), so only ~4% of 2025 complaints contain one. JSON_SCHEMA is passed to the teacher model as
a constrained-output format; validate() is the independent check run on every answer
(a server can ignore the format, and the student model in later phases has no such guard).
"""

PRODUCT_GROUPS = [
    "credit_reporting", "debt_collection", "bank_account", "mortgage", "credit_card",
    "money_transfer", "student_loan", "vehicle_loan", "personal_loan", "other",
]

# What went wrong. One label: the main problem, not every problem mentioned.
# Descriptions and rules below are prompt v2 (2026-11-24): v1 left overlaps open, and the 14B
# teacher agreed with the gold set on only 52% of issues (see NOTES, phase 3).
ISSUES = {
    "not_mine_identity_theft": "an account, loan, collection or hard inquiry the person says they never opened or authorized, or identity theft",
    "unauthorized_transaction": "money taken from or charged to the person's own account without permission, or sent because of a scam or impersonation",
    "debt_not_owed": "a collector or lender seeks a debt the person says is already paid, the wrong amount, or discharged",
    "collection_tactics": "collector calls, texts, threats, lawsuits, or a debt validation request that was not answered",
    "incorrect_information": "wrong or outdated information about an account the person accepts as theirs (balance, status, late payments, personal details), on a credit report or statement",
    "dispute_handling": "a dispute with a merchant or seller (chargeback, item not received or damaged) that the company rejected or mishandled",
    "fees_interest": "unexpected or excessive fees, interest, or charges",
    "payment_problem": "a payment not applied, misapplied, misdirected, delayed, or refused",
    "account_access": "an account frozen, restricted or closed, funds held, or the person cannot log in or withdraw",
    "loan_servicing": "escrow, insurance, loan modification, forbearance, repayment plan, payoff, transfer between servicers, or forgiveness",
    "application_denied": "an application for a loan, card or account refused, delayed, or changed",
    "service_problem": "only if nothing above fits: rude or unresponsive service, refusal to provide documents",
    "other": "fits none of the above",
}

ISSUE_RULES = """If several apply, use the FIRST that fits in this order:
  1. not_mine_identity_theft  2. unauthorized_transaction  3. debt_not_owed  4. collection_tactics
  5. incorrect_information  6. dispute_handling  7. fees_interest  8. payment_problem
  9. account_access  10. loan_servicing  11. application_denied  12. service_problem  13. other
  A scam or fraud loss is unauthorized_transaction even when the bank then denied the claim.
  A credit report entry the person does not recognize is not_mine_identity_theft, not incorrect_information."""

# What the person explicitly asks for. null = no outcome is asked for.
RESOLUTIONS = {
    "refund": "money back, a charge or fee reversed, funds released, or compensation",
    "deletion": "remove, delete or block an item, account, inquiry or late mark from a report",
    "correction": "fix information that stays on the account or report (balance, status, name, address)",
    "stop_contact": "stop calls, letters, collection activity or a lawsuit",
    "other": "another concrete outcome (e.g. reinstate an account, change a payment plan, cancel a loan)",
}

RESOLUTION_RULES = """Only an outcome the person explicitly asks for, now or earlier ("I want", "please remove",
  "I requested a refund", "stop calling me"); filing a dispute or fraud claim to get money back counts
  as refund. Do not infer what they would probably want. Asking only for an investigation, an
  explanation, or documents is null. If they ask for several outcomes, use the first in this order:
  refund, deletion, correction, stop_contact, other."""

FIELDS = ["product_group", "issue", "company", "amount_in_dispute",
          "contacted_company_first", "requested_resolution"]
COMPANY_MAX = 80  # characters; a longer answer is a sentence, not a name
NOT_COMPANIES = {"cfpb", "consumer financial protection bureau"}  # the complaint is filed *with* the CFPB

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "product_group": {"enum": PRODUCT_GROUPS},
        "issue": {"enum": list(ISSUES)},
        "company": {"type": ["string", "null"], "maxLength": COMPANY_MAX},
        "amount_in_dispute": {"type": ["number", "null"], "minimum": 0},
        "contacted_company_first": {"type": ["boolean", "null"]},
        "requested_resolution": {"enum": [*RESOLUTIONS, None]},
    },
    "required": FIELDS,
    "additionalProperties": False,
}


def validate(label):
    """Return a list of problems; empty means the label is valid."""
    if not isinstance(label, dict):
        return ["not a JSON object"]
    errors = [f"missing {f}" for f in FIELDS if f not in label]
    errors += [f"unexpected {k}" for k in label if k not in FIELDS]

    def bad(field, ok):
        if field in label and not ok(label[field]):
            errors.append(f"bad {field}: {label[field]!r}")

    is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    bad("product_group", lambda v: v in PRODUCT_GROUPS)
    bad("issue", lambda v: v in ISSUES)
    bad("amount_in_dispute", lambda v: v is None or (is_num(v) and v >= 0))
    bad("company", lambda v: v is None or (
        isinstance(v, str) and 0 < len(v.strip()) <= COMPANY_MAX
        and v.replace("X", "").strip() != ""            # a redaction is not a name: use null
        and v.strip().lower() not in NOT_COMPANIES))
    bad("contacted_company_first", lambda v: v is None or isinstance(v, bool))
    bad("requested_resolution", lambda v: v is None or v in RESOLUTIONS)
    return errors


def normalise(label):
    """Canonical form for storing: trimmed company, float amounts. (Company spelling variants,
    e.g. 'Capital One, N.A.' vs 'capital one', are matched at evaluation time, not here.)"""
    out = dict(label)
    if out["company"] is not None:
        out["company"] = " ".join(out["company"].split())
    if out["amount_in_dispute"] is not None:
        out["amount_in_dispute"] = round(float(out["amount_in_dispute"]), 2)
    return out


COMPANY_SUFFIXES = {"inc", "llc", "corp", "corporation", "co", "company", "na", "ltd", "lp", "plc", "the"}


def company_key(name):
    """Loose key for comparing company names: 'Capital One, N.A.' and 'capital one' match.
    Lowercase, punctuation removed, legal suffixes and a leading 'the' dropped."""
    if name is None:
        return None
    words = "".join(c if c.isalnum() else " " for c in name.lower().replace("n.a.", "na")).split()
    while words and words[-1] in COMPANY_SUFFIXES:
        words.pop()
    if words and words[0] == "the":
        words.pop(0)
    return " ".join(words) or None
