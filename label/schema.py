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
ISSUES = {
    "incorrect_information": "wrong or outdated information on a credit report or account (balance, status, late payment)",
    "not_mine_identity_theft": "account, inquiry or charge the person says is not theirs, or identity theft",
    "unauthorized_transaction": "money taken or charged without permission from the person's own account, or a scam",
    "debt_not_owed": "collector seeks a debt the person says is paid, wrong amount, or not theirs to pay",
    "collection_tactics": "calls, threats, harassment or no validation notice from a collector",
    "fees_interest": "unexpected or excessive fees, interest or charges",
    "payment_problem": "payment not applied, misapplied, delayed or refused",
    "account_access": "account frozen, closed, funds held, cannot log in or withdraw",
    "loan_servicing": "loan modification, forbearance, escrow, payoff, repayment plan or forgiveness",
    "dispute_handling": "company did not investigate or respond properly to a dispute",
    "application_denied": "application, loan or account opening refused or delayed",
    "service_problem": "company unresponsive, gives conflicting information, or will not provide documents or a service (outside a formal dispute)",
    "other": "anything that fits none of the above",
}

# What the person wants done. null = they do not say.
RESOLUTIONS = {
    "refund": "money back, reversed charge or fee, compensation",
    "correction": "fix wrong information or account details",
    "deletion": "remove an item, account or inquiry from a report",
    "stop_contact": "stop calls, letters or collection",
    "other": "some other concrete outcome (e.g. loan modification, unfreeze account)",
}

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
