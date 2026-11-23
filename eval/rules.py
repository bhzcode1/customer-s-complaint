"""Phase 4 baseline (a): keyword and regex rules, no model.

The floor every model must beat. If a fine-tuned LLM cannot clearly beat this on a field,
that field does not need an LLM.

Run: uv run python eval/rules.py --split val    # -> data/predictions/rules_val.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "label"))
from schema import validate  # noqa: E402
from teacher import truncate  # noqa: E402

PREDICTIONS = ROOT / "data" / "predictions"

# Keyword -> group; the group with the most hits wins (ties: earlier in this dict).
PRODUCT_KEYWORDS = {
    "credit_reporting": ["credit report", "equifax", "transunion", "experian", "fcra", "1681", "inquir",
                         "credit bureau", "consumer report"],
    "debt_collection": ["collection", "collector", "fdcpa", "1692", "validation", "validate the debt"],
    "mortgage": ["mortgage", "escrow", "foreclos", "loan modification", "servicer", "home loan", "heloc"],
    "student_loan": ["student loan", "navient", "nelnet", "mohela", "aidvantage", "fedloan", "edfinancial",
                     "forgiveness", "pslf", "department of education"],
    "credit_card": ["credit card", "card", "synchrony", "discover", "american express", "amex", "capital one"],
    "bank_account": ["checking", "savings", "overdraft", "deposit", "debit card", "atm", "branch"],
    "money_transfer": ["zelle", "venmo", "paypal", "cash app", "cashapp", "wire", "money transfer",
                       "western union", "moneygram", "crypto", "coinbase", "bitcoin", "remittance"],
    "vehicle_loan": ["vehicle", "car loan", "auto loan", "repossess", "lease", "dealership", "gap insurance"],
    "personal_loan": ["payday", "personal loan", "installment loan", "title loan", "affirm", "klarna",
                      "afterpay", "cash advance"],
    "other": ["credit repair", "debt settlement", "debt relief", "tax refund"],
}

# Checked in order; first match wins. Default: incorrect_information (the commonest kind).
ISSUE_RULES = [
    ("not_mine_identity_theft", r"identity theft|not mine|never opened|did not open|didn'?t open|fraudulent account|do not recognize"),
    ("collection_tactics", r"harass|keep calling|calling me|threaten|validation notice|cease and desist"),
    ("debt_not_owed", r"do not owe|don'?t owe|already paid|paid in full|not my debt|validate the debt"),
    ("unauthorized_transaction", r"unauthori[sz]ed|scam|stolen|without my (permission|consent|authori[sz]ation)"),
    ("account_access", r"frozen|freeze my|closed my account|locked|hold (on )?my funds|can ?not access|cannot withdraw"),
    ("fees_interest", r"\bfees?\b|interest rate|overdraft charge|late charge"),
    ("payment_problem", r"payment (was )?not (applied|posted|credited)|misapplied|autopay"),
    ("loan_servicing", r"forbearance|modification|escrow|forgiveness|payoff|repayment plan|deferment"),
    ("application_denied", r"denied|declined|application"),
    ("dispute_handling", r"did not investigate|no response to my dispute|failed to investigate|reinvestigat"),
    ("service_problem", r"no response|never responded|customer service|on hold|conflicting"),
]

# Common companies in CFPB complaints; matched as whole words, longest name first.
COMPANIES = [
    "Equifax", "Experian", "TransUnion", "Capital One", "Bank of America", "Wells Fargo", "JPMorgan Chase",
    "Chase", "Citibank", "Citi", "Synchrony", "Discover", "American Express", "Navient", "Nelnet", "Mohela",
    "Aidvantage", "PayPal", "Venmo", "Zelle", "Cash App", "Coinbase", "Comenity", "Credit One", "U.S. Bank",
    "PNC", "TD Bank", "Truist", "Ally", "Santander", "Mr. Cooper", "Rocket Mortgage", "Freedom Mortgage",
    "Ocwen", "Portfolio Recovery Associates", "Midland Credit Management", "LVNV", "Resurgent",
    "Westlake", "Credit Acceptance", "Affirm", "Klarna", "Navy Federal", "USAA", "Barclays", "Goldman Sachs",
    "Apple Card", "Chime", "Varo", "Current", "Block", "Carvana", "CarMax", "Toyota", "Honda", "Ford Credit",
]
COMPANY_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(COMPANIES, key=len, reverse=True)) + r")\b")
AMOUNT_RE = re.compile(r"\{\$([\d,]+(?:\.\d+)?)\}")
CONTACTED_RE = re.compile(r"\b(i|we) (have )?(called|contacted|emailed|e-mailed|wrote|spoke|disputed|reached out|"
                          r"filed a dispute|sent (them|a letter))", re.I)
RESOLUTION_RULES = [
    ("stop_contact", r"stop (calling|contacting)|cease (all )?(communication|contact)|leave me alone"),
    ("deletion", r"\b(remove|delete|deletion|removed|deleted)\b"),
    ("refund", r"refund|reimburse|money back|return (my|the) (money|funds)|credit (back|my account)"),
    ("correction", r"\b(correct|fix|update|rectify)\b"),
]


def predict(text):
    t = truncate(" ".join(text.split()))  # same input the models see
    low = t.lower()
    hits = {g: sum(low.count(k) for k in kws) for g, kws in PRODUCT_KEYWORDS.items()}
    product = max(hits, key=hits.get) if max(hits.values()) else "credit_reporting"
    issue = next((i for i, rx in ISSUE_RULES if re.search(rx, low)), "incorrect_information")
    company = COMPANY_RE.search(t)
    amount = AMOUNT_RE.search(t)  # first amount written
    resolution = next((r for r, rx in RESOLUTION_RULES if re.search(rx, low)), None)
    return {
        "product_group": product,
        "issue": issue,
        "company": company.group(1) if company else None,
        "amount_in_dispute": float(amount.group(1).replace(",", "")) if amount else None,
        "contacted_company_first": True if CONTACTED_RE.search(t) else None,
        "requested_resolution": resolution,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val", "replay"], default="val")
    a = p.parse_args()
    rows = duckdb.sql(f"SELECT complaint_id, narrative FROM "
                      f"'{(ROOT / 'data' / 'processed' / f'{a.split}.parquet').as_posix()}'").fetchall()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out = PREDICTIONS / f"rules_{a.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for cid, text in rows:
            label = predict(text)
            assert validate(label) == [], (cid, validate(label))
            f.write(json.dumps({"complaint_id": cid, "label": label, "model": "rules"}) + "\n")
    print(f"{len(rows):,} predictions -> {out}")


if __name__ == "__main__":
    main()
