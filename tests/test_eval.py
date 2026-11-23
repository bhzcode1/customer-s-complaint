import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "label"))
import evaluate  # noqa: E402
import rules  # noqa: E402
import schema  # noqa: E402


def test_macro_f1():
    assert evaluate.macro_f1([("a", "a"), ("b", "b")]) == 1.0
    # always "a": perfect recall on a (F1 2/3), zero on b -> mean 1/3
    assert abs(evaluate.macro_f1([("a", "a"), ("a", "b")]) - 1 / 3) < 1e-9
    assert evaluate.macro_f1([]) == 0.0


def test_rules_predict():
    text = ("I called Capital One about a charge of {$1,250.00} I did not authorize. "
            "They refused, so I want a refund of the money on my credit card.")
    label = rules.predict(text)
    assert schema.validate(label) == []
    assert label["product_group"] == "credit_card"
    assert label["company"] == "Capital One"
    assert label["amount_in_dispute"] == 1250.0
    assert label["contacted_company_first"] is True
    assert label["requested_resolution"] == "refund"


def test_rules_predict_empty_text_is_valid():
    label = rules.predict("XXXX XXXX")
    assert schema.validate(label) == []
    assert label["company"] is None and label["amount_in_dispute"] is None
