"""
Unit tests for app/excerpt.py — the passage-selection logic that replaced
plain start-of-chunk truncation. The payment-method case here is the exact
scenario reported during manual testing: a long document where the
answer-relevant line is far from the start of the text.
"""

from app.excerpt import select_excerpt

LONG_RECEIPT = """BIG CITY MEDICAL CENTER
123 Health Pkwy, Suite 400
Patient visit summary
Date of service: 04/02/2026
Provider: Dr. Example
Procedure: Routine checkup
Diagnosis code: Z00.00
Line item: Office visit, level 3     150.00
Line item: Lab panel                  85.00
Line item: Flu vaccine                45.00
Subtotal                             280.00
Insurance adjustment                 -80.00
Patient responsibility               200.00
Payment methods accepted: Visa, Mastercard, Cash, HSA
Paid via: Mastercard ending in 4477
Amount paid: 200.00
Thank you for visiting us."""


def test_excerpt_centers_on_answer_relevant_line_not_document_start():
    question = "What payment method was used?"
    answer = "The payment was made via Mastercard."
    excerpt = select_excerpt(LONG_RECEIPT, question, answer)

    assert "Mastercard" in excerpt
    assert "Paid via" in excerpt
    # The old behavior (first 200 chars) would have shown the facility
    # header instead — confirm that's NOT what we get.
    assert "BIG CITY MEDICAL CENTER" not in excerpt


def test_excerpt_falls_back_to_question_only_when_no_answer_available():
    question = "What payment method was used?"
    excerpt = select_excerpt(LONG_RECEIPT, question, answer=None)
    # Without an answer to ground against, question-word overlap alone
    # ("payment", "method") should still find the payment line.
    assert "Payment methods accepted" in excerpt or "Paid via" in excerpt


def test_excerpt_falls_back_to_start_when_nothing_matches():
    excerpt = select_excerpt(LONG_RECEIPT, question="xyz completely unrelated zzz", answer=None)
    assert excerpt.startswith("BIG CITY MEDICAL CENTER")


def test_excerpt_respects_max_chars():
    excerpt = select_excerpt(LONG_RECEIPT, "What is the subtotal?", "The subtotal is $280.00.", max_chars=50)
    assert len(excerpt) <= 51  # allows for the trailing "…"


def test_excerpt_handles_empty_text():
    assert select_excerpt("", "any question", "any answer") == ""
