"""
Picks the line(s) within a retrieved chunk that actually support the
answer, instead of blindly showing the first N characters of the chunk —
which, for a long document, is often an unrelated header rather than the
passage the answer is grounded in (e.g. a payment-method question should
show the payment lines, not the vendor's address block at the top).

Deliberately simple and explainable: lexical (word-overlap) scoring per
line, no extra embedding or LLM calls, so there's no added latency or cost
over the previous truncate-from-the-start behavior. Answer-word overlap is
weighted higher than question-word overlap, since the generated answer
text is the more precise grounding signal (it typically repeats the exact
value extracted from the source — a dollar amount, a vendor name, a
payment method), while the question's words are a weaker, more generic
signal on their own.
"""

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3}


def select_excerpt(text: str, question: str, answer: str | None = None,
                    context_lines: int = 1, max_chars: int = 280) -> str:
    """
    text: the full chunk text to excerpt from.
    question: the user's question (always available).
    answer: the generated answer, if one exists — None for the no-answer
        (insufficient-evidence / error) case, in which case this falls back
        to question-only matching.
    context_lines: how many lines of surrounding context to include above
        and below the best-matching line, so the excerpt reads naturally
        rather than as a single bare fragment.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return text[:max_chars]

    question_words = _content_words(question)
    answer_words = _content_words(answer) if answer else set()

    best_idx, best_score = 0, 0
    for i, line in enumerate(lines):
        line_words = _content_words(line)
        score = 2 * len(line_words & answer_words) + len(line_words & question_words)
        if score > best_score:
            best_score, best_idx = score, i

    if best_score == 0:
        # No real lexical match found (e.g. a generic "I don't know" case
        # with no answer to ground against) — fall back to the start of
        # the text, the previous behavior, rather than pick an arbitrary line.
        joined = "\n".join(lines)
        return joined if len(joined) <= max_chars else joined[:max_chars] + "…"

    start = max(0, best_idx - context_lines)
    end = min(len(lines), best_idx + context_lines + 1)
    excerpt = "\n".join(lines[start:end])

    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[:max_chars] + "…"
