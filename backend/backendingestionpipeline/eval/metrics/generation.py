# eval/metrics/generation.py

from dataclasses import dataclass
import re


@dataclass
class GenerationMetrics:
    answered: bool
    correctly_refused: bool | None
    citation_rate: float
    term_coverage: float
    forbidden_term_present: bool
    answer_length_chars: int


CITATION_PATTERN = re.compile(r'\[Src\s*\d+\]', re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def compute_generation_metrics(
    answer: str | None,
    was_answered: bool,
    citations: list[dict],
    citation_coverage: float | None,
    expected_contains: list[str],
    should_not_contain: list[str],
    should_be_refused: bool,
) -> GenerationMetrics:
    """
    Deterministic generation quality metrics. No LLM calls required.
    """

    correctly_refused = None
    if should_be_refused:
        correctly_refused = not was_answered

    if not was_answered or not answer:
        return GenerationMetrics(
            answered=False,
            correctly_refused=correctly_refused,
            citation_rate=0.0,
            term_coverage=0.0,
            forbidden_term_present=False,
            answer_length_chars=0,
        )

    answer_lower = answer.lower()

    # Fallback heuristic citation rate
    raw_sentences = SENTENCE_SPLIT_PATTERN.split(answer.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    cited_sentences = sum(
        1 for s in sentences
        if CITATION_PATTERN.search(s)
    )

    heuristic_rate = (
        cited_sentences / len(sentences)
        if sentences else 0.0
    )

    # Prefer generation.py metric if available
    citation_rate = (
        citation_coverage
        if citation_coverage is not None
        else heuristic_rate
    )

    # Expected term coverage
    found = sum(
        1 for term in expected_contains
        if term.lower() in answer_lower
    )

    term_coverage = (
        found / len(expected_contains)
        if expected_contains else 1.0
    )

    # Forbidden terms
    forbidden_present = any(
        t.lower() in answer_lower
        for t in should_not_contain
    )

    return GenerationMetrics(
        answered=True,
        correctly_refused=correctly_refused,
        citation_rate=round(citation_rate, 3),
        term_coverage=round(term_coverage, 3),
        forbidden_term_present=forbidden_present,
        answer_length_chars=len(answer),
    )