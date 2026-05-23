import re
import time
from dataclasses import dataclass

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.models.schemas import CitationSource, ChunkResult, HallucinationFlags
from app.services.prompt_assembler import (
    AssembledPrompt,
    INSUFFICIENT_EVIDENCE_MARKER,
    SourceBlock,
)

logger = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[Src\s*(\d+)\]", re.IGNORECASE)

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])(?:\s+|\n+)"
)

MIN_SENTENCE_CHARS = 15
MIN_CITATION_COVERAGE = 0.90


@dataclass
class GenerationResult:
    answer: str | None
    answered: bool
    refusal_reason: str | None
    citations: list[CitationSource]
    hallucination_flags: HallucinationFlags
    citation_coverage: float
    uncited_sentence_count: int
    prompt_tokens: int | None
    completion_tokens: int | None
    llm_ms: int
    model_used: str


def _get_openai_client() -> AsyncOpenAI:
    settings = get_settings()

    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
    )


def _extract_citations(
    answer_text: str,
    source_blocks: list[SourceBlock],
) -> tuple[list[CitationSource], list[int]]:
    cited_ns = set(
        int(m)
        for m in CITATION_PATTERN.findall(answer_text)
    )

    citations: list[CitationSource] = []
    out_of_range: list[int] = []

    for n in sorted(cited_ns):
        block_idx = n - 1

        if 0 <= block_idx < len(source_blocks):
            block = source_blocks[block_idx]

            citations.append(
                CitationSource(
                    chunk_index=block.chunk.chunk_index,
                    filename=block.chunk.filename,
                    page_number=block.chunk.page_number,
                    relevance_score=round(
                        block.chunk.similarity_score,
                        3,
                    ),
                    text_preview=block.chunk.text[:200],
                )
            )

        else:
            out_of_range.append(n)

            logger.warning(
                "citation_out_of_range",
                cited_n=n,
                source_block_count=len(source_blocks),
            )

    return citations, out_of_range


def _compute_citation_coverage(
    answer_text: str,
) -> tuple[float, int]:
    raw_sentences = SENTENCE_SPLIT_PATTERN.split(
        answer_text.strip()
    )

    sentences = [
        s.strip()
        for s in raw_sentences
        if len(s.strip()) >= MIN_SENTENCE_CHARS
    ]

    if not sentences:
        return 1.0, 0

    cited_count = sum(
        1
        for s in sentences
        if CITATION_PATTERN.search(s)
    )

    uncited_count = len(sentences) - cited_count

    coverage = cited_count / len(sentences)

    return round(coverage, 3), uncited_count


def _check_hallucination_signals(
    answer: str,
    source_blocks: list[SourceBlock],
    citations: list[CitationSource],
    out_of_range_citations: list[int],
    coverage: float,
) -> HallucinationFlags:
    flags = HallucinationFlags()

    if not citations and answer.strip():
        flags.answer_contains_no_citation = True

    if out_of_range_citations:
        flags.answer_contains_no_citation = True

    if coverage < 0.70:
        flags.low_term_overlap = True

    total_context_chars = sum(
        len(b.chunk.text)
        for b in source_blocks
    )

    if (
        answer
        and len(answer) > total_context_chars * 0.85
        and len(answer) > 400
    ):
        flags.suspiciously_long_answer = True

    return flags


async def generate(
    assembled: AssembledPrompt,
    ranked_chunks: list[ChunkResult],
    no_chunks_passed_threshold: bool = False,
    openai_client: AsyncOpenAI | None = None,
) -> GenerationResult:
    settings = get_settings()

    if no_chunks_passed_threshold:
        logger.info("generation_refused_threshold_gate")

        return GenerationResult(
            answer=None,
            answered=False,
            refusal_reason=(
                "No relevant information found "
                "in the uploaded documents for this query."
            ),
            citations=[],
            hallucination_flags=HallucinationFlags(
                no_chunks_passed_threshold=True
            ),
            citation_coverage=0.0,
            uncited_sentence_count=0,
            prompt_tokens=None,
            completion_tokens=None,
            llm_ms=0,
            model_used=settings.openai_chat_model,
        )

    if not assembled.source_blocks:
        logger.warning(
            "generation_called_with_empty_source_blocks"
        )

        return GenerationResult(
            answer=None,
            answered=False,
            refusal_reason=(
                "No source documents were prepared "
                "for this query."
            ),
            citations=[],
            hallucination_flags=HallucinationFlags(
                no_chunks_passed_threshold=True
            ),
            citation_coverage=0.0,
            uncited_sentence_count=0,
            prompt_tokens=None,
            completion_tokens=None,
            llm_ms=0,
            model_used=settings.openai_chat_model,
        )

    client = openai_client or _get_openai_client()

    t0 = time.perf_counter()

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=assembled.messages,
            temperature=0.0,
            max_tokens=1200,
        )

    except RateLimitError as exc:
        logger.error(
            "openai_rate_limit",
            error=str(exc),
        )
        raise

    except APITimeoutError as exc:
        logger.error(
            "openai_timeout",
            error=str(exc),
        )
        raise

    except APIError as exc:
        logger.error(
            "openai_api_error",
            status=exc.status_code,
            error=str(exc),
        )
        raise

    llm_ms = int(
        (time.perf_counter() - t0) * 1000
    )

    raw_answer = (
        response.choices[0]
        .message.content
        or ""
    ).strip()

    logger.info(
        "llm_call_completed",
        model=settings.openai_chat_model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        llm_ms=llm_ms,
        answer_chars=len(raw_answer),
    )

    if INSUFFICIENT_EVIDENCE_MARKER in raw_answer:
        logger.info(
            "model_refused_insufficient_evidence",
            evidence_summary=assembled.evidence_summary,
        )

        return GenerationResult(
            answer=None,
            answered=False,
            refusal_reason=(
                "The documents do not contain "
                "sufficient evidence to answer this question."
            ),
            citations=[],
            hallucination_flags=HallucinationFlags(),
            citation_coverage=0.0,
            uncited_sentence_count=0,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            llm_ms=llm_ms,
            model_used=settings.openai_chat_model,
        )

    citations, out_of_range = _extract_citations(
        raw_answer,
        assembled.source_blocks,
    )

    coverage, uncited_count = (
        _compute_citation_coverage(raw_answer)
    )

    if coverage < MIN_CITATION_COVERAGE:
        original_coverage = coverage

        logger.warning(
            "citation_repair_triggered",
            original_coverage=original_coverage,
        )

        repair_prompt = f"""
Rewrite the answer below.

RULES:
- Every sentence must contain at least one citation
- Use citation format exactly: [Src N]
- Preserve meaning exactly
- Do not add information
- Do not remove information

Answer:
{raw_answer}
"""

        repair_response = (
            await client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a citation repair engine."
                        ),
                    },
                    {
                        "role": "user",
                        "content": repair_prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=1200,
            )
        )

        repaired_answer = (
            repair_response.choices[0]
            .message.content
            .strip()
        )

        repaired_citations, repaired_out_of_range = (
            _extract_citations(
                repaired_answer,
                assembled.source_blocks,
            )
        )

        repaired_coverage, repaired_uncited = (
            _compute_citation_coverage(
                repaired_answer
            )
        )

        if repaired_coverage > original_coverage:
            raw_answer = repaired_answer

            citations = repaired_citations
            out_of_range = repaired_out_of_range

            coverage = repaired_coverage
            uncited_count = repaired_uncited

            logger.info(
                "citation_repair_success",
                improved_from=original_coverage,
                improved_to=repaired_coverage,
            )

    flags = _check_hallucination_signals(
        answer=raw_answer,
        source_blocks=assembled.source_blocks,
        citations=citations,
        out_of_range_citations=out_of_range,
        coverage=coverage,
    )

    if flags.any_flag_raised:
        logger.warning(
            "hallucination_signals_raised",
            flags=flags.model_dump(),
            citation_coverage=coverage,
            uncited_sentences=uncited_count,
            out_of_range_citations=out_of_range,
        )

    else:
        logger.info(
            "generation_clean",
            citation_coverage=coverage,
            citations_found=len(citations),
            llm_ms=llm_ms,
        )

    return GenerationResult(
        answer=raw_answer,
        answered=True,
        refusal_reason=None,
        citations=citations,
        hallucination_flags=flags,
        citation_coverage=coverage,
        uncited_sentence_count=uncited_count,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        llm_ms=llm_ms,
        model_used=settings.openai_chat_model,
    )