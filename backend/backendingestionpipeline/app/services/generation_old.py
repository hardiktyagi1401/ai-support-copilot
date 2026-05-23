"""
LLM generation service.

Responsibilities:
- Hold the OpenAI client configuration
- Make the API call with the assembled messages
- Parse the raw completion text
- Extract which sources were cited
- Run hallucination heuristics
- Return a structured GenerationResult

What this does NOT do:
- Build prompts (that is prompt_assembler.py)
- Retrieve chunks (that is retrieval.py)
- Log to the database (that is query_logger.py)
- Know about the HTTP request/response cycle
"""

import re
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.models.schemas import (
    CitationSource,
    ChunkResult,
    HallucinationFlags,
)
from app.services.prompt_assembler import AssembledPrompt

logger = get_logger(__name__)

# The refusal string the model is instructed to return.
# We check for this exactly so we can set answered=False cleanly.
REFUSAL_MARKER = "I don't have information about that in the provided documents."


@dataclass
class GenerationResult:
    answer: str | None          # None means answered=False
    answered: bool
    refusal_reason: str | None
    citations: list[CitationSource]
    hallucination_flags: HallucinationFlags
    prompt_tokens: int | None
    completion_tokens: int | None
    llm_ms: int
    model_used: str


def _get_openai_client() -> AsyncOpenAI:
    """
    Build the AsyncOpenAI client from settings.

    We do NOT cache this with lru_cache because AsyncOpenAI clients
    hold an internal httpx.AsyncClient with a connection pool.
    Creating one per request is fine for V1. For higher throughput,
    instantiate once at app startup in the lifespan function.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
    )


def _extract_citations(
    answer_text: str,
    ranked_chunks: list[ChunkResult],
) -> list[CitationSource]:
    """
    Parse [Source N] markers from the answer and map to actual chunks.

    We look for [Source 1], [Source 2] etc. (1-indexed, matching the
    numbering in the prompt). If the model cited a source number that
    doesn't exist (e.g. [Source 9] when we only had 5 chunks), we
    skip it — this is itself a hallucination signal.
    """
    # Find all [Source N] references in the answer
    cited_indices = set(
        int(m) for m in re.findall(r'\[Source\s+(\d+)\]', answer_text)
    )

    citations = []
    for idx in sorted(cited_indices):
        # Convert from 1-indexed citation to 0-indexed chunk list
        chunk_pos = idx - 1
        if 0 <= chunk_pos < len(ranked_chunks):
            chunk = ranked_chunks[chunk_pos]
            citations.append(CitationSource(
                chunk_index=chunk.chunk_index,
                filename=chunk.filename,
                page_number=chunk.page_number,
                relevance_score=round(chunk.similarity_score, 3),
                text_preview=chunk.text[:200],
            ))
        else:
            logger.warning(
                "citation_out_of_range",
                cited_index=idx,
                available_chunks=len(ranked_chunks),
            )

    return citations


def _check_hallucination_signals(
    answer: str,
    ranked_chunks: list[ChunkResult],
    citations: list[CitationSource],
) -> HallucinationFlags:
    """
    Deterministic heuristics for hallucination risk.
    Fast, no I/O. Each check catches a distinct failure mode.
    """
    flags = HallucinationFlags()

    # No citations found — model answered without referencing any source
    if not citations and len(ranked_chunks) > 0:
        flags.answer_contains_no_citation = True

    # Answer shares very few terms with the retrieved context
    all_chunk_words = set()
    for chunk in ranked_chunks:
        all_chunk_words.update(chunk.text.lower().split())

    answer_words = set(answer.lower().split())
    overlap = answer_words & all_chunk_words
    # Fewer than 8 common content words is suspicious
    if len(overlap) < 8:
        flags.low_term_overlap = True

    # Answer is implausibly long relative to context given
    total_context_chars = sum(len(c.text) for c in ranked_chunks)
    if len(answer) > total_context_chars * 0.9 and len(answer) > 500:
        flags.suspiciously_long_answer = True

    return flags


async def generate(
    assembled: AssembledPrompt,
    ranked_chunks: list[ChunkResult],
    no_chunks_passed_threshold: bool = False,
) -> GenerationResult:
    """
    Call the LLM with the assembled prompt and process the response.

    ranked_chunks is passed here (not just inside AssembledPrompt) because
    we need it for citation extraction and hallucination checks — operations
    that happen after the LLM call.
    """
    settings = get_settings()

    # Hard gate: if retrieval found nothing useful, refuse without calling LLM.
    # This is the most important hallucination prevention we have.
    if no_chunks_passed_threshold:
        logger.info("generation_refused_no_threshold_chunks")
        return GenerationResult(
            answer=None,
            answered=False,
            refusal_reason="No relevant information found in the uploaded documents for this query.",
            citations=[],
            hallucination_flags=HallucinationFlags(no_chunks_passed_threshold=True),
            prompt_tokens=None,
            completion_tokens=None,
            llm_ms=0,
            model_used=settings.openai_chat_model,
        )

    client = _get_openai_client()
    t0 = time.perf_counter()

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=assembled.messages,
            temperature=0.1,   # Low temperature = more faithful to context
            max_tokens=1024,   # Generous but bounded
        )
    except RateLimitError as exc:
        logger.error("openai_rate_limit", error=str(exc))
        raise
    except APITimeoutError as exc:
        logger.error("openai_timeout", error=str(exc))
        raise
    except APIError as exc:
        logger.error("openai_api_error", status=exc.status_code, error=str(exc))
        raise

    llm_ms = int((time.perf_counter() - t0) * 1000)
    raw_answer = response.choices[0].message.content or ""

    logger.info(
        "llm_call_completed",
        model=settings.openai_chat_model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        llm_ms=llm_ms,
    )

    # Detect model refusal (instructed in system prompt)
    if REFUSAL_MARKER in raw_answer:
        return GenerationResult(
            answer=None,
            answered=False,
            refusal_reason="The documents do not contain information relevant to this query.",
            citations=[],
            hallucination_flags=HallucinationFlags(),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            llm_ms=llm_ms,
            model_used=settings.openai_chat_model,
        )

    # Extract citations and run hallucination heuristics
    citations = _extract_citations(raw_answer, ranked_chunks)
    flags = _check_hallucination_signals(raw_answer, ranked_chunks, citations)

    if flags.any_flag_raised:
        logger.warning(
            "hallucination_signals_detected",
            flags=flags.model_dump(),
            answer_preview=raw_answer[:100],
        )

    return GenerationResult(
        answer=raw_answer,
        answered=True,
        refusal_reason=None,
        citations=citations,
        hallucination_flags=flags,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        llm_ms=llm_ms,
        model_used=settings.openai_chat_model,
    )