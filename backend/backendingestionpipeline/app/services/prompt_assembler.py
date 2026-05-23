import time
from dataclasses import dataclass
 
from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.models.schemas import ChunkResult
 
logger = get_logger(__name__)
 
CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 3000
STRONG_SCORE_THRESHOLD = 0.80
MODERATE_SCORE_THRESHOLD = 0.65
 
SYSTEM_PROMPT = SYSTEM_PROMPT = """\
You are a document-grounded answer engine.

RULES:
1. Answer ONLY using the provided source documents.
2. Every factual sentence MUST end with a citation like [Src 1].
3. Never use outside knowledge.
4. If multiple sources support a claim, cite all relevant sources.
5. If sources conflict, state the conflict and cite both.

If the documents do not contain enough evidence to answer the question, respond exactly:
"Insufficient evidence in the provided documents to answer this question."

Do not hallucinate.
"""
 
INSUFFICIENT_EVIDENCE_MARKER = "Insufficient evidence in the provided documents"
 
 
@dataclass
class SourceBlock:
    index: int
    chunk: ChunkResult
    score_band: str
    formatted: str
 
 
@dataclass
class AssembledPrompt:
    messages: list[dict]
    source_blocks: list[SourceBlock]
    chunks_used: int
    chunks_dropped: int
    estimated_prompt_tokens: int
    evidence_summary: str
    assembly_ms: int
 
 
def _score_band(score: float) -> str:
    if score >= STRONG_SCORE_THRESHOLD:
        return "strong"
    if score >= MODERATE_SCORE_THRESHOLD:
        return "moderate"
    return "weak"
 
 
def _format_source_block(index: int, chunk, band: str) -> str:
    """
    Format a retrieved chunk for prompt injection.

    Important:
    We trim chunk text BEFORE sending to the LLM.
    This reduces prompt tokens and improves latency significantly.

    Full chunk text still exists in:
    - PostgreSQL
    - ChromaDB
    - retrieval results

    Only the prompt gets shortened.
    """

    source_label = chunk.filename

    if chunk.page_number:
        source_label += f", page {chunk.page_number}"

    header = (
        f"[Src {index}] {source_label} "
        f"| relevance: {chunk.similarity_score:.2f} ({band})"
    )

    # ── Token optimization ─────────────────────────────
    # 800 chars ≈ 180–220 tokens
    # Enough semantic context for grounded answers
    # while reducing latency substantially.
    text = chunk.text[:800]

    # Avoid cutting mid-sentence if possible
    if len(chunk.text) > 800:
        last_period = text.rfind(". ")

        if last_period > 400:
            text = text[: last_period + 1]

    return f"{header}\n{text}"
 
 
def _build_evidence_summary(blocks: list[SourceBlock]) -> str:
    counts = {"strong": 0, "moderate": 0, "weak": 0}
    for b in blocks:
        counts[b.score_band] += 1
    return f"{counts['strong']} strong, {counts['moderate']} moderate, {counts['weak']} weak"
 
 
def build_prompt(
    query: str,
    chunks: list[ChunkResult],
    conversation_history: list[dict] | None = None,
) -> AssembledPrompt:
    t0 = time.perf_counter()
 
    ranked = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
 
    blocks: list[SourceBlock] = []
    tokens_used = 0
    dropped = 0
 
    for i, chunk in enumerate(ranked, start=1):
        band = _score_band(chunk.similarity_score)
        formatted = _format_source_block(i, chunk, band)
        block_tokens = len(formatted) // CHARS_PER_TOKEN
 
        if tokens_used + block_tokens > MAX_CONTEXT_TOKENS:
            dropped += 1
            logger.debug(
                "prompt_chunk_dropped_budget",
                chunk_index=chunk.chunk_index,
                score=chunk.similarity_score,
            )
            continue
 
        blocks.append(SourceBlock(
            index=i,
            chunk=chunk,
            score_band=band,
            formatted=formatted,
        ))
        tokens_used += block_tokens
 
    evidence_summary = _build_evidence_summary(blocks)
    context_str = "\n\n---\n\n".join(b.formatted for b in blocks)
 
    user_content = (
        f"Source documents:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Evidence strength: {evidence_summary}\n\n"
        f"Question: {query}"
    )
 
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    if conversation_history:
        messages.extend(conversation_history[-6:])
 
    messages.append({"role": "user", "content": user_content})
 
    estimated_tokens = sum(len(m["content"]) // CHARS_PER_TOKEN for m in messages)
    assembly_ms = int((time.perf_counter() - t0) * 1000)
 
    if dropped:
        logger.warning(
            "prompt_chunks_dropped",
            dropped=dropped,
            used=len(blocks),
            tokens_used=tokens_used,
        )
 
    logger.info(
        "prompt_assembled",
        chunks_used=len(blocks),
        chunks_dropped=dropped,
        evidence_summary=evidence_summary,
        estimated_prompt_tokens=estimated_tokens,
        assembly_ms=assembly_ms,
    )
 
    return AssembledPrompt(
        messages=messages,
        source_blocks=blocks,
        chunks_used=len(blocks),
        chunks_dropped=dropped,
        estimated_prompt_tokens=estimated_tokens,
        evidence_summary=evidence_summary,
        assembly_ms=assembly_ms,
    )
 