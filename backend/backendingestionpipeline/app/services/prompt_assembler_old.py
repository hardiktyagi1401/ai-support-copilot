"""
Prompt assembly for grounded RAG generation.

Responsibilities:
- Build the messages list that goes to the OpenAI API
- Enforce token budget so we never silently truncate
- Order chunks by relevance (highest score first)
- Format chunks with numbered source labels for citation
- Inject conversation history if present
- Produce consistent citation targets the parser can find

What this module does NOT do:
- Make any API calls
- Know about OpenAI models or their limits
- Parse the generated answer
- Check for hallucinations
"""

from dataclasses import dataclass
import time

from app.models.schemas import ChunkResult
from app.core.app_logging import get_logger

logger = get_logger(__name__)

# Approximate chars per token for budget estimation.
# GPT tokenizers average 3.8-4.2 chars/token. Using 4 is safe.
CHARS_PER_TOKEN = 4

# Reserve this many tokens for the answer.
# gpt-4o-mini max_tokens=4096. We use 16k context, reserve 1500 for answer.
ANSWER_TOKEN_RESERVE = 1500
MAX_CONTEXT_TOKENS = 3000  # chars / 4 = tokens

SYSTEM_PROMPT = """You are a precise document assistant. Your only job is to answer questions using the context documents provided below.

Strict rules you must follow:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. If the answer is not found in the context, respond with exactly: "I don't have information about that in the provided documents."
3. EVERY factual statement must include an inline citation using [Source N].
4. Do not write any sentence without a citation.
5. If different sources say different things, explicitly mention the conflict and cite both sources.
6. Prefer shorter sentences with citations throughout the answer.
7. Be concise and factual."""


@dataclass
class AssembledPrompt:
    messages: list[dict]
    chunks_used: int
    chunks_dropped: int  # dropped due to token budget
    estimated_prompt_tokens: int
    assembly_ms: int


def build_prompt(
    query: str,
    chunks: list[ChunkResult],
    conversation_history: list[dict] | None = None,
) -> AssembledPrompt:
    """
    Build the complete messages list for the OpenAI API.

    Ordering: highest similarity score first. The model weights
    early context more heavily, so the most relevant chunk goes first.

    Token budget: we estimate tokens by character count. If a chunk
    would push us over MAX_CONTEXT_TOKENS, we drop it and all subsequent
    chunks. The dropped count is surfaced in the response so callers
    know the budget was hit.
    """
    t0 = time.perf_counter()

    # Sort by score descending — best chunk first
    ranked = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)

    # Build context blocks with token budget enforcement
    context_blocks = []
    tokens_used = 0
    dropped = 0

    for i, chunk in enumerate(ranked, start=1):
        source_label = chunk.filename
        if chunk.page_number:
            source_label += f", page {chunk.page_number}"

        block = f"[Source {i}] {source_label}\n{chunk.text}"
        block_tokens = len(block) // CHARS_PER_TOKEN

        if tokens_used + block_tokens > MAX_CONTEXT_TOKENS:
            dropped += 1
            continue

        context_blocks.append(block)
        tokens_used += block_tokens

    if not context_blocks:
        # All chunks dropped — this should not happen in normal flow
        # because the endpoint checks for passed_threshold first.
        # But be defensive.
        logger.warning("prompt_assembler_no_chunks", total_chunks=len(chunks))

    context_str = "\n\n---\n\n".join(context_blocks)

    # Build the user message
    # Keep context and question structurally separate.
    # The separator "---" signals a hard boundary to the model.
    user_content = f"Context documents:\n\n{context_str}\n\n---\n\nQuestion: {query}"

    # Assemble messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject recent conversation history BETWEEN system and current user message.
    # This gives the model context on what was already discussed.
    # Cap at 3 turns (6 messages) to control token usage.
    if conversation_history:
        capped = conversation_history[-6:]  # last 3 user+assistant pairs
        messages.extend(capped)

    messages.append({"role": "user", "content": user_content})

    estimated_tokens = sum(len(m["content"]) // CHARS_PER_TOKEN for m in messages)
    assembly_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "prompt_assembled",
        chunks_used=len(context_blocks),
        chunks_dropped=dropped,
        estimated_prompt_tokens=estimated_tokens,
        assembly_ms=assembly_ms,
    )

    return AssembledPrompt(
        messages=messages,
        chunks_used=len(context_blocks),
        chunks_dropped=dropped,
        estimated_prompt_tokens=estimated_tokens,
        assembly_ms=assembly_ms,
    )