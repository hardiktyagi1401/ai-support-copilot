"""
Query endpoints — retrieval and inspection.

Two endpoints, one service:

  POST /query/retrieve
      Production retrieval path. Returns scored chunks ready to be injected
      into an LLM prompt. When we add generation, this endpoint grows to
      include the answer. For now, it returns raw retrieval results.

  POST /query/inspect
      Engineering/debug path. Returns everything: extended candidates,
      score distribution, quality signals, embedding preview.
      This endpoint powers the debug panel in the frontend.

Architecture note:
    Both endpoints call the same underlying retrieval service functions.
    The difference is how much data they return and the extra parameters
    inspect accepts (candidate_multiplier, include_embedding_preview).

    This pattern — thin endpoint, fat service — keeps business logic
    out of the HTTP layer. If we added a WebSocket interface or a CLI
    tool later, they'd call the same service functions.
"""
import time
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.core.app_logging import get_logger
from app.models.schemas import (
    InspectRequest,
    InspectResult,
    QueryRequest,
    RetrievalResult,
)
from app.services.retrieval import retrieve, retrieve_with_inspection


from app.models.schemas import (
    AskRequest, AskResponse,
    RetrievalStats, LatencyBreakdown,
)

from app.services.prompt_assembler import build_prompt
from app.services.generation import generate
from app.services.query_logger import log_query



logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/query/retrieve",
    response_model=RetrievalResult,
    summary="Semantic retrieval — returns scored chunks for a query",
)
async def retrieve_endpoint(
    body: QueryRequest,
    request: Request,
) -> RetrievalResult:
    """
    Embed the query and retrieve the most semantically relevant chunks.

    Returns all top_k chunks with similarity scores. Chunks are marked
    with passed_threshold=True/False — the generation stage uses only
    the passing chunks when building the LLM prompt.

    At this stage (retrieval-only), no LLM call is made.
    Typical latency: 100–400ms (dominated by the embedding API call).
    """
    logger.info(
        "retrieve_request",
        query_preview=body.query[:80],
        tenant_id=body.tenant_id,
        top_k=body.top_k,
    )

    try:
        result = await retrieve(
            cache=request.app.state.embedding_cache,
            query=body.query,
            tenant_id=body.tenant_id,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            document_ids=body.document_ids,
        )
    except Exception as exc:
        logger.error("retrieve_failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    return result


@router.post(
    "/query/inspect",
    response_model=InspectResult,
    summary="Debug retrieval — returns full diagnostic data for a query",
)
async def inspect_endpoint(
    body: InspectRequest,
    request: Request,
) -> InspectResult:
    """
    Extended retrieval with full diagnostic output.

    Use this endpoint to:
    - Evaluate whether the right chunks are being retrieved
    - Understand why a query might return poor results
    - Tune the score_threshold for your document set
    - See what is just below the top_k cutoff

    Not intended for production chat use — the extra ChromaDB calls
    and statistical computation add ~50ms overhead.
    """
    logger.info(
        "inspect_request",
        query_preview=body.query[:80],
        tenant_id=body.tenant_id,
        top_k=body.top_k,
        candidate_multiplier=body.candidate_multiplier,
    )

    try:
        data = await retrieve_with_inspection(
            cache=request.app.state.embedding_cache,
            query=body.query,
            tenant_id=body.tenant_id,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            document_ids=body.document_ids,
            candidate_multiplier=body.candidate_multiplier,
            include_embedding_preview=body.include_embedding_preview,
        )
    except Exception as exc:
        logger.error("inspect_failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {exc}")

    return InspectResult(**data)



@router.post(
    "/query/ask",
    response_model=AskResponse,
    summary="Grounded RAG answer with citations",
)
async def ask_endpoint(
    body: AskRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> AskResponse:
    """
    Full RAG pipeline: retrieve → assemble → generate → respond.

    Returns a grounded answer with inline citations, retrieval stats,
    latency breakdown, and hallucination signals.

    answered=False means the system withheld an answer due to low
    retrieval confidence. The frontend should show a graceful refusal.
    """
    total_start = time.perf_counter()
    query_id = uuid4()

    logger.info(
        "ask_request",
        query_id=str(query_id),
        query_preview=body.query[:80],
        tenant_id=body.tenant_id,
    )

    # ── Stage 1: Retrieval ─────────────────────────────────────────
    retrieval_start = time.perf_counter()
    try:
        retrieval = await retrieve(
            cache=request.app.state.embedding_cache,
            query=body.query,
            tenant_id=body.tenant_id,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            document_ids=body.document_ids,
        )
        if cache := getattr(request.app.state, "embedding_cache", None):
         stats = await cache.stats()

        logger.info(
          "cache_stats",
          hit_rate=stats["hit_rate"],
          size=stats["size"],
        )
        
    except Exception as exc:
        logger.error("ask_retrieval_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    retrieval_stats = RetrievalStats(
        chunks_retrieved=retrieval.chunks_retrieved,
        chunks_passed_threshold=retrieval.chunks_passed_threshold,
        top_similarity_score=(
            retrieval.chunks[0].similarity_score if retrieval.chunks else None
        ),
        score_threshold_used=retrieval.score_threshold_used,
        embed_ms=retrieval.query_embedding_ms,
        search_ms=retrieval.search_ms,
    )

    # Chunks that will go to the LLM — only those that passed threshold
    grounded_chunks = [c for c in retrieval.chunks if c.passed_threshold]
    no_chunks = len(grounded_chunks) == 0

    # ── Stage 2: Prompt assembly ───────────────────────────────────
    assembly_start = time.perf_counter()
    assembled = build_prompt(
        query=body.query,
        chunks=grounded_chunks,
        conversation_history=body.conversation_history,
    )
    assembly_ms = int((time.perf_counter() - assembly_start) * 1000)

    # ── Stage 3: Generation ────────────────────────────────────────
    try:
        gen_result = await generate(
             assembled=assembled,
             ranked_chunks=grounded_chunks,
             no_chunks_passed_threshold=no_chunks,
            openai_client=getattr(request.app.state, "openai_client", None),
         )
    except Exception as exc:
        logger.error("ask_generation_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}")

    total_ms = int((time.perf_counter() - total_start) * 1000)

    response = AskResponse(
        query_id=query_id,
        answered=gen_result.answered,
        answer=gen_result.answer,
        refusal_reason=gen_result.refusal_reason,
        citations=[c.model_dump() for c in gen_result.citations],
       hallucination_flags=gen_result.hallucination_flags.model_dump(),
        retrieval_stats=retrieval_stats,
        latency=LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            prompt_assembly_ms=assembly_ms,
            llm_ms=gen_result.llm_ms,
            total_ms=total_ms,
        ),
        citation_coverage=gen_result.citation_coverage,
        uncited_sentence_count=gen_result.uncited_sentence_count,
        model_used=gen_result.model_used,
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
    )

    # ── Stage 4: Log (non-blocking) ────────────────────────────────
    background_tasks.add_task(log_query, query_id, body, response)

    logger.info(
        "ask_completed",
        query_id=str(query_id),
        answered=response.answered,
        citations_count=len(response.citations),
        total_ms=total_ms,
        hallucination_flags=response.hallucination_flags.any_flag_raised,
    )

    return response
