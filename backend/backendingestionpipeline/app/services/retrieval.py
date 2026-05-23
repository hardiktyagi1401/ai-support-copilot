"""
Semantic retrieval service.

Architecture note:
    This service owns exactly one responsibility: given a query string and
    a tenant context, return the most semantically relevant chunks.

    It does NOT:
    - Generate answers (that is generation.py, not built yet)
    - Decide what to do if results are poor (that is the caller's job)
    - Log query analytics (that is the endpoint's job)

    This separation means retrieval can be called from multiple places
    (the /retrieve endpoint, the /inspect endpoint, a batch eval script)
    without duplicating logic.

Distance → similarity conversion:
    ChromaDB with hnsw:space="cosine" returns distances in [0, 2].
    - Distance 0.0 means the vectors are identical (cos similarity = 1.0)
    - Distance 1.0 means the vectors are orthogonal (cos similarity = 0.0)
    - Distance 2.0 means the vectors point in opposite directions (cos similarity = -1.0)

    Formula: similarity = 1.0 - (distance / 2.0)
    This maps distance [0, 2] → similarity [1, 0], which is what we want.

    WARNING: some ChromaDB versions / collection configs return distance
    in [0, 1] instead of [0, 2]. If your scores look wrong (all 0.5ish),
    check with a known-identical query to confirm the range. The log line
    "raw_distance=X" will show you the raw value.

Tenant isolation:
    Every ChromaDB query includes a 'where' filter on tenant_id.
    This is enforced at the service layer, not just the API layer,
    so even internal calls (from background jobs, etc.) are isolated.

    This is a defense-in-depth pattern: even if the API layer has a bug
    that passes the wrong tenant_id, the service layer still enforces it.
"""

import time
import statistics
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI
from app.services.embedding_cache import EmbeddingCacheProtocol
from app.services.cache_embedding_integration import embed_query_cached
from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.db.chroma import get_or_create_collection
from app.models.schemas import (
    ChunkResult,
    RetrievalResult,
    ScoreDistribution,
)

logger = get_logger(__name__)


@dataclass
class RawChromaResult:
    """Typed wrapper around ChromaDB's dict-of-lists response format."""
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]


# In retrieval.py — REPLACE the entire embed_query() function (lines ~40-75)
# with this. Everything else in the file stays untouched.

async def embed_query(query: str) -> tuple[list[float], int]:
    """
    Embed a single query string by delegating to the shared embedder.

    Reuses embed_chunks() with a synthetic TextChunk so the same
    OpenAI client instance, retry config, and model pin are used for
    both ingestion and retrieval. A mismatch here would produce
    silently wrong similarity scores.

    Returns (embedding_vector, latency_ms).
    """
    from app.services.chunker import TextChunk  # local import avoids circular

    t0 = time.perf_counter()

    # Wrap query in a minimal TextChunk — embedder only reads .text
    synthetic_chunk = TextChunk(
        text=query,
        chunk_index=0,
        page_number=None,
        token_count=len(query.split()),  # approximate — not stored
    )

    embedded = await embed_chunks([synthetic_chunk])

    if not embedded:
        raise ValueError("embed_chunks returned empty result for query")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    embedding = embedded[0].embedding

    logger.info(
        "query_embedded",
        query_preview=query[:80],
        dimensions=len(embedding),
        latency_ms=latency_ms,
    )

    return embedding, latency_ms


def _distance_to_similarity(distance: float) -> float:
    """
    Convert ChromaDB cosine distance [0, 2] to similarity score [0, 1].

    Clamps to [0, 1] to handle any floating point edge cases.
    """
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def _build_chroma_filter(
    tenant_id: str,
    document_ids: list[uuid.UUID] | None,
) -> dict:
    """
    Build the ChromaDB 'where' filter for a query.

    ChromaDB's filter syntax uses $and, $or for compound conditions.
    Always include tenant_id. Optionally restrict to specific document_ids.

    Single condition: {"tenant_id": tenant_id}
    Multiple conditions: {"$and": [{"tenant_id": ...}, {"document_id": {"$in": [...]}}]}
    """
    if not document_ids:
        return {"tenant_id": tenant_id}

    doc_id_strings = [str(d) for d in document_ids]

    return {
        "$and": [
            {"tenant_id": tenant_id},
            {"document_id": {"$in": doc_id_strings}},
        ]
    }


def _parse_chroma_response(
    chroma_result: dict,
    threshold: float,
) -> list[ChunkResult]:
    """
    Parse ChromaDB's response format into typed ChunkResult objects.

    ChromaDB returns dicts-of-lists:
        {
            "ids": [["id1", "id2"]],        # outer list is per-query (we send 1)
            "documents": [["text1", ...]],
            "metadatas": [[{...}, {...}]],
            "distances": [[0.12, 0.34]],
        }

    We unwrap the outer list (index [0]) and zip the parallel lists.
    """
    ids = chroma_result["ids"][0]
    documents = chroma_result["documents"][0]
    metadatas = chroma_result["metadatas"][0]
    distances = chroma_result["distances"][0]

    results = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        similarity = _distance_to_similarity(distance)
        results.append(ChunkResult(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", "unknown"),
            page_number=meta.get("page_number") or None,
            chunk_index=meta.get("chunk_index", 0),
            text=text,
            raw_distance=distance,
            similarity_score=round(similarity, 4),
            passed_threshold=similarity >= threshold,
            token_count=meta.get("token_count"),
            embedding_model=meta.get("embedding_model"),
        ))

    # Sort by similarity descending (ChromaDB already does this by distance,
    # but we re-sort after conversion to be explicit and safe)
    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results


async def retrieve(
    query: str,
    tenant_id: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
    cache: EmbeddingCacheProtocol | None = None,
) -> RetrievalResult:
    """
    Core retrieval function. Embeds query, searches ChromaDB, scores results.

    This is the production retrieval path used by the chat endpoint.
    For debugging, use retrieve_with_inspection() which returns more detail.
    """
    
    settings = get_settings()

    k = top_k or settings.retrieval_top_k

    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

    total_start = time.perf_counter()

    # Stage 1: embed the query
    if cache is not None:
     embedding, embed_ms, cache_hit = await embed_query_cached(
        query=query,
        cache=cache,
    )
    else:
     embedding, embed_ms = await embed_query(query)
     cache_hit = False

    # Stage 2: search ChromaDB
    collection = get_or_create_collection()
    where_filter = _build_chroma_filter(tenant_id, document_ids)

    search_start = time.perf_counter()
    try:
        chroma_result = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error(
            "chroma_query_failed",
            tenant_id=tenant_id,
            top_k=k,
            error=str(exc),
        )
        raise
    search_ms = int((time.perf_counter() - search_start) * 1000)

    # Stage 3: parse and score
    chunks = _parse_chroma_response(chroma_result, threshold)
    passed = [c for c in chunks if c.passed_threshold]
    total_ms = int((time.perf_counter() - total_start) * 1000)

    logger.info(
        "retrieval_complete",
        query_preview=query[:80],
        tenant_id=tenant_id,
        top_k=k,
        chunks_retrieved=len(chunks),
        chunks_passed=len(passed),
        top_score=chunks[0].similarity_score if chunks else None,
        threshold=threshold,
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
    )

    return RetrievalResult(
        query=query,
        query_embedding_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
        chunks_retrieved=len(chunks),
        chunks_passed_threshold=len(passed),
        score_threshold_used=threshold,
        chunks=chunks,
        embedding_model=settings.openai_embedding_model,
        tenant_id=tenant_id,
    )


async def retrieve_with_inspection(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    score_threshold: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
    candidate_multiplier: int = 3,
    include_embedding_preview: bool = False,
    cache: EmbeddingCacheProtocol | None = None,
) -> dict:
    """
    Extended retrieval that returns full diagnostic data.

    The candidate_multiplier is the key insight here:
    We fetch top_k × multiplier candidates internally, then return
    top_k to the caller. The "extended" candidates show what ChromaDB
    found just outside the top_k cutoff — this is essential for
    diagnosing whether your top_k setting is too restrictive.

    Example: top_k=5, multiplier=3 → fetch 15, return 5 + show next 10.
    If chunk #6 has a score of 0.71 and your threshold is 0.72, you
    can see exactly what you're missing and why.
    """
    settings = get_settings()
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

    total_start = time.perf_counter()
    if cache is not None:
      embedding, embed_ms, cache_hit = await embed_query_cached(
        query=query,
        cache=cache,
    )
    else:
      embedding, embed_ms = await embed_query(query)
      cache_hit = False

    # Fetch more candidates than needed for extended inspection
    extended_k = top_k * candidate_multiplier
    collection = get_or_create_collection()
    where_filter = _build_chroma_filter(tenant_id, document_ids)

    search_start = time.perf_counter()
    chroma_result = collection.query(
        query_embeddings=[embedding],
        n_results=min(extended_k, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    search_ms = int((time.perf_counter() - search_start) * 1000)
    total_ms = int((time.perf_counter() - total_start) * 1000)

    all_chunks = _parse_chroma_response(chroma_result, threshold)
    top_chunks = all_chunks[:top_k]
    extended_chunks = all_chunks[top_k:]

    # Build score distribution from all candidates
    score_dist = None
    if len(all_chunks) >= 2:
        scores = [c.similarity_score for c in all_chunks]
        top_scores = sorted(scores, reverse=True)
        score_dist = ScoreDistribution(
            min_score=round(min(scores), 4),
            max_score=round(max(scores), 4),
            mean_score=round(statistics.mean(scores), 4),
            median_score=round(statistics.median(scores), 4),
            std_dev=round(statistics.stdev(scores), 4),
            score_gap=round(top_scores[0] - top_scores[1], 4) if len(top_scores) > 1 else 0.0,
        )

    # Quality signal and diagnostics
    quality, diagnostics = _assess_retrieval_quality(top_chunks, threshold, score_dist)

    retrieval_result = RetrievalResult(
        query=query,
        query_embedding_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
        chunks_retrieved=len(top_chunks),
        chunks_passed_threshold=len([c for c in top_chunks if c.passed_threshold]),
        score_threshold_used=threshold,
        chunks=top_chunks,
        embedding_model=settings.openai_embedding_model,
        tenant_id=tenant_id,
    )

    return {
        "retrieval": retrieval_result,
        "extended_candidates": extended_chunks,
        "score_distribution": score_dist,
        "query_embedding_preview": embedding[:10] if include_embedding_preview else None,
        "retrieval_quality_signal": quality,
        "diagnostics": diagnostics,
    }


def _assess_retrieval_quality(
    chunks: list[ChunkResult],
    threshold: float,
    dist: ScoreDistribution | None,
) -> tuple[str, list[str]]:
    """
    Produce a human-readable quality signal and specific diagnostic observations.

    This is deterministic rule-based assessment — not LLM generated.
    It runs in microseconds and is fully explainable.

    Quality tiers:
        strong   — top score >= 0.85, at least 1 chunk passed threshold
        moderate — top score >= threshold, at least 1 chunk passed
        weak     — some chunks retrieved but none passed threshold
        no_results — ChromaDB returned nothing
    """
    diagnostics: list[str] = []

    if not chunks:
        return "no_results", ["No chunks retrieved. The vector store may be empty for this tenant/filter."]

    top_score = chunks[0].similarity_score
    passed = [c for c in chunks if c.passed_threshold]

    if not passed:
        quality = "weak"
        diagnostics.append(
            f"Top score {top_score:.3f} is below threshold {threshold:.3f}. "
            f"No chunks would be sent to the LLM. Consider lowering the threshold "
            f"or rephrasing the query."
        )
    elif top_score >= 0.85:
        quality = "strong"
        diagnostics.append(f"Strong match: top chunk scored {top_score:.3f}.")
    else:
        quality = "moderate"
        diagnostics.append(f"Moderate match: top score {top_score:.3f}.")

    if dist and dist.score_gap < 0.05 and len(chunks) > 1:
        diagnostics.append(
            f"Score gap between rank-1 and rank-2 is only {dist.score_gap:.3f}. "
            f"Retrieval is ambiguous — multiple chunks are nearly equally relevant."
        )

    if dist and dist.std_dev < 0.03:
        diagnostics.append(
            "Low score variance across chunks. The query may be too generic, "
            "causing many chunks to score similarly."
        )

    unique_docs = {c.document_id for c in chunks}
    if len(unique_docs) == 1 and len(chunks) > 1:
        diagnostics.append(
            f"All {len(chunks)} retrieved chunks come from the same document. "
            f"This is expected if you only have one document, but may indicate "
            f"imbalanced coverage if you have multiple."
        )

    if any(c.embedding_model != chunks[0].embedding_model for c in chunks if c.embedding_model):
        diagnostics.append(
            "WARNING: retrieved chunks have mixed embedding models. "
            "Scores are not comparable across chunks. Re-embed all documents with the same model."
        )

    return quality, diagnostics
