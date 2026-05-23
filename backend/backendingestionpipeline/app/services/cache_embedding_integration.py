import time

from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.services.embedding_cache import (
    EmbeddingCacheProtocol,
    make_cache_key,
)

logger = get_logger(__name__)


async def embed_query_cached(
    query: str,
    cache: EmbeddingCacheProtocol,
) -> tuple[list[float], int, bool]:

    settings = get_settings()
    model = settings.openai_embedding_model

    cache_key = make_cache_key(query, model)

    t0 = time.perf_counter()

    # CACHE HIT
    cached_embedding = await cache.get(cache_key)

    if cached_embedding is not None:
        latency_ms = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "embedding_cache_hit",
            key_prefix=cache_key[:8],
            latency_ms=latency_ms,
        )

        return cached_embedding, latency_ms, True

    # CACHE MISS
    from app.services.retrieval import embed_query

    embedding, api_latency_ms = await embed_query(query)

    try:
        await cache.set(cache_key, embedding)
    except Exception as exc:
        logger.warning(
            "embedding_cache_write_failed",
            error=str(exc),
        )

    total_latency_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "embedding_cache_miss",
        key_prefix=cache_key[:8],
        api_latency_ms=api_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    return embedding, total_latency_ms, False