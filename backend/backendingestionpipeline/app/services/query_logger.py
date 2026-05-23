"""
Query logging — fire-and-forget persistence.

This runs as a FastAPI BackgroundTask after the response is sent.
It must NEVER raise an exception that would affect the HTTP response.
Wrap everything in try/except. Log errors internally but swallow them.

Design choice: we write to QueryLog as a background task rather than
awaiting it inline. This keeps p99 response latency clean — a slow
database write doesn't penalize the user.

Tradeoff: if the server crashes between response and background task
execution, the log is lost. For V1 this is acceptable. For production,
use a proper queue (Redis, Postgres LISTEN/NOTIFY) for durability.
"""

import json
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.app_logging import get_logger
from app.db.session import AsyncSessionLocal
from app.models.database import QueryLog
from app.models.schemas import AskRequest, AskResponse

logger = get_logger(__name__)


async def log_query(
    query_id: UUID,
    request: AskRequest,
    response: AskResponse,
) -> None:
    """
    Persist a query log entry. Called as BackgroundTask — never awaited inline.
    """
    try:
        async with AsyncSessionLocal() as db:
            # Serialize retrieved chunk IDs for storage
            chunk_ids = ",".join(
                c.filename for c in response.citations
            ) if response.citations else None

            record = QueryLog(
                id=query_id,
                tenant_id=request.tenant_id,
                session_id=None,  # add when you implement conversation sessions
                query_text=request.query,
                response_text=response.answer[:2000] if response.answer else None,
                retrieved_chunk_ids=chunk_ids,
                top_retrieval_score=response.retrieval_stats.top_similarity_score,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency.total_ms,
                was_answered=response.answered,
            )
            db.add(record)
            await db.commit()

            logger.info(
                "query_logged",
                query_id=str(query_id),
                answered=response.answered,
                latency_ms=response.latency.total_ms,
                hallucination_flags_raised=response.hallucination_flags.any_flag_raised,
            )

    except Exception as exc:
        # Log the error but never propagate — this runs after the response is sent
        logger.error(
            "query_log_failed",
            query_id=str(query_id),
            error=str(exc),
            exc_info=True,
        )