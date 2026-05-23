"""
Structured logging configuration using structlog.

Architecture note:
    We use structlog rather than Python's built-in logging for two reasons:
    1. It emits JSON natively — parseable by every log aggregation platform
    2. It supports "context binding" — you can attach a request_id at the
       start of a request, and every subsequent log statement in that request
       automatically includes it, without passing it through every function call.

Production consideration:
    In development, logs are human-readable (colored, indented).
    In production (APP_ENV=production), logs are JSON — one object per line.
    The switch is automatic based on APP_ENV.

Failure modes:
    - Logging too much (DEBUG in production) can fill disk and create I/O pressure.
      Set LOG_LEVEL=INFO in production.
    - Logging sensitive data (API keys, PII in document content) in DEBUG mode.
      Never log raw document text or user queries at INFO or above.

Usage:
    from app.core.applogging import get_logger
    logger = get_logger(__name__)
    logger.info("chunk_embedded", chunk_index=3, duration_ms=145)
    logger.error("embedding_failed", error=str(exc), document_id=doc_id)
"""

import logging
import sys

import structlog
from structlog.types import EventDict, WrappedLogger

from app.core.config import get_settings


def _drop_color_message_key(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Uvicorn adds a 'color_message' key that duplicates 'event' with ANSI codes.
    This processor removes it so our JSON logs stay clean.
    """
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """
    Call once at application startup (in main.py lifespan).
    Idempotent — safe to call multiple times in tests.
    """
    settings = get_settings()
    is_production = settings.is_production
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Shared processors run on every log entry, regardless of renderer
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,        # injects request_id, etc.
        structlog.stdlib.add_log_level,                  # adds "level" field
        structlog.stdlib.add_logger_name,                # adds "logger" field
        structlog.processors.TimeStamper(fmt="iso"),     # adds "timestamp" field
        _drop_color_message_key,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_production:
        # JSON output for log aggregation platforms (Datadog, CloudWatch, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable colored output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            # ExceptionRenderer must come before the final renderer
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so that libraries (SQLAlchemy, uvicorn, etc.)
    # route through structlog and get the same format.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Quiet down chatty libraries in production
    if is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named logger. Use __name__ as the convention.

    Example:
        logger = get_logger(__name__)
        logger.info("document_uploaded", document_id=doc_id, size_bytes=size)

    The name becomes the "logger" field in the JSON output, letting you filter
    logs by module in your log aggregation platform.
    """
    return structlog.get_logger(name)


def bind_request_context(request_id: str, tenant_id: str | None = None) -> None:
    """
    Bind values to the current async context so all subsequent log calls
    in this request automatically include them — without passing them through
    every function signature.

    Call this in a middleware at the start of each request.
    structlog uses contextvars under the hood, so it's async-safe.
    """
    structlog.contextvars.clear_contextvars()
    ctx = {"request_id": request_id}
    if tenant_id:
        ctx["tenant_id"] = tenant_id
    structlog.contextvars.bind_contextvars(**ctx)
