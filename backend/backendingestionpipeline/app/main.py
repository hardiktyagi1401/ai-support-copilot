"""
FastAPI application entry point.

Architecture note:
    We use the application factory pattern — create_app() returns the configured
    FastAPI instance. This pattern enables:
    - Testing: tests can call create_app() to get a fresh instance each time
    - Multiple environments: different settings without modifying module-level state
    - Clean startup/shutdown ordering via the lifespan context manager

Lifespan ordering (critical — do not reorder):
    1. Configure logging first — so all subsequent startup logs are structured
    2. Verify external dependencies (DB, ChromaDB, OpenAI reachability)
    3. Initialize database tables (dev) or validate schema (prod)
    4. Register routers last

Middleware ordering:
    FastAPI middleware runs in reverse registration order (last registered = outermost).
    Our ordering (outermost first):
        CORS → Request ID injection → Logging → Route handlers

Failure modes:
    - External dependency down at startup: process crashes, clear error in logs
    - Router import fails: Python import error at startup — good
    - Missing env var: Pydantic ValidationError before lifespan runs — good
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.embedding_cache import build_cache
from app.core.config import get_settings
from app.core.app_logging import configure_logging, get_logger
from app.db.chroma import get_or_create_collection, verify_chroma_connection
from app.db.session import close_db, init_db
from app.middleware import RequestIDMiddleware, RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs startup code before the first request and shutdown code after the last.

    The `yield` separates startup (above) from shutdown (below).
    FastAPI guarantees the shutdown block runs even if a request handler raises.
    """

    # ── Startup ───────────────────────────────────────────────
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info(
        "application_starting",
        app_env=settings.app_env,
        log_level=settings.log_level,
    )

    # Shared OpenAI client (connection pooling + lower latency variance)
    app.state.openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
    )

    logger.info("openai_client_initialized")

    # Verify ChromaDB is reachable before accepting any requests.
    verify_chroma_connection()

    # Initialize ChromaDB collection
    collection = get_or_create_collection()
    app.state.chroma_collection = collection

    # Initialize PostgreSQL
    await init_db()

    # Embedding cache
    app.state.embedding_cache = build_cache()

    logger.info(
        "application_ready",
        vector_count=collection.count(),
    )

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("application_shutting_down")

    await close_db()

    logger.info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        # Disable docs in production — they expose your API schema publicly.
        # In production, use a separate internal docs deployment.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    _register_middleware(app, settings)
    _register_routers(app)

    return app


def _register_middleware(app: FastAPI, settings: any) -> None:
    """
    Register middleware in innermost-first order.
    FastAPI wraps them in reverse, so the last one registered is outermost.
    """
    # Innermost: request logging (closest to route handlers)
    app.add_middleware(RequestLoggingMiddleware)

    # Middle: inject a unique request_id into every request
    app.add_middleware(RequestIDMiddleware)

    # Outermost: CORS (first to handle preflight requests)
    allowed_origins = (
        ["*"] if not settings.is_production
        else [
            "https://your-frontend.vercel.app",
            "https://app.yourproduct.com",
        ]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def _register_routers(app: FastAPI) -> None:
    """Import and register all API routers."""
    from app.api.v1 import ingest, query, health

    # All routes are versioned under /api/v1/
    # This lets us introduce breaking changes under /api/v2/ without
    # removing /api/v1/ — critical for enterprise customers with long migration cycles.
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])


# The FastAPI app instance — imported by uvicorn
app = create_app()
