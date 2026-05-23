"""
Database engine and session management.

Architecture note:
    SQLAlchemy's async engine uses a connection pool under the hood.
    The pool keeps N connections open ("warm") so requests don't pay the
    TCP + SSL + auth cost on every query.

    Session lifecycle per request:
        1. FastAPI calls get_db() dependency → yields a session from the pool
        2. Route handler does work with the session
        3. If no exception: session.commit() is called
        4. Session is closed (returned to pool), not destroyed
        5. On exception: session.rollback() is called, then session closed

    The "yield" pattern in get_db() is crucial — it ensures cleanup (commit
    or rollback) runs even if an unhandled exception propagates out of the
    route handler.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)

_settings = get_settings()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    _settings.database_url,
    pool_size=_settings.database_pool_size,
    max_overflow=_settings.database_max_overflow,
    pool_timeout=_settings.database_pool_timeout,
    pool_pre_ping=True,
    echo=not _settings.is_production,
)

# ---------------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create all tables in development.
    """
    from app.models.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_tables_initialized")


async def close_db() -> None:
    """
    Dispose engine and pooled connections.
    """
    await engine.dispose()
    logger.info("database_connections_closed")

# ---------------------------------------------------------------------------
# FastAPI DB Dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides one DB session per request.
    """

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()

        except Exception:
            await session.rollback()
            raise