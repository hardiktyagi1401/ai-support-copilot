"""
Health check endpoints.

Architecture note:
    Two endpoints with different purposes:

    GET /health — liveness probe
        Returns 200 if the process is running. Does NOT check dependencies.
        Kubernetes/Railway uses this to know if the container is alive.
        If this returns non-200, the container is killed and restarted.
        Should NEVER check external dependencies — a DB outage shouldn't cause
        container restarts when the app itself is healthy.

    GET /health/ready — readiness probe
        Returns 200 only if all dependencies are reachable.
        Kubernetes uses this to know if the pod should receive traffic.
        If DB is down, readiness fails → traffic is routed elsewhere.
        If liveness still passes → container stays alive (no restart).

    This distinction is critical in production. Without it, a database blip
    causes container restarts, which makes the DB outage worse.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.db.chroma import get_chroma_client
from app.db.session import engine

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    app_env: str
    version: str = "0.1.0"


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness probe — is the process running?"""
    settings = get_settings()
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    """Readiness probe — are all dependencies reachable?"""
    checks: dict[str, str] = {}
    all_ok = True

    # Check PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgresql"] = "ok"
    except Exception as exc:
        checks["postgresql"] = f"error: {exc}"
        all_ok = False

    # Check ChromaDB
    try:
        get_chroma_client().heartbeat()
        checks["chromadb"] = "ok"
    except Exception as exc:
        checks["chromadb"] = f"error: {exc}"
        all_ok = False

    status = "ok" if all_ok else "degraded"
    return ReadinessResponse(status=status, checks=checks)
