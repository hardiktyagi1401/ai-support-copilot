"""
Custom ASGI middleware for request correlation and logging.

Architecture note:
    Middleware runs on every request, before any route handler.
    We have two pieces of middleware here:

    RequestIDMiddleware:
        Assigns a unique ID to every request. This ID propagates through:
        - The X-Request-ID response header (the client can use it for support tickets)
        - structlog's context vars (every log line in this request gets request_id)
        - The response header (frontend can log it too)

    RequestLoggingMiddleware:
        Logs every request with method, path, status code, and duration.
        This gives you a complete access log in structured format.

Why not use uvicorn's built-in access log?
    Uvicorn logs are unstructured strings. Our middleware emits JSON with all
    fields parseable — you can aggregate "avg latency for POST /api/v1/ingest"
    in Grafana or Datadog without regex parsing.

Failure modes:
    - Logging request body: never do this (PII, file content, secrets)
    - Logging headers: careful — Authorization header contains tokens
    - Exception in middleware: propagates and may return 500 without request_id header
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.app_logging import bind_request_context, get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request_id into every request.

    The ID is either taken from an incoming X-Request-ID header
    (so upstream load balancers or API gateways can set their own)
    or generated fresh as a UUID4.
    """

    async def dispatch(self, request: Request, call_next: any) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store on request.state so route handlers can access it
        request.state.request_id = request_id

        # Bind to structlog context — all log lines in this request get request_id
        bind_request_context(
            request_id=request_id,
            tenant_id=request.headers.get("X-Tenant-ID"),
        )

        response = await call_next(request)

        # Return the request_id to the caller — useful for support tickets
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with method, path, status code, and duration.

    Skip health check endpoints to avoid filling logs with noise
    from load balancer health probes (they hit /health every 5-30 seconds).
    """

    SKIP_PATHS = {"/api/v1/health", "/health", "/ping"}

    async def dispatch(self, request: Request, call_next: any) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000)

        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            # Do NOT log: request body, headers (may contain auth tokens)
        )

        return response
