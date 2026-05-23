"""
Application configuration — loaded once at startup, validated by Pydantic.

Architecture note:
    All configuration lives here. Other modules import `get_settings()` rather
    than `os.getenv()` directly. This gives us:
      - Type safety: bad env vars fail at startup, not at runtime
      - Testability: tests can override settings via dependency injection
      - Single source of truth: one place to see every configurable parameter
      - IDE support: autocomplete on settings attributes

Production consideration:
    The @lru_cache decorator means settings are loaded exactly once per process.
    In tests, call get_settings.cache_clear() between tests that mutate env vars.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings, sourced from environment variables.
    Pydantic validates types and raises ValidationError on startup if any
    required variable is missing or has the wrong type.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Extra fields in .env are ignored, not raised as errors.
        # This allows gradual migration without breaking older deployments.
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────
    app_name: str = "AI Support Copilot"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    secret_key: str = Field(min_length=32)

    # ── PostgreSQL ─────────────────────────────────────────────
    database_url: str  # postgresql+asyncpg://...
    database_sync_url: str  # postgresql+psycopg2://... (alembic only)
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=5)

    # ── ChromaDB ──────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_collection_name: str = "support_docs"

    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = Field(min_length=20)
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_dimensions: int = 1536
    openai_max_retries: int = Field(default=3, ge=0, le=10)
    openai_timeout: int = Field(default=30, ge=5)

    # ── Ingestion Pipeline ─────────────────────────────────────
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    allowed_mime_types: list[str] = [
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    retrieval_top_k: int = Field(default=3, ge=1, le=20)
    retrieval_score_threshold: float = Field(default=0.72, ge=0.0, le=1.0)

    # ── Rate Limiting ──────────────────────────────────────────
    rate_limit_requests_per_minute: int = Field(default=60, ge=1)
    rate_limit_upload_per_hour: int = Field(default=20, ge=1)

    # ── Computed properties ────────────────────────────────────
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk(cls, v: int, info: any) -> int:
        """
        Overlap >= chunk_size would mean chunks are 100% duplicated content —
        a degenerate configuration that would break retrieval scoring.
        """
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size}). "
                f"Use 10-15% of chunk_size as a rule of thumb."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.

    Usage in FastAPI:
        from app.core.config import get_settings
        settings = get_settings()

    Usage as FastAPI dependency:
        def my_endpoint(settings: Settings = Depends(get_settings)):
            ...

    In tests:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        get_settings.cache_clear()
        settings = get_settings()
    """
    return Settings()  # type: ignore[call-arg]
