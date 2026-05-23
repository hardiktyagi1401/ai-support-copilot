"""
SQLAlchemy ORM models — the shape of our PostgreSQL schema.

Architecture note:
    We use SQLAlchemy 2.0's "mapped column" style, which provides full type
    safety and IDE support. Each model maps to one PostgreSQL table.

    Two layers of data live in this system:
    - PostgreSQL (here): ownership, status, metadata, analytics, audit logs
    - ChromaDB (elsewhere): the actual embedding vectors + chunk text

    They're linked by document_id (UUID). This separation means:
    - You can query "all docs for tenant X" in SQL — fast, relational, filtered
    - You can retrieve vectors by document_id filter — isolating tenants in ChromaDB

Design decisions:
    - UUIDs as primary keys: avoids integer enumeration attacks, works across
      distributed systems, and is safe to expose in URLs.
    - created_at/updated_at on every table: required for audit trails and debugging.
    - Explicit status enum: avoids magic string bugs ("procesing" vs "processing").
    - Storing chunk_count after ingestion: lets us validate ChromaDB integrity.

Failure modes:
    - Not adding indexes on foreign keys: slow JOINs at scale. We add them explicitly.
    - Storing large blobs in PostgreSQL: document content goes to object storage
      (S3) in production, not in a TEXT column. We only store metadata here.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    All tables get created_at and updated_at automatically via server_default
    and onupdate, so we never forget to set them manually.
    """
    pass


class DocumentStatus(str, enum.Enum):
    """
    Lifecycle of a document through the ingestion pipeline.

    PENDING    → uploaded, waiting for background worker
    PROCESSING → worker has picked it up, pipeline running
    READY      → fully embedded, available for retrieval
    FAILED     → pipeline failed, error stored in error_message
    ARCHIVED   → soft-deleted, excluded from retrieval

    Using str enum means these serialize as plain strings in JSON responses,
    not as {"value": "ready", "name": "READY"} which is the default enum behavior.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class Document(Base):
    """
    Represents one uploaded document — the top-level entity in the ingestion pipeline.

    One document → many DocumentChunks (after processing).
    One document → many QueryLogs (usage tracking).
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Public-facing ID, safe to expose in URLs",
    )
    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,  # Every query filters by tenant — this index is critical
        comment="Business account identifier for multi-tenant isolation",
    )
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original filename as uploaded by the user",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="Original file size — for storage billing and display",
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status_enum"),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True,
        comment="Current stage in the ingestion pipeline",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Set when status=FAILED, contains the exception message",
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Set after processing — how many chunks were created",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Which embedding model was used — critical for retrieval correctness",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",  # deleting a document deletes its chunks
        lazy="selectin",               # avoids N+1 queries
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class DocumentChunk(Base):
    """
    Represents one chunk of a document after splitting.

    Stores chunk metadata in PostgreSQL. The actual embedding vector lives
    in ChromaDB, referenced by the same chunk_id.

    Why store chunks in Postgres at all if ChromaDB has them?
    - Postgres lets you JOIN chunks with documents, filter by tenant, sort by date
    - ChromaDB only supports vector similarity + metadata filters — no JOINs
    - Having chunks in Postgres means you can verify ChromaDB integrity:
      COUNT(chunks WHERE document_id=X) should equal ChromaDB's count for doc X
    """
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Also used as the ChromaDB vector ID — must match exactly",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position of this chunk within the document (0-indexed)",
    )
    text_preview: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="First 500 chars of chunk text — for debug panel display",
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Actual token count after chunking — for prompt budget tracking",
    )
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="PDF page number — enables source citation with page reference",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} index={self.chunk_index}>"


class QueryLog(Base):
    """
    Records every query made against the system.

    Purposes:
    1. Analytics dashboard — queries per day, avg latency, avg score
    2. Cost tracking — token usage per tenant for billing
    3. Quality monitoring — flag low-confidence answers for human review
    4. Debugging — reproduce exactly what happened for a given query

    In production, this table grows fast. Plan for:
    - Partitioning by created_at (monthly partitions)
    - Archiving records older than 90 days to cold storage
    - A read replica for analytics queries to avoid impacting the write path
    """
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Groups multiple queries in the same conversation",
    )
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    response_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Null if no context was retrieved (below threshold)",
    )
    retrieved_chunk_ids: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Comma-separated chunk UUIDs — for tracing which chunks answered this",
    )
    top_retrieval_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Highest cosine similarity score among retrieved chunks",
    )
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total end-to-end latency including embedding + retrieval + generation",
    )
    was_answered: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="False when confidence threshold not met — answer withheld",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
