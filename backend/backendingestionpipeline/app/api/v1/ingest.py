"""
Document ingestion endpoints.

Architecture note:
    The upload endpoint is intentionally minimal — it validates, saves, records,
    and returns. All heavy processing happens in a background task.

    Request lifecycle:
        POST /upload  →  validate  →  save to disk  →  DB record (PENDING)  →  200ms response
                                                              ↓
                                                    BackgroundTask runs:
                                                    extract → chunk → embed → store
                                                              ↓
                                                    DB record updated (READY or FAILED)

    The client polls GET /documents/{id}/status until status=READY.

File storage:
    In this implementation, files are saved to local disk (/tmp/uploads/).
    In production, upload directly to S3/GCS and store the object URL in the DB.
    Local disk doesn't survive container restarts, isn't shared across workers,
    and has no redundancy. Even on Railway, local disk is ephemeral.

    For now, local disk is fine for learning — just know this is the first
    thing to change before going to real production.

Validation layers:
    1. File size: checked before reading the entire file into memory
    2. MIME type: from the Content-Type header — client-controlled, can be spoofed
    3. Magic bytes: check actual file bytes to confirm type (not implemented here,
       but add python-magic for production to prevent MIME type spoofing attacks)

Failure modes:
    - Uploading the same file twice: currently creates duplicate documents.
      Production: hash the file content (SHA256), check for existing hash in DB.
    - Disk full: save() raises an OSError — return 507 Insufficient Storage.
    - Background task fails silently: we catch exceptions and set status=FAILED.
      The client polling /status sees FAILED and the error_message.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.db.session import get_db
from app.models.database import Document, DocumentStatus
from app.models.schemas import DocumentStatusResponse, DocumentUploadResponse
from app.services.ingestion_pipeline import run_ingestion_pipeline

logger = get_logger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("/tmp/copilot_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/ingest/upload",
    response_model=DocumentUploadResponse,
    status_code=202,  # 202 Accepted: request received, processing not complete
    summary="Upload a document for ingestion",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    x_tenant_id: str = Header(..., description="Tenant identifier for multi-tenant isolation"),
) -> DocumentUploadResponse:
    """
    Accept a document upload and queue it for background processing.

    Returns immediately (202) with a document_id.
    Poll GET /ingest/documents/{document_id}/status to track processing.
    """
    settings = get_settings()

    # ── Validation ──────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if file.content_type not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(settings.allowed_mime_types)}",
        )

    # Read file into memory to check size.
    # For very large files, use a streaming approach instead of reading all at once.
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # ── Persist file to disk ─────────────────────────────────────
    document_id = uuid.uuid4()
    safe_filename = f"{document_id}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_filename

    try:
        file_path.write_bytes(file_bytes)
    except OSError as exc:
        logger.error("file_save_failed", error=str(exc), filename=file.filename)
        raise HTTPException(status_code=507, detail="Failed to save file. Storage may be full.")

    # ── Create DB record ─────────────────────────────────────────
    document = Document(
        id=document_id,
        tenant_id=x_tenant_id,
        filename=file.filename,
        mime_type=file.content_type,
        size_bytes=len(file_bytes),
        status=DocumentStatus.PENDING,
    )
    db.add(document)

    await db.commit()
    await db.refresh(document)

    logger.info(
        "document_upload_received",
        document_id=str(document_id),
        filename=file.filename,
        size_bytes=len(file_bytes),
        tenant_id=x_tenant_id,
    )

    # ── Queue background processing ──────────────────────────────
    # BackgroundTasks runs after the response is sent.
    # The DB session from get_db() is already committed by then (via the yield),
    # so the background task needs its own session — we pass the IDs, not the session.
    background_tasks.add_task(
        run_ingestion_pipeline,
        document_id=document_id,
        file_path=file_path,
        tenant_id=x_tenant_id,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
    )


@router.get(
    "/ingest/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Poll document processing status",
)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    x_tenant_id: str = Header(...),
) -> DocumentStatusResponse:
    """
    Poll this endpoint after uploading to check if processing is complete.

    Status transitions:
        PENDING → PROCESSING → READY     (success)
        PENDING → PROCESSING → FAILED    (error)

    When status=READY, the document is available for queries.
    When status=FAILED, error_message contains the failure reason.
    """
    from sqlalchemy import select

    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == x_tenant_id,  # tenant isolation — can't see other tenants' docs
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found",
        )

    return DocumentStatusResponse(
    document_id=document.id,
    filename=document.filename,
    status=document.status,
    chunk_count=document.chunk_count,
    error_message=document.error_message,
    created_at=document.created_at,
    updated_at=document.updated_at,
)
