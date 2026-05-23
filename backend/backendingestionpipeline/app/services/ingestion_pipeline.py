"""
Ingestion pipeline orchestrator.

Architecture note:
    This module is the conductor of the ingestion pipeline. It:
    1. Opens its OWN database session (background task can't use the request's session)
    2. Updates document status at each stage transition
    3. Calls each pipeline stage in order
    4. Handles failures: catches exceptions, sets status=FAILED, logs the error
    5. Cleans up the temporary file when done

    The pipeline stages are intentionally independent modules:
        text_extractor.py  — knows about files and PDF/text formats
        chunker.py         — knows about text splitting
        embedder.py        — knows about OpenAI
        vector_store.py    — knows about ChromaDB

    This module knows the ORDER and the ERROR HANDLING. That's separation of concerns.

Why a separate DB session?
    The upload endpoint's session is tied to the HTTP request lifecycle. By the time
    BackgroundTasks runs, that session is already committed and closed. The background
    task must create a fresh session. We use AsyncSessionLocal directly (not via
    the get_db() dependency which is for FastAPI route handlers only).

Status updates timing:
    PENDING → PROCESSING: immediately when the background task starts
    PROCESSING → READY: after all vectors are stored and PostgreSQL chunks are saved
    PROCESSING → FAILED: any exception in any stage

    We update status in the DB at each transition so the client polling /status
    sees accurate state, not a stale PENDING while we're actively embedding.

Idempotency consideration:
    If the server restarts during processing, the document stays in PROCESSING.
    On the next startup, you'd want a "recovery job" that finds all PROCESSING
    documents older than N minutes and either retries them or marks them FAILED.
    Not implemented here — add this as a scheduled task in production.

Cleanup:
    The temporary file is deleted after successful processing or after a failure.
    We don't keep files on disk after embedding — they're no longer needed.
    In production (with S3), you'd move from a staging bucket to an archive bucket,
    or just delete from staging.
"""

import uuid
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.core.app_logging import get_logger
from app.db.session import AsyncSessionLocal
from app.models.database import Document, DocumentChunk, DocumentStatus
from app.services.chunker import chunk_document
from app.services.embedder import embed_chunks
from app.services.text_extractor import extract_text
from app.services.vector_store import store_embedded_chunks

logger = get_logger(__name__)


async def run_ingestion_pipeline(
    document_id: uuid.UUID,
    file_path: Path,
    tenant_id: str,
) -> None:
    """
    Run the full ingestion pipeline for one document.

    Stages:
        1. Extract text from file
        2. Split into chunks
        3. Embed all chunks (OpenAI API)
        4. Store vectors in ChromaDB
        5. Record chunks in PostgreSQL
        6. Mark document as READY

    This function is designed to be called as a FastAPI BackgroundTask.
    All exceptions are caught and result in status=FAILED — never a crash.
    """
    settings = get_settings()
    doc_id_str = str(document_id)

    logger.info("ingestion_pipeline_started", document_id=doc_id_str)

    # Each background task gets its own DB session
    async with AsyncSessionLocal() as db:
        try:
            # ── Stage 0: Fetch document and mark as PROCESSING ───────────
            result = await db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.error("document_not_found_in_pipeline", document_id=doc_id_str)
                return

            document.status = DocumentStatus.PROCESSING
            await db.commit()

            logger.info(
                "ingestion_stage_extract",
                document_id=doc_id_str,
                file_path=str(file_path),
                mime_type=document.mime_type,
            )

            # ── Stage 1: Text extraction ─────────────────────────────────
            extraction = extract_text(file_path, document.mime_type)

            if extraction.total_chars < 100:
                raise ValueError(
                    f"Document produced only {extraction.total_chars} characters. "
                    f"It may be a scanned image PDF. OCR is required for scanned documents."
                )

            logger.info(
                "ingestion_stage_chunk",
                document_id=doc_id_str,
                total_chars=extraction.total_chars,
            )

            # ── Stage 2: Chunking ─────────────────────────────────────────
            chunks = chunk_document(extraction)

            if not chunks:
                raise ValueError("Chunking produced no chunks. Document may be empty or corrupt.")

            logger.info(
                "ingestion_stage_embed",
                document_id=doc_id_str,
                chunk_count=len(chunks),
            )

            # ── Stage 3: Embedding ────────────────────────────────────────
            embedded_chunks = await embed_chunks(chunks)

            logger.info(
                "ingestion_stage_store_vectors",
                document_id=doc_id_str,
                embedded_count=len(embedded_chunks),
            )

            # ── Stage 4: Store vectors in ChromaDB ───────────────────────
            stored_count = store_embedded_chunks(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=document.filename,
                embedding_model=settings.openai_embedding_model,
                embedded_chunks=embedded_chunks,
            )

            # ── Stage 5: Record chunks in PostgreSQL ──────────────────────
            # This creates the chunk records that the debug panel uses.
            # We do this after ChromaDB storage succeeds — if ChromaDB fails,
            # we don't want orphaned chunk records in PostgreSQL.
            chunk_records = [
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=ec.chunk.chunk_index,
                    text_preview=ec.chunk.text[:500],  # first 500 chars for debug display
                    token_count=ec.chunk.token_count,
                    page_number=ec.chunk.page_number,
                )
                for ec in embedded_chunks
            ]
            db.add_all(chunk_records)

            # ── Stage 6: Mark document as READY ──────────────────────────
            document.status = DocumentStatus.READY
            document.chunk_count = stored_count
            document.embedding_model = settings.openai_embedding_model

            await db.commit()

            logger.info(
                "ingestion_pipeline_completed",
                document_id=doc_id_str,
                chunk_count=stored_count,
                status="ready",
            )

        except Exception as exc:
            # Catch ALL exceptions — a pipeline failure must always result in
            # status=FAILED, never in an indefinitely PROCESSING document.
            logger.error(
                "ingestion_pipeline_failed",
                document_id=doc_id_str,
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,  # includes traceback in structured logs
            )

            try:
                # Attempt to mark the document as FAILED in a fresh transaction.
                # The previous transaction may have already been rolled back.
                await db.rollback()
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                if document:
                    document.status = DocumentStatus.FAILED
                    document.error_message = f"{type(exc).__name__}: {str(exc)}"
                    await db.commit()
            except Exception as inner_exc:
                # If we can't even update the status, log it — don't crash
                logger.error(
                    "failed_to_update_document_status",
                    document_id=doc_id_str,
                    inner_error=str(inner_exc),
                )

        finally:
            # Always clean up the temporary file, regardless of success or failure
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info("temp_file_deleted", file_path=str(file_path))
                except OSError as exc:
                    logger.warning("temp_file_delete_failed", error=str(exc))


# Stub for query endpoint (to be implemented in next step)
async def run_query_pipeline(*args, **kwargs):
    raise NotImplementedError("Query pipeline coming in the next implementation step")
