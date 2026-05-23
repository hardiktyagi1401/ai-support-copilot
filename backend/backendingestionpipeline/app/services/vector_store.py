"""
Vector store operations — ChromaDB reads and writes.

Architecture note:
    This service is the bridge between the embedding pipeline and ChromaDB.
    It handles:
    - Storing embedded chunks with their metadata
    - Querying by vector similarity with tenant isolation
    - Deleting all vectors for a document (for re-processing or deletion)

Metadata schema stored per vector:
    Every vector in ChromaDB carries metadata that ChromaDB can filter on
    without doing a vector search. This lets us do things like:
    - "Retrieve only from this specific document" (document_id filter)
    - "Retrieve only for this tenant" (tenant_id filter — isolation!)
    - "Retrieve only from page 3-10" (page_number range filter)

    The metadata fields must be primitive types (str, int, float, bool).
    No nested objects, no lists — ChromaDB's metadata store is flat.

ChromaDB's add() call:
    We pass four parallel lists:
    - ids: unique string IDs for each vector (we use the chunk UUID)
    - embeddings: list of float[] vectors
    - documents: the chunk text (stored alongside the vector for retrieval)
    - metadatas: list of metadata dicts

    ChromaDB validates that all four lists have the same length.

Failure modes:
    - Duplicate IDs: if you add the same chunk_id twice, ChromaDB raises.
      Always delete existing vectors for a document before re-processing.
    - Metadata values that aren't primitives: ChromaDB raises a TypeError.
    - ChromaDB collection deleted: all vectors lost, must re-embed.
    - Network partition during add(): partial write — some chunks stored, some not.
      Detect by comparing chunk_count in PostgreSQL with ChromaDB's count.

Scaling consideration:
    ChromaDB's add() is not atomic for large batches. If the process is killed
    mid-batch, you'll have partial data. For production, add a vector_storage_status
    field to PostgreSQL chunks and mark each chunk as stored after confirmation.
"""

import uuid

from app.core.app_logging import get_logger
from app.db.chroma import get_or_create_collection
from app.services.embedder import EmbeddedChunk

logger = get_logger(__name__)


def store_embedded_chunks(
    document_id: uuid.UUID,
    tenant_id: str,
    filename: str,
    embedding_model: str,
    embedded_chunks: list[EmbeddedChunk],
) -> int:
    """
    Store all embedded chunks in ChromaDB.

    Returns the number of vectors successfully stored.

    We store in batches of 500 to avoid ChromaDB's internal limits and
    to give us progress logging on large documents.
    """
    if not embedded_chunks:
        return 0

    collection = get_or_create_collection()
    doc_id_str = str(document_id)
    STORAGE_BATCH_SIZE = 500

    total_stored = 0

    for batch_start in range(0, len(embedded_chunks), STORAGE_BATCH_SIZE):
        batch = embedded_chunks[batch_start: batch_start + STORAGE_BATCH_SIZE]

        ids = [str(uuid.uuid4()) for _ in batch]  # unique ID per vector
        embeddings = [ec.embedding for ec in batch]
        documents = [ec.chunk.text for ec in batch]
        metadatas = [
            {
                "document_id": doc_id_str,
                "tenant_id": tenant_id,
                "chunk_index": ec.chunk.chunk_index,
                "page_number": ec.chunk.page_number or 0,  # 0 = no page info
                "filename": filename,
                "embedding_model": embedding_model,
                "token_count": ec.chunk.token_count,
            }
            for ec in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total_stored += len(batch)
        logger.info(
            "vectors_stored",
            document_id=doc_id_str,
            batch_start=batch_start,
            batch_size=len(batch),
            total_stored=total_stored,
        )

    logger.info(
        "document_vectors_stored",
        document_id=doc_id_str,
        total_vectors=total_stored,
        collection_total=collection.count(),
    )

    return total_stored


def delete_document_vectors(document_id: uuid.UUID, tenant_id: str) -> int:
    """
    Delete all vectors for a document from ChromaDB.

    Used when:
    - Re-processing a document (delete old vectors before inserting new ones)
    - User deletes a document
    - Compliance: right-to-erasure requests

    Returns the number of vectors deleted.
    """
    collection = get_or_create_collection()
    doc_id_str = str(document_id)

    # ChromaDB's where filter uses the metadata we stored
    results = collection.get(
        where={"document_id": doc_id_str, "tenant_id": tenant_id},
        include=[],  # don't fetch embeddings or documents, just ids
    )

    if not results["ids"]:
        logger.warning("no_vectors_found_to_delete", document_id=doc_id_str)
        return 0

    collection.delete(ids=results["ids"])

    deleted_count = len(results["ids"])
    logger.info(
        "document_vectors_deleted",
        document_id=doc_id_str,
        count=deleted_count,
    )
    return deleted_count
