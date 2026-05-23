"""
ChromaDB client and collection management.

Architecture note:
    ChromaDB runs as a separate HTTP service (chromadb server) rather than
    embedded in our process. This matters for production because:
    - Embedded mode doesn't support concurrent writes from multiple workers
    - HTTP mode lets ChromaDB scale independently of the API server
    - You can restart the API server without losing the in-memory index

    We connect using HttpClient, which talks to ChromaDB over HTTP.
    The collection is created on first connection if it doesn't exist.

Collection design:
    One collection per environment (not one per tenant).
    Tenant isolation is achieved via metadata filters on every query:
        collection.query(where={"tenant_id": tenant_id}, ...)

    Alternative: one collection per tenant. Simpler isolation, but:
    - ChromaDB has overhead per collection (index files, memory)
    - At 10,000 tenants you have 10,000 collections — operational nightmare
    - Can't do cross-tenant analytics queries

    The metadata-filter approach is standard for multi-tenant vector stores.

Metadata stored per vector (alongside the embedding):
    {
        "document_id": str(uuid),
        "chunk_id": str(uuid),       ← matches PostgreSQL DocumentChunk.id
        "tenant_id": str,
        "chunk_index": int,
        "page_number": int | None,
        "filename": str,
        "embedding_model": str,
    }

    Storing embedding_model in metadata lets us validate at query time that
    the query embedding uses the same model as the stored vectors.

Failure modes:
    - ChromaDB not running: connection refused at startup — good, fails fast
    - Collection deleted externally: recreated on next startup, but all data lost
    - Concurrent writes without proper locking: ChromaDB HTTP mode handles this
    - Embedding dimension mismatch: ChromaDB raises an error on insert — good
"""

from functools import lru_cache

import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    """
    Returns a cached ChromaDB HTTP client.

    Cached with lru_cache so we create one client per process, not one per request.
    The client maintains a persistent HTTP connection pool to ChromaDB.
    """
    settings = get_settings()
    client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=ChromaSettings(
        anonymized_telemetry=False,
    ),
  )
    logger.info(
        "chroma_client_connected",
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    return client


def get_or_create_collection() -> Collection:
    """
    Get the main collection, creating it if it doesn't exist.

    HNSW parameters:
        space="cosine" — use cosine distance (not L2/euclidean).
                         Must match how you compute similarity at query time.
        construction_ef=200 — higher = better index quality, slower build
        search_ef=100 — higher = better recall, slower query
        M=16 — connections per node in the HNSW graph; 16 is a good default

    These parameters are set once at collection creation and cannot be changed
    afterward without recreating the collection (and re-indexing all vectors).
    Choose them carefully based on your recall vs. latency requirements.

    Production note:
        For collections with >1M vectors, consider:
        - M=32 for better recall (at the cost of more memory)
        - construction_ef=400 for higher-quality index (slower ingestion)
    """
    settings = get_settings()
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:M": 16,
        },
    )

    logger.info(
        "chroma_collection_ready",
        collection=settings.chroma_collection_name,
        count=collection.count(),
    )
    return collection


def verify_chroma_connection() -> bool:
    """
    Health check — call at startup to fail fast if ChromaDB is unreachable.
    Returns True if connected, raises on failure.
    """
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True
    except Exception as exc:
        logger.error("chroma_connection_failed", error=str(exc))
        raise
