"""
Embedding cache — V1 in-memory implementation with Redis migration path.

Design principles:
    1. Protocol-based abstraction — retrieval.py depends on the protocol,
       not the implementation. Swapping to Redis means adding a new class,
       not changing any call sites.

    2. Deterministic key generation — the same query string must always
       produce the same cache key, regardless of whitespace, casing, or
       unicode normalization quirks. We normalize before hashing.

    3. Async-safe — asyncio is single-threaded but the cache dict access
       must still be atomic. Python dict reads and writes are GIL-protected
       and therefore safe without locks in a single-process asyncio app.
       We use asyncio.Lock only for the eviction sweep to prevent two
       coroutines running eviction simultaneously.

    4. Observable — every hit and miss is logged with the full context
       needed to compute hit rate, latency savings, and memory pressure.
       The cache knows nothing about what it's caching except the key and
       a vector of floats.

    5. Bounded memory — the cache has a max size (entries, not bytes).
       When full, it evicts the oldest entry (FIFO). A more sophisticated
       implementation would use LRU, but FIFO is correct, simple, and
       has no per-access overhead.

Memory math for V1 defaults:
    max_size = 500 entries
    per entry = 1536 floats × 8 bytes + ~200 bytes overhead = ~12.5KB
    total = 500 × 12.5KB = ~6.25MB — negligible
    You can safely raise max_size to 5000 (62MB) without concern.

TTL design decision:
    TTL is measured from insertion time, not last access time.
    This means a very popular query will expire and get re-embedded
    at the TTL boundary. This is intentional — it prevents stale
    embeddings if you ever change models or model parameters.
    In practice, text-embedding-3-small is deterministic so "stale"
    doesn't apply, but the TTL prevents unbounded cache growth for
    long-running servers.
"""

import asyncio
import hashlib
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.core.config import get_settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Key generation — the most important correctness piece
# ---------------------------------------------------------------------------

def make_cache_key(query: str, model: str) -> str:
    """
    Produce a deterministic cache key for a (query, model) pair.

    Normalization steps applied before hashing:
        1. NFC unicode normalization — "café" and "cafe\u0301" hash identically
        2. Strip leading/trailing whitespace
        3. Collapse internal whitespace runs to single space
        4. Lowercase — "What is fmap?" and "what is fmap?" hit the same entry

    Model is included in the key because:
        - Different models produce incompatible vector spaces
        - A cache entry from text-embedding-ada-002 must not be returned
          for a text-embedding-3-small query, even for identical text

    We use SHA-256 truncated to 16 bytes (32 hex chars).
    Collision probability for 500 entries: astronomically low.
    Full SHA-256 would also be fine but is unnecessarily long for a dict key.
    """
    normalized = unicodedata.normalize("NFC", query)
    normalized = normalized.strip()
    normalized = " ".join(normalized.split())  # collapse whitespace
    normalized = normalized.lower()

    payload = f"{model}::{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Protocol — the interface both implementations satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingCacheProtocol(Protocol):
    """
    The cache contract. Any class implementing these three methods
    can be used as a drop-in replacement by retrieval.py.

    get() and set() are async because Redis operations are async.
    The in-memory implementation wraps sync access in trivial coroutines.
    """

    async def get(self, key: str) -> list[float] | None:
        """Return the cached embedding or None on miss/expiry."""
        ...

    async def set(self, key: str, embedding: list[float]) -> None:
        """Store an embedding. Evicts oldest entry if at capacity."""
        ...

    async def stats(self) -> dict:
        """Return current cache statistics for the /health or /metrics endpoint."""
        ...


# ---------------------------------------------------------------------------
# Cache entry — what lives in the dict
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    embedding: list[float]
    inserted_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: float | None) -> bool:
        if ttl_seconds is None:
            return False
        return (time.monotonic() - self.inserted_at) > ttl_seconds


# ---------------------------------------------------------------------------
# V1 implementation — in-memory, process-local
# ---------------------------------------------------------------------------

class InMemoryEmbeddingCache:
    """
    In-memory LRU-ish embedding cache using an OrderedDict.

    OrderedDict preserves insertion order, enabling O(1) FIFO eviction:
    - popitem(last=False) removes the oldest entry
    - move_to_end() on access would give true LRU (not implemented here
      to avoid the per-access overhead — FIFO is good enough for V1)

    Thread safety:
        Python's GIL makes dict.get() and dict.__setitem__() atomic.
        The eviction sweep uses asyncio.Lock to prevent two concurrent
        coroutines both finding the cache full and both evicting.

    This class is intentionally NOT a singleton. Instantiate it once
    in main.py's lifespan and store on app.state. This makes it
    testable (each test gets a fresh instance) and replaceable.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float | None = 86400.0,  # 24 hours default
    ) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> list[float] | None:
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired(self._ttl):
            # Expired — remove and count as miss
            del self._store[key]
            self._misses += 1
            logger.debug("embedding_cache_expired", key=key[:8])
            return None

        self._hits += 1
        return entry.embedding

    async def set(self, key: str, embedding: list[float]) -> None:
        async with self._lock:
            # If key already exists, remove it so we can re-insert at the end
            # (preserves FIFO ordering: existing entries don't get "refreshed")
            if key in self._store:
                del self._store[key]

            # Evict oldest entries until we're under capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("embedding_cache_evicted", key=evicted_key[:8])

            self._store[key] = _CacheEntry(embedding=embedding)

    async def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend": "in_memory",
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "ttl_seconds": self._ttl,
        }

    def clear(self) -> None:
        """Synchronous clear for use in tests and shutdown."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0


# ---------------------------------------------------------------------------
# Redis implementation stub — swap in when you're ready
# ---------------------------------------------------------------------------

class RedisEmbeddingCache:
    """
    Redis-backed embedding cache. Drop-in replacement for InMemoryEmbeddingCache.

    To activate: install redis[asyncio] and add REDIS_URL to settings.
    In main.py lifespan, replace InMemoryEmbeddingCache with this class.
    retrieval.py doesn't change at all.

    Key format: "emb:{key}" — namespaced to avoid collisions with
    other Redis keys in the same database.

    Serialization: JSON. Slower than msgpack or pickle but human-readable
    in redis-cli for debugging. Upgrade to msgpack if you measure
    serialization as a bottleneck (unlikely at V2 scale).

    TTL: handled natively by Redis EXPIRE — no background sweep needed.
    """

    def __init__(self, redis_url: str, ttl_seconds: float = 86400.0) -> None:
        # Lazy import — only required if this class is used
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise RuntimeError(
                "redis[asyncio] is required for RedisEmbeddingCache. "
                "Run: pip install redis[asyncio]"
            )
        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = int(ttl_seconds)
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> list[float] | None:
        import json
        raw = await self._client.get(f"emb:{key}")
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return json.loads(raw)

    async def set(self, key: str, embedding: list[float]) -> None:
        import json
        await self._client.setex(
            f"emb:{key}",
            self._ttl,
            json.dumps(embedding),
        )

    async def stats(self) -> dict:
        total = self._hits + self._misses
        info = await self._client.info("memory")
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "redis_used_memory_mb": round(
                info.get("used_memory", 0) / 1024 / 1024, 2
            ),
            "ttl_seconds": self._ttl,
        }


# ---------------------------------------------------------------------------
# Module-level factory — how retrieval.py gets the cache
# ---------------------------------------------------------------------------

def build_cache() -> InMemoryEmbeddingCache:
    """
    Build the cache from settings. Called once in main.py lifespan.

    Returns InMemoryEmbeddingCache for V1. To switch to Redis:
        1. Add REDIS_URL to .env and Settings
        2. Return RedisEmbeddingCache(settings.redis_url) here
        3. Nothing else changes
    """
    settings = get_settings()
    cache = InMemoryEmbeddingCache(
        max_size=getattr(settings, "embedding_cache_max_size", 500),
        ttl_seconds=getattr(settings, "embedding_cache_ttl_seconds", 86400.0),
    )
    logger.info(
        "embedding_cache_initialized",
        backend="in_memory",
        max_size=cache._max_size,
        ttl_seconds=cache._ttl,
    )
    return cache
