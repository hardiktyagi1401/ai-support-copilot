
"""
Unit tests for the embedding cache.
 
These tests require no network, no database, no running server.
They run in pure Python and complete in milliseconds.
 
Run: pytest tests/unit/test_embedding_cache.py -v
"""
 
import asyncio
import time
import pytest
from app.services.embedding_cache import (
    InMemoryEmbeddingCache,
    make_cache_key,
)
 
 
# ---------------------------------------------------------------------------
# Key generation tests — correctness of normalization
# ---------------------------------------------------------------------------
 
class TestMakeCacheKey:
 
    def test_identical_queries_produce_identical_keys(self):
        k1 = make_cache_key("What is fmap?", "text-embedding-3-small")
        k2 = make_cache_key("What is fmap?", "text-embedding-3-small")
        assert k1 == k2
 
    def test_whitespace_normalization(self):
        """Leading, trailing, and internal whitespace must not affect the key."""
        k1 = make_cache_key("What is fmap?", "model")
        k2 = make_cache_key("  What   is  fmap?  ", "model")
        assert k1 == k2
 
    def test_case_normalization(self):
        k1 = make_cache_key("what is fmap?", "model")
        k2 = make_cache_key("WHAT IS FMAP?", "model")
        k3 = make_cache_key("What Is Fmap?", "model")
        assert k1 == k2 == k3
 
    def test_unicode_normalization(self):
        """NFC and NFD forms of the same character produce the same key."""
        nfc = make_cache_key("caf\u00e9", "model")      # precomposed é
        nfd = make_cache_key("cafe\u0301", "model")     # combining accent
        assert nfc == nfd
 
    def test_different_models_produce_different_keys(self):
        """Same query, different model = different key. Critical for correctness."""
        k1 = make_cache_key("hello", "text-embedding-3-small")
        k2 = make_cache_key("hello", "text-embedding-ada-002")
        assert k1 != k2
 
    def test_different_queries_produce_different_keys(self):
        k1 = make_cache_key("What is fmap?", "model")
        k2 = make_cache_key("What is a monad?", "model")
        assert k1 != k2
 
    def test_key_is_deterministic_across_calls(self):
        """Multiple calls produce the same key — not time-dependent."""
        keys = [make_cache_key("test query", "model") for _ in range(10)]
        assert len(set(keys)) == 1
 
    def test_key_length_is_consistent(self):
        k1 = make_cache_key("short", "model")
        k2 = make_cache_key("a " * 500, "model")  # very long query
        assert len(k1) == len(k2) == 32  # SHA-256 truncated to 32 hex chars
 
 
# ---------------------------------------------------------------------------
# Cache behaviour tests
# ---------------------------------------------------------------------------
 
@pytest.mark.asyncio
class TestInMemoryEmbeddingCache:
 
    async def test_get_returns_none_on_empty_cache(self):
        cache = InMemoryEmbeddingCache()
        result = await cache.get("nonexistent_key")
        assert result is None
 
    async def test_set_and_get_roundtrip(self):
        cache = InMemoryEmbeddingCache()
        embedding = [0.1, 0.2, 0.3, -0.4, 0.5]
        await cache.set("key1", embedding)
        result = await cache.get("key1")
        assert result == embedding
 
    async def test_get_returns_none_for_different_key(self):
        cache = InMemoryEmbeddingCache()
        await cache.set("key1", [0.1, 0.2])
        result = await cache.get("key2")
        assert result is None
 
    async def test_stats_hit_miss_counting(self):
        cache = InMemoryEmbeddingCache()
        await cache.set("k", [1.0])
        await cache.get("k")          # hit
        await cache.get("k")          # hit
        await cache.get("missing")    # miss
 
        stats = await cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3, abs=0.001)
 
    async def test_eviction_at_max_size(self):
        """When cache reaches max_size, oldest entry is evicted."""
        cache = InMemoryEmbeddingCache(max_size=3)
        await cache.set("first", [1.0])
        await cache.set("second", [2.0])
        await cache.set("third", [3.0])
        # Cache is full — next set should evict "first"
        await cache.set("fourth", [4.0])
 
        assert await cache.get("first") is None   # evicted
        assert await cache.get("second") is not None
        assert await cache.get("third") is not None
        assert await cache.get("fourth") is not None
 
    async def test_eviction_counter_increments(self):
        cache = InMemoryEmbeddingCache(max_size=2)
        await cache.set("a", [1.0])
        await cache.set("b", [2.0])
        await cache.set("c", [3.0])  # evicts "a"
        await cache.set("d", [4.0])  # evicts "b"
 
        stats = await cache.stats()
        assert stats["evictions"] == 2
 
    async def test_ttl_expiry(self):
        """Entries older than TTL are treated as misses."""
        cache = InMemoryEmbeddingCache(ttl_seconds=0.05)  # 50ms TTL
        await cache.set("expiring", [1.0, 2.0])
 
        # Should hit immediately
        assert await cache.get("expiring") is not None
 
        # Wait for expiry
        await asyncio.sleep(0.1)
 
        # Should miss after TTL
        assert await cache.get("expiring") is None
 
    async def test_no_ttl_never_expires(self):
        """ttl_seconds=None means entries never expire."""
        cache = InMemoryEmbeddingCache(ttl_seconds=None)
        await cache.set("permanent", [1.0])
        # We can't wait forever, but verify the flag behaves correctly
        entry = cache._store.get("permanent")
        assert entry is not None
        assert not entry.is_expired(None)
 
    async def test_overwrite_existing_key(self):
        """Setting an existing key updates the value."""
        cache = InMemoryEmbeddingCache()
        await cache.set("key", [1.0, 2.0])
        await cache.set("key", [9.0, 8.0])  # overwrite
        result = await cache.get("key")
        assert result == [9.0, 8.0]
 
    async def test_size_does_not_grow_beyond_max(self):
        """Even with many sets, cache size never exceeds max_size."""
        cache = InMemoryEmbeddingCache(max_size=10)
        for i in range(50):
            await cache.set(f"key_{i}", [float(i)])
        assert len(cache._store) <= 10
 
    async def test_clear_resets_all_state(self):
        cache = InMemoryEmbeddingCache()
        await cache.set("k", [1.0])
        await cache.get("k")
        cache.clear()
 
        stats = await cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
 
    async def test_concurrent_sets_are_safe(self):
        """Concurrent coroutines writing to the cache should not corrupt state."""
        cache = InMemoryEmbeddingCache(max_size=100)
 
        async def write_many(prefix: str) -> None:
            for i in range(20):
                await cache.set(f"{prefix}_{i}", [float(i)] * 10)
 
        await asyncio.gather(
            write_many("a"),
            write_many("b"),
            write_many("c"),
        )
 
        # No assertion on exact contents — concurrent FIFO ordering is
        # non-deterministic — but the cache must not raise and must be
        # within size bounds.
        assert len(cache._store) <= 100
 
    async def test_stats_size_reflects_current_contents(self):
        cache = InMemoryEmbeddingCache(max_size=100)
        for i in range(5):
            await cache.set(f"k{i}", [float(i)])
        stats = await cache.stats()
        assert stats["size"] == 5
        assert stats["max_size"] == 100
 
 
# ---------------------------------------------------------------------------
# Key generation integration — cache uses normalized keys
# ---------------------------------------------------------------------------
 
@pytest.mark.asyncio
class TestCacheWithKeyNormalization:
 
    async def test_whitespace_variants_hit_same_entry(self):
        """
        Queries that normalize to the same string must hit the same cache entry.
        This test verifies the full key→cache pipeline, not just make_cache_key.
        """
        cache = InMemoryEmbeddingCache()
        model = "text-embedding-3-small"
        embedding = [0.42] * 5
 
        key1 = make_cache_key("What is fmap?", model)
        key2 = make_cache_key("  what   is  fmap?  ", model)
 
        assert key1 == key2  # keys match
 
        await cache.set(key1, embedding)
        result = await cache.get(key2)  # retrieves via normalized key
        assert result == embedding
 
    async def test_model_mismatch_is_a_miss(self):
        """
        A cache entry for model A must not be returned for a query with model B,
        even if the query text is identical.
        """
        cache = InMemoryEmbeddingCache()
        embedding_a = [1.0, 2.0, 3.0]
 
        key_a = make_cache_key("hello", "text-embedding-3-small")
        key_b = make_cache_key("hello", "text-embedding-ada-002")
 
        await cache.set(key_a, embedding_a)
        result = await cache.get(key_b)
        assert result is None  # different model — cache miss