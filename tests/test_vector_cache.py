from array import array

from dandelion_memory.store import SQLiteStore


def _upsert(store: SQLiteStore, chunk_id: str, value: float) -> None:
    store.upsert_chunk(
        chunk_id=chunk_id,
        user_id="u1",
        session_id="s1",
        parent_msg_id=None,
        seq=int(value),
        session_ts=None,
        kind="raw_slice",
        subject="user",
        tag=None,
        display_text=chunk_id,
        search_text=chunk_id,
        bm25_tokens=chunk_id,
        event_time_start=None,
        event_time_end=None,
        embedding=[value] * 1024,
    )


def test_vector_cache_is_float32_and_bounded() -> None:
    store = SQLiteStore(":memory:", vector_cache_max_entries=2)

    _upsert(store, "c1", 1.0)
    _upsert(store, "c2", 2.0)
    _upsert(store, "c3", 3.0)

    assert list(store._vector_cache) == ["c2", "c3"]
    assert all(
        isinstance(vec, array) and vec.itemsize == 4
        for vec in store._vector_cache.values()
    )


def test_vector_cache_hit_refreshes_lru_order() -> None:
    store = SQLiteStore(":memory:", vector_cache_max_entries=2)

    _upsert(store, "c1", 1.0)
    _upsert(store, "c2", 2.0)
    assert store._get_cached_vector("c1") is not None
    _upsert(store, "c3", 3.0)

    assert list(store._vector_cache) == ["c1", "c3"]
