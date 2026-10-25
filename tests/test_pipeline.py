from dandelion_memory.models import AddRequest, MemoryCandidate, SearchRequest
from dandelion_memory.pipeline import MemoryPipeline, pack_results

from .conftest import FakeEmbedder, FakeStore


async def test_forget_request_archives_matching_memory_in_same_user_scope():
    store = FakeStore()
    store.candidates = [
        MemoryCandidate(id="paint", content="[user] I love painting landscapes", score=0.03),
        MemoryCandidate(id="tea", content="[user] I drink tea", score=0.02),
    ]
    pipeline = MemoryPipeline(store=store, embedder=FakeEmbedder())
    request = AddRequest(
        request_id="forget-1",
        user_id="u1",
        session_id="s2",
        messages=[{"role": "user", "content": "Please forget that I love painting landscapes."}],
    )
    await pipeline.add(request)
    assert store.searches[0]["user_id"] == "u1"
    assert store.adds[0]["archive_ids"] == ["paint"]
    assert store.adds[0]["items"][0].searchable is False


async def test_relative_time_hint_is_written_next_to_original_text():
    store = FakeStore()
    pipeline = MemoryPipeline(store=store, embedder=FakeEmbedder())
    request = AddRequest(
        request_id="time-1",
        user_id="u1",
        session_id="s1",
        messages=[{"role": "user", "content": "I went yesterday.", "timestamp": 1704067200000}],
    )
    await pipeline.add(request)
    content = store.adds[0]["items"][0].content
    assert "I went yesterday." in content
    assert "yesterday=2023-12-31" in content


async def test_search_passes_user_scope_and_options_to_retrieval():
    store = FakeStore()
    pipeline = MemoryPipeline(store=store, embedder=FakeEmbedder())
    await pipeline.search(
        SearchRequest(query="Which one?", options=["A. tea", "B. coffee"], user_id="u9", top_k=5)
    )
    assert store.searches[0]["user_id"] == "u9"
    assert "A. tea" in store.searches[0]["query"]


def test_token_budget_never_returns_an_oversized_item():
    rows = [
        MemoryCandidate(id="big", content="x" * 1000, score=1),
        MemoryCandidate(id="small", content="ok", score=0.5),
    ]
    assert [row.id for row in pack_results(rows, top_k=10, token_budget=2)] == ["small"]
