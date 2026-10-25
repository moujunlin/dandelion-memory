from fastapi.testclient import TestClient

from dandelion_memory.api import create_app
from dandelion_memory.models import MemoryCandidate
from dandelion_memory.pipeline import MemoryPipeline

from .conftest import FakeEmbedder, FakeStore


def make_client(settings):
    store = FakeStore()
    pipeline = MemoryPipeline(store=store, embedder=FakeEmbedder())
    app = create_app(settings=settings, pipeline=pipeline)
    return TestClient(app), store


def auth(settings):
    return {"Authorization": f"Bearer {settings.memory_api_key}"}


def test_health_is_public(settings):
    client, _ = make_client(settings)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_requires_bearer_auth(settings):
    client, _ = make_client(settings)
    payload = {
        "request_id": "r1",
        "messages": [{"role": "user", "content": "hello"}],
        "user_id": "u1",
        "session_id": "s1",
    }
    assert client.post("/add", json=payload).status_code == 401
    assert (
        client.post("/add", json=payload, headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_add_echoes_all_ids_and_is_synchronous(settings):
    client, store = make_client(settings)
    payload = {
        "request_id": "r1",
        "messages": [{"role": "user", "content": "hello", "timestamp": 1704067200000}],
        "user_id": "u1",
        "session_id": "s1",
    }
    response = client.post("/add", json=payload, headers=auth(settings))
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
    }
    assert store.adds[0]["request_id"] == "r1"
    assert store.adds[0]["items"][0].embedding


def test_add_rejects_unknown_fields_and_blank_content(settings):
    client, _ = make_client(settings)
    base = {"request_id": "r1", "user_id": "u1", "session_id": "s1"}
    bad = {**base, "messages": [{"role": "user", "content": " "}]}
    assert client.post("/add", json=bad, headers=auth(settings)).status_code == 422
    extra = {**base, "messages": [{"role": "user", "content": "hello"}], "metadata": {}}
    assert client.post("/add", json=extra, headers=auth(settings)).status_code == 422


def test_search_enforces_top_k_and_response_shape(settings):
    client, store = make_client(settings)
    store.candidates = [
        MemoryCandidate(id=f"m{i}", content=f"memory {i}", score=1 / (i + 1)) for i in range(120)
    ]
    response = client.post(
        "/search",
        json={"query": "what happened?", "user_id": "u1", "top_k": 100},
        headers=auth(settings),
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data"}
    assert len(body["data"]) == 100
    assert all(item["id"] and item["content"] for item in body["data"])


def test_search_requires_positive_top_k(settings):
    client, _ = make_client(settings)
    response = client.post(
        "/search",
        json={"query": "q", "user_id": "u1", "top_k": 0},
        headers=auth(settings),
    )
    assert response.status_code == 422
