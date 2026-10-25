from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from dandelion_memory.api import create_app

from .conftest import FakeEmbedder, FakeExtractor, FakeReranker, build_v2_pipeline


def make_v2_client(
    settings,
    db_path: Path,
    *,
    embedder: FakeEmbedder | None = None,
    extractor: FakeExtractor | None = None,
    reranker: FakeReranker | None = None,
):
    pipeline, store, emb, ext, rer = build_v2_pipeline(
        db_path=db_path,
        embedder=embedder,
        extractor=extractor,
        reranker=reranker,
    )
    app = create_app(settings=settings, pipeline=pipeline)
    return TestClient(app), store, emb, ext, rer


def auth(settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.memory_api_key}"}


def test_health_is_public(settings, db_path: Path):
    client, *_ = make_v2_client(settings, db_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_requires_bearer_auth(settings, db_path: Path):
    client, *_ = make_v2_client(settings, db_path)
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


def test_add_echoes_all_ids_and_is_synchronous(settings, db_path: Path):
    client, *_ = make_v2_client(settings, db_path)
    payload = {
        "request_id": "r1",
        "messages": [{"role": "user", "content": "hello world", "timestamp": 1704067200000}],
        "user_id": "u1",
        "session_id": "s1",
    }
    response = client.post("/add", json=payload, headers=auth(settings))
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "r1"
    assert body["user_id"] == "u1"
    assert body["session_id"] == "s1"
    assert body.get("success") is True or body.get("status") == "ok"


def test_add_rejects_unknown_fields_and_blank_content(settings, db_path: Path):
    client, *_ = make_v2_client(settings, db_path)
    base = {"request_id": "r1", "user_id": "u1", "session_id": "s1"}
    bad = {**base, "messages": [{"role": "user", "content": " "}]}
    assert client.post("/add", json=bad, headers=auth(settings)).status_code == 422
    extra = {**base, "messages": [{"role": "user", "content": "hello"}], "metadata": {}}
    assert client.post("/add", json=extra, headers=auth(settings)).status_code == 422
    bad_role = {**base, "messages": [{"role": "tool", "content": "hello"}]}
    assert client.post("/add", json=bad_role, headers=auth(settings)).status_code == 422


def test_search_requires_positive_top_k(settings, db_path: Path):
    client, *_ = make_v2_client(settings, db_path)
    response = client.post(
        "/search",
        json={"query": "q", "user_id": "u1", "top_k": 0},
        headers=auth(settings),
    )
    assert response.status_code == 422


def test_add_supports_system_role_and_filters_from_storage(settings, db_path: Path):
    """F1, F5: /add accepts role='system' but filters it out of parent_messages & search_chunks."""
    client, *_ = make_v2_client(settings, db_path)
    payload = {
        "request_id": "req-mixed-sys",
        "user_id": "u_sys",
        "session_id": "s_sys",
        "messages": [
            {
                "role": "system",
                "content": "PersonaMem secret system profile: user has a hidden safe in Zurich.",
            },
            {"role": "user", "content": "我平时喜欢在周末去公园散步。"},
            {"role": "assistant", "content": "周末去公园散步确实很放松。"},
        ],
    }
    resp = client.post("/add", json=payload, headers=auth(settings))
    assert resp.status_code == 200

    with sqlite3.connect(db_path) as conn:
        parent_rows = conn.execute(
            "SELECT role, content, seq FROM parent_messages WHERE user_id = ? ORDER BY seq",
            ("u_sys",),
        ).fetchall()
        chunk_rows = conn.execute(
            "SELECT display_text, search_text FROM search_chunks WHERE user_id = ?",
            ("u_sys",),
        ).fetchall()

    assert [r[0] for r in parent_rows] == ["user", "assistant"]
    assert [r[2] for r in parent_rows] == [1, 2]
    assert all("Zurich" not in r[1] for r in parent_rows)
    assert len(chunk_rows) >= 2
    assert all("Zurich" not in (r[0] + " " + r[1]) for r in chunk_rows)

    search_resp = client.post(
        "/search",
        json={"query": "Zurich hidden safe", "user_id": "u_sys", "top_k": 10},
        headers=auth(settings),
    )
    assert search_resp.status_code == 200
    assert all("Zurich" not in item["content"] for item in search_resp.json()["data"])


def test_add_all_system_batch_records_idempotency_and_returns_200(settings, db_path: Path):
    """F1, F5: When an entire /add batch is role='system', record in add_requests and return 200."""
    client, _, embedder, extractor, _ = make_v2_client(settings, db_path)
    payload = {
        "request_id": "req-only-system",
        "user_id": "u_only_sys",
        "session_id": "s_only_sys",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "System instruction block #2."},
        ],
    }
    resp = client.post("/add", json=payload, headers=auth(settings))
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "req-only-system"
    assert body.get("success") is True or body.get("status") == "ok"

    assert extractor.call_count == 0
    assert embedder.call_count == 0

    with sqlite3.connect(db_path) as conn:
        req_row = conn.execute(
            "SELECT request_id, user_id, session_id, payload_hash FROM add_requests "
            "WHERE request_id = ?",
            ("req-only-system",),
        ).fetchone()
        parent_count = conn.execute(
            "SELECT COUNT(*) FROM parent_messages WHERE user_id = ?",
            ("u_only_sys",),
        ).fetchone()[0]
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM search_chunks WHERE user_id = ?",
            ("u_only_sys",),
        ).fetchone()[0]

    assert req_row is not None
    assert req_row[0] == "req-only-system"
    assert req_row[3]  # non-empty payload_hash
    assert parent_count == 0
    assert chunk_count == 0


def test_add_idempotent_replay_skips_llm_and_embedder(settings, db_path: Path):
    """F2.1: Replaying the exact same (request_id, payload_hash) returns 200 with 0 new calls."""
    client, _, embedder, extractor, _ = make_v2_client(settings, db_path)
    payload = {
        "request_id": "req-idem-1",
        "user_id": "u_idem",
        "session_id": "s_idem",
        "messages": [
            {"role": "user", "content": "我养了一只名叫团子的布偶猫。", "timestamp": 1700000000000}
        ],
    }
    r1 = client.post("/add", json=payload, headers=auth(settings))
    assert r1.status_code == 200
    calls_after_first_ext = extractor.call_count
    calls_after_first_emb = embedder.call_count
    assert calls_after_first_emb >= 1

    # Replay identical request
    r2 = client.post("/add", json=payload, headers=auth(settings))
    assert r2.status_code == 200
    assert r2.json()["request_id"] == "req-idem-1"
    assert extractor.call_count == calls_after_first_ext
    assert embedder.call_count == calls_after_first_emb

    with sqlite3.connect(db_path) as conn:
        parent_count = conn.execute(
            "SELECT COUNT(*) FROM parent_messages WHERE user_id = ?",
            ("u_idem",),
        ).fetchone()[0]
    assert parent_count == 1


def test_add_idempotency_conflict_returns_409_on_different_messages_or_scope(
    settings, db_path: Path
):
    """F2.1: Same request_id with different messages or user_id/session_id returns HTTP 409."""
    client, *_ = make_v2_client(settings, db_path)
    base_payload = {
        "request_id": "req-conflict-1",
        "user_id": "u1",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Original message content"}],
    }
    assert client.post("/add", json=base_payload, headers=auth(settings)).status_code == 200

    diff_msg_payload = {
        **base_payload,
        "messages": [{"role": "user", "content": "Tampered message content"}],
    }
    assert client.post("/add", json=diff_msg_payload, headers=auth(settings)).status_code == 409

    diff_user_payload = {
        **base_payload,
        "user_id": "u2",
    }
    assert client.post("/add", json=diff_user_payload, headers=auth(settings)).status_code == 409


def test_add_module_b_llm_timeout_and_invalid_json_fallback_to_regex(settings, db_path: Path):
    """F2.2: When Module B LLM times out (8s) or returns invalid JSON, degrade to regex."""
    extractor = FakeExtractor(
        responses=[
            httpx.ReadTimeout("Module B LLM timed out after 8.0s"),
            "{not valid json at all!!!",
            {"facts": [{"broken_schema": 123}], "forget_actions": "invalid_type"},
        ]
    )
    client, *_ = make_v2_client(settings, db_path, extractor=extractor)

    # Batch 1: LLM raises ReadTimeout -> fallback stores parent_messages & raw_slice + hint
    b1 = {
        "request_id": "req-fb-1",
        "user_id": "u_fb",
        "session_id": "s_fb_1",
        "messages": [
            {
                "role": "user",
                "content": "我昨天去了杭州西湖划船，还买了一盒龙井茶。",
                "timestamp": 1704153600000,  # 2024-01-02 00:00:00 UTC -> yesterday=2024-01-01
            }
        ],
    }
    r1 = client.post("/add", json=b1, headers=auth(settings))
    assert r1.status_code == 200

    # Batch 2: LLM returns malformed JSON string -> fallback still stores normal memory
    b2 = {
        "request_id": "req-fb-2",
        "user_id": "u_fb",
        "session_id": "s_fb_2",
        "messages": [
            {"role": "user", "content": "我的备用门禁卡放在玄关第二个抽屉里。"},
        ],
    }
    r2 = client.post("/add", json=b2, headers=auth(settings))
    assert r2.status_code == 200

    # Batch 3: LLM returns broken schema on explicit forget -> regex fallback tombstones target!
    b3 = {
        "request_id": "req-fb-3",
        "user_id": "u_fb",
        "session_id": "s_fb_3",
        "messages": [
            {"role": "user", "content": "请忘掉我的备用门禁卡放在玄关第二个抽屉里这件事。"},
        ],
    }
    r3 = client.post("/add", json=b3, headers=auth(settings))
    assert r3.status_code == 200

    with sqlite3.connect(db_path) as conn:
        active_parents = conn.execute(
            "SELECT content FROM parent_messages WHERE user_id = ? AND tombstoned = 0",
            ("u_fb",),
        ).fetchall()
        active_chunks = conn.execute(
            "SELECT display_text, search_text FROM search_chunks "
            "WHERE user_id = ? AND tombstoned = 0",
            ("u_fb",),
        ).fetchall()

    assert any("西湖划船" in row[0] for row in active_parents)
    assert all("备用门禁卡" not in row[0] for row in active_parents)
    assert any("2024-01-01" in (row[0] + " " + row[1]) for row in active_chunks)

    # Search for West Lake succeeds, while forgotten keycard is not returned
    s_lake = client.post(
        "/search",
        json={"query": "去杭州西湖划船买了什么茶", "user_id": "u_fb", "top_k": 5},
        headers=auth(settings),
    )
    assert s_lake.status_code == 200
    assert any("龙井茶" in item["content"] for item in s_lake.json()["data"])

    s_card = client.post(
        "/search",
        json={"query": "备用门禁卡放在哪个抽屉", "user_id": "u_fb", "top_k": 5},
        headers=auth(settings),
    )
    assert s_card.status_code == 200
    assert all("备用门禁卡" not in item["content"] for item in s_card.json()["data"])
