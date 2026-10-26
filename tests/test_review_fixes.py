from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from dandelion_memory.embedding import DashScopeEmbedder
from dandelion_memory.extractor import DashScopeExtractor
from dandelion_memory.models import AddRequest, SearchRequest
from dandelion_memory.reranker import DashScopeReranker, NoOpReranker

from .conftest import FakeEmbedder, FakeExtractor, FakeReranker, build_v2_pipeline


async def test_p0_1_unrecalled_chunks_do_not_trigger_neighbor_lookup_or_timeline_pollution(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """P0-1: Chunks outside top-K retrieval channels must not trigger neighbor SQL or pin."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "无关维度/冷门标签",
                        "fact_text": "用户在冷门维度上的状态是阿尔法值",
                        "prev_state": "",
                        "current_state": "阿尔法值",
                        "update_reason": "初始记录",
                        "event_date_start": "2023-01-01",
                        "event_date_end": "2023-01-01",
                    }
                ],
                "forget_actions": [],
            },
            {"facts": [], "forget_actions": []},
        ]
    )
    # Set semantic_candidates=2 and lexical_candidates=2 so only top-2 enter RRF
    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        semantic_candidates=2,
        lexical_candidates=2,
    )

    e_hit = [1.0, 0.0, 0.0, 0.0]
    e_miss = [0.0, 1.0, 0.0, 0.0]
    embedder.set_vector("目标查询：量子计算超导比特相干时间", e_hit)
    embedder.set_vector(
        "命中文档一：量子计算超导比特相干时间突破五百微秒", e_hit, match_substring=True
    )
    embedder.set_vector(
        "命中文档二：量子计算超导比特门保真度达到百分之九十九", e_hit, match_substring=True
    )
    embedder.set_vector("冷门无关消息", e_miss, match_substring=True)
    embedder.set_vector("冷门标签", e_miss, match_substring=True)

    await pipeline.add(
        AddRequest(
            request_id="req-p01-cold",
            user_id="u_p01",
            session_id="s_cold",
            messages=[
                {"role": "user", "content": "冷门无关消息一：今天买了一袋面粉。"},
                {"role": "user", "content": "冷门无关消息二：顺便买了一瓶酱油。"},
                {"role": "user", "content": "冷门无关消息三：晚上蒸了一锅馒头。"},
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-p01-hot",
            user_id="u_p01",
            session_id="s_hot",
            messages=[
                {
                    "role": "user",
                    "content": "命中文档一：量子计算超导比特相干时间突破五百微秒。",
                },
                {
                    "role": "user",
                    "content": "命中文档二：量子计算超导比特门保真度达到百分之九十九。",
                },
            ],
        )
    )

    queried_neighbor_seqs: list[int] = []
    orig_get_neighbors = store.get_session_neighbors

    def tracked_get_neighbors(user_id: str, session_id: str, seq: int):
        queried_neighbor_seqs.append(seq)
        return orig_get_neighbors(user_id, session_id, seq)

    monkeypatch.setattr(store, "get_session_neighbors", tracked_get_neighbors)

    resp = await pipeline.search(
        SearchRequest(query="目标查询：量子计算超导比特相干时间", user_id="u_p01", top_k=5)
    )

    # Cold messages (seq 1..3) were not in top-2 Dense/BM25, so neighbor lookup was not called
    assert all(s in (4, 5) for s in queried_neighbor_seqs)
    # Unrecalled timeline '无关维度/冷门标签' must not be pinned into results
    assert all("阿尔法值" not in item.content for item in resp.data)


async def test_p0_2_add_embedder_failure_rolls_back_and_retry_succeeds_cleanly(db_path: Path):
    """P0-2: If embedder.embed raises mid-add, SQLite rolls back completely; retry succeeds."""

    class FlakyEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = True

        async def embed(self, texts):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("Simulated network failure during embedding")
            return await super().embed(texts)

    flaky_emb = FlakyEmbedder()
    pipeline, *_ = build_v2_pipeline(db_path=db_path, embedder=flaky_emb)

    req = AddRequest(
        request_id="req-atomic-1",
        user_id="u_atomic",
        session_id="s_atomic",
        messages=[
            {"role": "user", "content": "第一条原子性测试消息：我在学习 Rust 异步运行时。"},
            {"role": "assistant", "content": "Tokio 是目前最主流的 Rust 异步运行时。"},
        ],
    )

    with pytest.raises(RuntimeError, match="Simulated network failure"):
        await pipeline.add(req)

    # Verify full rollback: 0 parent_messages, 0 search_chunks, 0 add_requests, last_seq == 0
    with sqlite3.connect(db_path) as conn:
        parent_cnt = conn.execute(
            "SELECT COUNT(*) FROM parent_messages WHERE user_id = ?", ("u_atomic",)
        ).fetchone()[0]
        chunk_cnt = conn.execute(
            "SELECT COUNT(*) FROM search_chunks WHERE user_id = ?", ("u_atomic",)
        ).fetchone()[0]
        req_cnt = conn.execute(
            "SELECT COUNT(*) FROM add_requests WHERE request_id = ?", ("req-atomic-1",)
        ).fetchone()[0]
        state_row = conn.execute(
            "SELECT last_seq FROM user_state WHERE user_id = ?", ("u_atomic",)
        ).fetchone()

    assert parent_cnt == 0
    assert chunk_cnt == 0
    assert req_cnt == 0
    assert state_row is None or state_row[0] == 0

    # Retry the exact same request: should succeed with seq 1 and 2 (no duplicates)
    resp = await pipeline.add(req)
    assert resp.success is True

    with sqlite3.connect(db_path) as conn:
        seqs = [
            r[0]
            for r in conn.execute(
                "SELECT seq FROM parent_messages WHERE user_id = ? ORDER BY seq",
                ("u_atomic",),
            ).fetchall()
        ]
    assert seqs == [1, 2]


async def test_p1_1_tombstoned_tag_timeline_does_not_resurrect_old_history(db_path: Path):
    """P1-1: Writing a new fact to a previously tombstoned tag starts a fresh history_chain."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "安全/口令",
                        "fact_text": "用户的旧版服务器口令是 OLD-PASS-111",
                        "prev_state": "",
                        "current_state": "OLD-PASS-111",
                        "update_reason": "初始设置",
                        "event_date_start": "2024-01-01",
                        "event_date_end": "2024-01-01",
                    }
                ],
                "forget_actions": [],
            },
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": ["安全/口令"],
                        "target_keywords": ["OLD-PASS-111"],
                        "target_description": "旧版服务器口令 OLD-PASS-111",
                        "batch_seqs_to_tombstone": [2],
                    }
                ],
            },
            {
                "facts": [
                    {
                        "source_seqs": [3],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "安全/口令",
                        "fact_text": "用户的新版服务器口令是 NEW-PASS-999",
                        "prev_state": "",
                        "current_state": "NEW-PASS-999",
                        "update_reason": "重新设置",
                        "event_date_start": "2024-06-01",
                        "event_date_end": "2024-06-01",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-res-1",
            user_id="u_res",
            session_id="s1",
            messages=[{"role": "user", "content": "我的旧版服务器口令是 OLD-PASS-111。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-res-2",
            user_id="u_res",
            session_id="s1",
            messages=[{"role": "user", "content": "请忘掉我的旧版服务器口令 OLD-PASS-111。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-res-3",
            user_id="u_res",
            session_id="s1",
            messages=[{"role": "user", "content": "我的新版服务器口令是 NEW-PASS-999。"}],
        )
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT current_state, history_chain, rendered_summary, tombstoned "
            "FROM tag_timelines WHERE user_id = ? AND tag = '安全/口令'",
            ("u_res",),
        ).fetchone()

    assert row is not None
    curr, chain_json, summary, tomb = row
    assert tomb == 0
    assert curr == "NEW-PASS-999"
    assert "OLD-PASS-111" not in chain_json
    assert "OLD-PASS-111" not in summary
    assert len(json.loads(chain_json)) == 1


async def test_p1_2_zero_lexical_overlap_high_cosine_tombstones_chunk_and_timeline(
    db_path: Path,
):
    """P1-2: Cosine >= 0.58 tombstones chunks & timelines even when lexical overlap == 0."""
    embedder = FakeEmbedder()
    secret_vec = [0.9, 0.4, 0.0, 0.0]
    embedder.set_vector("我有一辆银色帕拉梅拉停在地下车库 B2", secret_vec, match_substring=True)
    embedder.set_vector("银色帕拉梅拉", secret_vec, match_substring=True)
    # Cross-lingual / synonym description with zero token overlap but identical semantic vector
    embedder.set_vector("User's luxury sports sedan parked underground", secret_vec)

    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "资产/车辆",
                        "fact_text": "我有一辆银色帕拉梅拉停在地下车库 B2",
                        "prev_state": "",
                        "current_state": "银色帕拉梅拉",
                        "update_reason": "购车",
                        "event_date_start": "2024-03-01",
                        "event_date_end": "2024-03-01",
                    }
                ],
                "forget_actions": [],
            },
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": [],
                        "target_keywords": ["nonexistent_english_kw"],
                        "target_description": "User's luxury sports sedan parked underground",
                        "batch_seqs_to_tombstone": [2],
                    }
                ],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, embedder=embedder, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-cos-1",
            user_id="u_cos",
            session_id="s_cos",
            messages=[{"role": "user", "content": "我有一辆银色帕拉梅拉停在地下车库 B2。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-cos-2",
            user_id="u_cos",
            session_id="s_cos",
            messages=[
                {
                    "role": "user",
                    "content": "Please forget about my luxury sports sedan parked underground.",
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        active_chunks = conn.execute(
            "SELECT COUNT(*) FROM search_chunks WHERE user_id = ? AND tombstoned = 0",
            ("u_cos",),
        ).fetchone()[0]
        tl_tomb = conn.execute(
            "SELECT tombstoned FROM tag_timelines WHERE user_id = ? AND tag = '资产/车辆'",
            ("u_cos",),
        ).fetchone()[0]

    assert active_chunks == 0
    assert tl_tomb == 1


async def test_p1_4_rank_51_plus_does_not_invert_above_top_50_reranked_and_noop_uses_rrf(
    db_path: Path,
):
    """P1-4: Candidates outside top-50 get rerank_score=0.0; NoOpReranker uses pure RRF + 0.05."""
    reranker = FakeReranker(score_fn=lambda _q, _d, _i: 0.15)
    pipeline, *_ = build_v2_pipeline(db_path=db_path, reranker=reranker)

    messages = [
        {"role": "user", "content": f"测试候选文档编号 #{i}：讨论分布式共识算法 Raft 的日志复制。"}
        for i in range(1, 56)
    ]
    await pipeline.add(
        AddRequest(
            request_id="req-55-docs",
            user_id="u_55",
            session_id="s_55",
            messages=messages,
        )
    )

    resp = await pipeline.search(
        SearchRequest(query="分布式共识算法 Raft 日志复制", user_id="u_55", top_k=55)
    )
    assert len(resp.data) == 55
    # Top 50 had rerank_score=0.15 -> score = 0.80 * 0.15 + 0.20 * rrf >= 0.12 + 0.20 * rrf
    # Rank 51..55 had rerank_score=0.0 -> score = 0.20 * rrf
    # Verify scores are strictly non-increasing and top-50 beat 51..55
    assert resp.data[0].score > resp.data[-1].score

    # Also verify NoOpReranker falls back to rrf_norm + latest_bonus without calling rerank
    noop_pipeline, *_ = build_v2_pipeline(db_path=db_path, reranker=NoOpReranker())
    resp_noop = await noop_pipeline.search(
        SearchRequest(query="分布式共识算法 Raft 日志复制", user_id="u_55", top_k=5)
    )
    assert len(resp_noop.data) == 5
    assert resp_noop.data[0].score == pytest.approx(1.0, rel=1e-3)


async def test_p2_3_multiple_facts_on_same_seq_merge_bm25_tokens(db_path: Path):
    """P2-3: Multiple facts extracted from the same seq merge their tags/tokens into raw_slice."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "饮食/禁忌",
                        "fact_text": "用户对海鲜严重过敏",
                        "prev_state": "",
                        "current_state": "海鲜过敏",
                        "update_reason": "体质原因",
                        "event_date_start": "2024-05-01",
                        "event_date_end": "2024-05-01",
                    },
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "运动/习惯",
                        "fact_text": "用户每周三固定打羽毛球",
                        "prev_state": "",
                        "current_state": "每周三打羽毛球",
                        "update_reason": "健身",
                        "event_date_start": "2024-05-01",
                        "event_date_end": "2024-05-01",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-multi-fact",
            user_id="u_mf",
            session_id="s_mf",
            messages=[
                {
                    "role": "user",
                    "content": "我不仅对海鲜严重过敏，而且每周三晚上都会去体育馆打羽毛球。",
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        bm25_str = conn.execute(
            "SELECT bm25_tokens FROM search_chunks "
            "WHERE user_id = ? AND seq = 1 AND kind = 'raw_slice'",
            ("u_mf",),
        ).fetchone()[0]

    assert "饮食" in bm25_str and "禁忌" in bm25_str
    assert "运动" in bm25_str and "习惯" in bm25_str
    assert "羽毛" in bm25_str and "毛球" in bm25_str


async def test_dashscope_clients_with_mock_transport():
    """P1-5, P2-1: Test DashScopeExtractor, DashScopeReranker, DashScopeEmbedder."""
    ext_calls: list[dict[str, Any]] = []

    def ext_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        ext_calls.append(body)
        if len(ext_calls) == 1:
            return httpx.Response(503, json={"error": "transient error"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"facts": [], "forget_actions": []}, ensure_ascii=False
                            )
                        }
                    }
                ]
            },
        )

    ext_client = httpx.AsyncClient(transport=httpx.MockTransport(ext_handler))
    extractor = DashScopeExtractor(api_key="sk-test", client=ext_client)
    res = await extractor.extract(
        [
            {
                "seq": 1,
                "role": "user",
                "speaker": "user",
                "session_time_str": "2024-05-01",
                "content": "x" * 3500,
            }
        ],
        existing_tags=[("user", "喜好/音乐")],
    )
    assert res == {"facts": [], "forget_actions": []}
    assert len(ext_calls) == 2
    assert ext_calls[0]["enable_thinking"] is False
    assert "...[truncated]" in ext_calls[0]["messages"][1]["content"]
    await extractor.aclose()

    # Verify timeout does NOT retry
    timeout_calls = 0

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal timeout_calls
        timeout_calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    timeout_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
    timeout_extractor = DashScopeExtractor(api_key="sk-test", client=timeout_client)
    fb_res = await timeout_extractor.extract(
        [{"seq": 1, "role": "user", "content": "请忘掉我的旧密码 123456"}]
    )
    assert timeout_calls == 1
    assert len(fb_res["forget_actions"]) == 1
    await timeout_extractor.aclose()

    # Test DashScopeReranker
    def rerank_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "qwen3-rerank"
        assert len(body["input"]["documents"][0]) == 2400
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.92},
                        {"index": 0, "relevance_score": 0.31},
                    ]
                }
            },
        )

    rerank_client = httpx.AsyncClient(transport=httpx.MockTransport(rerank_handler))
    reranker = DashScopeReranker(api_key="sk-test", client=rerank_client)
    scores = await reranker.rerank("query", ["a" * 3000, "b" * 100])
    assert scores == [0.31, 0.92]
    await reranker.aclose()

    # Test DashScopeEmbedder retry + truncation
    emb_calls = 0

    def emb_handler(request: httpx.Request) -> httpx.Response:
        nonlocal emb_calls
        emb_calls += 1
        body = json.loads(request.content.decode("utf-8"))
        assert len(body["input"][0]) == 6000
        if emb_calls == 1:
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.25, 0.75, 0.0, 0.0]}]},
        )

    emb_client = httpx.AsyncClient(transport=httpx.MockTransport(emb_handler))
    embedder = DashScopeEmbedder(
        api_key="sk-test",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=4,
        client=emb_client,
    )
    vecs = await embedder.embed(["z" * 8000])
    assert emb_calls == 2
    assert vecs == [[0.25, 0.75, 0.0, 0.0]]
    await embedder.aclose()


def test_dashscope_clients_survive_across_multiple_closed_event_loops(
    settings, db_path: Path
):
    """Verify internal _get_client() recreates AsyncClient when event loop changes or closes."""
    import asyncio

    from fastapi.testclient import TestClient

    from dandelion_memory.api import create_app
    from dandelion_memory.pipeline import MemoryPipeline
    from dandelion_memory.store import SQLiteStore

    embedder = DashScopeEmbedder(
        api_key="sk-test",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=4,
    )
    extractor = DashScopeExtractor(api_key="sk-test")
    reranker = DashScopeReranker(api_key="sk-test")

    # Simulate binding _get_client() to a first event loop that then closes
    async def bind_in_first_loop():
        c1 = embedder._get_client()
        c2 = extractor._get_client()
        c3 = reranker._get_client()
        return c1, c2, c3, asyncio.get_running_loop()

    first_emb_c, first_ext_c, first_rer_c, loop1 = asyncio.run(bind_in_first_loop())
    assert loop1.is_closed()

    # Now run in a brand new event loop: _get_client() must detect loop1 is closed and recreate
    async def bind_in_second_loop():
        c1_new = embedder._get_client()
        c2_new = extractor._get_client()
        c3_new = reranker._get_client()
        assert c1_new is not first_emb_c
        assert c2_new is not first_ext_c
        assert c3_new is not first_rer_c
        await embedder.aclose()
        await extractor.aclose()
        await reranker.aclose()

    asyncio.run(bind_in_second_loop())

    # Also verify non-context-managed TestClient(app) across two consecutive requests
    pipeline = MemoryPipeline(
        store=SQLiteStore(db_path),
        embedder=FakeEmbedder(),
        extractor=extractor,
        reranker=NoOpReranker(),
    )
    app = create_app(settings=settings, pipeline=pipeline)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {settings.memory_api_key}"}

    r1 = client.post(
        "/add",
        json={
            "request_id": "loop-req-1",
            "user_id": "u_loop",
            "session_id": "s_loop",
            "messages": [{"role": "user", "content": "第一轮跨事件循环请求。"}],
        },
        headers=headers,
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/add",
        json={
            "request_id": "loop-req-2",
            "user_id": "u_loop",
            "session_id": "s_loop",
            "messages": [{"role": "user", "content": "第二轮跨事件循环请求。"}],
        },
        headers=headers,
    )
    assert r2.status_code == 200


async def test_same_message_mixed_new_fact_and_forget_directive_preserves_new_fact(
    db_path: Path,
):
    """Same seq containing both a new state update and an explicit forget command."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "居住地",
                        "fact_text": "用户居住在北京市海淀区",
                        "prev_state": "",
                        "current_state": "北京市海淀区",
                        "update_reason": "初始居住地",
                        "event_date_start": "2023-01-01",
                        "event_date_end": "2023-01-01",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "安全/密码",
                        "fact_text": "用户的备用保险箱密码是 849201",
                        "prev_state": "",
                        "current_state": "849201",
                        "update_reason": "初始记录",
                        "event_date_start": "2023-01-01",
                        "event_date_end": "2023-01-01",
                    },
                ],
                "forget_actions": [],
            },
            # Batch 2 (seq=3): user moves to Shanghai Xuhui AND asks to forget 849201 in ONE message
            {
                "facts": [
                    {
                        "source_seqs": [3],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "居住地",
                        "fact_text": "用户搬家到上海市徐汇区了",
                        "prev_state": "北京市海淀区",
                        "current_state": "上海市徐汇区",
                        "update_reason": "昨天正式搬家",
                        "event_date_start": "2024-05-01",
                        "event_date_end": "2024-05-01",
                    }
                ],
                "forget_actions": [
                    {
                        "trigger_seq": 3,
                        "target_existing_tags": ["安全/密码"],
                        "target_keywords": ["849201", "备用保险箱密码"],
                        "target_description": "用户的备用保险箱密码是 849201",
                        "batch_seqs_to_tombstone": [3],
                    }
                ],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-mix-1",
            user_id="u_mix",
            session_id="s_mix_1",
            messages=[
                {"role": "user", "content": "我目前住在北京市海淀区。", "timestamp": 1672531200000},
                {
                    "role": "user",
                    "content": "我的备用保险箱密码是 849201。",
                    "timestamp": 1672531260000,
                },
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-mix-2",
            user_id="u_mix",
            session_id="s_mix_2",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "我昨天正式搬家到上海市徐汇区了。"
                        "另外请彻底忘掉我的备用保险箱密码 849201，以后绝不要再提。"
                    ),
                    "timestamp": 1714608000000,
                }
            ],
        )
    )

    # 1. Secret 849201 must never leak
    s_secret = await pipeline.search(
        SearchRequest(query="备用保险箱密码是多少 849201", user_id="u_mix", top_k=10)
    )
    assert all("849201" not in item.content for item in s_secret.data)

    # 2. New residence "上海市徐汇区" timeline and fact_item must survive and be returned!
    s_home = await pipeline.search(
        SearchRequest(query="用户现在住在哪里", user_id="u_mix", top_k=5)
    )
    assert len(s_home.data) >= 1
    combined = "\n".join(item.content for item in s_home.data)
    assert "上海市徐汇区" in combined
    assert "849201" not in combined


