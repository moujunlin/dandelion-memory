from __future__ import annotations

import sqlite3
from pathlib import Path

from dandelion_memory.models import AddRequest, MemoryCandidate, SearchRequest
from dandelion_memory.pipeline import pack_results

from .conftest import FakeEmbedder, FakeExtractor, FakeReranker, build_v2_pipeline


async def test_timeline_rendered_summary_synced_to_search_chunks_and_tag_boosted(db_path: Path):
    """F14.2, F15: tag_timelines.rendered_summary syncs to search_chunks with TAG_BOOST=2."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "喜好/音乐",
                        "fact_text": "用户以前喜欢电子音乐，现在改听独立民谣，因为嫌音乐节太吵",
                        "prev_state": "喜欢电子音乐",
                        "current_state": "喜欢独立民谣",
                        "update_reason": "觉得电子音乐节太吵",
                        "event_date_start": "2023-05-07",
                        "event_date_end": "2023-05-07",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "全局指令/格式约束",
                        "fact_text": "所有代码示例必须使用 Python 3.12 类型注解且不写多余注释",
                        "prev_state": "",
                        "current_state": "Python 3.12 类型注解且无多余注释",
                        "update_reason": "用户全局格式要求",
                        "event_date_start": "2023-05-07",
                        "event_date_end": "2023-05-07",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-tl-sync",
            user_id="u_tl_sync",
            session_id="s_tl",
            messages=[
                {
                    "role": "user",
                    "content": "我以前挺迷电子音乐的，但今年觉得音乐节太吵，现在只听独立民谣了。",
                    "timestamp": 1683417600000,
                },
                {
                    "role": "user",
                    "content": (
                        "另外记住：以后给我写的所有代码都用 Python 3.12 类型注解，"
                        "不要写多余注释。"
                    ),
                    "timestamp": 1683417660000,
                },
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        tl_chunks = conn.execute(
            "SELECT tag, display_text, bm25_tokens, embedding, tombstoned "
            "FROM search_chunks WHERE user_id = ? AND kind = 'timeline_summary'",
            ("u_tl_sync",),
        ).fetchall()

    assert len(tl_chunks) >= 2
    music_tl = [row for row in tl_chunks if row[0] == "喜好/音乐"]
    assert len(music_tl) == 1
    _tag, display_text, bm25_tokens, embedding_blob, tombstoned = music_tl[0]
    assert tombstoned == 0
    assert "独立民谣" in display_text
    assert "电子音乐" in display_text
    assert embedding_blob is not None and len(embedding_blob) > 0
    # TAG_BOOST=2: tag tokens appear at least twice in bm25_tokens
    assert bm25_tokens.count("音乐") >= 2

    # Searching should surface both the music timeline evolution and the global format constraint
    resp = await pipeline.search(
        SearchRequest(query="用户现在的音乐喜好是什么", user_id="u_tl_sync", top_k=5)
    )
    combined = "\n".join(item.content for item in resp.data)
    assert "独立民谣" in combined
    assert "Python 3.12" in combined


async def test_parent_message_restoration_inlines_structured_fact_notes(db_path: Path):
    """F11: Restoring a normal parent message in Module D appends fact_notes at the end."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "宠物/信息",
                        "fact_text": "用户领养了一只三岁的柯基犬名叫花生，对鸡肉过敏",
                        "prev_state": "",
                        "current_state": "养柯基犬花生（对鸡肉过敏）",
                        "update_reason": "新增宠物信息",
                        "event_date_start": "2024-04-12",
                        "event_date_end": "2024-04-12",
                    }
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-fact-inline",
            user_id="u_fn",
            session_id="s_fn",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "上周六我去救助站领养了一只三岁的柯基，取名叫花生，"
                        "兽医说它不能吃鸡肉。"
                    ),
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        fact_notes = conn.execute(
            "SELECT fact_notes FROM parent_messages WHERE user_id = ? AND seq = 1",
            ("u_fn",),
        ).fetchone()[0]
    assert "对鸡肉过敏" in fact_notes

    resp = await pipeline.search(
        SearchRequest(query="领养的柯基花生有什么饮食禁忌", user_id="u_fn", top_k=5)
    )
    # Find the restored dialogue item containing the original raw sentence
    restored_items = [item for item in resp.data if "上周六我去救助站" in item.content]
    assert len(restored_items) >= 1
    restored_text = restored_items[0].content
    assert "对鸡肉过敏" in restored_text
    assert ("事实" in restored_text) or ("fact" in restored_text.casefold())


async def test_multiple_choice_options_07_03_vector_synthesis_bm25_weights_and_pure_rerank_query(
    db_path: Path,
):
    """F13: /search with options uses 0.7/0.3 Dense, 2.0/0.5 BM25, and pure Rerank query."""
    embedder = FakeEmbedder()
    reranker = FakeReranker(score_fn=lambda _q, _d, _i: 0.50)

    # Set orthogonal basis vectors to test 0.7 / 0.3 synthesis & BM25 2.0 vs 0.5 weighting:
    # e0 = [1, 0, 0, ...] for query stem
    # e1 = [0, 1, 0, ...] for options
    e0 = [1.0, 0.0, 0.0, 0.0]
    e1 = [0.0, 1.0, 0.0, 0.0]
    pure_query = "候选人曾获得哪项国际竞赛奖项"
    options = [
        "A. 国际数学奥林匹克金牌",
        "B. 国际物理奥林匹克银牌",
        "C. 国际信息学竞赛铜牌",
        "D. 国际化学竞赛优胜奖",
    ]

    embedder.set_vector(pure_query, e0)
    for opt in options:
        embedder.set_vector(opt, e1, match_substring=True)

    # Doc 1 matches the query stem vector e0 and query stem term "国际竞赛奖项"
    embedder.set_vector(
        "档案记录甲：候选人在高中阶段曾获得一项重要的国际竞赛奖项。", e0, match_substring=True
    )
    # Doc 2 matches the option vector e1 and option term "国际数学奥林匹克金牌"
    embedder.set_vector(
        "档案记录乙：高二那年他拿到了国际数学奥林匹克金牌。", e1, match_substring=True
    )
    # Doc 3 is unrelated (e2 = [0, 0, 1, 0])
    embedder.set_vector(
        "档案记录丙：大学期间他在校篮球队担任控球后卫。",
        [0.0, 0.0, 1.0, 0.0],
        match_substring=True,
    )

    pipeline, *_ = build_v2_pipeline(db_path=db_path, embedder=embedder, reranker=reranker)

    await pipeline.add(
        AddRequest(
            request_id="req-mcq-docs",
            user_id="u_mcq",
            session_id="s_mcq",
            messages=[
                {
                    "role": "user",
                    "content": "档案记录甲：候选人在高中阶段曾获得一项重要的国际竞赛奖项。",
                },
                {"role": "user", "content": "档案记录乙：高二那年他拿到了国际数学奥林匹克金牌。"},
                {"role": "user", "content": "档案记录丙：大学期间他在校篮球队担任控球后卫。"},
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(
            query=pure_query,
            options=options,
            user_id="u_mcq",
            top_k=3,
        )
    )

    # 1. Verify Reranker received the pure query stem WITHOUT options appended
    assert reranker.call_count >= 1
    rerank_query = reranker.calls[-1]["query"]
    assert rerank_query == pure_query
    assert "国际数学奥林匹克金牌" not in rerank_query

    # 2. Stem has 0.7 Dense & 2.0 BM25 weight; options have 0.3 Dense & 0.5 BM25 weight:
    # Doc 1 (stem match) ranks #1, Doc 2 (option match) ranks #2, and both beat Doc 3!
    assert len(resp.data) >= 2
    assert "档案记录甲" in resp.data[0].content
    assert "档案记录乙" in resp.data[1].content


async def test_weighted_rrf_rerank_fusion_and_is_latest_in_tag_boost(db_path: Path):
    """F12, F16: Weighted RRF + Rerank fusion and is_latest_in_tag boost over stale entries."""
    embedder = FakeEmbedder()
    # Give both old and new messages identical vectors and identical rerank scores (0.60)
    shared_vec = [0.6, 0.8, 0.0, 0.0]
    embedder.set_vector(
        "旧版状态记录：我的工作邮箱是 old_team@example.com。", shared_vec, match_substring=True
    )
    embedder.set_vector(
        "新版状态记录：我的工作邮箱是 new_team@example.com。", shared_vec, match_substring=True
    )
    embedder.set_vector("查询我的工作邮箱", shared_vec)

    reranker = FakeReranker(score_fn=lambda _q, _d, _i: 0.60)
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "联系方式/邮箱",
                        "fact_text": "用户的工作邮箱是 old_team@example.com",
                        "prev_state": "",
                        "current_state": "old_team@example.com",
                        "update_reason": "入职分配",
                        "event_date_start": "2023-01-01",
                        "event_date_end": "2023-01-01",
                    }
                ],
                "forget_actions": [],
            },
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "联系方式/邮箱",
                        "fact_text": "用户的工作邮箱更新为 new_team@example.com",
                        "prev_state": "old_team@example.com",
                        "current_state": "new_team@example.com",
                        "update_reason": "转岗变更",
                        "event_date_start": "2024-06-01",
                        "event_date_end": "2024-06-01",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )

    pipeline, *_ = build_v2_pipeline(
        db_path=db_path, embedder=embedder, extractor=extractor, reranker=reranker
    )

    await pipeline.add(
        AddRequest(
            request_id="req-boost-1",
            user_id="u_boost",
            session_id="s_b1",
            messages=[
                {
                    "role": "user",
                    "content": "旧版状态记录：我的工作邮箱是 old_team@example.com。",
                    "timestamp": 1672531200000,
                }
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-boost-2",
            user_id="u_boost",
            session_id="s_b2",
            messages=[
                {
                    "role": "user",
                    "content": "新版状态记录：我的工作邮箱是 new_team@example.com。",
                    "timestamp": 1717200000000,
                }
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(query="查询我的工作邮箱", user_id="u_boost", top_k=5)
    )
    dialogue_hits = [
        item
        for item in resp.data
        if "旧版状态记录" in item.content or "新版状态记录" in item.content
    ]
    assert len(dialogue_hits) == 2
    new_hit = next(item for item in dialogue_hits if "new_team@example.com" in item.content)
    old_hit = next(item for item in dialogue_hits if "old_team@example.com" in item.content)
    assert new_hit.score is not None and old_hit.score is not None
    assert new_hit.score > old_hit.score, (
        f"Expected is_latest_in_tag boost: new ({new_hit.score}) > old ({old_hit.score})"
    )
    assert dialogue_hits[0].id == new_hit.id


async def test_reranker_timeout_gracefully_falls_back_to_rrf(db_path: Path):
    """F12.2: When qwen3-rerank times out (4.0s), /search falls back to rrf_norm + latest boost."""
    reranker = FakeReranker(side_effect=TimeoutError("qwen3-rerank timed out after 4.0s"))
    pipeline, *_ = build_v2_pipeline(db_path=db_path, reranker=reranker)

    await pipeline.add(
        AddRequest(
            request_id="req-rerank-fb",
            user_id="u_rfb",
            session_id="s_rfb",
            messages=[
                {"role": "user", "content": "我的私人书单第一本是《百年孤独》。"},
                {"role": "user", "content": "周末我喜欢去奥林匹克森林公园慢跑十公里。"},
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(query="私人书单第一本是什么书", user_id="u_rfb", top_k=2)
    )
    assert len(resp.data) >= 1
    assert "百年孤独" in resp.data[0].content


def test_token_budget_never_returns_an_oversized_item():
    rows = [
        MemoryCandidate(id="big", content="x" * 1000, score=1.0),
        MemoryCandidate(id="small", content="ok", score=0.5),
    ]
    assert [row.id for row in pack_results(rows, top_k=10, token_budget=2)] == ["small"]
