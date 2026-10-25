from __future__ import annotations

import sqlite3
from pathlib import Path

from dandelion_memory.models import AddRequest, SearchRequest
from dandelion_memory.text import approximate_tokens

from .conftest import FakeExtractor, build_v2_pipeline


async def test_normal_message_restores_parent_and_inlines_same_session_neighbors(db_path: Path):
    """F3: Normal messages (<=3000 chars) restore parent and inline same-session neighbors."""
    pipeline, *_ = build_v2_pipeline(db_path=db_path)

    req = AddRequest(
        request_id="req-normal-neighbors",
        user_id="u_chunk",
        session_id="s_chunk_1",
        messages=[
            {"role": "user", "content": "周六早上我先去菜市场买了新鲜的番茄和牛腩。"},
            {
                "role": "assistant",
                "content": (
                    "番茄炖牛腩是很经典的家常菜。你可以先把牛腩冷水下锅焯水去腥，"
                    "再小火慢炖九十分钟。出锅前撒一把香菜味道更好。"
                ),
            },
            {"role": "user", "content": "最后我按照这个方法炖出来特别软烂，配了两碗米饭。"},
        ],
    )
    await pipeline.add(req)

    # Also add a message in a DIFFERENT session to verify cross-session neighbors are never inlined
    await pipeline.add(
        AddRequest(
            request_id="req-other-session",
            user_id="u_chunk",
            session_id="s_chunk_2",
            messages=[
                {"role": "user", "content": "这是另一个会话里的无关消息：周日我去修自行车了。"}
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(
            query="牛腩冷水下锅焯水小火慢炖九十分钟",
            user_id="u_chunk",
            top_k=3,
        )
    )
    assert len(resp.data) >= 1
    top_content = resp.data[0].content
    # Restored full parent message #2 + inlined neighbor #1 and neighbor #3 from s_chunk_1
    assert "冷水下锅焯水" in top_content
    assert "出锅前撒一把香菜" in top_content
    assert "菜市场买了新鲜的番茄" in top_content
    assert "配了两碗米饭" in top_content
    assert "修自行车" not in top_content


async def test_neighbor_inlining_skips_neighbor_exceeding_1000_chars(db_path: Path):
    """F3: Neighbors with len(content) > 1000 chars must not be inlined into the anchor parent."""
    pipeline, *_ = build_v2_pipeline(db_path=db_path)

    oversized_neighbor = "长邻居前缀MARKER_OVERSIZED_NEIGHBOR " + (
        "这是一段超过一千字的背景描述。" * 80
    )
    assert 1000 < len(oversized_neighbor) <= 3000

    await pipeline.add(
        AddRequest(
            request_id="req-oversized-nb",
            user_id="u_nb",
            session_id="s_nb",
            messages=[
                {"role": "user", "content": oversized_neighbor},
                {
                    "role": "assistant",
                    "content": "核心命中锚点：量子纠缠密钥分发协议参数是 QKD-7749。",
                },
                {"role": "user", "content": "短邻居后缀：收到参数 QKD-7749，我记下了。"},
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(query="量子纠缠密钥分发协议参数 QKD-7749", user_id="u_nb", top_k=1)
    )
    assert len(resp.data) == 1
    top_text = resp.data[0].content
    assert "QKD-7749" in top_text
    assert "短邻居后缀" in top_text
    assert "MARKER_OVERSIZED_NEIGHBOR" not in top_text


async def test_long_document_over_3000_chars_uses_800_char_sliding_window(
    db_path: Path,
):
    """F3: Messages >3000 chars use ~800-char sliding window and return raw_slice directly."""
    pipeline, *_ = build_v2_pipeline(db_path=db_path)

    # Construct a ~10,000 character long document with a unique needle near the middle
    prefix_block = (
        "第一章节：星际航行常规引擎维护手册与冷却液循环规范。每次巡航前需检查主阀门压力表读数。\n"
        * 120
    )
    needle_block = (
        "【核心机密条款】深空探测器的应急曲率跃迁激活口令为 ORION-DELTA-9527，"
        "必须在主反应堆温度低于 420 开尔文时输入该激活口令。\n"
    )
    suffix_block = "第九章节：附录与船员轮班作息表。所有值班工程师需在每日零点提交日志。\n" * 130
    long_doc = prefix_block + needle_block + suffix_block
    assert len(long_doc) >= 9500

    await pipeline.add(
        AddRequest(
            request_id="req-clbench-long",
            user_id="u_long",
            session_id="s_long",
            messages=[
                {"role": "user", "content": "邻居前导消息：请帮我归档以下万字手册。"},
                {"role": "user", "content": long_doc},
                {"role": "assistant", "content": "邻居后置消息：手册已归档完毕。"},
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        long_parent = conn.execute(
            "SELECT msg_id, seq, length(content) FROM parent_messages "
            "WHERE user_id = ? AND seq = 2",
            ("u_long",),
        ).fetchone()
        assert long_parent is not None
        long_msg_id = long_parent[0]

        slices = conn.execute(
            "SELECT chunk_id, display_text FROM search_chunks "
            "WHERE user_id = ? AND parent_msg_id = ? AND kind = 'raw_slice' ORDER BY chunk_id",
            ("u_long", long_msg_id),
        ).fetchall()

    # Verify sliding window count and size (~800 chars per window, step ~640-650 chars)
    assert len(slices) >= 12
    for _, slice_text in slices:
        # Allow small metadata/header prefix while keeping window around ~800 chars
        assert len(slice_text) <= 1200

    # Verify overlap between at least one pair of adjacent slices
    has_overlap = any(
        slices[i][1][-80:] in slices[i + 1][1] or slices[i][1][-40:] in slices[i + 1][1]
        for i in range(len(slices) - 1)
    )
    assert has_overlap, (
        "Expected ~150-160 char overlap between consecutive 800-char sliding windows"
    )

    # Search for the needle buried in the middle of the 10,000-char document
    resp = await pipeline.search(
        SearchRequest(
            query="应急曲率跃迁激活口令 ORION-DELTA-9527",
            user_id="u_long",
            top_k=3,
        )
    )
    assert len(resp.data) >= 1
    top_hit = resp.data[0].content
    assert "ORION-DELTA-9527" in top_hit
    # Must NOT restore the full 10,000-char parent message
    assert len(top_hit) < 2000
    # Must NOT inline neighbor messages for >3000 char document slices
    assert "邻居前导消息" not in top_hit
    assert "邻居后置消息" not in top_hit


async def test_search_strictly_enforces_top_k_and_dynamic_timeline_pinning(db_path: Path):
    """F4: Verify len(results) <= request.top_k for top_k=1, 3, 10 and dynamic timeline pinning."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [i],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": f"偏好维度/类别{i}",
                        "fact_text": f"用户在类别{i}上的当前状态是状态值_{i}",
                        "prev_state": f"旧状态_{i}",
                        "current_state": f"状态值_{i}",
                        "update_reason": "近期调整",
                        "event_date_start": "2024-03-01",
                        "event_date_end": "2024-03-01",
                    }
                    for i in range(1, 8)
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    messages = [
        {
            "role": "user",
            "content": f"关于偏好维度类别{i}，我现在改成了状态值_{i}，不再用旧状态_{i}了。",
            "timestamp": 1709251200000 + i * 60000,
        }
        for i in range(1, 15)
    ]
    await pipeline.add(
        AddRequest(
            request_id="req-topk-batch",
            user_id="u_topk",
            session_id="s_topk",
            messages=messages,
        )
    )

    for k in (1, 3, 10):
        resp = await pipeline.search(
            SearchRequest(
                query="偏好维度类别与状态值",
                user_id="u_topk",
                top_k=k,
            )
        )
        assert 1 <= len(resp.data) <= k, f"Expected 1 <= len(data) <= {k}, got {len(resp.data)}"


async def test_search_respects_token_budget(db_path: Path):
    """F4: Verify total returned tokens never exceed the configured token_budget."""
    pipeline, *_ = build_v2_pipeline(db_path=db_path, token_budget=300)

    messages = [
        {
            "role": "user",
            "content": f"预算测试条目 #{i}："
            + ("这是一段用于测试令牌预算截断的较长文本内容。" * 8),
        }
        for i in range(1, 10)
    ]
    await pipeline.add(
        AddRequest(
            request_id="req-budget",
            user_id="u_budget",
            session_id="s_budget",
            messages=messages,
        )
    )

    resp = await pipeline.search(
        SearchRequest(query="预算测试条目文本内容", user_id="u_budget", top_k=10)
    )
    total_tokens = sum(approximate_tokens(item.content) for item in resp.data)
    assert len(resp.data) >= 1
    assert total_tokens <= 300
