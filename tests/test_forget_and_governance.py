from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dandelion_memory.models import AddRequest, SearchRequest

from .conftest import FakeExtractor, build_v2_pipeline


async def test_explicit_forget_tombstones_chunks_parent_and_timeline(
    db_path: Path,
):
    """F6: Explicit forget tombstones search_chunks, parent_messages, and blocks revival."""
    extractor = FakeExtractor(
        responses=[
            # Batch 1 extraction: safe password recorded under tag '安全/密码'
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "安全/密码",
                        "fact_text": "用户的保险箱密码是 849201",
                        "prev_state": "",
                        "current_state": "保险箱密码 849201",
                        "update_reason": "首次记录",
                        "event_date_start": "2024-02-01",
                        "event_date_end": "2024-02-01",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2 extraction: explicit forget instruction at seq=4 + assistant confirm at seq=5
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 4,
                        "target_existing_tags": ["安全/密码"],
                        "target_keywords": ["保险箱密码", "849201"],
                        "target_description": "用户的保险箱密码是 849201",
                        "batch_seqs_to_tombstone": [4, 5],
                    }
                ],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    # Batch 1: 3 contiguous messages in session s1 (seq 1, 2, 3)
    await pipeline.add(
        AddRequest(
            request_id="req-fg-b1",
            user_id="u_fg",
            session_id="s1",
            messages=[
                {"role": "user", "content": "周三下午我把书房的红木书架重新擦了一遍。"},
                {"role": "user", "content": "我的保险箱密码是 849201，放在书架底层。"},
                {"role": "user", "content": "擦完书架之后我泡了一壶正山小种红茶。"},
            ],
        )
    )

    # Batch 2: user issues explicit forget instruction (seq 4) + assistant confirms (seq 5)
    await pipeline.add(
        AddRequest(
            request_id="req-fg-b2",
            user_id="u_fg",
            session_id="s2",
            messages=[
                {"role": "user", "content": "请忘掉我的保险箱密码 849201，不要再提了"},
                {
                    "role": "assistant",
                    "content": "好的，我已经把您的保险箱密码 849201 彻底抹除，不会再提及。",
                },
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        parent_statuses = dict(
            conn.execute(
                "SELECT seq, tombstoned FROM parent_messages WHERE user_id = ? ORDER BY seq",
                ("u_fg",),
            ).fetchall()
        )
        leaked_chunks = conn.execute(
            "SELECT chunk_id, display_text FROM search_chunks "
            "WHERE user_id = ? AND tombstoned = 0 "
            "AND (display_text LIKE '%849201%' OR search_text LIKE '%849201%')",
            ("u_fg",),
        ).fetchall()
        tl_tombstoned = conn.execute(
            "SELECT tombstoned FROM tag_timelines WHERE user_id = ? AND tag = '安全/密码'",
            ("u_fg",),
        ).fetchone()

    # seq 1 & 3 remain active; seq 2 (secret), seq 4 (directive), seq 5 (confirm) are tombstoned
    assert parent_statuses[1] == 0
    assert parent_statuses[2] == 1
    assert parent_statuses[3] == 0
    assert parent_statuses[4] == 1
    assert parent_statuses[5] == 1
    assert leaked_chunks == []
    if tl_tombstoned is not None:
        assert tl_tombstoned[0] == 1

    # 1) Direct search for the secret must not return 849201
    direct_resp = await pipeline.search(
        SearchRequest(query="我的保险箱密码是多少 849201", user_id="u_fg", top_k=10)
    )
    assert all("849201" not in item.content for item in direct_resp.data)

    # 2) Searching adjacent neighbors (seq 1 / seq 3) must NOT revive seq 2 via neighbor inlining
    neighbor_resp = await pipeline.search(
        SearchRequest(query="红木书架正山小种红茶", user_id="u_fg", top_k=5)
    )
    assert len(neighbor_resp.data) >= 1
    assert any(
        "红木书架" in item.content or "正山小种" in item.content for item in neighbor_resp.data
    )
    assert all("849201" not in item.content for item in neighbor_resp.data)


async def test_explicit_forget_on_long_document_masks_slice_in_parent_with_erased_marker(
    db_path: Path,
):
    """F6: Tombstoning a slice in a >3000 char parent replaces that slice with '[已擦除]'."""
    part_a = "第一部分：公司园区班车时刻表，早班车七点半从地铁口发车，每十五分钟一班。\n" * 65
    secret_slice = "第二部分：实验室门禁紧急超控密码为 LAB-SECRET-6630，仅限安全主管使用。\n"
    part_c = "第三部分：园区食堂每周五提供海鲜自助晚餐，员工可凭工牌免费领取酸奶。\n" * 65
    long_text = part_a + secret_slice + part_c
    assert len(long_text) > 3500

    extractor = FakeExtractor(
        responses=[
            {"facts": [], "forget_actions": []},
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": [],
                        "target_keywords": ["LAB-SECRET-6630", "门禁紧急超控密码"],
                        "target_description": "实验室门禁紧急超控密码 LAB-SECRET-6630",
                        "batch_seqs_to_tombstone": [2],
                    }
                ],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-long-fg-1",
            user_id="u_long_fg",
            session_id="s_long_fg",
            messages=[{"role": "user", "content": long_text}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-long-fg-2",
            user_id="u_long_fg",
            session_id="s_long_fg",
            messages=[{"role": "user", "content": "请忘掉实验室门禁紧急超控密码 LAB-SECRET-6630"}],
        )
    )

    with sqlite3.connect(db_path) as conn:
        parent_content, parent_tomb = conn.execute(
            "SELECT content, tombstoned FROM parent_messages WHERE user_id = ? AND seq = 1",
            ("u_long_fg",),
        ).fetchone()
        active_secret_chunks = conn.execute(
            "SELECT chunk_id FROM search_chunks WHERE user_id = ? AND tombstoned = 0 "
            "AND display_text LIKE '%LAB-SECRET-6630%'",
            ("u_long_fg",),
        ).fetchall()

    assert active_secret_chunks == []
    assert "LAB-SECRET-6630" not in parent_content
    assert "[已擦除]" in parent_content
    assert parent_tomb == 0  # Non-erased parts of the long document remain available

    # Unrelated slices of the same long document are still searchable
    innocent_resp = await pipeline.search(
        SearchRequest(query="园区食堂每周五海鲜自助晚餐", user_id="u_long_fg", top_k=5)
    )
    assert any("海鲜自助晚餐" in item.content for item in innocent_resp.data)
    assert all("LAB-SECRET-6630" not in item.content for item in innocent_resp.data)


async def test_cross_batch_pending_forget_confirm_auto_tombstones_next_batch_assistant_reply(
    db_path: Path,
):
    """F6, F10.2: Trailing user forget directive sets pending_forget_confirm=1 for next batch."""
    extractor = FakeExtractor(
        responses=[
            {"facts": [], "forget_actions": []},
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": [],
                        "target_keywords": ["银行卡尾号", "9981"],
                        "target_description": "我的工资银行卡尾号是 9981",
                        "batch_seqs_to_tombstone": [2],
                    }
                ],
            },
            {"facts": [], "forget_actions": []},
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    # Batch 0: store secret
    await pipeline.add(
        AddRequest(
            request_id="req-cb-0",
            user_id="u_cb",
            session_id="s_cb",
            messages=[{"role": "user", "content": "我的工资银行卡尾号是 9981。"}],
        )
    )

    # Batch 1: ends with user explicit forget directive (no assistant reply yet)
    await pipeline.add(
        AddRequest(
            request_id="req-cb-1",
            user_id="u_cb",
            session_id="s_cb",
            messages=[{"role": "user", "content": "请删除关于我工资银行卡尾号 9981 的记忆。"}],
        )
    )

    with sqlite3.connect(db_path) as conn:
        pending_flag = conn.execute(
            "SELECT pending_forget_confirm FROM user_state WHERE user_id = ?",
            ("u_cb",),
        ).fetchone()[0]
    assert pending_flag == 1

    # Batch 2: starts with assistant confirmation repeating 9981, followed by a new user message
    await pipeline.add(
        AddRequest(
            request_id="req-cb-2",
            user_id="u_cb",
            session_id="s_cb",
            messages=[
                {
                    "role": "assistant",
                    "content": "好的，已经为您彻底删除工资银行卡尾号 9981 的记录。",
                },
                {"role": "user", "content": "今天晚饭我想做清蒸鲈鱼。"},
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        pending_after = conn.execute(
            "SELECT pending_forget_confirm FROM user_state WHERE user_id = ?",
            ("u_cb",),
        ).fetchone()[0]
        seq3_tomb, seq4_tomb = [
            row[0]
            for row in conn.execute(
                "SELECT tombstoned FROM parent_messages "
                "WHERE user_id = ? AND seq IN (3, 4) ORDER BY seq",
                ("u_cb",),
            ).fetchall()
        ]
        seq3_active_chunks = conn.execute(
            "SELECT COUNT(*) FROM search_chunks WHERE user_id = ? AND seq = 3 AND tombstoned = 0",
            ("u_cb",),
        ).fetchone()[0]

    assert pending_after == 0
    assert seq3_tomb == 1
    assert seq3_active_chunks == 0
    assert seq4_tomb == 0

    resp = await pipeline.search(
        SearchRequest(query="银行卡尾号 9981 清蒸鲈鱼", user_id="u_cb", top_k=10)
    )
    assert all("9981" not in item.content for item in resp.data)
    assert any("清蒸鲈鱼" in item.content for item in resp.data)


async def test_contradiction_without_forget_keywords_is_blocked_from_tombstoning(
    db_path: Path,
):
    """F8.1: Natural state change without explicit forget roots blocks LLM forget_actions."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "居住/城市",
                        "fact_text": "用户目前居住在北京",
                        "prev_state": "",
                        "current_state": "居住在北京",
                        "update_reason": "初始居住地",
                        "event_date_start": "2022-06-01",
                        "event_date_end": "2022-06-01",
                    }
                ],
                "forget_actions": [],
            },
            # LLM mistakenly emits a forget_action on a normal state change!
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "居住/城市",
                        "fact_text": "用户搬家到上海了，以前住北京",
                        "prev_state": "居住在北京",
                        "current_state": "居住在上海",
                        "update_reason": "工作调动搬家",
                        "event_date_start": "2024-05-01",
                        "event_date_end": "2024-05-01",
                    }
                ],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": ["居住/城市"],
                        "target_keywords": ["北京"],
                        "target_description": "用户居住在北京",
                        "batch_seqs_to_tombstone": [1],
                    }
                ],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-contra-1",
            user_id="u_contra",
            session_id="s_c1",
            messages=[
                {"role": "user", "content": "我现在住在北京海淀区。", "timestamp": 1654041600000}
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-contra-2",
            user_id="u_contra",
            session_id="s_c2",
            messages=[
                {
                    "role": "user",
                    "content": "我搬家到上海徐汇区了，以前住北京。",
                    "timestamp": 1714521600000,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        tomb_counts = conn.execute(
            "SELECT COUNT(*) FROM parent_messages WHERE user_id = ? AND tombstoned = 1",
            ("u_contra",),
        ).fetchone()[0]
        tl_row = conn.execute(
            "SELECT current_state, history_chain, rendered_summary, tombstoned "
            "FROM tag_timelines WHERE user_id = ? AND subject = 'user' AND tag = '居住/城市'",
            ("u_contra",),
        ).fetchone()

    # Code-level validator blocked the false forget_action!
    assert tomb_counts == 0
    assert tl_row is not None
    current_state, history_json, rendered_summary, tl_tomb = tl_row
    assert tl_tomb == 0
    assert "上海" in current_state
    history = json.loads(history_json)
    assert len(history) >= 2
    assert "北京" in rendered_summary and "上海" in rendered_summary
