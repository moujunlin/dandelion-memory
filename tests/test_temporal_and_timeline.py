from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dandelion_memory.models import AddRequest, SearchRequest

from .conftest import FakeExtractor, build_v2_pipeline


async def test_timestamp_inherits_forward_within_same_session_and_extracts_speaker(db_path: Path):
    """F9: First message timestamp inherits forward in same session_id; speaker is extracted."""
    pipeline, *_ = build_v2_pipeline(db_path=db_path)

    ts_ms = 1683554160000  # 2023-05-08 13:56:00 UTC -> 1683554160 sec
    await pipeline.add(
        AddRequest(
            request_id="req-ts-inherit-1",
            user_id="u_ts",
            session_id="s_inherit",
            messages=[
                {
                    "role": "user",
                    "content": "Alice: We booked the flight to Kyoto.",
                    "timestamp": ts_ms,
                },
                {
                    "role": "assistant",
                    "content": "Great! Do you need hotel recommendations in Kyoto?",
                    "timestamp": None,
                },
                {
                    "role": "user",
                    "content": "Bob_Smith: Yes, somewhere near Gion would be wonderful.",
                    "timestamp": None,
                },
            ],
        )
    )

    # Also test cross-request inheritance within the same session_id when timestamp is omitted
    await pipeline.add(
        AddRequest(
            request_id="req-ts-inherit-2",
            user_id="u_ts",
            session_id="s_inherit",
            messages=[
                {
                    "role": "assistant",
                    "content": "I found three boutique ryokans in Gion.",
                    "timestamp": None,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT seq, role, speaker, session_ts, session_time_str, content "
            "FROM parent_messages WHERE user_id = ? ORDER BY seq",
            ("u_ts",),
        ).fetchall()
        chunk_ts_list = [
            r[0]
            for r in conn.execute(
                "SELECT session_ts FROM search_chunks WHERE user_id = ?",
                ("u_ts",),
            ).fetchall()
        ]

    assert len(rows) == 4
    for seq, _role, _speaker, session_ts, session_time_str, _content in rows:
        assert session_ts == 1683554160, f"seq={seq} failed to inherit session_ts=1683554160"
        assert session_time_str and "2023-05-08" in session_time_str

    # Check speaker extraction on seq 1 ("Alice"), seq 2 ("assistant"), seq 3 ("Bob_Smith")
    assert rows[0][2] == "Alice"
    assert "Alice: We booked" in rows[0][5]
    assert rows[1][2] == "assistant"
    assert rows[2][2] == "Bob_Smith"
    assert all(ts == 1683554160 for ts in chunk_ts_list)


async def test_out_of_order_session_ts_orders_timeline_history_by_session_ts_then_seq(
    db_path: Path,
):
    """F7: Out-of-order session_ts sorts tag_timelines by (session_ts, seq) ascending."""
    extractor = FakeExtractor(
        responses=[
            # Batch 1 (seq=1, session_ts=1700000000 [2023-11-14]): newer session ingested first
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "职业/职位",
                        "fact_text": "用户现在在字节跳动担任高级后端架构师",
                        "prev_state": "",
                        "current_state": "字节跳动高级后端架构师",
                        "update_reason": "晋升",
                        "event_date_start": "2023-11-14",
                        "event_date_end": "2023-11-14",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2 (seq=2, session_ts=1600000000 [2020-09-13]): older session ingested second!
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "职业/职位",
                        "fact_text": "用户在美团做测试开发实习生",
                        "prev_state": "",
                        "current_state": "美团测试开发实习生",
                        "update_reason": "校招实习",
                        "event_date_start": "2020-09-13",
                        "event_date_end": "2020-09-13",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    # First write: 2023-11-14 (1700000000000 ms)
    await pipeline.add(
        AddRequest(
            request_id="req-ooo-1",
            user_id="u_ooo",
            session_id="s_2023",
            messages=[
                {
                    "role": "user",
                    "content": "我现在在字节跳动担任高级后端架构师。",
                    "timestamp": 1700000000000,
                }
            ],
        )
    )

    # Second write (out-of-order): 2020-09-13 (1600000000000 ms)
    await pipeline.add(
        AddRequest(
            request_id="req-ooo-2",
            user_id="u_ooo",
            session_id="s_2020",
            messages=[
                {
                    "role": "user",
                    "content": "我在美团做测试开发实习生。",
                    "timestamp": 1600000000000,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        tl = conn.execute(
            "SELECT latest_seq, latest_session_ts, current_state, history_chain, rendered_summary "
            "FROM tag_timelines WHERE user_id = ? AND subject = 'user' AND tag = '职业/职位'",
            ("u_ooo",),
        ).fetchone()

    assert tl is not None
    latest_seq, latest_session_ts, current_state, history_json, rendered_summary = tl
    assert latest_session_ts == 1700000000
    assert latest_seq == 1
    assert "字节跳动" in current_state
    assert "美团" not in current_state

    chain = json.loads(history_json)
    assert len(chain) == 2
    assert chain[0]["session_ts"] == 1600000000
    assert "美团" in chain[0]["state"]
    assert chain[1]["session_ts"] == 1700000000
    assert "字节跳动" in chain[1]["state"]
    assert "字节跳动" in rendered_summary and "美团" in rendered_summary


async def test_other_person_fact_is_prefixed_and_isolated_from_user_timeline(db_path: Path):
    """F10, F15: is_other_person=True facts get prefixed and do not overwrite user timeline."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "喜好/饮品",
                        "fact_text": "用户最喜欢喝冰美式咖啡",
                        "prev_state": "",
                        "current_state": "最喜欢喝冰美式咖啡",
                        "update_reason": "日常习惯",
                        "event_date_start": "2024-01-10",
                        "event_date_end": "2024-01-10",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "妈妈",
                        "is_other_person": True,
                        "tag": "喜好/饮品",
                        "fact_text": "妈妈最喜欢喝茉莉花茶，从不喝咖啡",
                        "prev_state": "",
                        "current_state": "最喜欢喝茉莉花茶",
                        "update_reason": "长辈习惯",
                        "event_date_start": "2024-01-10",
                        "event_date_end": "2024-01-10",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-other-person",
            user_id="u_op",
            session_id="s_op",
            messages=[
                {"role": "user", "content": "我每天早上都必须喝一杯冰美式咖啡。"},
                {"role": "user", "content": "但我妈妈最喜欢喝茉莉花茶，她从不喝咖啡。"},
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        user_tl = conn.execute(
            "SELECT current_state FROM tag_timelines "
            "WHERE user_id = ? AND subject = 'user' AND tag = '喜好/饮品'",
            ("u_op",),
        ).fetchone()
        mom_tl = conn.execute(
            "SELECT current_state FROM tag_timelines "
            "WHERE user_id = ? AND subject = '妈妈' AND tag = '喜好/饮品'",
            ("u_op",),
        ).fetchone()
        seq2_notes = conn.execute(
            "SELECT fact_notes FROM parent_messages WHERE user_id = ? AND seq = 2",
            ("u_op",),
        ).fetchone()[0]

    assert user_tl is not None
    assert "冰美式" in user_tl[0]
    assert "茉莉花茶" not in user_tl[0]

    assert mom_tl is not None
    assert "茉莉花茶" in mom_tl[0]

    # Verify other-person prefix in parent_messages.fact_notes
    assert "妈妈" in seq2_notes
    assert ("他人属性" in seq2_notes) or ("非User本人" in seq2_notes)


async def test_event_date_range_parsed_to_utc_seconds_and_boosted_by_time_range_channel(
    db_path: Path,
):
    """F7, F10: event_date_start/end parse to UTC seconds and support time window filtering."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "行程/度假",
                        "fact_text": "用户在云南大理洱海骑行度假",
                        "prev_state": "",
                        "current_state": "云南大理洱海骑行",
                        "update_reason": "五一假期出行",
                        "event_date_start": "2023-05-01",
                        "event_date_end": "2023-05-07",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "行程/度假",
                        "fact_text": "用户在哈尔滨看冰雕展览",
                        "prev_state": "云南大理洱海骑行",
                        "current_state": "哈尔滨看冰雕",
                        "update_reason": "元旦出行",
                        "event_date_start": "2024-01-01",
                        "event_date_end": "2024-01-03",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, *_ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-event-range",
            user_id="u_ev",
            session_id="s_ev",
            messages=[
                {
                    "role": "user",
                    "content": "去年五一那周我去云南大理洱海骑行度假了。",
                    "timestamp": 1705000000000,
                },
                {
                    "role": "user",
                    "content": "今年元旦我去哈尔滨看了冰雕展览。",
                    "timestamp": 1705000060000,
                },
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        may_chunks = conn.execute(
            "SELECT event_time_start, event_time_end FROM search_chunks "
            "WHERE user_id = ? AND seq = 1 AND kind IN ('raw_slice', 'fact_item')",
            ("u_ev",),
        ).fetchall()

    assert len(may_chunks) >= 1
    # 2023-05-01 00:00:00 UTC = 1682899200; 2023-05-07 23:59:59 UTC = 1683503999
    assert any(start == 1682899200 and end == 1683503999 for start, end in may_chunks)

    # Query specifically for 2023-05-03 (falls inside 2023-05-01 .. 2023-05-07)
    resp = await pipeline.search(
        SearchRequest(
            query="2023-05-03 那天我在哪里度假旅行",
            user_id="u_ev",
            top_k=3,
        )
    )
    assert len(resp.data) >= 1
    assert "大理洱海" in resp.data[0].content
