# 契约：Dandelion Memory v2.2 统一流水线、切片级来源分层、外科手术式遗忘与按需父级组装契约
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from dandelion_memory.models import AddRequest, SearchRequest
from dandelion_memory.text import slice_message_content, tokenize_bm25

from .conftest import FakeEmbedder, FakeExtractor, FakeReranker, build_v2_pipeline


def test_t1_english_stemming_and_by_the_way_split():
    """T1: tokenize_bm25 stems English words (bike/bake/run) and

    slice_message_content splits on by-the-way markers.
    """
    # 1. Stemming verification: biking, bikes, baked, running -> bike, bake, run
    tokens_biking = tokenize_bm25("biking")
    tokens_bikes = tokenize_bm25("bikes")
    tokens_baked = tokenize_bm25("baked")
    tokens_running = tokenize_bm25("running")

    assert "bike" in tokens_biking, (
        f"Expected 'bike' stem in tokenize_bm25('biking'), got {tokens_biking}"
    )
    assert "bike" in tokens_bikes, (
        f"Expected 'bike' stem in tokenize_bm25('bikes'), got {tokens_bikes}"
    )
    assert "bake" in tokens_baked, (
        f"Expected 'bake' stem in tokenize_bm25('baked'), got {tokens_baked}"
    )
    assert "run" in tokens_running, (
        f"Expected 'run' stem in tokenize_bm25('running'), got {tokens_running}"
    )

    # Mutual hit: query "bike" and document containing "biking" must share the stem token
    tokens_query_bike = set(tokenize_bm25("bike"))
    tokens_doc_biking = set(tokenize_bm25("I enjoy biking every morning"))
    assert "bike" in (tokens_query_bike & tokens_doc_biking), (
        "Query 'bike' and doc with 'biking' must hit common stem 'bike'"
    )

    # 2. Independent slicing on by-the-way markers
    # Chinese marker: 顺便说
    text_zh = (
        "这是一篇关于魔法世界帝国历史的草稿。第一章讲述了古老王朝的兴衰与魔法师议会的成立。"
        "顺便说，我下周要搬家去上海了。"
    )
    slices_zh = slice_message_content(text_zh)
    assert len(slices_zh) >= 2, (
        f"Expected at least 2 slices for text with '顺便说', got {slices_zh}"
    )
    by_the_way_slice_zh = [s for s in slices_zh if "顺便说" in s]
    assert len(by_the_way_slice_zh) == 1, (
        f"Expected exactly 1 slice with '顺便说', got {by_the_way_slice_zh}"
    )
    assert by_the_way_slice_zh[0].startswith("顺便说"), (
        f"Slice must start with '顺便说', got {by_the_way_slice_zh[0]}"
    )
    assert "魔法师议会" not in by_the_way_slice_zh[0], (
        "Draft content must not leak into the '顺便说' slice"
    )

    # English marker: By the way
    text_en = (
        "This is a fantasy fiction draft about dragons. "
        "The hero enters the dark cave with a torch. "
        "By the way, I got a new job in Seattle yesterday."
    )
    slices_en = slice_message_content(text_en)
    assert len(slices_en) >= 2, (
        f"Expected at least 2 slices for text with 'By the way', got {slices_en}"
    )
    by_the_way_slice_en = [s for s in slices_en if "By the way" in s]
    assert len(by_the_way_slice_en) == 1, (
        f"Expected exactly 1 slice with 'By the way', got {by_the_way_slice_en}"
    )
    assert by_the_way_slice_en[0].startswith("By the way"), (
        f"Slice must start with 'By the way', got {by_the_way_slice_en[0]}"
    )
    assert "dark cave" not in by_the_way_slice_en[0], (
        "Draft content must not leak into the 'By the way' slice"
    )


async def test_t2_sliding_window_concurrency_and_read_only_isolation(db_path: Path):
    """T2: Single /add with >16 messages runs concurrent sliding windows,

    discarding facts & forget_actions on read-only context (S_ro).
    """
    # 20 valid user messages: seq 1..20
    # Window 1: S_win = [1..12], S_ro = []
    # Window 2: S_win = [13..20], S_ro = [11, 12]
    # Message 11 has explicit forget root ("请忘掉")
    # Window 2 LLM extractor tries to return a fact on seq 11 and a forget action on seq 11
    extractor = FakeExtractor(
        responses=[
            # Window 1 response: facts for seq 2 and 10
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/生活",
                        "fact_text": "用户在第2句提到的生活事实",
                        "current_state": "第2句事实",
                    },
                    {
                        "source_seqs": [10],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/生活",
                        "fact_text": "用户在第10句提到的重要旧琐事",
                        "current_state": "第10句旧琐事",
                    },
                ],
                "forget_actions": [],
            },
            # Window 2 response:
            # Fact on seq 11 (in S_ro, MUST be discarded, NEVER fallback to seq 20)
            # Forget action on seq 11 (in S_ro, MUST be discarded, so seq 10 MUST NOT be tombstoned)
            # Fact on seq 15 (in S_win, MUST be accepted)
            {
                "facts": [
                    {
                        "source_seqs": [11],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/生活",
                        "fact_text": "用户在只读上文第11句提到的琐事",
                        "current_state": "第11句琐事",
                    },
                    {
                        "source_seqs": [15],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/生活",
                        "fact_text": "用户在第15句提到的有效事实",
                        "current_state": "第15句事实",
                    },
                ],
                "forget_actions": [
                    {
                        "trigger_seq": 11,
                        "target_existing_tags": ["日常/生活"],
                        "target_keywords": ["旧琐事", "琐事"],
                        "target_description": "之前的旧琐事",
                        "batch_seqs_to_tombstone": [10, 11],
                    }
                ],
            },
        ]
    )

    pipeline, store, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    messages = [
        {"role": "user", "content": f"这是批量对话消息的第 {i} 条内容。"}
        for i in range(1, 21)
    ]
    # Message 10 & 11 specialized: message 11 has explicit forget keyword "请忘掉"
    messages[9] = {"role": "user", "content": "第 10 条消息：之前的重要旧琐事。"}
    messages[10] = {"role": "user", "content": "第 11 条消息：请忘掉之前的旧琐事。"}

    await pipeline.add(
        AddRequest(
            request_id="req-t2-window-20",
            user_id="u_t2",
            session_id="s_t2",
            messages=messages,
        )
    )

    # 1. Extractor must be called twice (sliding window for 20 messages > 16)
    assert extractor.call_count == 2, (
        f"Expected 2 extractor calls for 20 messages, got {extractor.call_count}"
    )

    # 2. Window 2 call must receive seq 11, 12 as read-only context
    call2 = extractor.calls[1]
    call2_msgs = call2["args"][0] if call2["args"] else call2["kwargs"].get("messages", [])
    call2_seqs = [m.get("seq") for m in call2_msgs if isinstance(m, dict)]
    assert 11 in call2_seqs and 12 in call2_seqs, (
        f"Window 2 must include read-only context seq 11, 12, got: {call2_seqs}"
    )

    with sqlite3.connect(db_path) as conn:
        fact_rows = conn.execute(
            "SELECT seq, display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? AND kind = 'fact_item'",
            ("u_t2",),
        ).fetchall()
        parents = dict(
            conn.execute(
                "SELECT seq, tombstoned FROM parent_messages WHERE user_id = ? ORDER BY seq",
                ("u_t2",),
            ).fetchall()
        )

    fact_seqs = [row[0] for row in fact_rows]
    fact_texts = [row[1] for row in fact_rows]

    # Valid facts from window 1 (seq 2, 10) and window 2 (seq 15) must be present
    assert 2 in fact_seqs, f"Fact at seq 2 should exist, got {fact_seqs}"
    assert 10 in fact_seqs, f"Fact at seq 10 should exist, got {fact_seqs}"
    assert 15 in fact_seqs, f"Fact at seq 15 should exist, got {fact_seqs}"

    # Hard isolation: fact on read-only seq 11 MUST be discarded!
    assert not any("第11句提到的琐事" in t for t in fact_texts), (
        "Fact on read-only context S_ro must be discarded"
    )
    assert 20 not in fact_seqs, (
        f"Fact on S_ro must never fallback to window end seq 20, but found fact_seqs={fact_seqs}"
    )

    # Hard isolation: forget action on read-only seq 11 MUST be discarded!
    # Therefore seq 10 MUST NOT be tombstoned
    assert parents[10] == 0, (
        "seq 10 must NOT be tombstoned because trigger_seq 11 is in read-only context S_ro"
    )


async def test_t3_cold_tag_merging_and_draft_prohibition_in_timeline(db_path: Path):
    """T3: '其他/*' cold tags merge automatically (>0.85 cosine / literal)

    and user_draft_fiction never enters tag_timelines.
    """
    embedder = FakeEmbedder()
    # Configure high cosine similarity (> 0.85) between literal-non-overlapping tags
    # "其他/守宫" vs "其他/爬宠小黄"
    v_reptile = [1.0, 0.1, 0.0, 0.0]
    embedder.set_vector("守宫", v_reptile, match_substring=True)
    embedder.set_vector("爬宠小黄", v_reptile, match_substring=True)

    extractor = FakeExtractor(
        responses=[
            # Message 1 & 2: cold tags with 0 literal overlap but cosine > 0.85
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "其他/守宫",
                        "fact_text": "用户养了一只守宫名为小黄",
                        "current_state": "养守宫小黄",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "其他/爬宠小黄",
                        "fact_text": "用户的爬宠小黄今天顺利蜕皮",
                        "current_state": "爬宠小黄脱皮顺利",
                    },
                    # Message 3: novel draft fiction fact
                    {
                        "source_seqs": [3],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "其他/玄幻宗门",
                        "fact_text": "主角林风持有青霜古剑",
                        "current_state": "持有青霜剑",
                        "source_tier": "user_draft_fiction",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )

    pipeline, store, _, _, _ = build_v2_pipeline(
        db_path=db_path, embedder=embedder, extractor=extractor
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t3-tags",
            user_id="u_t3",
            session_id="s_t3",
            messages=[
                {"role": "user", "content": "我最近养了一只守宫，取名叫小黄。"},
                {"role": "user", "content": "我的爬宠小黄今天顺利蜕皮了，状态很好。"},
                {
                    "role": "user",
                    "content": "【小说草稿】第二章：青云宗弟子林风拔出了腰间的青霜古剑。",
                },
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        tl_rows = conn.execute(
            "SELECT subject, tag, current_state, history_chain "
            "FROM tag_timelines WHERE user_id = ?",
            ("u_t3",),
        ).fetchall()

        draft_chunks = conn.execute(
            "SELECT chunk_id, display_text FROM search_chunks "
            "WHERE user_id = ? AND display_text LIKE '%青霜剑%'",
            ("u_t3",),
        ).fetchall()

    # 1. Cold tag merging: "其他/守宫" and "其他/爬宠小黄" have 0 literal overlap,
    # but cosine > 0.85 -> must merge into ONE timeline!
    reptile_timelines = [
        row
        for row in tl_rows
        if row[0] == "user" and row[1].startswith("其他/") and "玄幻" not in row[1]
    ]
    assert len(reptile_timelines) == 1, (
        f"Expected '其他/守宫' and '其他/爬宠小黄' to merge into 1 timeline, "
        f"got {len(reptile_timelines)}: {reptile_timelines}"
    )
    chain = json.loads(reptile_timelines[0][3])
    assert len(chain) == 2, (
        f"Expected 2 events in merged reptile timeline history_chain, got {len(chain)}"
    )

    # 2. Strict draft prohibition: user_draft_fiction must NEVER enter tag_timelines
    sect_timelines = [row for row in tl_rows if "玄幻宗门" in row[1] or "青霜剑" in row[2]]
    assert len(sect_timelines) == 0, (
        f"user_draft_fiction must never enter tag_timelines, but found: {sect_timelines}"
    )

    # 3. But the fact chunk exists in search_chunks
    assert len(draft_chunks) >= 1


async def test_t4_causal_forget_boundary_and_short_denominator_guard(db_path: Path):
    """T4: In-memory serial replay respects seq <= trigger_seq causal boundary,

    tombstones AI confirm, and prevents false-positives via short denominator guard.
    """
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/出行",
                        "fact_text": "用户上个月买了一辆公路自行车经常骑行",
                        "current_state": "拥有公路自行车",
                    },
                    {
                        "source_seqs": [3],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "喜好/音乐",
                        "fact_text": "用户喜欢古典交响 music",
                        "current_state": "喜欢古典交响 music",
                    },
                    {
                        "source_seqs": [9],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/出行",
                        "fact_text": "用户今天在车行看中了一辆新折叠自行车打算购买",
                        "current_state": "打算买新折叠自行车",
                    },
                ],
                "forget_actions": [
                    # Forget action 1: at seq 5, forget cycling/bicycle
                    {
                        "trigger_seq": 5,
                        "target_existing_tags": ["日常/出行"],
                        "target_keywords": ["自行车", "骑行"],
                        "target_description": "关于自行车和骑行的事情",
                        "batch_seqs_to_tombstone": [5, 6],
                    },
                    # Forget action 2: at seq 7, 2-word target ["electronic", "music"]
                    # target_existing_tags MUST be empty so tag-match doesn't bypass guard!
                    {
                        "trigger_seq": 7,
                        "target_existing_tags": [],
                        "target_keywords": ["electronic", "music"],
                        "target_description": "electronic music",
                        "batch_seqs_to_tombstone": [7, 8],
                    },
                ],
            }
        ]
    )

    pipeline, store, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    messages = [
        {"role": "user", "content": "今天天气不错，早上出去散步。"},  # seq 1
        {"role": "user", "content": "我上个月买了一辆公路自行车，经常去骑行锻炼身体。"},  # seq 2
        {"role": "user", "content": "昨晚听了贝多芬的古典交响 music，非常震撼。"},  # seq 3
        {"role": "user", "content": "今天下午喝了一杯拿铁咖啡。"},  # seq 4
        # seq 5 (trigger)
        {"role": "user", "content": "请忘掉关于自行车和骑行的事情，不要再提了。"},
        {
            "role": "assistant",
            "content": "好的，我已经为您清除所有关于自行车和骑行的记录。",
        },  # seq 6 (AI confirm)
        {
            "role": "user",
            "content": "Please forget everything about electronic music.",
        },  # seq 7 (2-word target)
        {
            "role": "assistant",
            "content": "Understood, I have removed all records about electronic music.",
        },  # seq 8 (AI confirm)
        {
            "role": "user",
            "content": "我今天去车行看中了一辆新的折叠自行车，打算入手。",
        },  # seq 9 (re-discussion > 5)
    ]

    await pipeline.add(
        AddRequest(
            request_id="req-t4-causal",
            user_id="u_t4",
            session_id="s_t4",
            messages=messages,
        )
    )

    with sqlite3.connect(db_path) as conn:
        chunks = conn.execute(
            "SELECT seq, kind, display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? ORDER BY seq",
            ("u_t4",),
        ).fetchall()
        parents = dict(
            conn.execute(
                "SELECT seq, tombstoned FROM parent_messages WHERE user_id = ? ORDER BY seq",
                ("u_t4",),
            ).fetchall()
        )

    # 1. Causal boundary:
    # seq 2 (bicycle before forget) must be tombstoned
    seq2_chunks = [c for c in chunks if c[0] == 2]
    assert len(seq2_chunks) > 0
    assert all(c[3] == 1 for c in seq2_chunks), f"seq 2 chunks should be tombstoned: {seq2_chunks}"
    assert parents[2] == 1, "seq 2 parent message must be tombstoned"

    # seq 5 (trigger) and seq 6 (AI confirm) must be tombstoned
    assert parents[5] == 1, "seq 5 forget command must be tombstoned"
    assert parents[6] == 1, "seq 6 AI confirm message must be tombstoned"

    # seq 9 (bicycle re-introduced AFTER trigger_seq 5) must SURVIVE (tombstoned == 0)
    seq9_chunks = [c for c in chunks if c[0] == 9]
    assert len(seq9_chunks) > 0
    assert all(c[3] == 0 for c in seq9_chunks), (
        f"seq 9 (seq > trigger_seq) must survive: {seq9_chunks}"
    )
    assert parents[9] == 0, "seq 9 parent message must survive"

    # 2. Short denominator protection:
    # Target was ["electronic", "music"] (2 words).
    # seq 3 had classical music ("古典交响 music"), matching 1/2 = 50% < 100%.
    # It must NOT be mistakenly killed!
    seq3_chunks = [c for c in chunks if c[0] == 3]
    assert len(seq3_chunks) > 0
    assert all(c[3] == 0 for c in seq3_chunks), (
        f"seq 3 classical music must NOT be killed by 2-word target: {seq3_chunks}"
    )
    assert parents[3] == 0


async def test_t5_surgical_forget_and_on_demand_assembly(db_path: Path):
    """T5: Surgical forget tombstones individual slice without mutating parent_messages.content,

    rendering [已擦除] on search.
    """
    extractor = FakeExtractor(
        responses=[
            # Batch 1: mixed parent message with bread (2 sentences) & cycling (with '顺便说')
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "喜好/烘焙",
                        "fact_text": "用户周末在家烤法式酸种面包",
                        "current_state": "烤法式酸种面包",
                    },
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/运动",
                        "fact_text": "用户周日去东湖骑行三十公里",
                        "current_state": "东湖骑行三十公里",
                    },
                ],
                "forget_actions": [],
            },
            # Batch 2: forget cycling
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": ["日常/运动"],
                        "target_keywords": ["东湖骑行", "骑行"],
                        "target_description": "东湖骑行三十公里",
                        "batch_seqs_to_tombstone": [2, 3],
                    }
                ],
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "酸种面包": 0.95,
            "法式酸种面包": 0.95,
        }
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    # 3 sentences with '顺便说' ensuring 2 independent slices:
    # Slice 0: bread (2 sentences)
    # Slice 1: cycling (starts with '顺便说')
    orig_content = (
        "周末我在家烤了法式酸种面包。它的外皮非常酥脆，口感很好。"
        "顺便说，周日我还去东湖骑行了三十公里。"
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t5-add",
            user_id="u_t5",
            session_id="s_t5",
            messages=[{"role": "user", "content": orig_content}],
        )
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t5-forget",
            user_id="u_t5",
            session_id="s_t5",
            messages=[
                {"role": "user", "content": "请把东湖骑行这件事彻底忘掉，不要再提了。"},
                {"role": "assistant", "content": "好的，我已经清除东湖骑行相关记忆。"},
            ],
        )
    )

    # 1. parent_messages.content is strictly read-only and unmutated
    with sqlite3.connect(db_path) as conn:
        parent_row = conn.execute(
            "SELECT content, tombstoned FROM parent_messages WHERE user_id = ? AND seq = 1",
            ("u_t5",),
        ).fetchone()
        chunk_rows = conn.execute(
            "SELECT kind, display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? AND seq = 1",
            ("u_t5",),
        ).fetchall()

    assert parent_row[0] == orig_content, "parent_messages.content must remain strictly unmodified"
    assert parent_row[1] == 0, (
        "parent message is not completely tombstoned since bread slice survives"
    )

    # Cycling slice & cycling fact are tombstoned; bread slice & bread fact survive
    bread_chunks = [c for c in chunk_rows if "酸种面包" in c[1]]
    cycling_chunks = [c for c in chunk_rows if "骑行" in c[1]]
    assert len(bread_chunks) >= 1 and all(c[2] == 0 for c in bread_chunks)
    assert len(cycling_chunks) >= 1 and all(c[2] == 1 for c in cycling_chunks)

    # 2. Search assembly:
    resp = await pipeline.search(
        SearchRequest(query="周末烤了什么面包", user_id="u_t5", top_k=5)
    )
    assert len(resp.data) >= 1
    found_item = resp.data[0]

    # Must contain bread text
    assert "法式酸种面包" in found_item.content
    # The forgotten cycling slice position must be rendered as [已擦除]
    assert "[已擦除]" in found_item.content, (
        f"Expected '[已擦除]' in assembled message, got:\n{found_item.content}"
    )
    # Cycling fact must NOT leak into [提炼事实]
    assert "东湖骑行" not in found_item.content, (
        f"Forgotten cycling fact leaked in content:\n{found_item.content}"
    )


async def test_t6_homogeneous_grouping_and_fiction_shelter_prevention(db_path: Path):
    """T6: Mixed parent message splits into (pmid, source_tier) homogeneous groups;

    D_direct skips fiction slices and inlines neighbors, while fiction does not inline neighbors.
    """
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "其他/草稿小说",
                        "fact_text": "深渊魔龙喷吐龙息摧毁了魔法要塞",
                        "current_state": "魔法要塞被毁",
                        "source_tier": "user_draft_fiction",
                    },
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "日常/住址",
                        "fact_text": "用户上周搬家到了杭州西湖区的新居",
                        "current_state": "居于杭州西湖区",
                        "source_tier": "user_direct",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )

    keyword_scores = {
        "杭州西湖区": 0.92,
        "搬家": 0.88,
        "深渊魔龙": 0.90,
    }

    def score_fn(query: str, doc: str, idx: int) -> float:
        if any(m in doc for m in ("状态演进汇总", "状态时间线", "【状态")):
            return 0.15
        for kw, val in keyword_scores.items():
            if kw in doc:
                return val
        return 0.10

    reranker = FakeReranker(score_fn=score_fn)

    pipeline, store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    # Message 1 has novel draft in front, followed by '顺便说' real life moving update
    content = (
        "深渊魔龙张开巨大的黑色双翼，喷吐出毁灭一切的龙息，整个魔法要塞化为灰烬。"
        "顺便说，我上周已经搬家到了杭州西湖区的新居。"
    )
    # Message 2 is an assistant neighbor in same session
    neighbor_content = "好的，搬家需要帮忙随时说。"

    await pipeline.add(
        AddRequest(
            request_id="req-t6-mixed",
            user_id="u_t6",
            session_id="s_t6",
            messages=[
                {"role": "user", "content": content},  # seq 1
                {"role": "assistant", "content": neighbor_content},  # seq 2 (neighbor)
            ],
        )
    )

    # 1. Search for real life moving update:
    resp_direct = await pipeline.search(
        SearchRequest(query="用户搬家到哪个城市哪个区了", user_id="u_t6", top_k=5)
    )
    direct_candidates = [item for item in resp_direct.data if not item.id.startswith("tl:")]
    assert len(direct_candidates) >= 1
    direct_item = direct_candidates[0]

    # The group ID for user_direct is pmid (without :user_draft_fiction)
    assert ":user_draft_fiction" not in direct_item.id
    # Assembled text contains moving house info
    assert "杭州西湖区" in direct_item.content
    # Novel draft slice must be SKIPPED during assembly (fiction shelter prevention!)
    assert "深渊魔龙" not in direct_item.content, (
        f"Fiction draft slice must not hitchhike into direct evidence:\n{direct_item.content}"
    )
    # Direct candidate inlines session neighbor seq 2 (§5.4.3)
    assert "搬家需要帮忙随时说" in direct_item.content, (
        f"user_direct evidence must inline session neighbor, got:\n{direct_item.content}"
    )

    # 2. Search for novel draft:
    resp_fiction = await pipeline.search(
        SearchRequest(query="深渊魔龙摧毁了什么", user_id="u_t6", top_k=5)
    )
    fiction_candidates = [item for item in resp_fiction.data if not item.id.startswith("tl:")]
    assert len(fiction_candidates) >= 1
    fiction_item = fiction_candidates[0]
    # The group ID for fiction draft carries :user_draft_fiction suffix
    assert ":user_draft_fiction" in fiction_item.id
    assert "深渊魔龙" in fiction_item.content
    # Moving house info must NOT be in the fiction draft group
    assert "杭州西湖区" not in fiction_item.content
    # Fiction candidate NEVER inlines session neighbors (§5.4.3)
    assert "搬家需要帮忙随时说" not in fiction_item.content, (
        f"user_draft_fiction evidence must NEVER inline neighbors, got:\n{fiction_item.content}"
    )


async def test_t7_timeline_relevance_gating_and_draft_fallback_with_backfill(db_path: Path):
    """T7: Irrelevant timelines (<0.40) not pinned; D_direct >=0.55 blocks drafts;

    insufficient D_direct triggers draft fallback (max 2) and backfills remaining slots.
    """
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    # Irrelevant timeline
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "工作/历史",
                        "fact_text": "用户五年前曾在北京做过运维工程师",
                        "current_state": "曾任运维",
                    },
                    # 5 direct memories with varying scores
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "tag": "日常/事实",
                        "fact_text": "真实直接记忆条目A",
                        "source_tier": "user_direct",
                    },
                    {
                        "source_seqs": [3],
                        "subject": "user",
                        "tag": "日常/事实",
                        "fact_text": "真实直接记忆条目B",
                        "source_tier": "user_direct",
                    },
                    {
                        "source_seqs": [4],
                        "subject": "user",
                        "tag": "日常/事实",
                        "fact_text": "真实直接记忆条目C",
                        "source_tier": "user_direct",
                    },
                    {
                        "source_seqs": [5],
                        "subject": "user",
                        "tag": "日常/事实",
                        "fact_text": "真实直接记忆条目D",
                        "source_tier": "user_direct",
                    },
                    {
                        "source_seqs": [6],
                        "subject": "user",
                        "tag": "日常/事实",
                        "fact_text": "真实直接记忆条目E",
                        "source_tier": "user_direct",
                    },
                    # 2 draft memories: Draft 1 qualifies (>= 0.30), Draft 2 fails (< 0.30)
                    {
                        "source_seqs": [7],
                        "subject": "user",
                        "tag": "小说/草稿",
                        "fact_text": "小说文稿条目Alpha",
                        "source_tier": "user_draft_fiction",
                    },
                    {
                        "source_seqs": [8],
                        "subject": "user",
                        "tag": "小说/草稿",
                        "fact_text": "小说文稿条目Beta",
                        "source_tier": "user_draft_fiction",
                    },
                ],
                "forget_actions": [],
            }
        ]
    )

    keyword_scores = {
        # Timeline score < 0.40
        "五年前曾在北京做过运维": 0.20,
        # Direct memories:
        # High score >= 0.55:
        "真实直接记忆条目A": 0.90,
        "真实直接记忆条目B": 0.85,
        # Lower score direct memories:
        "真实直接记忆条目C": 0.42,
        "真实直接记忆条目D": 0.38,
        "真实直接记忆条目E": 0.32,
        # Draft memories:
        "小说文稿条目Alpha": 0.45,  # >= 0.30 (qualifies for fallback)
        "小说文稿条目Beta": 0.15,  # < 0.30 (disqualified)
    }

    # Custom score_fn to ensure timeline summaries never mistakenly score >= 0.40
    def score_fn(query: str, doc: str, idx: int) -> float:
        if any(m in doc for m in ("状态演进汇总", "状态时间线", "【状态")):
            return 0.15
        for kw, val in keyword_scores.items():
            if kw in doc:
                return val
        return 0.10

    reranker = FakeReranker(score_fn=score_fn)

    pipeline, store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    messages = [
        {"role": "user", "content": "五年前我在北京做过运维工程师。"},  # seq 1
        {"role": "user", "content": "这是真实直接记忆条目A相关的内容。"},  # seq 2
        {"role": "user", "content": "这是真实直接记忆条目B相关的内容。"},  # seq 3
        {"role": "user", "content": "这是真实直接记忆条目C相关的内容。"},  # seq 4
        {"role": "user", "content": "这是真实直接记忆条目D相关的内容。"},  # seq 5
        {"role": "user", "content": "这是真实直接记忆条目E相关的内容。"},  # seq 6
        {"role": "user", "content": "【小说草稿】这是小说文稿条目Alpha的内容。"},  # seq 7
        {"role": "user", "content": "【小说草稿】这是小说文稿条目Beta的内容。"},  # seq 8
    ]

    await pipeline.add(
        AddRequest(
            request_id="req-t7-gating",
            user_id="u_t7",
            session_id="s_t7",
            messages=messages,
        )
    )

    # 1. Timeline relevance gating:
    # Querying "真实直接记忆条目": timeline for 工作/历史 has score 0.15 (< 0.40).
    # It must NOT be pinned at the top!
    resp = await pipeline.search(
        SearchRequest(query="查询真实直接记忆条目", user_id="u_t7", top_k=5)
    )
    first_item = resp.data[0]
    assert "五年前曾在北京做过运维" not in first_item.content, (
        "Irrelevant timeline (<0.40) must NOT be pinned at top"
    )

    # 2. High-score direct memories count n_high:
    # Only A (0.90) and B (0.85) have score >= 0.55 (n_high = 2 < top_k = 5).
    # Insufficient direct memory triggers draft fallback!
    # Draft Alpha (0.45 >= 0.30) qualifies; Draft Beta (0.15 < 0.30) does not qualify.
    # Total available slots rem = 5.
    # direct_keep = max(n_high=2, rem - 2=3) = 3 -> keeps A, B, C.
    # selected_draft = [Alpha] (1 draft).
    # Current count = 3 + 1 = 4 < 5.
    # UNFILLED BACKFILL: D (0.38) must be backfilled to fill the 5th slot!
    # Result must contain 5 items: A, B, C, D, and Alpha.
    returned_contents = [item.content for item in resp.data]
    assert len(resp.data) == 5, f"Expected 5 items backfilled to fill top_k=5, got {len(resp.data)}"

    # Alpha must be in results as draft fallback at bottom
    assert any("小说文稿条目Alpha" in c for c in returned_contents), (
        "Draft Alpha must fall back into results"
    )
    # Beta must NOT be in results (below 0.30)
    assert not any("小说文稿条目Beta" in c for c in returned_contents), (
        "Draft Beta (<0.30) must not enter results"
    )
    # D must be backfilled to satisfy top_k
    assert any("真实直接记忆条目D" in c for c in returned_contents), (
        "Direct D must be backfilled to fill remaining slot"
    )

    # 3. When D_direct >= 0.55 is enough to fill top_k (e.g. top_k=2, n_high=2):
    resp_top2 = await pipeline.search(
        SearchRequest(query="查询真实直接记忆条目", user_id="u_t7", top_k=2)
    )
    assert len(resp_top2.data) == 2
    assert not any("小说文稿" in item.content for item in resp_top2.data), (
        "When direct memories >= 0.55 can fill top_k, NO draft memories should be returned"
    )
