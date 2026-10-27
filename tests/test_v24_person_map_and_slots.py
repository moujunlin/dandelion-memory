# 契约：Dandelion Memory v2.4 全局人物状态 Map、稳定 slot_id 主键与归属贴标契约
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dandelion_memory.pipeline as pipeline_mod
from dandelion_memory.extractor import DashScopeExtractor, sanitize_extraction_result
from dandelion_memory.models import AddRequest, SearchRequest

from .conftest import FakeExtractor, FakeReranker, build_v2_pipeline


@pytest.mark.asyncio
async def test_slot_id_primary_key_and_no_hash_f_suffix(db_path: Path):
    """1. 稳定 slot_id 主键与零 #F1 标签拼接契约。

    - 使用 build_v2_pipeline + pipeline.add 写入；
    - FakeExtractor 返回两条 who="user"、同标签 "资产/物品与车辆" 的不同事实；
    - 断言：
      - tag_timelines 中生成 slot_id = 'F1', 'F2' 且 tag == '资产/物品与车辆'（严格无 #F 后缀）；
      - search_chunks 中同时存在 tl:{user_id}:F1, F2 与 fact:{user_id}:F1, F2，
        且 slot_id 列均为 'F1', 'F2'。
    """
    user_id = "test_user_slot_pk"
    session_id = "sess_slot_pk_01"

    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "资产/物品与车辆",
                        "fact": "用户买了一辆山地自行车",
                        "current_state": "买了一辆山地自行车",
                        "is_fiction": False,
                    },
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "资产/物品与车辆",
                        "fact": "用户买了一台 M3 MacBook Pro 笔记本电脑",
                        "current_state": "买了一台 M3 MacBook Pro 笔记本电脑",
                        "is_fiction": False,
                    },
                ],
                "forget_actions": [],
            }
        ]
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
    )

    await pipeline.add(
        AddRequest(
            request_id="req-slot-pk-01",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "content": "我今天买了一辆山地自行车，还买了一台 M3 MacBook Pro 笔记本电脑。",
                }
            ],
        )
    )

    conn = store._connect()
    # 断言 A：表结构必须包含 slot_id 列
    tt_cols = {r["name"] for r in conn.execute("PRAGMA table_info(tag_timelines)").fetchall()}
    assert "slot_id" in tt_cols, "tag_timelines 表必须包含 slot_id 列作为唯一标识"

    sc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(search_chunks)").fetchall()}
    assert "slot_id" in sc_cols, "search_chunks 表必须包含 slot_id 列"

    # 断言 B：tag_timelines 中以 (user_id, slot_id) 存储了 F1 和 F2，且 tag 均保持纯净（零 #F 后缀）
    tt_rows = conn.execute(
        "SELECT slot_id, subject, tag, current_state FROM tag_timelines "
        "WHERE user_id = ? ORDER BY slot_id ASC",
        (user_id,),
    ).fetchall()
    assert len(tt_rows) == 2, f"同 tag 写入 2 条事实应保存 2 条独立槽位，实际为 {len(tt_rows)} 条"
    assert tt_rows[0]["slot_id"] == "F1"
    assert tt_rows[1]["slot_id"] == "F2"
    assert tt_rows[0]["tag"] == "资产/物品与车辆"
    assert tt_rows[1]["tag"] == "资产/物品与车辆"
    assert "#F" not in tt_rows[0]["tag"], f"tag_timelines tag 包含非法 #F 后缀: {tt_rows[0]['tag']}"
    assert "#F" not in tt_rows[1]["tag"], f"tag_timelines tag 包含非法 #F 后缀: {tt_rows[1]['tag']}"

    # 断言 C：search_chunks 中状态块与事实块均生成，且 slot_id 列正确写入
    tl_chunks = conn.execute(
        "SELECT chunk_id, slot_id, tag FROM search_chunks "
        "WHERE user_id = ? AND kind = 'timeline_summary' ORDER BY slot_id ASC",
        (user_id,),
    ).fetchall()
    assert len(tl_chunks) == 2, f"search_chunks 应有 2 个状态块，实际为 {len(tl_chunks)}"
    assert tl_chunks[0]["chunk_id"] == f"tl:{user_id}:F1"
    assert tl_chunks[1]["chunk_id"] == f"tl:{user_id}:F2"
    assert tl_chunks[0]["slot_id"] == "F1"
    assert tl_chunks[1]["slot_id"] == "F2"

    fact_chunks = conn.execute(
        "SELECT chunk_id, slot_id, tag FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' ORDER BY slot_id ASC",
        (user_id,),
    ).fetchall()
    assert len(fact_chunks) == 2, f"search_chunks 应有 2 个事实块，实际为 {len(fact_chunks)}"
    assert fact_chunks[0]["chunk_id"] == f"fact:{user_id}:F1"
    assert fact_chunks[1]["chunk_id"] == f"fact:{user_id}:F2"
    assert fact_chunks[0]["slot_id"] == "F1"
    assert fact_chunks[1]["slot_id"] == "F2"


@pytest.mark.asyncio
async def test_single_restored_parent_card_with_dynamic_attribution(db_path: Path):
    """2. 原始对话卡不拆卡、不降权、实时动态渲染头部与底部归属契约。

    - 写入一轮包含 user 事实与 other_person (Priya) 事实的混合对话
      （以及纯案例 Lt. Meera Kapoor 对话）；
    - pipeline.search 检索：
      - 同一 turn_pair_id 只生成唯一一张 restored_parent 卡片
        （绝不拆出 :other_person:Priya 副本卡）；
      - 原始对话卡的分数未被乘以 0.35 降权；
      - 即使 parent_messages.fact_notes 为空，卡片头部正文前也显式包含
        【注意：本段包含非User本人（系Priya）的信息，请勿与User本人混淆】；
      - 卡片底部 [提炼事实: ...] 同时包含 user 事实和带【非User本人，系Priya的信息】的事实标注。
    """
    user_id = "user_dyn_attr_test"
    session_id = "sess_attr_01"

    extractor = FakeExtractor(
        responses=[
            # Round 1: 混合对话 (User 考驾照 + Priya 排灯节)
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "技能/驾照",
                        "fact": "用户正在准备考驾照",
                        "is_fiction": False,
                    },
                    {
                        "seq": 1,
                        "who": "Priya",
                        "aliases": ["Priya"],
                        "relations": ["colleague"],
                        "tag": "家庭与节日",
                        "fact": "Priya在老家和家人一起过排灯节",
                        "is_fiction": False,
                    },
                ],
                "forget_actions": [],
            },
            # Round 2: 纯案例对话 (Lt. Meera Kapoor)
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "Lt. Meera Kapoor",
                        "aliases": ["Lt. Meera Kapoor", "Meera Kapoor"],
                        "relations": ["rescue_officer"],
                        "tag": "工作/案例",
                        "fact": "Lt. Meera Kapoor 在暴风雨救援中指挥撤离了 20 位渔民",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "排灯节": 0.90,
            "考驾照": 0.85,
            "Meera Kapoor": 0.92,
            "暴风雨救援": 0.88,
        }
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        reranker=reranker,
    )

    # 写入 Round 1 对话 (seq 1 & 2)
    await pipeline.add(
        AddRequest(
            request_id="req-attr-01",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "我最近在准备考驾照。另外我的同事 Priya 跟我说她在老家和家人一起过排灯节。"
                    ),
                    "timestamp": 1000,
                },
                {
                    "role": "assistant",
                    "content": "考驾照加油！排灯节也是非常温馨热闹的节日。",
                    "timestamp": 1001,
                },
            ],
        )
    )

    # 写入 Round 2 案例对话 (seq 3 & 4)
    await pipeline.add(
        AddRequest(
            request_id="req-attr-02",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "今天读了一篇搜救案例，Lt. Meera Kapoor "
                        "在暴风雨救援中指挥撤离了 20 位渔民。"
                    ),
                    "timestamp": 2000,
                },
                {
                    "role": "assistant",
                    "content": "这样的搜救行动展现了非凡的勇气与专业指挥。",
                    "timestamp": 2001,
                },
            ],
        )
    )

    # 核心契约检验前提：清空 parent_messages.fact_notes 死文本缓存
    # 证明头部与底部的归属完全由检索阶段实时动态挂载，绝不依赖写入时固化的 dead text
    conn = store._connect()
    conn.execute("UPDATE parent_messages SET fact_notes = '' WHERE user_id = ?", (user_id,))
    conn.commit()

    # 检索 1: 针对 Round 1
    res1 = await pipeline.search(
        SearchRequest(
            user_id=user_id,
            query="Priya 排灯节",
            top_k=10,
        )
    )

    r1_cards = [
        c for c in res1.data
        if not c.id.startswith("tl:") and ("考驾照" in c.content or "排灯节" in c.content)
    ]

    # 断言 1：同一轮问答只生成唯一一张 restored_parent 卡片（绝不拆出 :other_person:Priya）
    assert len(r1_cards) == 1, (
        f"同一轮问答应只生成唯一一张 restored_parent 卡片，实际发现 {len(r1_cards)} 张: "
        f"{[c.id for c in r1_cards]}"
    )
    card1 = r1_cards[0]
    assert ":other_person" not in card1.id, (
        f"绝不应拆出 :other_person 副本卡，实际卡片 ID 为: {card1.id}"
    )

    # 断言 2：原始对话卡未被乘以 0.35 降权（FakeReranker 命中排灯节=0.90，若乘 0.35 会跌至 ~0.315）
    assert card1.score >= 0.50, f"原始对话卡不应被乘 0.35 降权，实际得分为: {card1.score}"

    # 断言 3：即使 fact_notes 为空，卡片头部正文前也显式包含归属警示
    expected_header_1 = "【注意：本段包含非User本人（系Priya）的信息，请勿与User本人混淆】"
    assert expected_header_1 in card1.content, (
        f"卡片头部应包含动态归属警示，实际卡片内容:\n{card1.content}"
    )

    # 断言 4：卡片底部 [提炼事实: ...] 同时包含 User 本人事实和带【非User本人，系Priya的信息】的事实
    assert "[提炼事实:" in card1.content, f"卡片底部应包含事实提炼，实际卡片内容:\n{card1.content}"
    assert "准备考驾照" in card1.content
    assert "【非User本人，系Priya的信息】" in card1.content or "系Priya" in card1.content

    # 检索 2: 针对 Round 2 (纯 Lt. Meera Kapoor 案例)
    res2 = await pipeline.search(
        SearchRequest(
            user_id=user_id,
            query="暴风雨救援 渔民",
            top_k=10,
        )
    )
    r2_cards = [
        c for c in res2.data
        if not c.id.startswith("tl:") and ("20 位渔民" in c.content or "Meera Kapoor" in c.content)
    ]
    assert len(r2_cards) == 1
    card2 = r2_cards[0]
    assert ":other_person" not in card2.id
    assert card2.score >= 0.50
    expected_header_2 = (
        "【注意：本段包含非User本人（系Lt. Meera Kapoor）的信息，请勿与User本人混淆】"
    )
    assert expected_header_2 in card2.content
    assert (
        "【非User本人，系Lt. Meera Kapoor的信息】" in card2.content
        or "系Lt. Meera Kapoor" in card2.content
    )


@pytest.mark.asyncio
async def test_summarizer1_merge_slots_preserves_fact_item_inv1(db_path: Path):
    """3. 一号模型同人状态合并与 INV-1 历史事实保护契约。

    - 为 subject="Rachel" 写入早先状态 F1 (seq=1, "搬去芝加哥市区")
      与后续状态 F2 (seq=3, "搬回郊区")；
    - 调用 store.merge_timeline_slots(user_id, session_id,
        [{"older_fid": "F1", "newer_fid": "F2", "latest_state": "搬回郊区（原搬去芝加哥市区）"}])；
    - 验证：
      - 活跃状态栏中只剩 1 条属于 Rachel 的状态槽（存活槽位为 slot_id == "F2"，
        当前状态为最新，history_chain 包含 #1 与 #3，superseded_seqs 包含 [1]）；
      - 被合并的旧状态卡 tl:{user_id}:F1 的 tombstoned == 1；
      - INV-1 铁律：旧事实块 fact:{user_id}:F1 的 tombstoned == 0（依然活跃），
        且检索召回历史对话 #1 时，其 TurnPairCard 依然能实时渲染出 Rachel 的头部提醒与底部事实注脚！
    """
    user_id = "user_inv1_test"
    session_id = "sess_inv1_01"

    extractor = FakeExtractor(
        responses=[
            # Round 1 (seq=1, 2): Rachel 搬去芝加哥市区
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "Rachel",
                        "aliases": ["Rachel"],
                        "relations": ["friend"],
                        "tag": "居住/城市",
                        "fact": "朋友 Rachel 搬去芝加哥市区",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 2 (seq=3, 4): Rachel 搬回郊区
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "Rachel",
                        "aliases": ["Rachel"],
                        "relations": ["friend"],
                        "tag": "居住/城市",
                        "fact": "朋友 Rachel 搬回郊区",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )

    reranker = FakeReranker(keyword_scores={"芝加哥市区": 0.95, "郊区": 0.90})
    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        reranker=reranker,
    )

    # 写入 Round 1 对话 (分配 seq 1, 2)
    await pipeline.add(
        AddRequest(
            request_id="req-inv1-01",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "content": "朋友 Rachel 搬去芝加哥市区了，她说市区生活很方便。",
                    "timestamp": 1000,
                },
                {
                    "role": "assistant",
                    "content": "市区生活确实配套更完善。",
                    "timestamp": 1001,
                },
            ],
        )
    )

    # 写入 Round 2 对话 (分配 seq 3, 4)
    await pipeline.add(
        AddRequest(
            request_id="req-inv1-02",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {
                    "role": "user",
                    "content": "Rachel 跟我抱怨市区太吵，她又搬回郊区了。",
                    "timestamp": 2000,
                },
                {
                    "role": "assistant",
                    "content": "郊区确实更安静宜居。",
                    "timestamp": 2001,
                },
            ],
        )
    )

    conn = store._connect()

    # 执行一号总结模型：将 F1 状态并入 F2，参数键名对齐架构文档
    merged_count = store.merge_timeline_slots(
        user_id,
        session_id,
        [
            {
                "older_fid": "F1",
                "newer_fid": "F2",
                "latest_state": "搬回郊区（原搬去芝加哥市区）",
            }
        ],
    )
    assert merged_count == 1, f"应成功合并 1 对槽位，实际为: {merged_count}"

    # 验证活跃状态栏：属于 Rachel 且 tombstoned=0 的槽位仅剩 1 条，且 slot_id 为 F2
    active_timelines = conn.execute(
        "SELECT * FROM tag_timelines WHERE user_id = ? AND subject = 'Rachel' AND tombstoned = 0",
        (user_id,),
    ).fetchall()
    assert len(active_timelines) == 1, f"活跃状态槽应只剩 1 条，实际为: {len(active_timelines)}"
    surviving = active_timelines[0]
    assert surviving["slot_id"] == "F2", (
        f"存活槽位必须为 newer_fid F2，实际为: {surviving['slot_id']}"
    )
    assert surviving["current_state"] == "搬回郊区（原搬去芝加哥市区）"

    sup_seqs = json.loads(str(surviving["superseded_seqs"] or "[]"))
    assert 1 in sup_seqs, f"superseded_seqs 必须包含旧事实 seq=1，实际为: {sup_seqs}"
    history_chain = str(surviving["history_chain"] or "")
    assert "搬去芝加哥市区" in history_chain and "搬回郊区" in history_chain

    # 验证旧状态卡 tl:{user_id}:F1 tombstoned == 1
    old_tl_chunk = conn.execute(
        "SELECT tombstoned FROM search_chunks WHERE chunk_id = ?",
        (f"tl:{user_id}:F1",),
    ).fetchone()
    assert old_tl_chunk is not None, "旧状态块 tl:{user_id}:F1 应该存在"
    assert int(old_tl_chunk["tombstoned"]) == 1, "旧状态块必须被标为 tombstoned == 1"

    # INV-1 铁律验证：旧事实块 fact:{user_id}:F1 tombstoned == 0 (绝不随状态演进而销毁历史事实！)
    old_fact_chunk = conn.execute(
        "SELECT tombstoned FROM search_chunks WHERE chunk_id = ?",
        (f"fact:{user_id}:F1",),
    ).fetchone()
    assert old_fact_chunk is not None, "旧事实块 fact:{user_id}:F1 必须存在"
    assert int(old_fact_chunk["tombstoned"]) == 0, (
        "INV-1 铁律违反：更新不等于销毁，历史事实块 fact:{user_id}:F1 的 tombstoned 必须保持为 0！"
    )

    # 检索历史对话（包含芝加哥市区）时，
    # TurnPairCard 依然能实时渲染出 Rachel 的头部提醒与底部事实注脚
    res = await pipeline.search(
        SearchRequest(
            user_id=user_id,
            query="Rachel 芝加哥市区 生活方便",
            top_k=5,
        )
    )
    card_seq1 = next(
        (c for c in res.data if not c.id.startswith("tl:") and "芝加哥市区" in c.content), None
    )
    assert card_seq1 is not None, "历史对话必须能够被检索召回"
    assert "【注意：本段包含非User本人（系Rachel）的信息，请勿与User本人混淆】" in card_seq1.content
    assert "系Rachel" in card_seq1.content


@pytest.mark.asyncio
async def test_summarizer2_person_map_merge_and_slot_reassignment(db_path: Path):
    """4. 二号模型跨 Key 合并、零主键冲突与误标移交契约。

    - 场景 A（同名 tag 下的跨 Key 合并零主键冲突）：
      - 先写入 subject="my sister", tag="居住/城市" (F1: "住在波士顿")，
        再写入 subject="Priya", tag="居住/城市" (F2: "搬到剑桥区")；
      - 调用 store.apply_person_map_updates(...,
          merge_persons=[{"from": "my sister", "to": "Priya",
                          "aliases": ["my sister Priya"], "relations": ["sister"]}])；
      - 验证：不会抛出 IntegrityError，F1 和 F2 的 subject 均变为 "Priya"，
        person_profiles 中 Priya 记录更新且移除 "my sister"；原对话 #1 实时显示 系Priya！
    - 场景 B（误标槽位移交 move_slots）：
      - F3 原被误标为 subject="user", tag="社交/人际", fact="朋友 Rachel 搬回郊区" (seq=3)；
      - 调用 apply_person_map_updates(...,
          move_slots=[{"fid": "F3", "to": "Rachel",
                       "aliases": ["Friend Rachel"], "relations": ["friend"]}])；
      - 验证：F3 的 subject 变为 "Rachel"、source_tier 变为 "other_person"，
        检索原对话卡时实时渲染出 系Rachel！
    - 场景 C（匿名泛称多候选防乱合保护）：
      - 已存在两个 relations 包含 "friend" 的具名人物 "Rachel" 和 "Sarah" 时，
        若传入将纯泛称 "a friend" 合并到 "Rachel" 且无状态内容重叠，
        防护逻辑拒绝盲目合并，保持 "a friend" 独立。
    """
    user_id = "user_p2_test"
    session_id = "sess_p2_01"

    extractor = FakeExtractor(
        responses=[
            # Round 1 (seq 1): my sister 住在波士顿
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "my sister",
                        "tag": "居住/城市",
                        "fact": "my sister 住在波士顿",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 2 (seq 2): Priya 搬到剑桥区
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "Priya",
                        "tag": "居住/城市",
                        "fact": "Priya 搬到剑桥区",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 3 (seq 3): 误标槽位 F3 (原本是朋友 Rachel 搬回郊区，误标成 who="user")
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "user",
                        "tag": "社交/人际",
                        "fact": "朋友 Rachel 搬回郊区",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 4 (seq 4): 具名人物 Sarah (friend)
            {
                "facts": [
                    {
                        "seq": 4,
                        "who": "Sarah",
                        "relations": ["friend"],
                        "tag": "社交/人际",
                        "fact": "Sarah 送了一盆绿植",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 5 (seq 5): 纯匿名泛称 a friend
            {
                "facts": [
                    {
                        "seq": 5,
                        "who": "a friend",
                        "relations": ["friend"],
                        "tag": "社交/礼物",
                        "fact": "a friend 送了一盒手工巧克力",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "波士顿": 0.90,
            "Rachel": 0.92,
        }
    )
    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        reranker=reranker,
    )

    # 写入 Round 1 & Round 2
    await pipeline.add(
        AddRequest(
            request_id="req-p2-01",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "我姐姐 my sister 住在波士顿。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-p2-02",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "Priya 搬到剑桥区了。"}],
        )
    )

    # --- 场景 A: 同名 tag 下的跨 Key 合并零主键冲突 ---
    assert hasattr(
        store, "apply_person_map_updates"
    ), "store 必须提供 apply_person_map_updates 方法"

    store.apply_person_map_updates(
        user_id=user_id,
        session_id=session_id,
        merge_persons=[
            {
                "from": "my sister",
                "to": "Priya",
                "aliases": ["my sister Priya"],
                "relations": ["sister"],
            }
        ],
        move_slots=[],
        profile_updates=[],
    )

    conn = store._connect()
    # 验证两槽位的 subject 均变为 Priya，且 tag 仍为干净的 "居住/城市"
    f1_row = conn.execute(
        "SELECT subject, tag FROM tag_timelines WHERE user_id = ? AND slot_id = 'F1'",
        (user_id,),
    ).fetchone()
    f2_row = conn.execute(
        "SELECT subject, tag FROM tag_timelines WHERE user_id = ? AND slot_id = 'F2'",
        (user_id,),
    ).fetchone()
    assert f1_row is not None and f1_row["subject"] == "Priya"
    assert f2_row is not None and f2_row["subject"] == "Priya"
    assert f1_row["tag"] == "居住/城市"
    assert f2_row["tag"] == "居住/城市"

    # 验证 person_profiles 表中 Priya 的 profiles
    priya_prof = conn.execute(
        "SELECT aliases, relations FROM person_profiles WHERE user_id = ? AND person = 'Priya'",
        (user_id,),
    ).fetchone()
    assert priya_prof is not None, "person_profiles 中必须存在 Priya 记录"
    aliases = json.loads(str(priya_prof["aliases"] or "[]"))
    relations = json.loads(str(priya_prof["relations"] or "[]"))
    assert "my sister" in aliases
    assert "my sister Priya" in aliases
    assert "sister" in relations

    # 验证 "my sister" 已从 person_profiles 移除
    old_prof = conn.execute(
        "SELECT * FROM person_profiles WHERE user_id = ? AND person = 'my sister'",
        (user_id,),
    ).fetchone()
    assert old_prof is None, "合并完成后旧人名 my sister 必须从 person_profiles 中移除"

    # 原对话 #1 (seq=1) 组装卡片时实时动态显示 系Priya
    res_s1 = await pipeline.search(
        SearchRequest(user_id=user_id, query="my sister 波士顿", top_k=5)
    )
    s1_card = next(
        (c for c in res_s1.data if not c.id.startswith("tl:") and "波士顿" in c.content), None
    )
    assert s1_card is not None
    assert "系Priya" in s1_card.content

    # --- 场景 B: 误标槽位移交 move_slots ---
    await pipeline.add(
        AddRequest(
            request_id="req-p2-03",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "朋友 Rachel 搬回郊区。"}],
        )
    )

    # 移交 F3 给 Rachel
    store.apply_person_map_updates(
        user_id=user_id,
        session_id=session_id,
        merge_persons=[],
        move_slots=[
            {
                "fid": "F3",
                "to": "Rachel",
                "aliases": ["Friend Rachel"],
                "relations": ["friend"],
            }
        ],
        profile_updates=[],
    )

    f3_tl = conn.execute(
        "SELECT subject FROM tag_timelines WHERE user_id = ? AND slot_id = 'F3'",
        (user_id,),
    ).fetchone()
    assert f3_tl is not None and f3_tl["subject"] == "Rachel"

    # 严格校验：slot_id=F3 的 fact 和 tl 两块 search_chunks
    # 的 subject 和 source_tier 均更新为 Rachel / other_person
    f3_chunks = conn.execute(
        "SELECT chunk_id, kind, subject, source_tier FROM search_chunks "
        "WHERE user_id = ? AND slot_id = 'F3' ORDER BY kind ASC",
        (user_id,),
    ).fetchall()
    assert len(f3_chunks) == 2, (
        f"slot_id=F3 应对应 fact 和 tl 两块 search_chunks，实际为 {len(f3_chunks)}"
    )
    for ch in f3_chunks:
        assert ch["subject"] == "Rachel", f"chunk {ch['chunk_id']} 的 subject 应更新为 Rachel"
        assert ch["source_tier"] == "other_person", (
            f"chunk {ch['chunk_id']} 的 source_tier 应更新为 other_person"
        )

    # 检索原对话卡：头部和底部自动渲染出 Rachel 归属（无需改写 parent_messages.fact_notes）
    res_s10 = await pipeline.search(
        SearchRequest(user_id=user_id, query="Rachel 搬回郊区", top_k=5)
    )
    s10_card = next(
        (c for c in res_s10.data if not c.id.startswith("tl:") and "搬回郊区" in c.content), None
    )
    assert s10_card is not None
    assert "【注意：本段包含非User本人（系Rachel）的信息，请勿与User本人混淆】" in s10_card.content
    assert "系Rachel" in s10_card.content

    # --- 场景 C: 匿名泛称多候选防乱合保护 ---
    # 写入具名 friend Sarah
    await pipeline.add(
        AddRequest(
            request_id="req-p2-04",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "Sarah 送了我一盆绿植。"}],
        )
    )
    # 写入纯匿名泛称 a friend
    await pipeline.add(
        AddRequest(
            request_id="req-p2-05",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "a friend 送了我一盒手工巧克力。"}],
        )
    )

    # 试图把纯泛称 "a friend" 盲目合并到 "Rachel"
    store.apply_person_map_updates(
        user_id=user_id,
        session_id=session_id,
        merge_persons=[
            {
                "from": "a friend",
                "to": "Rachel",
            }
        ],
        move_slots=[],
        profile_updates=[],
    )

    # 验证防乱合机制介入：存在多个候选 friend (Rachel, Sarah) 且无重叠证据时，保持 "a friend" 独立
    a_friend_tl = conn.execute(
        "SELECT subject FROM tag_timelines WHERE user_id = ? AND subject = 'a friend'",
        (user_id,),
    ).fetchall()
    assert len(a_friend_tl) > 0, "防乱合机制应拒绝将纯匿名泛称 a friend 盲目并入特定具名人物"


@pytest.mark.asyncio
async def test_other_person_timeline_word_boundary_gate(db_path: Path):
    """5. 独立他人状态卡检索词边界门控契约。

    - 验证 _is_subj_in_query(query, person_key, aliases, relations) 及排序门控：
      - Query 为 "What city did my sister move to?" 或 "What is Col. Sen doing?" 时，
        通过 relations=["sister"] / aliases=["Col. Sen"]
        精准命中 "Priya" / "Lt. Col. Raghav Sen"（不降权）；
      - Query 为 "Wait a moment please" 或 "He is a senior engineer" 时，
        绝不误命中 "Mom" 或 "Col. Sen"（因为 "moment" != \bmom\b, "senior" != \bsen\b）；
    - 端到端检索排序：通过 pipeline.add 正常写入 Mom 的状态卡（带向量与 BM25）；
      - Query A: "Wait a moment please 体检" -> 成功召回 Mom 的状态卡，
        但因未匹配 \bmom\b 被 ×0.35 降权；
      - Query B: "What did my mother say about 体检?" -> 命中 relations=["mother"] 未被降权，
        且得分显著高于 Query A。
    """
    user_id = "user_wb_test"
    session_id = "sess_wb_01"

    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "Mom",
                        "aliases": ["my mother"],
                        "relations": ["mother"],
                        "tag": "生活/健康",
                        "fact": "妈妈准备下周去体检",
                        "current_state": "准备下周去体检",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            }
        ]
    )

    reranker = FakeReranker(keyword_scores={"体检": 0.85})
    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        reranker=reranker,
    )

    # 1. 词边界门控函数契约测试
    hit_fn = (
        getattr(pipeline_mod, "_is_subj_in_query", None)
        or getattr(pipeline, "_is_subj_in_query", None)
        or getattr(pipeline, "is_subject_hit_in_query", None)
    )
    assert hit_fn is not None, "pipeline 模块或类中必须提供 _is_subj_in_query 词边界判定函数"

    # 验证 Query: "What city did my sister move to?" 命中 Priya (relations=["sister"])
    assert (
        hit_fn("What city did my sister move to?", "Priya", aliases=[], relations=["sister"])
        is True
    )

    # 验证 Query: "What is Col. Sen doing?" 命中 Lt. Col. Raghav Sen (aliases=["Col. Sen"])
    assert (
        hit_fn(
            "What is Col. Sen doing?",
            "Lt. Col. Raghav Sen",
            aliases=["Col. Sen", "Raghav Sen"],
            relations=["case_officer"],
        )
        is True
    )

    # 验证 Query: "Wait a moment please" 绝不误命中 Mom（\bmom\b 词边界保护）
    assert (
        hit_fn(
            "Wait a moment please",
            "Mom",
            aliases=["my mother"],
            relations=["mother"],
        )
        is False
    )

    # 验证 Query: "He is a senior engineer" 绝不误命中 Col. Sen（\bsen\b 词边界保护）
    assert (
        hit_fn(
            "He is a senior engineer",
            "Lt. Col. Raghav Sen",
            aliases=["Col. Sen", "Sen"],
            relations=["case_officer"],
        )
        is False
    )

    # 2. 检索排序端到端门控验证：通过 pipeline.add 正常写入带向量和 BM25 的状态卡
    await pipeline.add(
        AddRequest(
            request_id="req-wb-01",
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": "我妈妈 Mom 跟我说她准备下周去医院体检。"}],
        )
    )

    # Query A: "Wait a moment please 体检" -> 必须召回 Mom 的状态卡，
    # 但未匹配 \bmom\b 被乘以 0.35 降权
    res_a = await pipeline.search(
        SearchRequest(user_id=user_id, query="Wait a moment please 体检", top_k=5)
    )
    mom_cand_a = next(
        (c for c in res_a.data if c.id.startswith("tl:") and "体检" in c.content), None
    )
    assert mom_cand_a is not None, "包含'体检'的 Query A 必须成功召回 Mom 的状态卡"
    assert mom_cand_a.score <= 0.45, (
        f"未命中的他人状态卡必须被乘 0.35 降权，实际得分: {mom_cand_a.score}"
    )

    # Query B: "What did my mother say about 体检?" -> 命中 relations=["mother"]，不降权
    res_b = await pipeline.search(
        SearchRequest(user_id=user_id, query="What did my mother say about 体检?", top_k=5)
    )
    mom_cand_b = next(
        (c for c in res_b.data if c.id.startswith("tl:") and "体检" in c.content), None
    )
    assert mom_cand_b is not None, "包含'体检'的 Query B 必须成功召回 Mom 的状态卡"
    assert mom_cand_b.score >= 0.50, (
        f"命中的他人状态卡不应被乘 0.35 降权，实际得分: {mom_cand_b.score}"
    )
    assert mom_cand_b.score > mom_cand_a.score * 1.5, (
        f"命中的状态卡得分 ({mom_cand_b.score}) 应显著高于降权后的得分 ({mom_cand_a.score})"
    )


@pytest.mark.asyncio
async def test_extractor_three_model_prompts_and_parsers():
    """6. 三模型极简协议解析与二号模型接口契约。

    - 验证 DashScopeExtractor 支持解析新极简字段
      {seq, who, aliases, relations, tag, fact, is_fiction}；
    - 验证调用 sanitize_extraction_result 归一化后的 ExtractedFact 对象具有正确的字段；
    - 验证向下兼容旧协议字段名；
    - 验证提供二号总结模型方法 reconcile_person_map(person_map_snapshot)。
    """
    extractor = DashScopeExtractor(api_key="test_key")

    # 1. 验证新极简协议解析：{seq, who, aliases, relations, tag, fact, is_fiction}
    new_protocol_json = {
        "facts": [
            {
                "seq": 28,
                "who": "Priya",
                "aliases": ["Priya Sharma"],
                "relations": ["colleague"],
                "tag": "家庭与节日",
                "fact": "在老家和家人一起过排灯节",
                "is_fiction": False,
            },
            {
                "seq": 29,
                "who": "user",
                "aliases": [],
                "relations": [],
                "tag": "工作/汇报",
                "fact": "完成了 Q3 季度的业绩复盘报告",
                "is_fiction": False,
            },
            {
                "seq": 30,
                "who": "user",
                "tag": "创作/小说草稿",
                "fact": "主角驾驶星际飞船跃迁进入暗物质星云",
                "is_fiction": True,
            },
        ],
        "forget_actions": [],
    }

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(new_protocol_json, ensure_ascii=False)}}
                ]
            }

    class MockClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            return MockResponse()

        async def aclose(self):
            pass

    extractor._client = MockClient()

    messages = [
        {"seq": 28, "role": "user", "speaker": "user", "content": "Priya 在老家过节"},
        {"seq": 29, "role": "user", "speaker": "user", "content": "完成 Q3 复盘"},
        {"seq": 30, "role": "user", "speaker": "user", "content": "写了飞船小说草稿"},
    ]
    raw_parsed = await extractor.extract(messages=messages)
    facts, _ = sanitize_extraction_result(raw_parsed, batch_messages=messages)

    assert len(facts) == 3, f"应成功解析并清洗出 3 条事实，实际为: {len(facts)}"
    f1, f2, f3 = facts[0], facts[1], facts[2]

    # 验证第 1 条 (Priya: other_person)
    assert f1.subject == "Priya"
    assert getattr(f1, "aliases", []) == ["Priya Sharma"]
    assert getattr(f1, "relations", []) == ["colleague"]
    assert f1.tag == "家庭与节日"
    assert f1.fact_text == "在老家和家人一起过排灯节"
    assert f1.is_other_person is True
    assert f1.source_tier == "other_person"

    # 验证第 2 条 (User: user_direct 现实汇报)
    assert f2.subject == "user"
    assert f2.is_other_person is False
    assert f2.source_tier == "user_direct"

    # 验证第 3 条 (User: user_draft_fiction 虚构创作)
    assert f3.subject == "user"
    assert f3.source_tier == "user_draft_fiction"

    # 2. 验证向下兼容旧协议字段名 (v2.3 living facts / ops 格式)
    old_protocol_json = {
        "facts": [
            {
                "source_seqs": [5],
                "subject": "user",
                "is_other_person": False,
                "tag": "爱好/水彩",
                "fact_text": "每周六在画室练习水彩画",
                "current_state": "每周六练习水彩画",
            }
        ]
    }

    class MockOldClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": json.dumps(old_protocol_json)}}]}

            return Resp()

        async def aclose(self):
            pass

    extractor._client = MockOldClient()
    messages_old = [{"seq": 5, "role": "user", "speaker": "user", "content": "练习水彩画"}]
    raw_parsed_old = await extractor.extract(messages=messages_old)
    facts_old, _ = sanitize_extraction_result(raw_parsed_old, batch_messages=messages_old)
    assert len(facts_old) == 1
    assert facts_old[0].tag == "爱好/水彩"

    # 3. 验证二号总结模型接口：reconcile_person_map
    assert hasattr(
        extractor, "reconcile_person_map"
    ), "DashScopeExtractor 必须提供 reconcile_person_map 方法"

    p2_resp_json = {
        "merge_persons": [
            {
                "from": "my sister",
                "to": "Priya",
                "aliases": ["my sister Priya"],
                "relations": ["sister"],
            }
        ],
        "move_slots": [
            {
                "fid": "F4",
                "to": "Rachel",
                "aliases": ["Friend Rachel"],
                "relations": ["friend"],
            }
        ],
        "profile_updates": [
            {
                "person": "Lt. Col. Raghav Sen",
                "aliases": ["Raghav Sen", "Col. Sen", "Raghav"],
                "relations": ["case_officer"],
            }
        ],
    }

    class MockP2Client:
        is_closed = False

        async def post(self, *args, **kwargs):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": json.dumps(p2_resp_json)}}]}

            return Resp()

        async def aclose(self):
            pass

    extractor._client = MockP2Client()
    snapshot = [
        {"person": "my sister", "slots": [{"fid": "F1", "state": "住在波士顿"}]},
        {"person": "Priya", "slots": [{"fid": "F2", "state": "搬到剑桥区"}]},
    ]
    p2_result = await extractor.reconcile_person_map(snapshot)
    assert isinstance(p2_result, dict)
    assert len(p2_result.get("merge_persons", [])) == 1
    assert p2_result["merge_persons"][0]["from"] == "my sister"
    assert len(p2_result.get("move_slots", [])) == 1
    assert p2_result["move_slots"][0]["fid"] == "F4"
    assert len(p2_result.get("profile_updates", [])) == 1


def test_state_summary_prompt_fusion_and_implicit_refresh_contract():
    """验证一号总结模型 Prompt 中包含隐式刷新与状态融合的指令规则。

    针对 docs/DESIGN.md 6.3 改动点 1：
    1. 包含「隐式刷新/隐式状态刷新（如打破 PB / 数量增加 / 额度变更）」识别规则；
    2. 包含「前后状态完整融合（禁止只照抄半截、需保留旧槽位主语与旧值演进）」指令。
    """
    from dandelion_memory.extractor import _STATE_SUMMARY_PROMPT

    prompt = _STATE_SUMMARY_PROMPT

    # 1. 隐式刷新识别规则
    assert (
        "隐式刷新" in prompt or "隐式状态刷新" in prompt
    ), "_STATE_SUMMARY_PROMPT 必须明确包含「隐式刷新」或「隐式状态刷新」指令规则"

    # 包含打破 PB、数量增加、额度变更等具体场景指引
    has_pb_metric = "PB" in prompt or "5K" in prompt
    has_amount_or_count_metric = any(kw in prompt for kw in ("额度", "预算", "数量", "bikes"))
    assert has_pb_metric and has_amount_or_count_metric, (
        "_STATE_SUMMARY_PROMPT 必须包含打破 PB 与额度/数量变更等隐式状态刷新场景指引"
    )

    # 2. 前后状态完整融合指令（禁止只照抄半截、需保留旧槽位主语与旧值演进）
    assert "融合" in prompt, "_STATE_SUMMARY_PROMPT 必须明确要求前后状态完整「融合」"
    has_prohibit_half = any(
        kw in prompt
        for kw in ("禁止只照抄", "严禁只照抄", "严禁只复制", "禁止半截", "严禁复制片断", "半截")
    )
    assert has_prohibit_half, "_STATE_SUMMARY_PROMPT 必须明确禁止只照抄/复制 newer_fid 的半截片断"

    has_subject_preservation = "主语" in prompt or "背景" in prompt
    has_evolution_preservation = any(kw in prompt for kw in ("旧值", "演进", "旧基准", "更新前"))
    assert has_subject_preservation and has_evolution_preservation, (
        "_STATE_SUMMARY_PROMPT 必须要求保留旧槽位完整背景/主语，并注明旧值演进"
    )


@pytest.mark.asyncio
async def test_merge_timeline_slots_syncs_surviving_fact_item_and_preserves_inv1(
    db_path: Path,
):
    """验证 merge_timeline_slots 合槽后同步存活 fact_item 且严守 INV-1。

    针对 docs/DESIGN.md 6.3 改动点 2：
    ① 存活槽位 F2 对应的 search_chunks (kind='fact_item', slot_id='F2') 的 display_text 和
       search_text 已同步更新为融合后的 latest_state；
    ② 若合并的是非 User 人物（如 Priya 的 F3 -> F4），存活 fact_item 更新后仍保留
       【非User本人，系Priya的信息】前缀与 [Priya | ...] 的 search_text 格式；
    ③ INV-1 严守：被合并的旧槽位 F1/F3 的 timeline_summary 被标为 tombstoned = 1，
       但历史 fact_item 保持 tombstoned = 0，且 parent_messages.content 未被篡改。
    """
    user_id = "test_user_merge_sync"
    session_id = "sess_merge_sync_01"

    extractor = FakeExtractor(
        responses=[
            # Round 1 (seq 1): User 5K PB 27:12 -> 生成 F1
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "兴趣/运动与休闲活动",
                        "fact": "PB in charity 5K run is 27:12",
                        "current_state": "PB in charity 5K run is 27:12",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 2 (seq 3): User training to beat 25:50 -> 生成 F2
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "user",
                        "tag": "兴趣/运动与休闲活动",
                        "fact": "Training to beat 25:50",
                        "current_state": "Training to beat 25:50",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 3 (seq 5): Priya 租房 -> 生成 F3
            {
                "facts": [
                    {
                        "seq": 5,
                        "who": "Priya",
                        "aliases": ["Priya"],
                        "relations": ["colleague"],
                        "tag": "居住/城市与住房",
                        "fact": "Priya rents a 1B apartment in downtown",
                        "current_state": "Rents a 1B apartment in downtown",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 4 (seq 7): Priya 买房 -> 生成 F4
            {
                "facts": [
                    {
                        "seq": 7,
                        "who": "Priya",
                        "aliases": ["Priya"],
                        "relations": ["colleague"],
                        "tag": "居住/城市与住房",
                        "fact": "Priya bought a condo in Cambridge",
                        "current_state": "Bought a condo in Cambridge",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
    )

    content_r1 = "我参加了慈善 5K 跑步，最好成绩是 27:12。"
    content_r2 = "我目前正在加紧训练，目标是打破 25:50 的成绩。"
    content_r3 = "Priya 跟我说她在市中心租了一套一居室公寓。"
    content_r4 = "Priya 刚在剑桥区买了一套公寓并搬了进去。"

    # 依次写入 4 轮问答对（seq 1..8）
    await pipeline.add(
        AddRequest(
            request_id="req-msync-01",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {"role": "user", "content": content_r1, "timestamp": 1000},
                {"role": "assistant", "content": "祝贺你的 5K 成绩！", "timestamp": 1001},
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-msync-02",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {"role": "user", "content": content_r2, "timestamp": 2000},
                {"role": "assistant", "content": "加油，祝训练突破目标！", "timestamp": 2001},
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-msync-03",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {"role": "user", "content": content_r3, "timestamp": 3000},
                {"role": "assistant", "content": "收到 Priya 租房的近况。", "timestamp": 3001},
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-msync-04",
            user_id=user_id,
            session_id=session_id,
            messages=[
                {"role": "user", "content": content_r4, "timestamp": 4000},
                {"role": "assistant", "content": "恭喜 Priya 在剑桥置业！", "timestamp": 4001},
            ],
        )
    )

    conn = store._connect()

    # 验证初始槽位 F1, F2, F3, F4 均已生成
    init_facts = conn.execute(
        "SELECT slot_id, kind, display_text FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' ORDER BY slot_id",
        (user_id,),
    ).fetchall()
    assert {r["slot_id"] for r in init_facts} == {"F1", "F2", "F3", "F4"}

    # 1. 合并 User 槽位 F1 -> F2
    f2_fused_state = (
        "Current personal best time in the charity 5K run is 25:50 "
        "(updated from previous PB of 27:12)"
    )
    merged_user = store.merge_timeline_slots(
        user_id=user_id,
        session_id=session_id,
        merges=[
            {
                "older_fid": "F1",
                "newer_fid": "F2",
                "latest_state": f2_fused_state,
            }
        ],
    )
    assert merged_user == 1

    # 2. 合并非 User 人物 Priya 槽位 F3 -> F4
    f4_fused_state = (
        "Moved and bought a condo in Cambridge "
        "(updated from renting a 1B apartment in downtown)"
    )
    merged_priya = store.merge_timeline_slots(
        user_id=user_id,
        session_id=session_id,
        merges=[
            {
                "older_fid": "F3",
                "newer_fid": "F4",
                "latest_state": f4_fused_state,
            }
        ],
    )
    assert merged_priya == 1

    # 断言 ①：存活槽位 F2 对应的 search_chunks (kind='fact_item', slot_id='F2') 已同步更新
    f2_chunk = conn.execute(
        "SELECT display_text, search_text, tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F2'",
        (user_id,),
    ).fetchone()
    assert f2_chunk is not None, "存活槽位 F2 的 fact_item 必须存在"
    assert int(f2_chunk["tombstoned"]) == 0
    assert f2_chunk["display_text"] == f2_fused_state, (
        f"存活 fact_item F2 display_text 必须同步更新为 latest_state，"
        f"实际为: {f2_chunk['display_text']}"
    )
    assert f2_fused_state in f2_chunk["search_text"], (
        f"存活 fact_item F2 search_text 必须包含 latest_state，实际为: {f2_chunk['search_text']}"
    )

    # 断言 ②：非 User 人物 Priya 存活槽位 F4 保留 【非User本人，系Priya的信息】 前缀与检索格式
    f4_chunk = conn.execute(
        "SELECT display_text, search_text, tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F4'",
        (user_id,),
    ).fetchone()
    assert f4_chunk is not None, "存活槽位 F4 的 fact_item 必须存在"
    assert int(f4_chunk["tombstoned"]) == 0
    assert "【非User本人，系Priya的信息】" in f4_chunk["display_text"], (
        "非 User 存活 fact_item 必须保留 【非User本人，系Priya的信息】 前缀: "
        f"{f4_chunk['display_text']}"
    )
    assert f4_fused_state in f4_chunk["display_text"]
    assert "[Priya | 居住/城市与住房]" in f4_chunk["search_text"]
    assert f4_fused_state in f4_chunk["search_text"]

    # 断言 ③：INV-1 严守：被合并的旧槽位 F1 和 F3
    # A. timeline_summary 标为 tombstoned = 1
    old_tl1 = conn.execute(
        "SELECT tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'timeline_summary' AND slot_id = 'F1'",
        (user_id,),
    ).fetchone()
    old_tl3 = conn.execute(
        "SELECT tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'timeline_summary' AND slot_id = 'F3'",
        (user_id,),
    ).fetchone()
    assert old_tl1 and int(old_tl1["tombstoned"]) == 1, (
        "旧槽位 F1 的 timeline_summary 必须 tombstoned=1"
    )
    assert old_tl3 and int(old_tl3["tombstoned"]) == 1, (
        "旧槽位 F3 的 timeline_summary 必须 tombstoned=1"
    )

    # B. 历史 fact_item 保持 tombstoned = 0，且保留历史原貌
    old_fact1 = conn.execute(
        "SELECT display_text, tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F1'",
        (user_id,),
    ).fetchone()
    old_fact3 = conn.execute(
        "SELECT display_text, tombstoned FROM search_chunks "
        "WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F3'",
        (user_id,),
    ).fetchone()
    assert old_fact1 and int(old_fact1["tombstoned"]) == 0, (
        "INV-1 违背：被合并的旧槽位 F1 历史 fact_item 必须保持 tombstoned=0"
    )
    assert old_fact3 and int(old_fact3["tombstoned"]) == 0, (
        "INV-1 违背：被合并的旧槽位 F3 历史 fact_item 必须保持 tombstoned=0"
    )
    assert "27:12" in old_fact1["display_text"]
    assert "downtown" in old_fact3["display_text"]

    # C. parent_messages.content 未被篡改
    p_msgs = conn.execute(
        "SELECT content FROM parent_messages WHERE user_id = ? AND role = 'user' ORDER BY seq ASC",
        (user_id,),
    ).fetchall()
    p_contents = [r["content"] for r in p_msgs]
    assert content_r1 in p_contents[0], "parent_messages.content (Round 1) 严禁被篡改"
    assert content_r2 in p_contents[1], "parent_messages.content (Round 2) 严禁被篡改"
    assert content_r3 in p_contents[2], "parent_messages.content (Round 3) 严禁被篡改"
    assert content_r4 in p_contents[3], "parent_messages.content (Round 4) 严禁被篡改"


@pytest.mark.asyncio
async def test_fine_grained_slot_latest_and_superseded_hint_with_mixed_turn_guard(
    db_path: Path,
):
    """验证检索卡片组装层细粒度 is_latest_in_tag、合槽取代提示与同轮混合事实保护。

    针对 docs/DESIGN.md 6.3 改动点 3：
    ① 细粒度 is_latest_in_tag（同 tag 不同事项互不覆盖）：同一用户先后写入同 tag 下的 F1（跑步 PB）
       与 F2（网球），未合并时检索 F1 对话卡，其 is_latest_in_tag 必须为 True；
    ② 被合槽取代的旧对话卡追加 [状态更新提示]：F1（$350,000）被 #5 的 F3（$400,000）合槽取代后，
       检索出的 #1 对话卡末尾必须包含
       [状态更新提示: 本段提及的旧状态已于后续 #5 更新为: ... $400,000 ...]，
       且因 #1 无其它活跃最新槽位，其 is_latest_in_tag 为 False、is_superseded 为 True；
    ③ 同轮混合事实保护（1-to-N mixed turn guard）：若 #1 同时包含 F1（旧房贷 $350k，后在 #5 被取代）
       和未取代的独立活跃槽位 F2（山地车），
       则 #1 对话卡末尾追加 [状态更新提示]，同时保留 is_latest_in_tag = True。
    """
    # -------------------------------------------------------------
    # ① 细粒度 is_latest_in_tag（同 tag 不同事项互不覆盖）
    # -------------------------------------------------------------
    user1 = "user_fg_tag_latest"
    extractor1 = FakeExtractor(
        responses=[
            # Round 1 (seq 1): 5K PB 27:12 -> F1
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "兴趣/运动与休闲活动",
                        "fact": "PB in charity 5K run is 27:12",
                        "current_state": "PB in charity 5K run is 27:12",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 2 (seq 3): 周末打网球 -> F2
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "user",
                        "tag": "兴趣/运动与休闲活动",
                        "fact": "周末打网球锻炼",
                        "current_state": "周末打网球锻炼",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    reranker1 = FakeReranker(keyword_scores={"5K": 0.95, "网球": 0.90})
    pipeline1, store1, _, _, _ = build_v2_pipeline(
        db_path=db_path.parent / "fg1.sqlite3",
        extractor=extractor1,
        reranker=reranker1,
    )
    await pipeline1.add(
        AddRequest(
            request_id="req-fg-01",
            user_id=user1,
            session_id="s1",
            messages=[
                {"role": "user", "content": "我的慈善 5K 跑步最好成绩 PB 是 27:12。"},
                {"role": "assistant", "content": "这个成绩很不错！保持节奏继续加油。"},
            ],
        )
    )
    await pipeline1.add(
        AddRequest(
            request_id="req-fg-02",
            user_id=user1,
            session_id="s1",
            messages=[
                {"role": "user", "content": "我周末一般去网球场打网球。"},
                {"role": "assistant", "content": "网球对心肺和敏捷度都很有帮助。"},
            ],
        )
    )

    # 挂载钩子捕获检索卡片组装层（_build_turn_pair_cards）产出的候选卡片
    captured_cards1: dict[str, Any] = {}
    orig_build1 = pipeline1._build_turn_pair_cards

    def hook_build1(*args: Any, **kwargs: Any) -> Any:
        d_cands, tl_cands = orig_build1(*args, **kwargs)
        captured_cards1.update(d_cands)
        return d_cands, tl_cands

    pipeline1._build_turn_pair_cards = hook_build1  # type: ignore[method-assign]

    # 检索 5K PB 对应的 #1 对话卡
    res1 = await pipeline1.search(
        SearchRequest(user_id=user1, query="慈善 5K PB 成绩", top_k=5)
    )
    assert len(res1.data) > 0, "必须检索召回卡片"
    card1 = next(
        (
            c
            for c in captured_cards1.values()
            if c.kind == "restored_parent" and "27:12" in c.content
        ),
        None,
    )
    assert card1 is not None, "检索卡片组装层必须生成第 1 轮 5K 对话卡"
    assert card1.metadata.get("is_latest_in_tag") is True, (
        "细粒度槽位契约：F1 与 F2 均为活跃槽位且未发生合槽取代，"
        "#1 对话卡的 is_latest_in_tag 必须保持为 True，严禁被同 tag 的后续网球覆盖为 False！"
    )

    # -------------------------------------------------------------
    # ② 被合槽取代的旧对话卡追加 [状态更新提示] 与 is_superseded
    # -------------------------------------------------------------
    user2 = "user_superseded_single_fact"
    extractor2 = FakeExtractor(
        responses=[
            # Round 1 (seq 1): 房贷预批额度 $350,000 -> F1
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "财务/贷款与预算",
                        "fact": "房贷预批额度为 $350,000",
                        "current_state": "房贷预批额度 $350,000",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
            # Round 2 (seq 3): 闲聊
            {"facts": [], "forget_actions": []},
            # Round 3 (seq 5): 房贷预批额度提高到 $400,000 -> F2
            {
                "facts": [
                    {
                        "seq": 5,
                        "who": "user",
                        "tag": "财务/贷款与预算",
                        "fact": "房贷预批额度提高到 $400,000",
                        "current_state": "房贷预批额度提高到 $400,000",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    reranker2 = FakeReranker(
        keyword_scores={"350,000": 0.95, "400,000": 0.92, "预批": 0.90}
    )
    pipeline2, store2, _, _, _ = build_v2_pipeline(
        db_path=db_path.parent / "fg2.sqlite3",
        extractor=extractor2,
        reranker=reranker2,
    )
    await pipeline2.add(
        AddRequest(
            request_id="req-sup-01",
            user_id=user2,
            session_id="s2",
            messages=[
                {"role": "user", "content": "我拿到了银行的房贷预批额度 $350,000。"},
                {"role": "assistant", "content": "收到，这是很关键的买房预算依据。"},
            ],
        )
    )
    await pipeline2.add(
        AddRequest(
            request_id="req-sup-02",
            user_id=user2,
            session_id="s2",
            messages=[
                {"role": "user", "content": "今天天气不错，下午去散步了。"},
                {"role": "assistant", "content": "散步是个放松心情的好方式。"},
            ],
        )
    )
    await pipeline2.add(
        AddRequest(
            request_id="req-sup-03",
            user_id=user2,
            session_id="s2",
            messages=[
                {
                    "role": "user",
                    "content": "银行重新核准后，把我的房贷预批额度提高到了 $400,000。",
                },
                {"role": "assistant", "content": "恭喜，预算空间更充裕了！"},
            ],
        )
    )

    conn2 = store2._connect()
    slots_u2 = conn2.execute(
        "SELECT slot_id, latest_seq FROM tag_timelines WHERE user_id = ? ORDER BY slot_id ASC",
        (user2,),
    ).fetchall()
    f_older = slots_u2[0]["slot_id"]
    f_newer = slots_u2[1]["slot_id"]

    merged_state_2 = "房贷预批额度为 $400,000（从 $350,000 更新）"
    store2.merge_timeline_slots(
        user_id=user2,
        session_id="s2",
        merges=[
            {
                "older_fid": f_older,
                "newer_fid": f_newer,
                "latest_state": merged_state_2,
            }
        ],
    )

    captured_cards2: dict[str, Any] = {}
    orig_build2 = pipeline2._build_turn_pair_cards

    def hook_build2(*args: Any, **kwargs: Any) -> Any:
        d_cands, tl_cands = orig_build2(*args, **kwargs)
        captured_cards2.update(d_cands)
        return d_cands, tl_cands

    pipeline2._build_turn_pair_cards = hook_build2  # type: ignore[method-assign]

    res2 = await pipeline2.search(
        SearchRequest(user_id=user2, query="350000 房贷 预批额度", top_k=5)
    )
    assert len(res2.data) > 0, "必须检索召回卡片"
    card2 = next(
        (
            c
            for c in captured_cards2.values()
            if c.kind == "restored_parent" and "350,000" in c.content
        ),
        None,
    )
    assert card2 is not None, "检索卡片组装层必须生成第 1 轮 #1 旧额度对话卡"
    assert "[状态更新提示: 本段提及的旧状态已于后续 #5 更新为:" in card2.content, (
        "被取代的旧对话卡末尾必须包含状态更新提示，实际内容为:\n"
        f"{card2.content}"
    )
    assert "$400,000" in card2.content
    assert card2.metadata.get("is_latest_in_tag") is False, (
        "因 #1 中的 F1 槽位已被合槽取代且无其它活跃最新槽位，is_latest_in_tag 必须为 False"
    )
    assert card2.metadata.get("is_superseded") is True, (
        "被取代的旧对话卡 metadata['is_superseded'] 必须为 True"
    )

    # -------------------------------------------------------------
    # ③ 同轮混合事实保护（1-to-N mixed turn guard）
    # -------------------------------------------------------------
    user3 = "user_mixed_turn_guard"
    extractor3 = FakeExtractor(
        responses=[
            # Round 1 (seq 1): 同一轮消息同时包含 F1(旧房贷) 与 F2(山地车)
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "财务/贷款与预算",
                        "fact": "房贷预批额度为 $350,000",
                        "current_state": "房贷预批额度 $350,000",
                        "is_fiction": False,
                    },
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "资产/物品与车辆",
                        "fact": "买了一辆山地自行车",
                        "current_state": "买了一辆山地自行车",
                        "is_fiction": False,
                    },
                ],
                "forget_actions": [],
            },
            # Round 2 (seq 3): 闲聊
            {"facts": [], "forget_actions": []},
            # Round 3 (seq 5): 房贷预批额度提高到 $400,000
            {
                "facts": [
                    {
                        "seq": 5,
                        "who": "user",
                        "tag": "财务/贷款与预算",
                        "fact": "房贷预批额度提高到 $400,000",
                        "current_state": "房贷预批额度提高到 $400,000",
                        "is_fiction": False,
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    reranker3 = FakeReranker(
        keyword_scores={"350,000": 0.95, "山地自行车": 0.93, "400,000": 0.90}
    )
    pipeline3, store3, _, _, _ = build_v2_pipeline(
        db_path=db_path.parent / "fg3.sqlite3",
        extractor=extractor3,
        reranker=reranker3,
    )
    await pipeline3.add(
        AddRequest(
            request_id="req-mix-01",
            user_id=user3,
            session_id="s3",
            messages=[
                {
                    "role": "user",
                    "content": "我确认了房贷预批额度是 $350,000，另外下午还买了一辆山地自行车。",
                },
                {"role": "assistant", "content": "收到，记下了预批额度和买的山地车。"},
            ],
        )
    )
    await pipeline3.add(
        AddRequest(
            request_id="req-mix-02",
            user_id=user3,
            session_id="s3",
            messages=[
                {"role": "user", "content": "周末过得挺充实。"},
                {"role": "assistant", "content": "充实的周末很棒。"},
            ],
        )
    )
    await pipeline3.add(
        AddRequest(
            request_id="req-mix-03",
            user_id=user3,
            session_id="s3",
            messages=[
                {
                    "role": "user",
                    "content": "银行将房贷预批额度提升到了 $400,000。",
                },
                {"role": "assistant", "content": "额度提升到了 40 万美元。"},
            ],
        )
    )

    conn3 = store3._connect()
    slots_u3 = conn3.execute(
        "SELECT slot_id, tag, latest_seq FROM tag_timelines WHERE user_id = ? ORDER BY slot_id ASC",
        (user3,),
    ).fetchall()
    f_mortgage_old = next(
        r["slot_id"]
        for r in slots_u3
        if r["tag"] == "财务/贷款与预算" and r["latest_seq"] == 1
    )
    f_mortgage_new = next(
        r["slot_id"]
        for r in slots_u3
        if r["tag"] == "财务/贷款与预算" and r["latest_seq"] == 5
    )

    # 仅将 F1 合入 F3，F2 保持活跃最新
    store3.merge_timeline_slots(
        user_id=user3,
        session_id="s3",
        merges=[
            {
                "older_fid": f_mortgage_old,
                "newer_fid": f_mortgage_new,
                "latest_state": "房贷预批额度为 $400,000（从 $350,000 更新）",
            }
        ],
    )

    captured_cards3: dict[str, Any] = {}
    orig_build3 = pipeline3._build_turn_pair_cards

    def hook_build3(*args: Any, **kwargs: Any) -> Any:
        d_cands, tl_cands = orig_build3(*args, **kwargs)
        captured_cards3.update(d_cands)
        return d_cands, tl_cands

    pipeline3._build_turn_pair_cards = hook_build3  # type: ignore[method-assign]

    # 检索第 1 轮 #1 对话卡
    res3 = await pipeline3.search(
        SearchRequest(user_id=user3, query="350000 房贷 预批 山地自行车", top_k=5)
    )
    assert len(res3.data) > 0, "必须检索召回卡片"
    card3 = next(
        (
            c
            for c in captured_cards3.values()
            if c.kind == "restored_parent" and "350,000" in c.content
        ),
        None,
    )
    assert card3 is not None, "检索卡片组装层必须生成包含 $350,000 和山地自行车的第 1 轮对话卡"

    # 断言 A：卡片末尾既追加针对 F1 的 [状态更新提示: ...]
    assert "[状态更新提示: 本段提及的旧状态已于后续 #5 更新为:" in card3.content, (
        f"#1 对话卡针对已取代的旧房贷额度必须包含更新提示，实际内容为:\n{card3.content}"
    )
    assert "$400,000" in card3.content

    # 断言 B：同轮混合事实保护（1-to-N mixed turn guard）：
    # 因为同轮中的 F2 (山地自行车) 仍处于活跃最新状态，#1 卡片的 is_latest_in_tag 依然保持为 True！
    assert card3.metadata.get("is_latest_in_tag") is True, (
        "1-to-N 混合事实保护契约：同轮对话中虽有一条事实被合槽取代，"
        "但仍包含另一条未过期的独立活跃最新槽位（山地自行车），"
        "该对话卡的 is_latest_in_tag 必须保留为 True，严禁被整卡标记为已过期！"
    )

