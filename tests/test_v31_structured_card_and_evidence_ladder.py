# 契约：Dandelion Memory v3.1 结构化卡片视图与细粒度证据阶梯测试集（TDD 红灯用例）
# 依据 docs/DESIGN.md 第九节「v3.1 结构化卡片视图与细粒度证据阶梯设计」
# 与第十节「历次迭代踩坑与永久禁回退红线总表」
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

import dandelion_memory.extractor as extractor_mod
from dandelion_memory.extractor import (
    DashScopeExtractor,
    ExtractedFact,
    ExtractedForgetAction,
    sanitize_extraction_result,
)
from dandelion_memory.models import AddRequest, Message, SearchRequest
from dandelion_memory.text import extract_keywords_from_target

from .conftest import FakeExtractor, FakeReranker, build_v2_pipeline


class ExtractionResult(NamedTuple):
    facts: list[ExtractedFact]
    forget_actions: list[ExtractedForgetAction]


@pytest.mark.asyncio
async def test_core1_implied_cue_extraction_and_whitelist_isolation(tmp_path: Path):
    """核心一.1 & 坑 7/9 红线：ExtractedFact 扩展 implied 字段、物理隔离与合槽保留。

    1. ExtractedFact 支持 implied: str = "" 字段；
    2. extractor._parse_v24_payload 解析 JSON 含 implied 字段；
    3. 入库后 search_chunks(kind='fact_item') 的 search_text 包含 implied 内容，
       display_text 包含 [隐式推导 / Implied (非显式确认事实，优先级低于直接事实): <implied>]；
    4. 白名单隔离（坑 9）：store.get_living_facts 返回的 current_state 绝不包含 implied 内容；
    5. Stage 3 合槽保留（坑 7）：merge_timeline_slots 合槽后，存活槽位的 fact_item.display_text
       同步更新最新事实的同时，必须保留原有的 [隐式推导 / Implied ...] 标记，未被裸覆盖冲掉。
    """
    # 1. ExtractedFact schema 扩展检查
    assert hasattr(ExtractedFact, "implied"), "ExtractedFact 必须支持 implied: str = ''"
    sample_fact = ExtractedFact(
        source_seqs=[1],
        subject="user",
        tag="健康/医疗与饮食禁忌",
        fact_text="阅读屏幕时扶了扶鼻梁上的细金属镜框",
        implied="eyeglasses, vision correction",
    )
    assert getattr(sample_fact, "implied", "") == "eyeglasses, vision correction"

    # 2. extractor._parse_v24_payload 解析
    parse_fn = getattr(extractor_mod, "_parse_v24_payload", None)
    if parse_fn is None:
        parse_fn = getattr(DashScopeExtractor, "_parse_v24_payload", None)
        if parse_fn is not None:
            parse_fn = parse_fn.__get__(DashScopeExtractor(api_key="test_key"))
    assert parse_fn is not None, "extractor._parse_v24_payload 必须存在"

    raw_json_item = {
        "seq": 1,
        "subject": "user",
        "tag": "健康/医疗与饮食禁忌",
        "fact": "阅读屏幕时扶了扶鼻梁上的细金属镜框",
        "implied": "eyeglasses, vision correction",
    }
    parsed = parse_fn(raw_json_item)
    fact = parsed[0] if isinstance(parsed, list) else parsed
    assert getattr(fact, "implied", None) == "eyeglasses, vision correction"

    db_path = tmp_path / "test_core1_implied.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "健康/医疗与饮食禁忌",
                        "fact": "阅读屏幕时扶了扶鼻梁上的细金属镜框",
                        "current_state": "阅读屏幕时扶了扶鼻梁上的细金属镜框",
                        "implied": "eyeglasses, vision correction",
                    }
                ],
                "forget_actions": [],
            },
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "user",
                        "tag": "健康/医疗与饮食禁忌",
                        "fact": "去眼科验光确定为175度近视散光",
                        "current_state": "175度近视散光配防疲劳镜",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    pipeline, store, embedder, _, reranker = build_v2_pipeline(
        db_path=db_path,
        extractor=fake_ext,
    )
    try:
        user_id = "u_core1_implied"
        session_id = "s_core1_01"

        # Batch 1: Ingest explicit fact with implied cue
        await pipeline.add(
            AddRequest(
                request_id="req-core1-01",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(
                        role="user",
                        content="阅读屏幕时扶了扶鼻梁上的细金属镜框，屏幕太晃眼了。",
                    )
                ],
            )
        )

        conn = store._connect()
        # 3. 验证 search_chunks 中的 fact_item
        f_row = conn.execute(
            "SELECT * FROM search_chunks WHERE user_id = ? AND kind = 'fact_item'",
            (user_id,),
        ).fetchone()
        assert f_row is not None, "fact_item 必须写入 search_chunks"
        assert "eyeglasses, vision correction" in str(f_row["search_text"]), (
            "fact_item.search_text 必须拼入 implied 词以供向量/BM25 检索命中"
        )
        expected_implied_tag = (
            "[隐式推导 / Implied: eyeglasses, vision correction]"
        )
        assert expected_implied_tag in str(f_row["display_text"]), (
            f"fact_item.display_text 必须以显式标签格式存储 implied，"
            f"实际为: {f_row['display_text']}"
        )

        # 4. 白名单隔离断言（坑 9）
        living_facts = store.get_living_facts(user_id)
        assert len(living_facts) >= 1
        for lf in living_facts:
            cs = str(lf.get("current_state") or "")
            assert "eyeglasses" not in cs, (
                f"坑 9 红线：implied 绝不能进入 tag_timelines.current_state 状态表，实际为: {cs}"
            )
            assert "Implied" not in cs, (
                f"坑 9 红线：Implied 标签绝不能污染 current_state，实际为: {cs}"
            )

        # Batch 2: Ingest second fact to create slot F2 under same tag
        await pipeline.add(
            AddRequest(
                request_id="req-core1-02",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(
                        role="user",
                        content="今天去眼科验光确定为175度近视散光，配了防疲劳镜。",
                    )
                ],
            )
        )

        slots = conn.execute(
            "SELECT slot_id, current_state FROM tag_timelines "
            "WHERE user_id = ? AND tombstoned = 0 ORDER BY slot_id",
            (user_id,),
        ).fetchall()
        assert len(slots) >= 2, "应当有两个存活槽位 (F1, F2)"

        # 5. Stage 3 合槽保留断言（坑 7 & 坑 9）：将 F2 合并入 F1（F1 为存活槽位）
        merged_state = "阅读屏幕时配戴175度防疲劳眼镜（原扶鼻梁镜框）"
        store.merge_timeline_slots(
            user_id=user_id,
            session_id=session_id,
            merges=[
                {
                    "older_fid": "F2",
                    "newer_fid": "F1",
                    "latest_state": merged_state,
                }
            ],
        )

        f1_chunk = conn.execute(
            "SELECT display_text, search_text FROM search_chunks "
            "WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F1'",
            (user_id,),
        ).fetchone()
        assert f1_chunk is not None, "F1 fact_item 必须存在"
        assert merged_state in f1_chunk["display_text"], "合槽后 display_text 必须包含最新显式事实"
        assert expected_implied_tag in f1_chunk["display_text"], (
            f"坑 7 & 坑 9 红线：合槽刷新时必须保留原有的 [隐式推导 / Implied ...] 标记，"
            f"不能被裸覆盖冲掉！实际 display_text: {f1_chunk['display_text']}"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core1_person_alias_merge_to_user_cleans_prefix_and_restores_tier(tmp_path: Path):
    """核心一.2：人物别名（The User -> user）归并、剥离前缀与恢复本人直接证据层级。

    - 先入库一条被 Extractor 误标为
      subject="The User", is_other_person=True, source_tier="other_person" 的事实；
    - 验证底表 search_chunks.display_text 与 parent_messages.fact_notes 初始包含
      【非User本人，系The User的信息】；
    - 调用 store.apply_person_map_updates 进行人物别名归并至 "user"；
    - 断言归并后：
      1. 对应 search_chunks 的 subject == "user"、source_tier == "user_direct"；
      2. search_chunks.display_text 中的 【非User本人，系The User的信息】 前缀被彻底剥除；
      3. parent_messages.fact_notes 中的 【非User本人，系The User的信息】 前缀同样被彻底剥除。
    """
    db_path = tmp_path / "test_core1_alias.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "The User",
                        "subject": "The User",
                        "is_other_person": True,
                        "source_tier": "other_person",
                        "tag": "居住/城市",
                        "fact": "在西雅图有一套两居室公寓",
                        "current_state": "在西雅图有一套两居室公寓",
                    }
                ],
                "forget_actions": [],
            }
        ]
    )
    pipeline, store, *_ = build_v2_pipeline(db_path=db_path, extractor=fake_ext)
    try:
        user_id = "u_core1_alias"
        session_id = "s_core1_alias_01"

        await pipeline.add(
            AddRequest(
                request_id="req-core1-alias-01",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(role="user", content="我在西雅图有一套两居室公寓，平时自己住。")
                ],
            )
        )

        conn = store._connect()
        # 验证初始状态：包含第三方前缀
        c_pre = conn.execute(
            "SELECT * FROM search_chunks WHERE user_id = ? AND kind = 'fact_item'",
            (user_id,),
        ).fetchone()
        assert c_pre is not None
        assert "【非User本人，系The User的信息】" in str(c_pre["display_text"])
        assert str(c_pre["subject"]) == "The User"
        assert str(c_pre["source_tier"]) == "other_person"

        p_pre = conn.execute(
            "SELECT fact_notes FROM parent_messages WHERE user_id = ? AND seq = 1",
            (user_id,),
        ).fetchone()
        assert p_pre is not None
        assert "【非User本人，系The User的信息】" in str(p_pre["fact_notes"])

        # 调用 apply_person_map_updates 归并 The User -> user
        person_update_payload = {
            "persons": [{"name": "user", "aliases": ["The User"], "relations": []}],
            "merge_persons": [
                {"from": "user", "to": "user", "aliases": ["The User"]},
                {"from": "The User", "to": "user", "aliases": ["The User"]},
            ],
        }
        try:
            store.apply_person_map_updates(user_id, session_id, person_update_payload)
        except (TypeError, AttributeError):
            store.apply_person_map_updates(
                user_id,
                session_id,
                merge_persons=person_update_payload["merge_persons"],
            )

        # 归并后断言
        c_post = conn.execute(
            "SELECT * FROM search_chunks WHERE user_id = ? AND kind = 'fact_item'",
            (user_id,),
        ).fetchone()
        assert c_post is not None
        assert str(c_post["subject"]).casefold() == "user", (
            f"归并后 subject 必须更新为 user，实际为: {c_post['subject']}"
        )
        assert str(c_post["source_tier"]) == "user_direct", (
            f"归并后 source_tier 必须恢复为 user_direct，实际为: {c_post['source_tier']}"
        )
        assert "【非User本人" not in str(c_post["display_text"]), (
            f"search_chunks.display_text 中的第三方前缀必须被彻底剥除，"
            f"实际为: {c_post['display_text']}"
        )
        assert "在西雅图有一套两居室公寓" in str(c_post["display_text"])

        p_post = conn.execute(
            "SELECT fact_notes FROM parent_messages WHERE user_id = ? AND seq = 1",
            (user_id,),
        ).fetchone()
        assert p_post is not None
        assert "【非User本人" not in str(p_post["fact_notes"]), (
            f"parent_messages.fact_notes 中的第三方前缀必须被彻底剥除，"
            f"实际为: {p_post['fact_notes']}"
        )
        assert "在西雅图有一套两居室公寓" in str(p_post["fact_notes"])
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core1_structured_card_header_and_neighbor_preservation(tmp_path: Path):
    """核心一.3 & 坑 1 / 坑 10 红线：卡片头防硬编码 Owner 污染，死守邻居 100% 完整保留。

    1. 默认模型写死 gpt-4o-mini（坑 10 红线）；
    2. 用户本人卡片（seq=2）：
       - 断言 "Owner: User (Self)" not in card_user.content（坑 10 严禁硬编码置顶行）；
       - 包含 [提炼事实: 与 [隐式推导 / Implied (非显式确认事实，优先级低于直接事实):
         eyeglasses, vision correction]；
       - 坑 1 永久禁回退红线：前后邻居 #1 与 #3 100% 完整保留；
    3. 第三方人物卡片（seq=5，系 Colleague）：
       - 断言 "Do NOT attribute to User" not in card_other.content；
       - 顶部包含 【注意：本段包含非User本人（系Colleague）的信息，请勿与User本人混淆】；
       - 包含 [提炼事实: 【非User本人，系Colleague的信息】；
       - 坑 1 邻居完整保留。
    """
    # 0. 验证默认模型写死为 gpt-4o-mini（坑 10 红线）
    from dandelion_memory.config import Settings
    assert Settings.model_fields["llm_model"].default == "gpt-4o-mini", (
        f"坑 10 红线：Settings.llm_model 默认必须写死为 'gpt-4o-mini'，严禁为 'qwen3.5-flash'，"
        f"实际为: {Settings.model_fields['llm_model'].default}"
    )
    assert DashScopeExtractor(api_key="test").model == "gpt-4o-mini", (
        f"坑 10 红线：DashScopeExtractor.model 默认必须写死为 'gpt-4o-mini'，"
        f"实际为: {DashScopeExtractor(api_key='test').model}"
    )

    db_path = tmp_path / "test_core1_struct_card.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            # Session 1: User self message with implied cue (seq=2, with neighbors seq=1 and seq=3)
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "user",
                        "tag": "健康/医疗与饮食禁忌",
                        "fact": "阅读屏幕时扶了扶鼻梁上的细金属镜框",
                        "current_state": "阅读屏幕时扶了扶鼻梁上的细金属镜框",
                        "implied": "eyeglasses, vision correction",
                    }
                ],
                "forget_actions": [],
            },
            # Session 2: Colleague (third-party) message (seq=5, with neighbors seq=4 and seq=6)
            {
                "facts": [
                    {
                        "seq": 5,
                        "who": "Colleague",
                        "subject": "Colleague",
                        "is_other_person": True,
                        "source_tier": "other_person",
                        "tag": "健康/运动与体征",
                        "fact": "换了人体工学椅后颈椎舒服很多",
                        "current_state": "换人体工学椅改善颈椎",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    pipeline, store, *_ = build_v2_pipeline(db_path=db_path, extractor=fake_ext)
    try:
        user_id = "u_core1_struct"
        # Add User conversation: seq=1, seq=2, seq=3
        await pipeline.add(
            AddRequest(
                request_id="req-struct-user",
                user_id=user_id,
                session_id="s_user",
                messages=[
                    Message(role="user", content="最近连续对着电脑工作，眼睛很容易发涩。"),
                    Message(
                        role="user",
                        content="阅读屏幕时我扶了扶鼻梁上的细金属镜框，屏幕实在太晃眼了。",
                    ),
                    Message(
                        role="assistant",
                        content="长时间盯屏幕确实容易疲劳，建议适当远眺放松。",
                    ),
                ],
            )
        )
        # Add Colleague conversation: seq=4, seq=5, seq=6
        await pipeline.add(
            AddRequest(
                request_id="req-struct-colleague",
                user_id=user_id,
                session_id="s_colleague",
                messages=[
                    Message(role="user", content="今天例会大家讨论了工位健康问题。"),
                    Message(role="user", content="Colleague 提到他换了人体工学椅后颈椎舒服很多。"),
                    Message(
                        role="assistant",
                        content="换一把符合人体工学的椅子确实能有效改善坐姿。",
                    ),
                ],
            )
        )

        # 1. 检索用户卡片 (seq=2)
        resp_user = await pipeline.search(
            SearchRequest(
                query="阅读屏幕时扶了扶鼻梁上的细金属镜框",
                user_id=user_id,
                top_k=3,
            )
        )
        assert len(resp_user.data) >= 1
        card_user = resp_user.data[0]
        user_card_content = card_user.content

        # 坑 10 修正：严禁在卡片顶行硬编码 "Owner: User (Self)"
        assert "Owner: User (Self)" not in user_card_content, (
            f"坑 10 红线：本人卡片严禁在顶部硬加 'Owner: User (Self)'，"
            f"实际内容:\n{user_card_content}"
        )
        assert "[提炼事实:" in user_card_content, "底部必须保留 [提炼事实: ...]"
        expected_implied_tag = (
            "[隐式推导 / Implied: eyeglasses, vision correction]"
        )
        assert expected_implied_tag in user_card_content, (
            f"卡片提炼事实区必须包含隐式推导标签，实际内容:\n{user_card_content}"
        )

        # 坑 1 永久禁回退红线：前后邻居 #1 与 #3 完整保留
        assert "#1" in user_card_content and "最近连续对着电脑工作" in user_card_content, (
            f"坑 1 永久禁回退红线：前邻居 #1 必须 100% 完整保留！实际内容:\n{user_card_content}"
        )
        assert "#3" in user_card_content and "长时间盯屏幕确实容易疲劳" in user_card_content, (
            f"坑 1 永久禁回退红线：后邻居 #3 必须 100% 完整保留！实际内容:\n{user_card_content}"
        )

        # 2. 检索第三方 Colleague 卡片 (seq=5)
        resp_colleague = await pipeline.search(
            SearchRequest(
                query="Colleague 人体工学椅 颈椎",
                user_id=user_id,
                top_k=3,
            )
        )
        assert len(resp_colleague.data) >= 1
        card_other = resp_colleague.data[0]
        other_card_content = card_other.content

        # 坑 10 修正：严禁在顶部硬加 "Do NOT attribute to User"
        assert "Do NOT attribute to User" not in card_other.content, (
            f"坑 10 红线：第三方卡严禁包含 'Do NOT attribute to User' 死命令，"
            f"实际内容:\n{other_card_content}"
        )
        # 顶部包含 【注意：本段包含非User本人（系Colleague）的信息，请勿与User本人混淆】
        warn_msg = "【注意：本段包含非User本人（系Colleague）的信息，请勿与User本人混淆】"
        assert warn_msg in other_card_content, (
            f"第三方卡必须包含混淆警告头，实际内容:\n{other_card_content}"
        )
        top_lines = [ln.strip() for ln in other_card_content.splitlines() if ln.strip()][:3]
        assert any(warn_msg in ln for ln in top_lines), (
            f"第三方卡顶部必须包含混淆警告头，实际前三行: {top_lines}"
        )
        # 提炼事实包含第三方前缀
        assert "[提炼事实: 【非User本人，系Colleague的信息】" in other_card_content, (
            f"第三方卡提炼事实必须标明非本人主体，实际内容:\n{other_card_content}"
        )
        # 坑 1 邻居完整保留
        assert ("#4" in other_card_content and "工位健康问题" in other_card_content) or (
            "#6" in other_card_content and "改善坐姿" in other_card_content
        ), (
            f"坑 1 永久禁回退红线：第三方卡的前后邻居必须完整保留，"
            f"实际内容:\n{other_card_content}"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core2_multi_slice_rerank_text_refresh_and_rrf_floor(tmp_path: Path):
    """核心二.1 & 坑 6 / 坑 10 红线：多切片高分同步刷新 rerank_text 与线性融合排噪。

    1. 多切片长消息：raw:0000 为首段寒暄，raw:0001 含核心机密条款；
       当 raw:0001 的 RRF 分高于 raw:0000 时，_build_turn_pair_cards 必须同步将
       metadata["rerank_text"] 刷新为 raw:0001 的文本，不再锁死在首段寒暄（坑 6 红线）；
    2. 坑 10 红线线性融合：
       严禁加无条件 0.80 * rrf_val 死保底（保持 0.55 * r + 0.45 * rrf 线性融合）；
       当 rrf_norm = 1.0、reranker 返回 0.10 时，融合分为 0.55 * 0.10 + 0.45 * 1.0 = 0.505，
       断言 0.48 <= top_item.score <= 0.65，严禁被 0.80 * rrf_val 死保底锁在 >= 0.80。
    """
    db_path = tmp_path / "test_core2_multi_slice.sqlite3"
    pipeline, store, embedder, _, reranker = build_v2_pipeline(db_path=db_path)
    try:
        user_id = "u_core2_rerank"
        session_id = "s_core2_01"

        pipeline.chunk_target_chars = 300
        pipeline.chunk_max_chars = 450

        greeting_prefix = "首段寒暄问候：你好，今天天气不错，祝你工作顺利愉快！\n" * 15
        needle_content = (
            "【关键机密工程代码】代号为 PROJECT-QUANTUM-X99，采用超导量子退火架构设计。\n" * 15
        )
        full_content = greeting_prefix + needle_content
        assert 800 <= len(full_content) <= 1800

        await pipeline.add(
            AddRequest(
                request_id="req-multi-slice-01",
                user_id=user_id,
                session_id=session_id,
                messages=[Message(role="user", content=full_content)],
            )
        )

        conn = store._connect()
        slices = conn.execute(
            "SELECT * FROM search_chunks "
            "WHERE user_id = ? AND kind = 'raw_slice' ORDER BY chunk_id",
            (user_id,),
        ).fetchall()
        assert len(slices) >= 2, f"超长消息必须切出至少 2 个切片，实际切出: {len(slices)}"

        c0 = dict(slices[0])
        c1 = dict(slices[1])
        assert "天气不错" in c0["display_text"]

        # 1. 模拟 _build_turn_pair_cards 接收 c0 (低 RRF) 与 c1 (高 RRF)
        recalled_chunks = [c0, c1]
        rrf_norm_map = {
            c0["chunk_id"]: 0.20,
            c1["chunk_id"]: 0.95,
        }
        dialogue_cands, _ = pipeline._build_turn_pair_cards(
            user_id=user_id,
            recalled_chunks=recalled_chunks,
            rrf_norm_map=rrf_norm_map,
            bm25_scores={},
            time_hit_cids=set(),
        )
        pmid = str(c0["parent_msg_id"])
        cand = dialogue_cands.get(pmid)
        assert cand is not None, "必须生成对话候选卡"

        rerank_txt = str(cand.metadata.get("rerank_text") or "")
        assert "PROJECT-QUANTUM-X99" in rerank_txt, (
            f"坑 6 红线：高分切片 raw:0001 必须同步刷新 rerank_text，不能锁死在首段寒暄！"
            f"实际 rerank_text 前 200 字: {rerank_txt[:200]}"
        )

        # 2. 坑 10 修正：严禁无条件 0.80 * rrf_val 死保底，保持 0.55 * r + 0.45 * rrf 线性融合
        # 当 rrf_norm = 1.0、reranker 返回 0.10 时，线性融合应为 0.55 * 0.10 + 0.45 * 1.0 = 0.505
        cand.metadata["rrf_norm"] = 1.0
        cand.score = 1.0
        reranker.score_fn = lambda q, d, i: 0.10

        scored_cards, rerank_ok = await pipeline._score_and_rerank_cards(
            pure_query="PROJECT-QUANTUM-X99",
            dialogue_candidates={pmid: cand},
            timeline_candidates={},
            user_id=user_id,
        )
        assert rerank_ok is True
        scored_cand = scored_cards[0]
        assert 0.48 <= scored_cand.score <= 0.65, (
            f"坑 10 红线：当 rrf_norm = 1.0、reranker 返回 0.10 时，必须使用线性融合 "
            f"0.55 * 0.10 + 0.45 * 1.0 = 0.505，严禁被 0.80 * rrf_val 死保底锁在 >= 0.80！"
            f"实际得分为: {scored_cand.score}"
        )
        assert scored_cand.score < 0.80, (
            f"坑 10 红线：严禁加 0.80 * rrf_val 死保底！实际得分为: {scored_cand.score}"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core2_evidence_tiering_n_high_and_draft_quota(tmp_path: Path):
    """核心二.2 & 坑 5 红线：证据按需分级排序，无主证据时不搞一刀切降权或低分填坑。

    - 子场景 A（无高分本人直接证据 n_high == 0 且未点名他人）：
      库内仅有 1 张低分无关本人日常卡（score < 0.4）、1 张高相关第三方卡（subject="Colleague"）
      和 1 张高相关文稿卡（source_tier="user_draft_fiction"）；
      检索未点名他人的通用问题时：
      1. 第三方状态卡不被 ×0.35 压死（保留原相关性参与排序）；
      2. 文稿配额 direct_keep == n_high == 0，高相关文稿卡不被低分无关日常卡挤出前 2。
    - 子场景 B（有高分本人直接证据 n_high > 0）：
      当存在高分纯净本人直接证据卡（score >= 0.55）时，未点名的第三方 timeline_summary
      仍被 ×0.35 降权且排除在 pinned_tl 之外。
    """
    db_path = tmp_path / "test_core2_evidence_tier.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            # Batch 1: Low-relevance user daily card (user_direct)
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "日常/杂项",
                        "fact": "早上喝了一杯温开水",
                        "current_state": "早上喝了一杯温开水",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2: Highly relevant colleague card (other_person)
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "Colleague",
                        "subject": "Colleague",
                        "is_other_person": True,
                        "source_tier": "other_person",
                        "tag": "生活/厨具",
                        "fact": "同事使用高压锅压牛肉炖汤很入味",
                        "current_state": "使用高压锅压牛肉炖汤",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 3: Highly relevant fiction draft card (user_draft_fiction)
            {
                "facts": [
                    {
                        "seq": 3,
                        "who": "user",
                        "subject": "user",
                        "tag": "创作/小说设定",
                        "fact": "小说中探险队在高压锅原理的密封舱中烹饪炖牛肉",
                        "current_state": "密封舱高压烹饪炖牛肉",
                        "is_fiction": True,
                    }
                ],
                "forget_actions": [],
            },
            # Batch 4: For Sub-scenario B: High-confidence clean user direct fact
            {
                "facts": [
                    {
                        "seq": 4,
                        "who": "user",
                        "tag": "生活/厨具",
                        "fact": "用户亲自购买了九阳双胆高压锅并用来做炖牛肉",
                        "current_state": "购买九阳双胆高压锅炖牛肉",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    fake_reranker = FakeReranker(
        keyword_scores={
            "高压锅": 0.85,
            "温开水": 0.15,
        }
    )
    pipeline, store, embedder, _, reranker = build_v2_pipeline(
        db_path=db_path,
        extractor=fake_ext,
        reranker=fake_reranker,
    )
    try:
        user_id = "u_core2_tier"
        # Ingest first 3 batches
        await pipeline.add(
            AddRequest(
                request_id="req-tier-01",
                user_id=user_id,
                session_id="s1",
                messages=[Message(role="user", content="早上起床后喝了一杯温开水。")],
            )
        )
        await pipeline.add(
            AddRequest(
                request_id="req-tier-02",
                user_id=user_id,
                session_id="s2",
                messages=[
                    Message(
                        role="user",
                        content="办公室同事分享说用高压锅压牛肉炖汤特别软烂入味。",
                    )
                ],
            )
        )
        await pipeline.add(
            AddRequest(
                request_id="req-tier-03",
                user_id=user_id,
                session_id="s3",
                messages=[
                    Message(
                        role="user",
                        content="小说草稿第十二章：探险队在参考高压锅原理设计的密封舱里烹饪炖牛肉。",
                    )
                ],
            )
        )

        # --- 子场景 A: n_high == 0 且未点名他人 ---
        resp_a = await pipeline.search(
            SearchRequest(
                query="用高压锅做炖牛肉有什么诀窍？",
                user_id=user_id,
                top_k=2,
            )
        )
        assert len(resp_a.data) >= 1
        results_a = [item.content for item in resp_a.data]
        colleague_tl = [x for x in resp_a.data if "【状态时间线 | 主体: Colleague" in x.content]
        assert (
            colleague_tl
            and colleague_tl[0].score is not None
            and colleague_tl[0].score >= 0.65
        ), (
            f"坑 5 红线：无本人直接证据 (n_high==0) 且未点名他人时，第三方状态卡不得被 ×0.35 压死！"
            f"实际得分: {colleague_tl[0].score if colleague_tl else '未在前 2 名召回'}"
        )
        assert not any("温开水" in r for r in results_a), (
            f"坑 5 红线：n_high==0 时 direct_keep 必须为 0，不能拿低分无关日常卡填坑挤出高相关卡！"
            f"实际前 2 结果: {results_a}"
        )
        assert any("高压锅" in r for r in results_a), (
            f"子场景 A：高压锅相关记忆必须排在前面，实际为: {results_a}"
        )

        # --- 子场景 B: n_high > 0 存在高置信度本人直接证据 ---
        await pipeline.add(
            AddRequest(
                request_id="req-tier-04",
                user_id=user_id,
                session_id="s4",
                messages=[
                    Message(
                        role="user",
                        content="我上周自己买了一台九阳双胆高压锅，亲自做了顿西红柿炖牛肉。",
                    )
                ],
            )
        )

        resp_b = await pipeline.search(
            SearchRequest(
                query="我自己买的高压锅炖牛肉效果怎么样？",
                user_id=user_id,
                top_k=3,
            )
        )
        assert len(resp_b.data) >= 1
        top_b = resp_b.data[0].content
        assert "九阳双胆高压锅" in top_b or "亲自" in top_b, (
            f"子场景 B：存在本人直接证据时，本人直接卡必须置顶优先，实际首名: {top_b}"
        )
        if len(resp_b.data) > 1:
            for item in resp_b.data:
                if "【状态时间线 | 主体: Colleague" in item.content:
                    assert item.score is not None and item.score < 0.60
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core2_specific_forget_no_collateral_damage_and_broad_category_fn_cue(tmp_path: Path):
    """核心二.3 & 坑 8 / 坑 10 红线：显式具体遗忘零误杀同大类其他记忆，
    FN 物理隔离且绝不注入宽泛污染词。

    1. 同一「饮食/喜好」标签下先后写入：
       - #1: "平时我最喜欢做菠菜咖喱配糙米饭，并且每周按 schedule 推进维护工作。"
         （包含普通词 schedule）
       - #2: "使用 MealPrepPro 软件规划每周备餐"
    2. #3 用户显式要求忘掉 "MealPrepPro" 与
       "weekly meal planning and prep scheduling"；
    3. 坑 8 & 坑 10 永久禁回退红线断言：
       - #2 (MealPrepPro) 必须被墓碑化（tombstoned == 1）；
       - #1 (菠菜咖喱与 schedule) 保持存活（tombstoned == 0），
         绝不因 "scheduling" 长度 >= 8 或带空格短语被单字误杀；
    4. FN 槽位规范与物理隔离断言（坑 10 红线）：
       - 生成的 FN 槽位（tl: 与 fact: 两条记录）的 parent_msg_id 均为 None 且 seq == 0；
       - search_text 包含 neutral_topic ("weekly meal planning")；
       - search_text 绝对不包含宽泛大类污染词 "cooking"
         （assert "cooking" not in fn_search_text）。
    """
    db_path = tmp_path / "test_core2_forget.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            # Batch 1: Spinach curry with schedule mention under 饮食/喜好
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "饮食/喜好",
                        "fact": "喜欢做菠菜咖喱配糙米饭，并且每周按 schedule 推进维护工作",
                        "current_state": "喜欢做菠菜咖喱，每周按 schedule 推进维护工作",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2: MealPrepPro under 饮食/喜好
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "user",
                        "tag": "饮食/喜好",
                        "fact": "使用 MealPrepPro 软件规划每周备餐",
                        "current_state": "使用 MealPrepPro 备餐软件",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 3: Explicit forget command targeting MealPrepPro & prep scheduling
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 3,
                        "target_existing_tags": ["饮食/喜好"],
                        "target_keywords": [
                            "MealPrepPro",
                            "weekly meal planning and prep scheduling",
                        ],
                        "target_description": (
                            "MealPrepPro and weekly meal planning and prep scheduling"
                        ),
                        "neutral_topic": "weekly meal planning and prep scheduling",
                        "batch_seqs_to_tombstone": [3],
                    }
                ],
            },
        ]
    )
    pipeline, store, *_ = build_v2_pipeline(db_path=db_path, extractor=fake_ext)
    try:
        user_id = "u_core2_forget"
        session_id = "s_core2_forget_01"

        # Batch 1: Ingest spinach curry with 'schedule'
        await pipeline.add(
            AddRequest(
                request_id="req-fg-01",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(
                        role="user",
                        content=(
                            "平时我最喜欢做菠菜咖喱配糙米饭，"
                            "并且每周按 schedule 推进维护工作。"
                        ),
                    )
                ],
            )
        )
        # Batch 2: Ingest MealPrepPro
        await pipeline.add(
            AddRequest(
                request_id="req-fg-02",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(role="user", content="我一直在用 MealPrepPro 软件规划每周备餐。")
                ],
            )
        )
        # Batch 3: Forget action
        await pipeline.add(
            AddRequest(
                request_id="req-fg-03",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(
                        role="user",
                        content=(
                            "请帮我忘掉 MealPrepPro 和所有 "
                            "weekly meal planning and prep scheduling 的记录。"
                        ),
                    )
                ],
            )
        )

        conn = store._connect()
        # 1. 验证 #2 (MealPrepPro) 被墓碑化
        row_seq2 = conn.execute(
            "SELECT tombstoned, display_text FROM search_chunks WHERE user_id = ? AND seq = 2",
            (user_id,),
        ).fetchall()
        assert len(row_seq2) > 0, "seq=2 切片必须存在"
        assert all(int(r["tombstoned"]) == 1 for r in row_seq2), (
            f"被遗忘的具体项 seq=2 必须全部墓碑化 (tombstoned == 1)，"
            f"实际状态: {[r['tombstoned'] for r in row_seq2]}"
        )

        # 2. 坑 8 & 坑 10 红线：同属「饮食/喜好」且含普通词 schedule 的 seq=1
        # 保持存活 (tombstoned == 0)，绝不因 "scheduling" 长度 >= 8 或带空格短语被单字误杀
        row_seq1 = conn.execute(
            "SELECT tombstoned, display_text FROM search_chunks WHERE user_id = ? AND seq = 1",
            (user_id,),
        ).fetchall()
        assert len(row_seq1) > 0, "seq=1 切片必须存在"
        assert all(int(r["tombstoned"]) == 0 for r in row_seq1), (
            f"坑 8 & 坑 10 红线：同属「饮食/喜好」且含普通词 schedule 的菠菜咖喱 (seq=1) "
            f"必须保持存活 (tombstoned == 0)，绝不因 'scheduling' 长度 >= 8 "
            f"或带空格短语被单字误杀！实际状态: {[r['tombstoned'] for r in row_seq1]}"
        )

        # 3. 坑 10 红线：验证生成的 FN 槽位 (tl: 与 fact: 两条记录) 的物理隔离与无宽泛词污染
        fn_chunks = conn.execute(
            "SELECT chunk_id, kind, slot_id, parent_msg_id, seq, search_text, display_text "
            "FROM search_chunks WHERE user_id = ? "
            "AND (slot_id LIKE 'FN%' OR chunk_id LIKE '%:FN%')",
            (user_id,),
        ).fetchall()
        assert len(fn_chunks) >= 2, (
            f"必须生成 tl: 与 fact: 至少两条 FN 记录，实际为: {len(fn_chunks)}"
        )

        tl_fn = next((r for r in fn_chunks if str(r["chunk_id"]).startswith("tl:")), None)
        fact_fn = next((r for r in fn_chunks if str(r["chunk_id"]).startswith("fact:")), None)
        assert tl_fn is not None, "tl: FN 槽位记录必须存在"
        assert fact_fn is not None, "fact: FN 槽位记录必须存在"

        # parent_msg_id 均为 None 且 seq == 0（杜绝原话拉取）
        assert tl_fn["parent_msg_id"] is None, (
            f"坑 10 红线：tl: FN 槽位 parent_msg_id 必须为 None，实际为: {tl_fn['parent_msg_id']}"
        )
        assert tl_fn["seq"] == 0, f"tl: FN 槽位 seq 必须为 0，实际为: {tl_fn['seq']}"

        assert fact_fn["parent_msg_id"] is None, (
            f"坑 10 红线：fact: FN 槽位 parent_msg_id 必须为 None（绝不挂载 trigger_seq），"
            f"实际为: {fact_fn['parent_msg_id']}"
        )
        assert fact_fn["seq"] == 0, f"fact: FN 槽位 seq 必须为 0，实际为: {fact_fn['seq']}"

        # search_text 包含 neutral_topic ("weekly meal planning")，且不包含宽泛污染词 "cooking"
        for fn_c in (tl_fn, fact_fn):
            fn_search_text = str(fn_c["search_text"]).lower()
            assert "weekly meal planning" in fn_search_text, (
                f"FN 槽位 search_text 必须包含 neutral_topic 核心词，实际为: {fn_c['search_text']}"
            )
            assert "cooking" not in fn_search_text, (
                f"坑 10 红线：FN 槽位严禁注入宽泛大类词 cooking！"
                f"实际 search_text 为: {fn_c['search_text']}"
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_forget_target_keywords_preserved_and_stopword_cleaned(tmp_path: Path):
    """验证遗忘动作关键词清洗与显式遗忘轮事实跳过（TDD 红灯用例）。

    1. extract_keywords_from_target 必须将 'User' 与 'prefers' 作为停用词过滤；
    2. sanitize_extraction_result 在 seq=3 显式遗忘轮下：
       - 不为已在 forget_actions 的显式遗忘轮从 facts 重复生成第二条 ExtractedForgetAction；
       - forget_actions[0].target_keywords 不包含 'User' 与 'prefers'；
       - len(res.facts) == 0（跳过 seq=3 显式遗忘轮的事实抽取）。
    """
    db_path = tmp_path / "test_forget_kw.sqlite3"
    _, store, *_ = build_v2_pipeline(db_path=db_path)
    try:
        # 1. extract_keywords_from_target 停用词清洗测试
        target_text = "User prefers not to use meal-prep apps to automate cooking schedules."
        extracted_kws = extract_keywords_from_target(target_text)
        assert "User" not in extracted_kws, (
            f"'User' 必须作为停用词被剔除，实际提取: {extracted_kws}"
        )
        assert "prefers" not in extracted_kws, (
            f"'prefers' 必须作为停用词被剔除，实际提取: {extracted_kws}"
        )

        # 2. sanitize_extraction_result 防重复抽取与停用词过滤测试
        messages = [
            {
                "seq": 3,
                "role": "user",
                "content": (
                    "Please forget my preference for meal-prep apps to automate cooking schedules."
                ),
            }
        ]
        raw_payload = {
            "facts": [
                {
                    "seq": 3,
                    "who": "user",
                    "tag": "全局指令/格式约束",
                    "fact": (
                        "User prefers not to use meal-prep apps to automate cooking schedules."
                    ),
                }
            ],
            "forget_actions": [
                {
                    "trigger_seq": 3,
                    "target_description": (
                        "preference for meal-prep apps to automate cooking schedules"
                    ),
                    "target_keywords": [
                        "meal-prep apps",
                        "automate cooking schedules",
                    ],
                }
            ],
        }
        facts, forgets = sanitize_extraction_result(raw_payload, batch_messages=messages)
        res = ExtractionResult(facts=facts, forget_actions=forgets)

        # 断言 1: 必须仅保留 1 条 forget_action（不从 seq=3 的 facts 项重复生成）
        assert len(res.forget_actions) == 1, (
            f"必须仅保留 1 条 forget_action，不能从 seq=3 的 facts 项重复生成第二条！"
            f"实际数量: {len(res.forget_actions)}, 内容: {res.forget_actions}"
        )

        # 断言 2: "User" 与 "prefers" 绝不能出现在 target_keywords 中
        assert "User" not in res.forget_actions[0].target_keywords, (
            f"'User' 绝不能出现在 forget_actions[0].target_keywords，"
            f"实际为: {res.forget_actions[0].target_keywords}"
        )
        assert "prefers" not in res.forget_actions[0].target_keywords, (
            f"'prefers' 绝不能出现在 forget_actions[0].target_keywords，"
            f"实际为: {res.forget_actions[0].target_keywords}"
        )

        # 断言 3: seq=3 系显式遗忘轮，其事实项必须被跳过 (len == 0)
        assert len(res.facts) == 0, (
            f"seq=3 系显式遗忘轮，facts 必须被跳过 (len == 0)，"
            f"实际包含 {len(res.facts)} 条: {res.facts}"
        )
    finally:
        store.close()
