# 契约：Dandelion Memory v3.1 结构化卡片视图与细粒度证据阶梯测试集（TDD 红灯用例）
# 依据 docs/DESIGN.md 第九节「v3.1 结构化卡片视图与细粒度证据阶梯设计」与第十节「历次迭代踩坑与永久禁回退红线总表」
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dandelion_memory.extractor as extractor_mod
import dandelion_memory.pipeline as pipeline_mod
from dandelion_memory.extractor import (
    DashScopeExtractor,
    ExtractedFact,
    ExtractedForgetAction,
    sanitize_extraction_result,
)
from dandelion_memory.models import AddRequest, Message, SearchRequest

from .conftest import FakeExtractor, FakeReranker, build_v2_pipeline


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
    if isinstance(parsed, list):
        fact = parsed[0]
    else:
        fact = parsed
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
                    Message(role="user", content="阅读屏幕时扶了扶鼻梁上的细金属镜框，屏幕太晃眼了。")
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
            "[隐式推导 / Implied (非显式确认事实，优先级低于直接事实): eyeglasses, vision correction]"
        )
        assert expected_implied_tag in str(f_row["display_text"]), (
            f"fact_item.display_text 必须以显式标签格式存储 implied，实际为: {f_row['display_text']}"
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
                    Message(role="user", content="今天去眼科验光确定为175度近视散光，配了防疲劳镜。")
                ],
            )
        )

        slots = conn.execute(
            "SELECT slot_id, current_state FROM tag_timelines WHERE user_id = ? AND tombstoned = 0 ORDER BY slot_id",
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
            "SELECT display_text, search_text FROM search_chunks WHERE user_id = ? AND kind = 'fact_item' AND slot_id = 'F1'",
            (user_id,),
        ).fetchone()
        assert f1_chunk is not None, "F1 fact_item 必须存在"
        assert merged_state in f1_chunk["display_text"], "合槽后 display_text 必须包含最新显式事实"
        assert expected_implied_tag in f1_chunk["display_text"], (
            f"坑 7 & 坑 9 红线：合槽刷新时必须保留原有的 [隐式推导 / Implied ...] 标记，不能被裸覆盖冲掉！"
            f"实际 display_text: {f1_chunk['display_text']}"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core1_person_alias_merge_to_user_cleans_prefix_and_restores_tier(tmp_path: Path):
    """核心一.2：人物别名（The User -> user）归并、剥离前缀与恢复本人直接证据层级。

    - 先入库一条被 Extractor 误标为 subject="The User", is_other_person=True, source_tier="other_person" 的事实；
    - 验证底表 search_chunks.display_text 与 parent_messages.fact_notes 初始包含 【非User本人，系The User的信息】；
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
            f"search_chunks.display_text 中的第三方前缀必须被彻底剥除，实际为: {c_post['display_text']}"
        )
        assert "在西雅图有一套两居室公寓" in str(c_post["display_text"])

        p_post = conn.execute(
            "SELECT fact_notes FROM parent_messages WHERE user_id = ? AND seq = 1",
            (user_id,),
        ).fetchone()
        assert p_post is not None
        assert "【非User本人" not in str(p_post["fact_notes"]), (
            f"parent_messages.fact_notes 中的第三方前缀必须被彻底剥除，实际为: {p_post['fact_notes']}"
        )
        assert "在西雅图有一套两居室公寓" in str(p_post["fact_notes"])
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core1_structured_card_header_and_neighbor_preservation(tmp_path: Path):
    """核心一.3 & 坑 1 红线：统一置顶 Owner / Fact / Implied 结构化卡片头，死守邻居 100% 完整保留。

    1. 用户本人卡片（User Self）：
       - 顶部必须包含置顶结构化头：Owner: User (Self) 与 Fact: <直接事实>；
       - 若存在 implied，必须紧跟：Implied: [隐式推导 ... <implied>；
       - 底部仍保留原有的 [提炼事实: <直接事实>] 标记；
    2. 第三方人物卡片（如 Priya）：
       - 顶部必须包含置顶结构化头：
         Owner: Priya (NOT User — Third-Party Trait, Do NOT attribute to User)
         【注意：本段包含非User本人（系Priya）的信息，请勿与User本人混淆】
         Fact: 【非User本人，系Priya的信息】<事实>
       - 底部保留：[提炼事实: 【非User本人，系Priya的信息】...]
    3. 坑 1 永久禁回退红线：
       - 配对消息的 prev_line / next_line 问答对邻居 100% 完整保留，严禁裁剪或丢弃！
    """
    db_path = tmp_path / "test_core1_struct_card.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            # Session 1: User self message with implied cue
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "健康/运动与体征",
                        "fact": "每天慢跑后习惯按压手腕查看静息心率",
                        "current_state": "每天慢跑后按压手腕查看心率",
                        "implied": "smartwatch, heart rate monitor",
                    }
                ],
                "forget_actions": [],
            },
            # Session 2: Priya (third-party) message with conversation pair
            {
                "facts": [
                    {
                        "seq": 2,
                        "who": "Priya",
                        "subject": "Priya",
                        "is_other_person": True,
                        "source_tier": "other_person",
                        "tag": "居住/搬迁",
                        "fact": "在剑桥区新购置了一套公寓",
                        "current_state": "在剑桥区购置公寓",
                    }
                ],
                "forget_actions": [],
            },
        ]
    )
    pipeline, store, *_ = build_v2_pipeline(db_path=db_path, extractor=fake_ext)
    try:
        user_id = "u_core1_struct"
        # Add User card
        await pipeline.add(
            AddRequest(
                request_id="req-struct-user",
                user_id=user_id,
                session_id="s_user",
                messages=[
                    Message(role="user", content="每天慢跑五公里后，我习惯按压手腕查看静息心率是否恢复。")
                ],
            )
        )
        # Add Priya conversation with assistant reply (to test neighbor preservation)
        await pipeline.add(
            AddRequest(
                request_id="req-struct-priya",
                user_id=user_id,
                session_id="s_priya",
                messages=[
                    Message(role="user", content="Priya 上周在剑桥区新购置了一套采光很好的公寓。"),
                    Message(role="assistant", content="恭喜 Priya！剑桥区的交通和社区环境确实非常方便。"),
                ],
            )
        )

        # 1. 检索用户卡片
        resp_user = await pipeline.search(
            SearchRequest(
                query="慢跑后按压手腕查看静息心率",
                user_id=user_id,
                top_k=3,
            )
        )
        assert len(resp_user.data) >= 1
        user_card_content = resp_user.data[0].content
        lines_user = [ln.strip() for ln in user_card_content.splitlines() if ln.strip()]

        # 置顶头断言
        assert lines_user[0] == "Owner: User (Self)", (
            f"用户卡片首行必须为 Owner: User (Self)，实际为: {lines_user[0] if lines_user else 'None'}"
        )
        assert any(ln.startswith("Fact:") for ln in lines_user[:3]), (
            f"卡片置顶区必须包含 Fact:，实际前三行: {lines_user[:3]}"
        )
        assert any("Implied:" in ln and "smartwatch" in ln for ln in lines_user[:4]), (
            f"卡片置顶区有 implied 时必须包含 Implied: [隐式推导 ...]，实际内容:\n{user_card_content}"
        )
        assert "[提炼事实:" in user_card_content, "底部必须保留 [提炼事实: ...]"

        # 2. 检索第三方 Priya 卡片
        resp_priya = await pipeline.search(
            SearchRequest(
                query="Priya 剑桥区新购置公寓",
                user_id=user_id,
                top_k=3,
            )
        )
        assert len(resp_priya.data) >= 1
        priya_card_content = resp_priya.data[0].content
        lines_priya = [ln.strip() for ln in priya_card_content.splitlines() if ln.strip()]

        # 第三方置顶头断言
        assert "Owner: Priya (NOT User — Third-Party Trait, Do NOT attribute to User)" in lines_priya[0], (
            f"第三方卡片首行必须包含专属 Owner 标签，实际为: {lines_priya[0] if lines_priya else 'None'}"
        )
        assert any("【注意：本段包含非User本人（系Priya）的信息，请勿与User本人混淆】" in ln for ln in lines_priya[:3]), (
            f"第三方卡片置顶区必须包含注意混淆警告行，实际前三行: {lines_priya[:3]}"
        )
        assert any(ln.startswith("Fact:") and "【非User本人，系Priya的信息】" in ln for ln in lines_priya[:4]), (
            f"第三方卡片置顶区 Fact 必须带非本人前缀，实际前四行: {lines_priya[:4]}"
        )
        assert "[提炼事实: 【非User本人，系Priya的信息】" in priya_card_content

        # 3. 坑 1 红线断言：邻居 100% 完整保留
        assert "恭喜 Priya！剑桥区的交通和社区环境确实非常方便。" in priya_card_content, (
            "坑 1 永久禁回退红线：配对消息的 assistant next_line 问答对邻居必须 100% 完整保留！"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_core2_multi_slice_rerank_text_refresh_and_rrf_floor(tmp_path: Path):
    """核心二.1 & 坑 6 红线：多切片高分同步刷新 rerank_text 与 RRF 下界融合保底。

    1. 多切片长消息：raw:0000 为首段寒暄，raw:0001 含核心机密条款；
       当 raw:0001 的 RRF 分高于 raw:0000 时，_build_turn_pair_cards 必须同步将
       metadata["rerank_text"] 刷新为 raw:0001 的文本，不再锁死在首段寒暄；
    2. RRF 下界保护（坑 6 红线）：
       当 RRF 归一化满分 1.00，而 Reranker 给分极低（如 0.10）时，
       融合得分 item.score 必须受到下界保护 item.score >= 0.80（max(0.55*r + 0.45*rrf, 0.80*rrf)），
       防止单看短题干的 Reranker 把 RRF 满分的精准命中反杀出前 10。
    """
    db_path = tmp_path / "test_core2_multi_slice.sqlite3"
    pipeline, store, embedder, _, reranker = build_v2_pipeline(db_path=db_path)
    try:
        user_id = "u_core2_rerank"
        session_id = "s_core2_01"

        pipeline.chunk_target_chars = 300
        pipeline.chunk_max_chars = 450

        greeting_prefix = "首段寒暄问候：你好，今天天气不错，祝你工作顺利愉快！\n" * 15
        needle_content = "【关键机密工程代码】代号为 PROJECT-QUANTUM-X99，采用超导量子退火架构设计。\n" * 15
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

        # 2. 验证 RRF 下界融合公式：RRF=1.0, Rerank=0.10 时 item.score >= 0.80
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
        assert scored_cand.score >= 0.80, (
            f"坑 6 红线：RRF 下界保护生效时，RRF=1.0 即使 Reranker=0.10 融合分也必须 >= 0.80，"
            f"实际得分为: {scored_cand.score}"
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
                messages=[Message(role="user", content="办公室同事分享说用高压锅压牛肉炖汤特别软烂入味。")],
            )
        )
        await pipeline.add(
            AddRequest(
                request_id="req-tier-03",
                user_id=user_id,
                session_id="s3",
                messages=[Message(role="user", content="小说草稿第十二章：探险队在参考高压锅原理设计的密封舱里烹饪炖牛肉。")],
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
        assert colleague_tl and colleague_tl[0].score is not None and colleague_tl[0].score >= 0.65, (
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
                messages=[Message(role="user", content="我上周自己买了一台九阳双胆高压锅，亲自做了顿西红柿炖牛肉。")],
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
    """核心二.3 & 坑 8 红线：显式具体遗忘零误杀同大类其他记忆，撤回提示槽注入双语大类别。

    1. 同一「饮食/喜好」标签下先后写入：
       - #1: "喜爱菠菜咖喱配糙米饭"
       - #2: "使用 MealPrepPro 备餐软件"
    2. #3 用户显式要求忘掉 "MealPrepPro / meal-prep apps"（指定了具体关键词与标签）；
    3. 坑 8 永久禁回退红线断言：
       - #2 (MealPrepPro) 必须被墓碑化（tombstoned == 1）；
       - 同标签下的 #1 (菠菜咖喱) 必须保持 tombstoned == 0 完好存活，严禁按大类 Tag 连坐误杀！
    4. 撤回提示槽（FN）大类提示断言：
       - 生成的 FN 槽位的 search_text 必须包含由「饮食/喜好」映射出的通用双语大类别词
         （如 diet / food / meal / cooking），确保用户问及相关大类时必能召回提示。
    """
    db_path = tmp_path / "test_core2_forget.sqlite3"
    fake_ext = FakeExtractor(
        responses=[
            # Batch 1: Spinach curry under 饮食/喜好
            {
                "facts": [
                    {
                        "seq": 1,
                        "who": "user",
                        "tag": "饮食/喜好",
                        "fact": "平时最喜欢吃菠菜咖喱配糙米饭",
                        "current_state": "喜爱菠菜咖喱配糙米饭",
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
            # Batch 3: Explicit forget command targeting MealPrepPro
            {
                "facts": [],
                "forget_actions": [
                    {
                        "trigger_seq": 3,
                        "target_existing_tags": ["饮食/喜好"],
                        "target_keywords": ["MealPrepPro", "meal-prep apps"],
                        "target_description": "MealPrepPro meal-prep apps",
                        "neutral_topic": "MealPrepPro",
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

        # Batch 1
        await pipeline.add(
            AddRequest(
                request_id="req-fg-01",
                user_id=user_id,
                session_id=session_id,
                messages=[Message(role="user", content="平时我最喜欢吃菠菜咖喱配糙米饭。")],
            )
        )
        # Batch 2
        await pipeline.add(
            AddRequest(
                request_id="req-fg-02",
                user_id=user_id,
                session_id=session_id,
                messages=[Message(role="user", content="我一直在用 MealPrepPro 软件规划每周备餐。")],
            )
        )
        # Batch 3: Forget action
        await pipeline.add(
            AddRequest(
                request_id="req-fg-03",
                user_id=user_id,
                session_id=session_id,
                messages=[
                    Message(role="user", content="请帮我忘掉 MealPrepPro 和所有 meal-prep apps 的记录。")
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
            f"被遗忘的具体项 seq=2 必须全部墓碑化 (tombstoned == 1)，实际状态: {[r['tombstoned'] for r in row_seq2]}"
        )

        # 2. 坑 8 红线：验证同属「饮食/喜好」的 #1 (菠菜咖喱) 严禁被连坐误杀！
        row_seq1 = conn.execute(
            "SELECT tombstoned, display_text FROM search_chunks WHERE user_id = ? AND seq = 1",
            (user_id,),
        ).fetchall()
        assert len(row_seq1) > 0, "seq=1 切片必须存在"
        assert any(int(r["tombstoned"]) == 0 for r in row_seq1), (
            f"坑 8 永久禁回退红线：同属「饮食/喜好」的菠菜咖喱 (seq=1) 必须保持存活 (tombstoned == 0)，"
            f"严禁按大类标签连坐误杀！实际状态: {[r['tombstoned'] for r in row_seq1]}"
        )

        # 3. 验证 FN 撤回提示槽挂上了双语大类提示词
        fn_chunk = conn.execute(
            "SELECT chunk_id, search_text, display_text FROM search_chunks "
            "WHERE user_id = ? AND (slot_id = 'FN3' OR chunk_id LIKE '%FN3%')",
            (user_id,),
        ).fetchone()
        assert fn_chunk is not None, "必须生成 FN3 撤回状态槽"
        fn_search_text = str(fn_chunk["search_text"]).lower()
        broad_keywords = ("diet", "food", "meal", "cooking")
        has_broad_cue = any(kw in fn_search_text for kw in broad_keywords)
        assert has_broad_cue, (
            f"FN 撤回槽必须注入由「饮食/喜好」映射的双语大类别词（如 {broad_keywords}），"
            f"实际 search_text 为: {fn_chunk['search_text']}"
        )
    finally:
        store.close()
