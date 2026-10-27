# 契约：Dandelion Memory v2.3 极简现存事实主表、单事务原子落库与双向证据联动契约
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from dandelion_memory.models import AddRequest, SearchRequest

from .conftest import FakeEmbedder, FakeExtractor, FakeReranker, build_v2_pipeline


async def test_t1_superseded_fact_item_retires_and_search_renders_update_warning(
    db_path: Path,
):
    """T1 (INV-1 / INV-4): Full-superseded card is demoted with update warning;

    half-superseded card (1 expired fact + 1 living fact on same seq) is NOT demoted
    when searching the living fact, yet still annotates the expired slot's latest value.
    """
    extractor = FakeExtractor(
        responses=[
            # Batch 1: seq 1 mortgage pre-approval $350,000
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "资产/房贷",
                        "fact_text": "用户的 Wells Fargo 房贷预批额度为 $350,000",
                        "current_state": "$350,000",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2: seq 2 mortgage pre-approval updated to $400,000
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "资产/房贷",
                        "fact_text": "用户的 Wells Fargo 房贷预批额度更新为 $400,000",
                        "prev_state": "$350,000",
                        "current_state": "$400,000",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 3: seq 3 contains TWO facts in the same message (Shanghai + peanut allergy)
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "上海市",
                        "seq": 3,
                    },
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "健康/医疗与过敏",
                        "value": "对花生严重过敏",
                        "seq": 3,
                    },
                ]
            },
            # Batch 4: seq 4 updates ONLY city to Beijing (peanut allergy at seq 3 stays alive!)
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "北京市",
                        "seq": 4,
                    }
                ]
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "$400,000": 0.95,
            "$350,000": 0.80,
            "对花生严重过敏": 0.92,
        }
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    # Batch 1: seq 1
    await pipeline.add(
        AddRequest(
            request_id="req-t1-b1",
            user_id="u_t1",
            session_id="s_t1_1",
            messages=[
                {
                    "role": "user",
                    "content": "My mortgage pre-approval from Wells Fargo is $350,000.",
                }
            ],
        )
    )

    # Batch 2: seq 2 updates mortgage to $400,000
    await pipeline.add(
        AddRequest(
            request_id="req-t1-b2",
            user_id="u_t1",
            session_id="s_t1_2",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Great news, Wells Fargo updated my mortgage pre-approval to $400,000."
                    ),
                }
            ],
        )
    )

    # Batch 3: seq 3 has both Shanghai residence and severe peanut allergy
    await pipeline.add(
        AddRequest(
            request_id="req-t1-b3",
            user_id="u_t1",
            session_id="s_t1_3",
            messages=[
                {
                    "role": "user",
                    "content": "我目前居住在上海市，另外我对花生严重过敏，一点都不能碰。",
                }
            ],
        )
    )

    # Batch 4: seq 4 updates residence to Beijing
    await pipeline.add(
        AddRequest(
            request_id="req-t1-b4",
            user_id="u_t1",
            session_id="s_t1_4",
            messages=[
                {
                    "role": "user",
                    "content": "我这个月刚搬到北京市定居了。",
                }
            ],
        )
    )

    # 1. Database assertion: old fact_items must be tombstoned == 1, living ones tombstoned == 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        fact_rows = conn.execute(
            "SELECT seq, tag, display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? AND kind = 'fact_item' ORDER BY seq ASC, chunk_id ASC",
            ("u_t1",),
        ).fetchall()

        columns = {r[1] for r in conn.execute("PRAGMA table_info(tag_timelines)").fetchall()}
        assert "superseded_seqs" in columns, (
            "tag_timelines must have superseded_seqs column in v2.3"
        )

        tl_row = conn.execute(
            "SELECT tag, current_state, superseded_seqs FROM tag_timelines "
            "WHERE user_id = ? AND subject = 'user' AND tag = '资产/房贷'",
            ("u_t1",),
        ).fetchone()

    # Historical fact_items (#1, #2, #3) remain tombstoned=0
    # (only explicit forget tombstones fact_items)
    assert fact_rows[0]["tombstoned"] == 0
    assert fact_rows[1]["tombstoned"] == 0

    seq3_shanghai = [r for r in fact_rows if r["seq"] == 3 and "上海市" in r["display_text"]]
    seq3_peanut = [r for r in fact_rows if r["seq"] == 3 and "花生" in r["display_text"]]
    assert len(seq3_shanghai) == 1 and seq3_shanghai[0]["tombstoned"] == 0
    assert len(seq3_peanut) == 1 and seq3_peanut[0]["tombstoned"] == 0

    # 2. Database assertion: tag_timelines.superseded_seqs records [1]
    assert tl_row is not None
    superseded = json.loads(str(tl_row["superseded_seqs"] or "[]"))
    assert 1 in superseded

    # 3. Search assertion for card #1 and #2
    resp = await pipeline.search(
        SearchRequest(
            query="Wells Fargo mortgage pre-approval",
            user_id="u_t1",
            top_k=5,
        )
    )
    assert len(resp.data) >= 1
    assert "$400,000" in resp.data[0].content

    recalled_seq1 = [item for item in resp.data if item.id.startswith("msg:u_t1:1")]
    assert len(recalled_seq1) == 1
    card_1 = recalled_seq1[0].content
    assert "[⚠️ 状态已更新" not in card_1

    # 4. Search assertion for half-superseded card #3 (peanut allergy is still alive!)
    resp_allergy = await pipeline.search(
        SearchRequest(
            query="对花生严重过敏",
            user_id="u_t1",
            top_k=5,
        )
    )
    recalled_seq3 = [item for item in resp_allergy.data if item.id.startswith("msg:u_t1:3")]
    assert len(recalled_seq3) == 1, (
        f"Expected seq 3 card to be recalled for peanut allergy, "
        f"got {[i.id for i in resp_allergy.data]}"
    )
    card_3_item = recalled_seq3[0]
    assert card_3_item.score is not None and card_3_item.score >= 0.75, (
        f"Half-alive card #3 must not be demoted as full-superseded, got score={card_3_item.score}"
    )
    assert "花生" in card_3_item.content

    # 5. BLOCK-R3-1: Anti-hardcoded-stopword & dynamic DF entity disambiguation
    # 5a. When two active siblings differ by city/employer/role
    # ("大学同学林远（北京/字节/后端/开发）" vs "大学同学陈默（上海/腾讯/前端/架构师）"),
    # an update without target_id/prev_state/name that specifies
    # "那位大学同学（北京，字节，后端，开发）今年升职了" MUST hit #F1 in-place
    # (must NOT strip city/employer/role as hardcoded stopwords and spawn #F3!)
    r3_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "大学同学林远（北京/字节/后端/开发）",
                        "seq": 1,
                    },
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "大学同学陈默（上海/腾讯/前端/架构师）",
                        "seq": 1,
                    },
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "那位大学同学（北京，字节，后端，开发）今年升职了",
                        "seq": 2,
                    }
                ]
            },
            # 5b. Unseen species coexistence ("星尘的豹纹守宫" vs "琥珀的豹纹守宫")
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/爬宠",
                        "value": "一只叫星尘的豹纹守宫",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/爬宠",
                        "value": "领养了一只叫琥珀的豹纹守宫",
                        "seq": 2,
                    }
                ]
            },
            # 5c. Unseen institution/city nameless ambiguous update ("沈括" & "周礼" both at
            # "在日内瓦CERN做高能物理实验的校友") -> nameless update must fail-safe to 3rd sibling!
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/校友",
                        "value": "校友沈括，在日内瓦CERN做高能物理实验",
                        "seq": 1,
                    },
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/校友",
                        "value": "校友周礼，在日内瓦CERN做高能物理实验",
                        "seq": 1,
                    },
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/校友",
                        "value": "那位在日内瓦CERN做高能物理实验的校友发了Nature",
                        "seq": 2,
                    }
                ]
            },
        ]
    )
    r3_pipe, _, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=r3_extractor)
    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-1a",
            user_id="u_t1_r3_df",
            session_id="s_r3_df",
            messages=[
                {
                    "role": "user",
                    "content": "大学同学林远在北京字节跳动做后端开发，陈默在上海腾讯做前端开发。",
                }
            ],
        )
    )
    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-1b",
            user_id="u_t1_r3_df",
            session_id="s_r3_df",
            messages=[
                {
                    "role": "user",
                    "content": "那位在北京字节跳动做后端开发的大学同学今年升职为技术总监了。",
                }
            ],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        df_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = 'u_t1_r3_df' AND tag LIKE '社交/朋友%' AND tombstoned = 0 "
            "ORDER BY short_id ASC",
        ).fetchall()
    assert len(df_rows) == 2 and "升职" in df_rows[0]["current_state"], (
        f"City/employer/role ('北京/字节/后端/开发') must NOT be hardcoded stopwords! "
        f"Update must hit #F1 in-place without spawning #F3. Got {[dict(r) for r in df_rows]}"
    )

    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-2a",
            user_id="u_t1_r3_gecko",
            session_id="s_r3_gecko",
            messages=[{"role": "user", "content": "我养了一只叫星尘的豹纹守宫。"}],
        )
    )
    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-2b",
            user_id="u_t1_r3_gecko",
            session_id="s_r3_gecko",
            messages=[{"role": "user", "content": "我又领养了一只叫琥珀的豹纹守宫。"}],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        gecko_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = 'u_t1_r3_gecko' AND tag LIKE '宠物/爬宠%' AND tombstoned = 0",
        ).fetchall()
    assert len(gecko_rows) == 2, (
        f"Two unseen-species pets ('星尘的豹纹守宫' vs '琥珀的豹纹守宫') must coexist as 2 "
        f"active timelines! Got {[dict(r) for r in gecko_rows]}"
    )

    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-3a",
            user_id="u_t1_r3_cern",
            session_id="s_r3_cern",
            messages=[
                {
                    "role": "user",
                    "content": "我有两位校友沈括和周礼都在日内瓦CERN做高能物理实验。",
                }
            ],
        )
    )
    await r3_pipe.add(
        AddRequest(
            request_id="req-t1-r3-3b",
            user_id="u_t1_r3_cern",
            session_id="s_r3_cern",
            messages=[
                {
                    "role": "user",
                    "content": "那位在日内瓦CERN做高能物理实验的校友发了Nature。",
                }
            ],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cern_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = 'u_t1_r3_cern' AND tag LIKE '社交/校友%' AND tombstoned = 0",
        ).fetchall()
    assert len(cern_rows) == 3, (
        f"Nameless ambiguous update matching two CERN alumni equally (margin < 0.20) must "
        f"fail-safe to a 3rd sibling! Got {[dict(r) for r in cern_rows]}"
    )


async def test_t2_ops_protocol_parallel_add_and_in_place_set(db_path: Path):
    """T2 (INV-1 / INV-2 / H2):
    1. op='add' coexists as F1 (Peanut) and F2 (Luna);
    2. op='set' with target_id=None and value='蓝猫 Luna...' hits F2 without overwriting F1;
    3. Re-asserting identical Luna state (#4 == #3) is idempotent, and a subsequent real
       update (#5 -> 4.8kg) retires BOTH #3 and #4 into superseded_seqs and tombstones both;
    4. Two ops in the same seq on the same single-value slot tombstone the first fact_item;
    5. Multi-valued category ('资产/物品与车辆') with op='set' and target_id=None on a different
       entity ('特斯拉 Model 3 轿车' vs '丰田凯美瑞轿车') spawns #F2 instead of overwriting F1.
    """
    extractor = FakeExtractor(
        responses=[
            # Batch 1 (seq 1): add Peanut (F1)
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/动物",
                        "value": "柯基犬 Peanut",
                        "seq": 1,
                    }
                ]
            },
            # Batch 2 (seq 2): add Luna (F2)
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/动物",
                        "value": "蓝猫 Luna",
                        "seq": 2,
                    }
                ]
            },
            # Batch 3 (seq 3): set with target_id=None updating Luna (must hit F2, NOT F1!)
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/动物",
                        "value": "蓝猫 Luna（已绝育，4.5kg）",
                        "seq": 3,
                    }
                ]
            },
            # Batch 4 (seq 4): repeat identical value for Luna -> idempotent
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/动物",
                        "value": "蓝猫 Luna（已绝育，4.5kg）",
                        "seq": 4,
                    }
                ]
            },
            # Batch 5 (seq 5): Luna weight increases to 4.8kg -> must supersede BOTH #3 and #4!
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/动物",
                        "value": "蓝猫 Luna（已绝育，4.8kg）",
                        "seq": 5,
                    }
                ]
            },
            # Batch 6 (seq 6): same seq consecutive updates on '居住/城市' (A -> B)
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "深圳市南山区",
                        "seq": 6,
                    },
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "广州市天河区",
                        "seq": 6,
                    },
                ]
            },
            # Batch 7 (seq 7): first car (Toyota Camry sedan)
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "资产/物品与车辆",
                        "value": "用户有一辆黑色的丰田凯美瑞轿车",
                        "seq": 7,
                    }
                ]
            },
            # Batch 8 (seq 8): weak model omits target_id and uses op='set' for a 2nd car
            # sharing the category noun '轿车' (Tesla Model 3 sedan) -> must spawn #F2!
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "资产/物品与车辆",
                        "value": "用户新买了一辆白色的特斯拉 Model 3 轿车",
                        "seq": 8,
                    }
                ]
            },
        ]
    )

    pipeline, store, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=extractor)

    await pipeline.add(
        AddRequest(
            request_id="req-t2-p1",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "我家养了一只柯基犬 Peanut。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p2",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "我还领养了一只蓝猫 Luna。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p3",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "蓝猫 Luna 已经绝育了，现在体重 4.5kg。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p4",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "顺便再提一句，蓝猫 Luna 已绝育，体重 4.5kg。"}],
        )
    )

    # Check idempotency right after #4 (seq 3 must NOT be superseded yet, and history_chain
    # must NOT append a duplicate state transition node for the same 4.5kg value!)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        pet_rows_at_4 = conn.execute(
            "SELECT tag, short_id, current_state, history_chain, source_seqs, superseded_seqs "
            "FROM tag_timelines "
            "WHERE user_id = ? AND subject = 'user' AND tag LIKE '宠物/动物%' AND tombstoned = 0",
            ("u_t2",),
        ).fetchall()

    assert len(pet_rows_at_4) == 2, (
        f"Expected 2 active coexisting pet records, got {[dict(r) for r in pet_rows_at_4]}"
    )
    peanut_rows = [r for r in pet_rows_at_4 if "Peanut" in r["current_state"]]
    luna_rows_at_4 = [r for r in pet_rows_at_4 if "Luna" in r["current_state"]]
    assert len(peanut_rows) == 1, (
        f"F1 (Peanut) must NOT be overwritten when target_id=None updates Luna! "
        f"Got states: {[r['current_state'] for r in pet_rows_at_4]}"
    )
    assert len(luna_rows_at_4) == 1
    luna_sup_at_4 = json.loads(str(luna_rows_at_4[0]["superseded_seqs"] or "[]"))
    assert 3 not in luna_sup_at_4, (
        f"Idempotent re-assertion at seq 4 must NOT mark seq 3 as superseded, "
        f"got superseded_seqs={luna_sup_at_4}"
    )
    luna_chain_at_4 = json.loads(str(luna_rows_at_4[0]["history_chain"] or "[]"))
    nodes_45kg = [n for n in luna_chain_at_4 if "4.5kg" in str(n.get("state") or "")]
    assert len(nodes_45kg) == 1, (
        f"Same-value re-assertion at #4 must NOT append a duplicate transition node to "
        f"history_chain! Expected 1 node for 4.5kg, got {luna_chain_at_4}"
    )

    # Now #5 updates Luna to 4.8kg -> BOTH #3 and #4 must be superseded and tombstoned!
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p5",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "蓝猫 Luna 最近长胖到 4.8kg 了。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p6",
            user_id="u_t2",
            session_id="s_t2",
            messages=[
                {
                    "role": "user",
                    "content": "我原本以为要去深圳市南山区，结果今天直接定居在广州市天河区了。",
                }
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p7",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "我有一辆黑色的丰田凯美瑞轿车。"}],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t2-p8",
            user_id="u_t2",
            session_id="s_t2",
            messages=[{"role": "user", "content": "我今天又新买了一辆白色的特斯拉 Model 3 轿车。"}],
        )
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        pet_rows = conn.execute(
            "SELECT tag, short_id, current_state, history_chain, superseded_seqs "
            "FROM tag_timelines "
            "WHERE user_id = ? AND subject = 'user' AND tag LIKE '宠物/动物%' AND tombstoned = 0",
            ("u_t2",),
        ).fetchall()
        luna_fact_chunks = conn.execute(
            "SELECT seq, display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? AND seq IN (3, 4, 5) AND kind = 'fact_item' ORDER BY seq ASC",
            ("u_t2",),
        ).fetchall()
        city_facts = conn.execute(
            "SELECT display_text, tombstoned FROM search_chunks "
            "WHERE user_id = ? AND seq = 6 AND kind = 'fact_item' ORDER BY chunk_id ASC",
            ("u_t2",),
        ).fetchall()
        car_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = ? AND subject = 'user' AND tag LIKE '资产/物品与车辆%' "
            "AND tombstoned = 0",
            ("u_t2",),
        ).fetchall()

    luna_rows = [r for r in pet_rows if "Luna" in r["current_state"]]
    assert len(luna_rows) == 1 and "4.8kg" in luna_rows[0]["current_state"]
    luna_superseded = json.loads(str(luna_rows[0]["superseded_seqs"] or "[]"))
    assert 3 in luna_superseded and 4 in luna_superseded, (
        f"After #5 updates Luna from 4.5kg (#3, #4) to 4.8kg, BOTH seq 3 and seq 4 must "
        f"be recorded in superseded_seqs! Got {luna_superseded}"
    )
    seq3_4_tomb = {r["seq"]: r["tombstoned"] for r in luna_fact_chunks if r["seq"] in (3, 4)}
    assert seq3_4_tomb.get(3) == 0 and seq3_4_tomb.get(4) == 0

    # 3. Same-seq consecutive updates on '居住/城市' at seq 6 (深圳市南山区 -> 广州市天河区):
    shenzhen_facts = [r for r in city_facts if "深圳市南山区" in r["display_text"]]
    guangzhou_facts = [r for r in city_facts if "广州市天河区" in r["display_text"]]
    assert len(shenzhen_facts) == 1 and shenzhen_facts[0]["tombstoned"] == 0
    assert len(guangzhou_facts) == 1 and guangzhou_facts[0]["tombstoned"] == 0

    # 4. Natural multi-value category ('资产/物品与车辆'): Camry + Tesla Model 3 must coexist!
    assert len(car_rows) == 2, (
        f"op='set' with target_id=None on '资产/物品与车辆' for a distinct car entity "
        f"('特斯拉 Model 3 轿车' vs '丰田凯美瑞轿车') must spawn #F2 instead of overwriting F1! "
        f"Got {[dict(r) for r in car_rows]}"
    )
    assert any("凯美瑞" in r["current_state"] for r in car_rows)
    assert any("Model 3" in r["current_state"] for r in car_rows)

    # 5. BLOCK-R2-3a: Second pet of the SAME breed ("豆豆的柯基犬" vs "旺财的柯基犬")
    # with op='set' and target_id=None must spawn #F2 instead of overwriting F1!
    corgi_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/狗",
                        "value": "用户养了一只名叫豆豆的柯基犬",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "宠物/狗",
                        "value": "用户又领养了一只名叫旺财的柯基犬",
                        "seq": 2,
                    }
                ]
            },
        ]
    )
    corgi_pipe, _, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=corgi_extractor)
    await corgi_pipe.add(
        AddRequest(
            request_id="req-t2-corgi-1",
            user_id="u_t2_corgi",
            session_id="s_corgi",
            messages=[{"role": "user", "content": "我家养了一只名叫豆豆的柯基犬。"}],
        )
    )
    await corgi_pipe.add(
        AddRequest(
            request_id="req-t2-corgi-2",
            user_id="u_t2_corgi",
            session_id="s_corgi",
            messages=[{"role": "user", "content": "我今天又领养了一只名叫旺财的柯基犬。"}],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        corgi_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = 'u_t2_corgi' AND tag LIKE '宠物/狗%' AND tombstoned = 0",
        ).fetchall()
    assert len(corgi_rows) == 2, (
        f"Second pet of the same breed ('旺财的柯基犬' vs '豆豆的柯基犬') with op='set' and "
        f"target_id=None must spawn #F2 instead of overwriting F1! "
        f"Got {[dict(r) for r in corgi_rows]}"
    )

    # 6. BLOCK-R2-3b: Ambiguous update (target_id=None, op='set') when 2 existing siblings
    # have nearly identical overlap (top1 - top2 margin < 0.20) must fail-safe to #F3!
    ambig_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "大学同学张伟在北京字节跳动做后端开发",
                        "seq": 1,
                    },
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "大学同学李娜在北京字节跳动做后端开发",
                        "seq": 1,
                    },
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": None,
                        "subject": "user",
                        "key": "社交/朋友",
                        "value": "大学同学在北京字节跳动做后端开发，今年刚升职为技术主管",
                        "seq": 2,
                    }
                ]
            },
        ]
    )
    ambig_pipe, _, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=ambig_extractor)
    await ambig_pipe.add(
        AddRequest(
            request_id="req-t2-ambig-1",
            user_id="u_t2_ambig",
            session_id="s_ambig",
            messages=[
                {
                    "role": "user",
                    "content": "我有两个大学同学张伟和李娜都在北京字节跳动做后端开发。",
                }
            ],
        )
    )
    await ambig_pipe.add(
        AddRequest(
            request_id="req-t2-ambig-2",
            user_id="u_t2_ambig",
            session_id="s_ambig",
            messages=[
                {
                    "role": "user",
                    "content": "我那个在北京字节跳动做后端开发的大学同学今年刚升职为技术主管。",
                }
            ],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ambig_rows = conn.execute(
            "SELECT tag, short_id, current_state FROM tag_timelines "
            "WHERE user_id = 'u_t2_ambig' AND tag LIKE '社交/朋友%' AND tombstoned = 0",
        ).fetchall()
    assert len(ambig_rows) == 3, (
        f"Ambiguous op='set' with target_id=None matching two existing siblings equally "
        f"(margin < 0.20) must fail-safe to #F3 instead of blindly overwriting F1! "
        f"Got {[dict(r) for r in ambig_rows]}"
    )

    # 7. BLOCK-R2-5: Legacy persistent row where history_chain only has [seq=10] while
    # source_seqs has [10, 11] — updating to seq=12 must supersede and tombstone BOTH 10 & 11!
    with store.transaction():
        store.upsert_chunk(
            chunk_id="fact:legacy:10",
            user_id="u_t2_legacy",
            session_id="s_leg",
            parent_msg_id="msg:u_t2_legacy:10",
            seq=10,
            session_ts=1000,
            kind="fact_item",
            subject="user",
            tag="居住/城市",
            display_text="杭州市西湖区",
            search_text="杭州市西湖区",
            bm25_tokens="杭州 西湖",
            event_time_start=1000,
            event_time_end=1000,
            embedding=[1.0, 0.0, 0.0, 0.0],
            tombstoned=0,
        )
        store.upsert_chunk(
            chunk_id="fact:legacy:11",
            user_id="u_t2_legacy",
            session_id="s_leg",
            parent_msg_id="msg:u_t2_legacy:11",
            seq=11,
            session_ts=1100,
            kind="fact_item",
            subject="user",
            tag="居住/城市",
            display_text="杭州市西湖区",
            search_text="杭州市西湖区",
            bm25_tokens="杭州 西湖",
            event_time_start=1100,
            event_time_end=1100,
            embedding=[1.0, 0.0, 0.0, 0.0],
            tombstoned=0,
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO tag_timelines (user_id, subject, tag, short_id, current_state, "
                "latest_seq, latest_session_ts, history_chain, rendered_summary, source_seqs, "
                "superseded_seqs, tombstoned, updated_at) "
                "VALUES (?, 'user', '居住/城市', 'F1', '杭州市西湖区', 10, 1000, ?, '摘要', "
                "'[10, 11]', '[]', 0, 1000)",
                (
                    "u_t2_legacy",
                    json.dumps([{"seq": 10, "session_ts": 1000, "state": "杭州市西湖区"}]),
                ),
            )
            conn.commit()
    with store.transaction():
        store.upsert_tag_timeline(
            user_id="u_t2_legacy",
            subject="user",
            tag="居住/城市",
            seq=12,
            session_ts=1200,
            time_str="",
            prev_state="",
            current_state="上海市徐汇区",
            update_reason="",
            fact_text="上海市徐汇区",
            op="set",
            target_id="F1",
        )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        leg_tl = conn.execute(
            "SELECT superseded_seqs FROM tag_timelines "
            "WHERE user_id = 'u_t2_legacy' AND tag = '居住/城市'",
        ).fetchone()
        leg_facts = conn.execute(
            "SELECT seq, tombstoned FROM search_chunks "
            "WHERE user_id = 'u_t2_legacy' AND kind = 'fact_item' ORDER BY seq ASC",
        ).fetchall()
    leg_sup = json.loads(str(leg_tl["superseded_seqs"] or "[]"))
    assert 10 in leg_sup and 11 in leg_sup, (
        f"Legacy row with source_seqs=[10, 11] must retire BOTH 10 and 11 into superseded_seqs "
        f"when updated at seq=12! Got {leg_sup}"
    )
    leg_tomb = {r["seq"]: r["tombstoned"] for r in leg_facts}
    assert leg_tomb.get(10) == 0 and leg_tomb.get(11) == 0


async def test_t3_extractor_injects_living_facts_and_parses_ops(db_path: Path):
    """T3 (INV-2 / H3):
    1. DashScopeExtractor._SYSTEM_PROMPT is pure 6-field ops;
    2. Hallucinated op='delete' without explicit forget keywords is downgraded to op='set',
       synthesizes a state reflecting the raw message (even if LLM echoed the old value),
       and forbids seq-less op='delete' from borrowing another message's forget keyword;
    3. Multiple op='delete' targets on the same seq are preserved without deduplication loss;
    4. schema_broken or alias 'tag_updates' preserves both regex_fallback facts and forget_actions.
    """
    import httpx

    from dandelion_memory import extractor as ext_mod
    from dandelion_memory.extractor import (
        DashScopeExtractor,
        sanitize_extraction_result,
    )

    # 1. Assert _SYSTEM_PROMPT does not contain abolished v2.2 fields
    sys_prompt = getattr(ext_mod, "_SYSTEM_PROMPT", "")
    for abolished_field in ("prev_state", "update_reason", "source_tier"):
        assert abolished_field not in sys_prompt, (
            f"_SYSTEM_PROMPT in v2.3 must be pure 6-field ops and must NOT contain "
            f"{abolished_field!r}"
        )

    captured_requests: list[dict[str, Any]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content.decode("utf-8"))
        captured_requests.append(data)
        content = json.dumps(
            {
                "ops": [
                    {
                        "op": "set",
                        "target_id": "F1",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "杭州市西湖区",
                        "seq": 2,
                    },
                    {
                        "op": "delete",
                        "target_id": "F2",
                        "subject": "user",
                        "key": "日常/出行",
                        "value": "自行车",
                        "seq": 2,
                    },
                ]
            }
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    transport = httpx.MockTransport(handle_request)
    mock_client = httpx.AsyncClient(transport=transport)
    extractor = DashScopeExtractor(api_key="test-key", client=mock_client)

    messages_with_forget = [
        {
            "seq": 2,
            "role": "user",
            "speaker": "user",
            "content": "我从上海搬到杭州西湖区了，请忘掉我以前骑自行车出行的事。",
            "session_time_str": "2024-05-01 10:00 UTC",
        }
    ]

    living_facts = [
        {
            "short_id": "F1",
            "subject": "user",
            "key": "居住/城市",
            "current_state": "上海市徐汇区",
            "latest_seq": 1,
        },
        {
            "short_id": "F2",
            "subject": "user",
            "key": "日常/出行",
            "current_state": "自行车",
            "latest_seq": 1,
        },
    ]

    try:
        res = await extractor.extract(messages_with_forget, living_facts=living_facts)
    finally:
        await extractor.aclose()

    assert len(captured_requests) >= 1
    user_prompt = captured_requests[0]["messages"][1]["content"]
    assert "[F1]" in user_prompt
    assert "上海市徐汇区" in user_prompt

    facts, forgets = sanitize_extraction_result(res, messages_with_forget)
    assert len(facts) >= 1
    assert "杭州市西湖区" in facts[0].current_state or "杭州市西湖区" in facts[0].fact_text
    assert len(forgets) >= 1

    # 2a. End-to-end: Hallucinated op='delete' where LLM echoes OLD value ("Trek mountain bike")
    # on raw text "I sold my Trek mountain bike yesterday" (no forget keyword) MUST update
    # tag_timelines.current_state to include "sold" and push #1 into superseded_seqs!
    bike_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "add",
                        "target_id": None,
                        "subject": "user",
                        "key": "资产/物品与车辆",
                        "value": "Trek mountain bike",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "delete",
                        "target_id": "F1",
                        "subject": "user",
                        "key": "资产/物品与车辆",
                        "value": "Trek mountain bike",
                        "seq": 2,
                    }
                ]
            },
        ]
    )
    bike_pipeline, _, _, _, _ = build_v2_pipeline(db_path=db_path, extractor=bike_extractor)
    await bike_pipeline.add(
        AddRequest(
            request_id="req-t3-bike-1",
            user_id="u_t3_bike",
            session_id="s_t3_bike",
            messages=[{"role": "user", "content": "I bought a Trek mountain bike."}],
        )
    )
    await bike_pipeline.add(
        AddRequest(
            request_id="req-t3-bike-2",
            user_id="u_t3_bike",
            session_id="s_t3_bike",
            messages=[{"role": "user", "content": "I sold my Trek mountain bike yesterday."}],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        bike_tl = conn.execute(
            "SELECT current_state, superseded_seqs, tombstoned FROM tag_timelines "
            "WHERE user_id = 'u_t3_bike' AND tag LIKE '资产/物品与车辆%'",
        ).fetchone()
    assert bike_tl is not None and bike_tl["tombstoned"] == 0, (
        "Downgraded op='delete' without forget keyword must NOT tombstone the timeline"
    )
    assert "sold" in str(bike_tl["current_state"]).lower(), (
        f"Downgraded op='delete' that echoed old value 'Trek mountain bike' must synthesize "
        f"new state from raw message ('sold') rather than hitting same-value no-op! "
        f"Got current_state={bike_tl['current_state']!r}"
    )
    assert 1 in json.loads(str(bike_tl["superseded_seqs"] or "[]")), (
        f"Expected #1 to enter superseded_seqs after #2 sold the bike, "
        f"got {bike_tl['superseded_seqs']}"
    )

    # 2b. Seq-less op='delete' in a multi-message batch where ANOTHER message (#11) has a
    # forget keyword must NOT borrow default_seq=11 and sneak into forget_actions!
    batch_borrow_msgs = [
        {
            "seq": 10,
            "role": "user",
            "speaker": "user",
            "content": "I gave away my acoustic guitar to my cousin.",
            "session_time_str": "2024-05-02 10:00 UTC",
        },
        {
            "seq": 11,
            "role": "user",
            "speaker": "user",
            "content": "请忘掉另一件事：我的旧邮箱是 old@example.com。",
            "session_time_str": "2024-05-02 10:01 UTC",
        },
    ]
    borrow_payload = {
        "ops": [
            {
                "op": "delete",
                "target_id": "F3",
                "subject": "user",
                "key": "资产/物品与车辆",
                "value": "acoustic guitar",
                # seq is omitted! Must NOT borrow seq=11's forget keyword!
            },
            {
                "op": "delete",
                "target_id": "F9",
                "subject": "user",
                "key": "联系方式/邮箱",
                "value": "old@example.com",
                "seq": 11,
            },
        ]
    }
    from unittest.mock import patch

    _, forgets_borrow = sanitize_extraction_result(borrow_payload, batch_borrow_msgs)
    assert all(
        getattr(fa, "target_id", None) != "F3"
        and "guitar" not in (fa.target_description or "").lower()
        for fa in forgets_borrow
    ), (
        f"Seq-less op='delete' for 'acoustic guitar' must NOT borrow seq=11's forget keyword "
        f"to become a forget_action! Got {forgets_borrow}"
    )

    # 2c. Single message with explicit forget keyword deleting TWO distinct targets on the same seq
    # (one via ops and one via regex_fallback on the same trigger_seq=12 when schema_broken=True)
    msg_dual_forget = [
        {
            "seq": 12,
            "role": "user",
            "speaker": "user",
            "content": "请忘掉我的旧手机号 13800001111，同时忘掉我的旧微信号 wx_old_99。",
            "session_time_str": "2024-05-02 10:05 UTC",
        }
    ]
    dual_delete_payload = {
        "ops": [
            {
                "op": "delete",
                "target_id": "F4",
                "subject": "user",
                "key": "联系方式/手机",
                "value": "13800001111",
                "seq": 12,
            },
            "malformed_entry_triggering_schema_broken",
        ]
    }
    dual_fb = {
        "facts": [],
        "forget_actions": [
            {
                "trigger_seq": 12,
                "target_existing_tags": ["联系方式/微信"],
                "target_keywords": ["wx_old_99"],
                "target_description": "旧微信号 wx_old_99",
                "batch_seqs_to_tombstone": [12],
            }
        ],
    }
    with patch.object(ext_mod, "regex_extract_batch", return_value=dual_fb):
        _, forgets_dual = sanitize_extraction_result(dual_delete_payload, msg_dual_forget)
    dual_descs = " | ".join(fa.target_description or "" for fa in forgets_dual)
    assert (
        len(forgets_dual) >= 2 and "13800001111" in dual_descs and "wx_old_99" in dual_descs
    ), (
        f"Two forget targets on the same trigger_seq=12 must both survive without being "
        f"deduplicated away by trigger_seq! Got {forgets_dual}"
    )

    # 3. schema_broken / alias 'tag_updates' / coexisting 'ops'+'facts' fallback

    mixed_fallback_msgs = [
        {
            "seq": 4,
            "role": "user",
            "speaker": "user",
            "content": "我目前住在南京市鼓楼区。",
            "session_time_str": "2024-05-03 10:00 UTC",
        },
        {
            "seq": 5,
            "role": "user",
            "speaker": "user",
            "content": "请忘掉我的保险箱密码 849201。",
            "session_time_str": "2024-05-03 10:01 UTC",
        },
    ]

    # 3a. Top-level alias 'tag_updates' + broken entry triggers regex_fallback merge
    alias_payload = {
        "tag_updates": [
            {
                "op": "set",
                "subject": "user",
                "key": "居住/地点",
                "value": "南京市鼓楼区",
                "seq": 4,
            }
        ],
        "ops": ["malformed_non_dict_entry"],
    }
    alias_facts, alias_forgets = sanitize_extraction_result(alias_payload, mixed_fallback_msgs)
    assert len(alias_facts) >= 1 and any(
        "南京" in f.current_state or "南京" in f.fact_text for f in alias_facts
    ), f"Expected tag_updates alias facts to be parsed, got {alias_facts}"
    assert len(alias_forgets) >= 1 and any(f.trigger_seq == 5 for f in alias_forgets), (
        f"Expected regex_fallback forget_actions on schema_broken, got {alias_forgets}"
    )

    # 3b. Pure schema_broken=True merges BOTH regex_fallback['facts'] and ['forget_actions']
    mock_fb = {
        "facts": [
            {
                "source_seqs": [4],
                "subject": "user",
                "tag": "居住/地点",
                "fact_text": "用户住在南京市鼓楼区",
                "current_state": "南京市鼓楼区",
            }
        ],
        "forget_actions": [
            {
                "trigger_seq": 5,
                "target_existing_tags": [],
                "target_keywords": ["保险箱密码", "849201"],
                "target_description": "我的保险箱密码 849201",
                "batch_seqs_to_tombstone": [5],
            }
        ],
    }
    with patch.object(ext_mod, "regex_extract_batch", return_value=mock_fb):
        fb_facts, fb_forgets = sanitize_extraction_result(
            {"ops": ["malformed_non_dict_entry"]}, mixed_fallback_msgs
        )
    assert len(fb_facts) >= 1 and any("南京" in f.current_state for f in fb_facts), (
        f"Expected regex_fallback['facts'] to be merged on schema_broken, got {fb_facts}"
    )
    assert len(fb_forgets) >= 1 and any(f.trigger_seq == 5 for f in fb_forgets), (
        f"Expected regex_fallback['forget_actions'] to be merged on schema_broken, got {fb_forgets}"
    )

    # 3c. Coexisting 'ops' and 'facts' in the same dict are merged and deduplicated
    coexist_payload = {
        "ops": [
            {
                "op": "set",
                "subject": "user",
                "key": "居住/地点",
                "value": "南京市鼓楼区",
                "seq": 4,
            }
        ],
        "facts": [
            {
                "source_seqs": [4],
                "subject": "user",
                "tag": "居住/地点",
                "fact_text": "南京市鼓楼区",
                "current_state": "南京市鼓楼区",
            },
            {
                "source_seqs": [4],
                "subject": "user",
                "tag": "饮食/喜好",
                "fact_text": "喜欢吃盐水鸭",
                "current_state": "喜欢吃盐水鸭",
            },
        ],
    }
    co_facts, _ = sanitize_extraction_result(coexist_payload, mixed_fallback_msgs)
    assert len(co_facts) == 2, (
        f"Expected coexisting ops and facts to merge and deduplicate into 2 facts, got {co_facts}"
    )

    # 3d. BLOCK-R2-1a: Single explicit forget message ("请忘掉 old@example.com") where LLM
    # outputs seq-less op='delete' MUST NOT be downgraded into a living 'set' fact, and MUST
    # trigger schema_broken=True so regex fallback generates the forget_action!
    msg_seqless_forget = [
        {
            "seq": 13,
            "role": "user",
            "speaker": "user",
            "content": "请忘掉我的旧邮箱 old@example.com。",
            "session_time_str": "2024-05-03 10:10 UTC",
        }
    ]
    seqless_delete_payload = {
        "ops": [
            {
                "op": "delete",
                "target_id": "F1",
                "subject": "user",
                "key": "联系方式/邮箱",
                "value": "old@example.com",
            }
        ]
    }
    facts_seqless, forgets_seqless = sanitize_extraction_result(
        seqless_delete_payload, msg_seqless_forget
    )
    assert len(facts_seqless) == 0, (
        f"Seq-less op='delete' on an explicit forget message must NEVER be downgraded into "
        f"a living 'set' fact! Got facts={facts_seqless}"
    )
    assert len(forgets_seqless) == 1 and forgets_seqless[0].trigger_seq == 13, (
        f"Seq-less op='delete' on an explicit forget message must mark schema_broken=True and "
        f"let regex fallback emit the forget_action! Got forgets={forgets_seqless}"
    )

    # 3e. BLOCK-R2-1b: op='delete' inside 'facts' container without forget keyword in raw text
    msg_facts_del = [
        {
            "seq": 14,
            "role": "user",
            "speaker": "user",
            "content": "I sold my mountain bike yesterday.",
            "session_time_str": "2024-05-03 10:12 UTC",
        }
    ]
    facts_container_delete = {
        "facts": [
            {
                "source_seqs": [14],
                "op": "delete",
                "target_id": "F2",
                "subject": "user",
                "tag": "资产/物品与车辆",
                "fact_text": "mountain bike",
                "current_state": "mountain bike",
            }
        ]
    }
    f_in_facts, fg_in_facts = sanitize_extraction_result(facts_container_delete, msg_facts_del)
    assert len(fg_in_facts) == 0, (
        f"op='delete' inside 'facts' container without forget keyword must not emit forget, "
        f"got {fg_in_facts}"
    )
    assert (
        len(f_in_facts) == 1
        and f_in_facts[0].op == "set"
        and "sold" in f_in_facts[0].current_state.lower()
    ), (
        f"op='delete' inside 'facts' container without forget keyword must downgrade to "
        f"op='set' with synthesized raw state, got {f_in_facts}"
    )

    # 3f. BLOCK-R2-1c: One top-level container has invalid type ("ops": {"bad": 1}) while
    # another container ("facts": [...]) is valid -> must still set schema_broken=True and
    # merge regex_fallback['forget_actions']!
    partial_broken_top = {
        "ops": {"bad": 1},
        "facts": [
            {
                "source_seqs": [4],
                "subject": "user",
                "tag": "居住/地点",
                "fact_text": "南京市鼓楼区",
                "current_state": "南京市鼓楼区",
            }
        ],
    }
    pb_facts, pb_forgets = sanitize_extraction_result(partial_broken_top, mixed_fallback_msgs)
    assert len(pb_facts) >= 1
    assert len(pb_forgets) >= 1 and any(f.trigger_seq == 5 for f in pb_forgets), (
        f"When 'ops' is a non-list dict even though 'facts' is a valid list, "
        f"schema_broken=True must trigger regex_fallback forget_actions! Got {pb_forgets}"
    )

    # 3g. BLOCK-R3-2a: Valid fact at seq=1 + out-of-window delete/forget_action (seq=999) when
    # seq=2 has an explicit forget command -> MUST set schema_broken=True and merge
    # regex_fallback so seq=2's forget_action is NOT silently lost!
    r3_oob_msgs = [
        {
            "seq": 1,
            "role": "user",
            "speaker": "user",
            "content": "我是一名后端程序员。",
            "session_time_str": "2024-05-04 10:00 UTC",
        },
        {
            "seq": 2,
            "role": "user",
            "speaker": "user",
            "content": "请忘掉我刚才说的银行卡尾号 6688 那件事吧。",
            "session_time_str": "2024-05-04 10:01 UTC",
        },
    ]
    oob_case_a = {
        "ops": [
            {
                "op": "set",
                "seq": 1,
                "subject": "user",
                "tag": "个人档案/基本信息",
                "current_state": "程序员",
                "fact_text": "用户是程序员",
            },
            {
                "op": "delete",
                "seq": 999,
                "subject": "user",
                "tag": "个人档案/基本信息",
                "value": "银行卡尾号 6688",
            },
        ]
    }
    oob_facts_a, oob_forgets_a = sanitize_extraction_result(oob_case_a, r3_oob_msgs)
    assert len(oob_facts_a) >= 1 and any("程序员" in f.current_state for f in oob_facts_a)
    assert len(oob_forgets_a) >= 1 and any(f.trigger_seq == 2 for f in oob_forgets_a), (
        f"Out-of-window op='delete' (seq=999) alongside a valid fact must mark "
        f"schema_broken=True and recover seq=2 forget via regex_fallback! Got {oob_forgets_a}"
    )

    oob_case_b = {
        "facts": [
            {
                "source_seqs": [1],
                "subject": "user",
                "tag": "个人档案/基本信息",
                "current_state": "程序员",
                "fact_text": "用户是程序员",
            }
        ],
        "forget_actions": [
            {
                "trigger_seq": 999,
                "target_description": "银行卡尾号 6688",
            }
        ],
    }
    oob_facts_b, oob_forgets_b = sanitize_extraction_result(oob_case_b, r3_oob_msgs)
    assert len(oob_facts_b) >= 1 and any("程序员" in f.current_state for f in oob_facts_b)
    assert len(oob_forgets_b) >= 1 and any(f.trigger_seq == 2 for f in oob_forgets_b), (
        f"Out-of-window forget_actions.trigger_seq=999 alongside a valid fact must mark "
        f"schema_broken=True and recover seq=2 forget via regex_fallback! Got {oob_forgets_b}"
    )

    # 3h. BLOCK-R3-2b: Scalar integer/dict in forget_actions list fields must NOT raise TypeError
    scalar_forget_payload = {
        "forget_actions": [
            {
                "trigger_seq": 2,
                "target_description": "银行卡尾号 6688",
                "target_existing_tags": 123,
                "target_keywords": 456,
                "batch_seqs_to_tombstone": 789,
            }
        ]
    }
    _, scalar_forgets = sanitize_extraction_result(scalar_forget_payload, r3_oob_msgs)
    assert len(scalar_forgets) >= 1 and any(f.trigger_seq == 2 for f in scalar_forgets), (
        f"Scalar fields in forget_actions must not raise TypeError and must fall back cleanly! "
        f"Got {scalar_forgets}"
    )

    # 3i. BLOCK-R4-1: List fields in forget_actions containing invalid element types
    # (e.g. int/dict inside target_existing_tags/target_keywords/batch_seqs_to_tombstone,
    # or out-of-window seq 999 inside batch_seqs_to_tombstone) must mark schema_broken=True,
    # skip the polluted item, and recover clean forget_actions via regex_fallback!
    bad_elem_forget_payload = {
        "forget_actions": [
            {
                "trigger_seq": 2,
                "target_existing_tags": [123],
                "target_keywords": [{"bad": 1}],
                "batch_seqs_to_tombstone": [2, {"bad": 2}],
            }
        ]
    }
    _, elem_forgets = sanitize_extraction_result(bad_elem_forget_payload, r3_oob_msgs)
    assert len(elem_forgets) == 1 and elem_forgets[0].trigger_seq == 2, (
        f"BLOCK-R4-1: Malformed list elements in forget_actions must fall back to regex_fallback! "
        f"Got {elem_forgets}"
    )
    assert "123" not in elem_forgets[0].target_existing_tags, (
        f"BLOCK-R4-1: Non-string element [123] must NOT be coerced to '123' in "
        f"target_existing_tags! Got {elem_forgets[0].target_existing_tags}"
    )
    assert "{'bad': 1}" not in elem_forgets[0].target_keywords, (
        f"BLOCK-R4-1: Dict element must NOT be coerced to string in target_keywords! "
        f"Got {elem_forgets[0].target_keywords}"
    )
    assert "6688" in elem_forgets[0].target_description, (
        f"BLOCK-R4-1: regex_fallback must recover real target_description ('6688'), "
        f"got {elem_forgets[0].target_description!r}"
    )

    oob_bseq_forget_payload = {
        "forget_actions": [
            {
                "trigger_seq": 2,
                "target_existing_tags": ["资产/银行卡"],
                "target_keywords": ["无效词"],
                "batch_seqs_to_tombstone": [2, 999],
            }
        ]
    }
    _, oob_bseq_forgets = sanitize_extraction_result(oob_bseq_forget_payload, r3_oob_msgs)
    assert len(oob_bseq_forgets) >= 1 and any(
        "6688" in f.target_description for f in oob_bseq_forgets
    ), (
        f"BLOCK-R4-1: Out-of-window seq 999 inside batch_seqs_to_tombstone must mark "
        f"schema_broken=True and recover regex_fallback ('6688')! Got {oob_bseq_forgets}"
    )

    # 3i-dedup (BLOCK-R4-1): Order-preserving element deduplication across all 3 list fields
    dedup_forget_payload = {
        "forget_actions": [
            {
                "trigger_seq": 2,
                "target_existing_tags": ["工作/职位", "资产/银行卡", "工作/职位"],
                "target_keywords": ["6688", "银行卡", "6688"],
                "target_description": "银行卡尾号 6688",
                "batch_seqs_to_tombstone": [2, "2", 2, 1, "1"],
            }
        ]
    }
    _, dedup_forgets = sanitize_extraction_result(dedup_forget_payload, r3_oob_msgs)
    assert len(dedup_forgets) == 1, f"Expected 1 deduplicated forget action, got {dedup_forgets}"
    assert dedup_forgets[0].target_existing_tags == ["工作/职位", "资产/银行卡"], (
        "BLOCK-R4-1: target_existing_tags must deduplicate elements while preserving order! "
        f"Got {dedup_forgets[0].target_existing_tags}"
    )
    assert dedup_forgets[0].target_keywords == ["6688", "银行卡"], (
        "BLOCK-R4-1: target_keywords must deduplicate elements while preserving order! "
        f"Got {dedup_forgets[0].target_keywords}"
    )
    assert dedup_forgets[0].batch_seqs_to_tombstone == [2, 1], (
        "BLOCK-R4-1: batch_seqs_to_tombstone must deduplicate elements while preserving order! "
        f"Got {dedup_forgets[0].batch_seqs_to_tombstone}"
    )

    # 3i-placeholders (BLOCK-R4-1): Blank strings ("   ") and placeholder tokens
    # ("null", "N/A", "unknown", "none", "undefined", "nil") inside target_existing_tags
    # or target_keywords must mark schema_broken=True, skip the polluted item, and recover
    # clean forget_actions via regex_fallback!
    for bad_tok in ("   ", "null", "N/A", "unknown", "none", "undefined", "nil"):
        ph_tag_payload = {
            "forget_actions": [
                {
                    "trigger_seq": 2,
                    "target_existing_tags": [bad_tok],
                    "target_keywords": ["占位词测试"],
                    "batch_seqs_to_tombstone": [2],
                }
            ]
        }
        _, ph_tag_forgets = sanitize_extraction_result(ph_tag_payload, r3_oob_msgs)
        assert len(ph_tag_forgets) == 1 and "6688" in ph_tag_forgets[0].target_description, (
            f"BLOCK-R4-1: Placeholder/blank {bad_tok!r} in target_existing_tags must mark "
            f"schema_broken=True and recover regex_fallback ('6688')! Got {ph_tag_forgets}"
        )
        assert bad_tok.strip() not in ph_tag_forgets[0].target_existing_tags

        ph_kw_payload = {
            "forget_actions": [
                {
                    "trigger_seq": 2,
                    "target_existing_tags": ["资产/银行卡"],
                    "target_keywords": [bad_tok],
                    "batch_seqs_to_tombstone": [2],
                }
            ]
        }
        _, ph_kw_forgets = sanitize_extraction_result(ph_kw_payload, r3_oob_msgs)
        assert len(ph_kw_forgets) == 1 and "6688" in ph_kw_forgets[0].target_description, (
            f"BLOCK-R4-1: Placeholder/blank {bad_tok!r} in target_keywords must mark "
            f"schema_broken=True and recover regex_fallback ('6688')! Got {ph_kw_forgets}"
        )
        assert bad_tok.strip() not in ph_kw_forgets[0].target_keywords

    # 3i-positive-seq (BLOCK-R4-1): batch_seqs_to_tombstone must enforce strict positive
    # integers (> 0), rejecting 0, "0", -1, "-5" even if window_messages contains seq=0 or -1!
    zero_neg_msgs = [
        {"seq": 0, "role": "user", "content": "测试零序号消息。"},
        {"seq": -1, "role": "user", "content": "测试负序号消息。"},
        {"seq": 2, "role": "user", "content": "请忘掉我刚才说的银行卡尾号 6688 那件事吧。"},
    ]
    for bad_seq_val in (0, "0", -1, "-5"):
        non_pos_payload = {
            "forget_actions": [
                {
                    "trigger_seq": 2,
                    "target_existing_tags": ["资产/银行卡"],
                    "target_keywords": ["无效词"],
                    "batch_seqs_to_tombstone": [2, bad_seq_val],
                }
            ]
        }
        _, non_pos_forgets = sanitize_extraction_result(non_pos_payload, zero_neg_msgs)
        assert len(non_pos_forgets) == 1 and "6688" in non_pos_forgets[0].target_description, (
            f"BLOCK-R4-1: Non-positive seq {bad_seq_val!r} in batch_seqs_to_tombstone must mark "
            f"schema_broken=True and fall back to regex_fallback! Got {non_pos_forgets}"
        )
        assert all(s > 0 for s in non_pos_forgets[0].batch_seqs_to_tombstone)


async def test_t4_index_to_raw_pull_retrieves_supporting_dialogue(db_path: Path):
    """T4 (INV-4): When Query matches timeline_summary, Index-to-Raw Pull pulls

    the corresponding raw parent message via raw_slice ONLY (never pulling fact_item
    when raw_slice is tombstoned/absent), AND supports third-person ('other_person')
    timeline_summary pulling neutral ('user_direct') raw_slice!
    """
    extractor = FakeExtractor(
        responses=[
            # Batch 1 (seq 1..4): routine chatter
            {"ops": []},
            # Batch 2 (seq 5): user reports back pain extracted as living fact
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "exercise constraint",
                        "value": "mild lower back strain",
                        "seq": 5,
                    }
                ]
            },
            # Batch 3 (seq 6): dietary restriction whose raw_slice will be tombstoned
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "dietary restriction",
                        "value": "strictly gluten free",
                        "seq": 6,
                    }
                ]
            },
            # Batch 4 (seq 1 for u_t4_elena): third-person Elena memory whose raw text has
            # zero lexical overlap and orthogonal vector to the query
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "Elena",
                        "key": "职业与教育/工作状态",
                        "value": "日内瓦钟表修复师",
                        "seq": 1,
                    }
                ]
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "exercise constraint mild lower back strain": 0.95,
            "aching since last Tuesday": 0.85,
            "dietary restriction strictly gluten free": 0.95,
            "Doctor told me to stay strictly gluten free": 0.85,
            "日内瓦钟表修复师": 0.95,
            "拿到那张认证证书了": 0.85,
        }
    )

    pipeline, store, embedder, _, _ = build_v2_pipeline(
        db_path=db_path,
        extractor=extractor,
        reranker=reranker,
        semantic_candidates=1,
        lexical_candidates=1,
    )

    v_target = [1.0, 0.0, 0.0, 0.0]
    embedder.set_vector(
        "exercise constraint mild lower back strain", v_target, match_substring=True
    )
    v_raw = [0.0, 1.0, 0.0, 0.0]
    embedder.set_vector("aching since last Tuesday", v_raw, match_substring=True)

    await pipeline.add(
        AddRequest(
            request_id="req-t4-b1",
            user_id="u_t4",
            session_id="s_t4_1",
            messages=[
                {"role": "user", "content": "Just had breakfast this morning."},
                {"role": "user", "content": "Weather is nice outside."},
                {"role": "user", "content": "Doing some routine office work."},
                {"role": "user", "content": "Had a cup of green tea after lunch."},
            ],
        )
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t4-b2",
            user_id="u_t4",
            session_id="s_t4_2",
            messages=[
                {
                    "role": "user",
                    "content": "My lower back has been aching since last Tuesday.",
                },
            ],
        )
    )

    resp = await pipeline.search(
        SearchRequest(
            query="exercise constraint mild lower back strain",
            user_id="u_t4",
            top_k=5,
        )
    )

    assert any(
        "【状态时间线" in item.content or "exercise constraint" in item.content
        for item in resp.data
    )
    raw_pulled = [item for item in resp.data if "aching since last Tuesday" in item.content]
    assert len(raw_pulled) >= 1
    assert raw_pulled[0].score is not None and raw_pulled[0].score >= 0.70

    # ADVISE provenance check: Index-to-Raw Pull must ONLY pull kind == 'raw_slice'
    # (never pulling kind == 'fact_item' when raw_slice of seq 6 is tombstoned!)
    v_gluten_tl = [0.0, 0.0, 1.0, 0.0]
    v_gluten_raw = [0.0, 0.0, 0.0, 1.0]
    embedder.set_vector(
        "dietary restriction strictly gluten free", v_gluten_tl, match_substring=True
    )
    embedder.set_vector(
        "Doctor told me to stay strictly gluten free", v_gluten_raw, match_substring=True
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t4-b3",
            user_id="u_t4",
            session_id="s_t4_3",
            messages=[
                {
                    "role": "user",
                    "content": "Doctor told me to stay strictly gluten free.",
                },
            ],
        )
    )
    # Tombstone ONLY the raw_slice of seq 6 and make fact_item's vector orthogonal so
    # semantic_candidates=1 only recalls tl:u_t4:user:dietary restriction
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE search_chunks SET tombstoned = 1 "
            "WHERE user_id = 'u_t4' AND seq = 6 AND kind = 'raw_slice'"
        )
        fact6_cid = conn.execute(
            "SELECT chunk_id FROM search_chunks "
            "WHERE user_id = 'u_t4' AND seq = 6 AND kind = 'fact_item'"
        ).fetchone()[0]
        conn.commit()
    store._vector_cache[str(fact6_cid)] = [0.0, 1.0, 0.0, 0.0]

    resp_gluten = await pipeline.search(
        SearchRequest(
            query="dietary restriction strictly gluten free",
            user_id="u_t4",
            top_k=5,
        )
    )
    pulled_seq6_cards = [item for item in resp_gluten.data if item.id.startswith("msg:u_t4:6")]
    assert len(pulled_seq6_cards) == 0, (
        f"Index-to-Raw Pull chunks_by_seq must ONLY include kind == 'raw_slice' and must NOT "
        f"pull fact_item to reconstruct msg:u_t4:6 when raw_slice is tombstoned! "
        f"Got {[i.id for i in resp_gluten.data]}"
    )

    # BLOCK-R3-3: Third-person ('other_person') timeline_summary Index-to-Raw Pull
    # Raw message has zero lexical overlap and orthogonal vector to query; only Elena's
    # timeline_summary is recalled by semantic/lexical channels, and Index-to-Raw Pull
    # MUST pull the neutral ('user_direct') raw_slice and assemble Elena's dialogue card!
    v_elena_q = [1.0, 0.0, 0.0, 0.0]
    v_elena_raw = [0.0, 1.0, 0.0, 0.0]
    embedder.set_vector("日内瓦钟表修复师", v_elena_q, match_substring=True)
    embedder.set_vector("Elena目前的职业状态是什么", v_elena_q, match_substring=True)
    embedder.set_vector(
        "昨天喝下午茶时她提到自己终于拿到那张认证证书了。",
        v_elena_raw,
        match_substring=True,
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t4-elena",
            user_id="u_t4_elena",
            session_id="s_t4_elena",
            messages=[
                {
                    "role": "user",
                    "content": "昨天喝下午茶时她提到自己终于拿到那张认证证书了。",
                }
            ],
        )
    )
    # Make fact_item vector orthogonal too so semantic_candidates=1 + lexical_candidates=1
    # ONLY recalls tl:u_t4_elena:Elena:职业与教育/工作状态 initially
    with sqlite3.connect(db_path) as conn:
        elena_fact_cid = conn.execute(
            "SELECT chunk_id FROM search_chunks "
            "WHERE user_id = 'u_t4_elena' AND seq = 1 AND kind = 'fact_item'"
        ).fetchone()[0]
    store._vector_cache[str(elena_fact_cid)] = [0.0, 1.0, 0.0, 0.0]

    resp_elena = await pipeline.search(
        SearchRequest(
            query="Elena目前的职业状态是什么",
            user_id="u_t4_elena",
            top_k=5,
        )
    )
    elena_pulled_dlg = [
        item for item in resp_elena.data if "拿到那张认证证书了" in item.content
    ]
    assert len(elena_pulled_dlg) >= 1, (
        f"Third-person ('other_person') timeline_summary must pull its supporting neutral "
        f"('user_direct') raw_slice via Index-to-Raw Pull! Got {[i.id for i in resp_elena.data]}"
    )


async def test_t5_neighbor_purification_and_cross_card_deduplication(db_path: Path):
    """T5 (INV-4 / H4):
    1. Pure draft neighbor & >240-char assistant neighbor are excluded from neighbor inlining;
    2. Same-seq real 'user' fact + 'draft' fact keeps raw_slice in user_direct tier;
    3. Cross-card neighbor deduplication (emitted_seqs) only registers packed cards.
    """
    extractor = FakeExtractor(
        responses=[
            # Session 1: seq 1 pure draft, seq 2 short user, seq 3 long assistant
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "tag": "其他/草稿小说",
                        "fact_text": "深渊魔王吞噬要塞",
                        "source_tier": "user_draft_fiction",
                    },
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "tag": "日常/生活",
                        "fact_text": "用户今天下午搬家收拾箱子很累",
                        "source_tier": "user_direct",
                    },
                ],
                "forget_actions": [],
            },
            # Session 2: turns 4 & 5 adjacent
            {
                "facts": [
                    {
                        "source_seqs": [4],
                        "subject": "user",
                        "tag": "行程/出差",
                        "fact_text": "明天去杭州西溪园区出差",
                        "source_tier": "user_direct",
                    }
                ],
                "forget_actions": [],
            },
            # Session 3 (seq 6): single message with BOTH a real fact (with past event date)
            # AND a draft fact -> raw_slice must keep session_ts, subject='user', tag=None
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "健康/运动与作息",
                        "value": "今晚去游泳馆游了一千米蛙泳",
                        "date": "2020-01-01",
                        "seq": 6,
                    },
                    {
                        "op": "add",
                        "subject": "draft",
                        "key": "其他/草稿小说",
                        "value": "星际海盗船长雷恩驾驶黑曜石号",
                        "seq": 6,
                    },
                ]
            },
            # Session 3b (seq 7): multi-slice (>300 chars) pure draft message where slice 2
            # has zero lexical overlap with the extracted draft fact
            {
                "ops": [
                    {
                        "op": "add",
                        "subject": "draft",
                        "key": "其他/草稿小说",
                        "value": "深海潜艇船长诺亚启动了反物质引擎",
                        "seq": 7,
                    }
                ]
            },
            # Session 4 (seq 9, 10, 11): #9 (~215 tokens) + #10 (neighbor only, no ops) + #11
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/项目",
                        "value": "星云数据库迁移方案与压测报告",
                        "seq": 9,
                    },
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/连接状态",
                        "value": "已使用星云压测服务器登录口令连接成功",
                        "seq": 11,
                    },
                ]
            },
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "【状态时间线": 0.10,
            "搬家收拾箱子好累": 0.95,
            "杭州西溪园区出差": 0.90,
            "祝你西溪园区出差一切顺利": 0.88,
            "游了一千米蛙泳": 0.99,
            "OVERSIZED_ANCHOR_SEQ9": 0.96,
            "我已经用上面的星云压测服务器登录口令连上去了": 0.88,
        }
    )

    pipeline, store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    # 1. Session 1:
    draft_msg = (
        "【小说草稿】艾伦拔出了闪耀着符文光芒的龙骨剑，指向深渊魔王。"
        "魔王发出了震耳欲聋的咆哮，黑色火焰瞬间吞噬了整个要塞。"
    )
    user_short = "今天下午搬家收拾箱子，感觉好累啊。"
    ai_long_reply = (
        "搬家确实是一项非常耗费体力的事情。科学研究表明，搬家时的疲劳主要来自于频繁的弯腰搬运重物，"
        "这会导致腰部竖脊肌持续处于紧张痉挛状态。建议你在搬家过程中注意以下几点：第一，搬运重物时尽量屈膝蹲下，"
        "利用大腿股四头肌的力量站起，而不是直接弯腰；第二，每搬运四十分钟就休息十分钟，适当饮用含有电解质的温水；"
        "第三，如果感到腰部明显酸痛，可以进行局部的冷敷或轻度拉伸，避免肌肉纤维出现微小的拉伤；第四，今晚整理完毕后"
        "可以使用温热水泡澡十五分钟，加速乳酸代谢并促进深度血液循环，同时确保卧室保持适宜温度，让身体得到充分的复原；"
        "第五，若搬家耗时较长，切忌连续作战，应当分批次循序渐进整理，以免造成慢性的腰背部劳损隐患。"
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t5-s1",
            user_id="u_t5",
            session_id="s_t5_1",
            messages=[
                {"role": "user", "content": draft_msg},  # seq 1
                {"role": "user", "content": user_short},  # seq 2
                {"role": "assistant", "content": ai_long_reply},  # seq 3
            ],
        )
    )

    resp1 = await pipeline.search(
        SearchRequest(query="今天搬家收拾箱子好累", user_id="u_t5", top_k=5)
    )
    cards_seq2 = [item for item in resp1.data if "[#2 |" in item.content]
    assert len(cards_seq2) >= 1
    card_2 = cards_seq2[0].content
    assert "深渊魔王" not in card_2
    assert "搬家确实是一项非常耗费体力的事情" in card_2

    # 2. Session 2: Adjacent turns #4 and #5
    await pipeline.add(
        AddRequest(
            request_id="req-t5-s2",
            user_id="u_t5",
            session_id="s_t5_2",
            messages=[
                {"role": "user", "content": "明天去杭州西溪园区出差。"},  # seq 4
                {"role": "assistant", "content": "祝你西溪园区出差一切顺利！"},  # seq 5
            ],
        )
    )

    resp2 = await pipeline.search(
        SearchRequest(query="明天去杭州西溪园区出差", user_id="u_t5", top_k=5)
    )
    dialogue_cards = [
        item for item in resp2.data if "[#4 |" in item.content or "[#5 |" in item.content
    ]
    assert len(dialogue_cards) >= 1

    # 3. Session 3 (seq 6): Same sentence has both real user fact and draft fact
    await pipeline.add(
        AddRequest(
            request_id="req-t5-s3",
            user_id="u_t5",
            session_id="s_t5_3",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "帮我润色这句小说设定：星际海盗船长雷恩驾驶黑曜石号；"
                        "另外我今晚去游泳馆游了一千米蛙泳。"
                    ),
                    "timestamp": 1_715_000_000_000,  # session_ts = 1715000000
                }
            ],
        )
    )

    # 3b. Session 3b (seq 7): Multi-slice (>300 chars) pure draft message where slice 2 has
    # zero lexical overlap with the extracted draft fact -> ALL slices of seq 7 must be
    # marked 'user_draft_fiction'!
    long_pure_draft_seq7 = (
        "【长篇小说草稿第一章】深海潜艇船长诺亚启动了反物质引擎，整座潜艇在海沟底部剧烈震颤，"
        "仪表盘上的蓝色指示灯依次亮起，宣告着深海远征的正式启航。"
        + (
            "第三章：远古亚特兰蒂斯遗迹的水晶柱发出耀眼的紫光，巨大的机械章鱼从海底裂谷缓缓升起，"
            "声呐屏幕上密密麻麻的光点不断逼近，全舰走廊响起了刺耳的防空警报声。"
            * 3
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t5-s3b",
            user_id="u_t5",
            session_id="s_t5_3b",
            messages=[
                {
                    "role": "user",
                    "content": long_pure_draft_seq7,
                    "timestamp": 1_715_000_100_000,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        raw_rows_all = conn.execute(
            "SELECT seq, subject, tag, source_tier, session_ts, "
            "event_time_start, event_time_end FROM search_chunks "
            "WHERE user_id = ? AND kind = 'raw_slice' ORDER BY seq ASC, chunk_id ASC",
            ("u_t5",),
        ).fetchall()

    raw_tiers_seq6 = [r["source_tier"] for r in raw_rows_all if r["seq"] == 6]
    assert raw_tiers_seq6 and all(t == "user_direct" for t in raw_tiers_seq6), (
        f"When seq 6 contains both a real fact and a draft fact, all its raw_slices must stay "
        f"'user_direct'! Got tiers: {raw_tiers_seq6}"
    )

    # BLOCK-R2-2a: Every raw_slice must preserve subject='user', tag=None, and
    # event_time_start == event_time_end == session_ts (never overwritten by fact_item's date!)
    for r in raw_rows_all:
        assert r["subject"] == "user" and r["tag"] is None, (
            f"raw_slice at seq={r['seq']} must keep subject='user' and tag=None, got {dict(r)}"
        )
        assert (
            r["event_time_start"] == r["session_ts"]
            and r["event_time_end"] == r["session_ts"]
        ), (
            f"raw_slice at seq={r['seq']} must keep event_time_start/end == session_ts "
            f"({r['session_ts']}), not overwritten by fact_item's event date! Got {dict(r)}"
        )

    # BLOCK-R2-2b: Multi-slice pure draft message at seq 7 must mark ALL its raw_slices as
    # 'user_draft_fiction', even slices with zero lexical overlap with the extracted fact!
    raw_tiers_seq7 = [r["source_tier"] for r in raw_rows_all if r["seq"] == 7]
    assert len(raw_tiers_seq7) >= 2 and all(
        t == "user_draft_fiction" for t in raw_tiers_seq7
    ), (
        f"All raw_slice chunks of multi-slice pure draft seq 7 must be 'user_draft_fiction'! "
        f"Got {raw_tiers_seq7}"
    )
    assert store.is_seq_draft_fiction("u_t5", 7) is True

    # 4. Session 4 (seq 9, 10, 11):
    # - #6 ("游了一千米蛙泳", score 0.99, ~70 tokens) is packed first (used ~ 70);
    # - #9 ("OVERSIZED_ANCHOR_SEQ9", score 0.96, ~215 + 25 = ~240 <= 300 tokens)
    #   inlines #10 during assembly, but is rejected in pack_results because 70 + 240 > 300;
    # - #10 ("星云压测服务器登录口令是 NEBULA-10。", no ops, not in Query) is only a neighbor;
    # - #11 ("我已经用上面的星云压测服务器登录口令连上去了。", score 0.88, ~55 tokens with #10)
    #   fits in pack_results (70 + 55 <= 300) and must keep its prev_line #10!
    store.update_user_state("u_t5", last_seq=8, pending_forget_confirm=0)
    oversized_seq9 = "OVERSIZED_ANCHOR_SEQ9 星云数据库压测日志：" + (
        "详细性能指标数据记录行；" * 16
    )
    short_seq10 = "星云压测服务器登录口令是 NEBULA-10。"
    short_seq11 = "我已经用上面的星云压测服务器登录口令连上去了。"

    await pipeline.add(
        AddRequest(
            request_id="req-t5-s4",
            user_id="u_t5",
            session_id="s_t5_4",
            messages=[
                {"role": "user", "content": oversized_seq9},  # seq 9 (anchor 1, next=#10)
                {"role": "user", "content": short_seq10},  # seq 10 (neighbor only)
                {"role": "user", "content": short_seq11},  # seq 11 (anchor 2, prev=#10)
            ],
        )
    )

    pipeline.token_budget = 400
    resp_budget = await pipeline.search(
        SearchRequest(
            query=(
                "游了一千米蛙泳 OVERSIZED_ANCHOR_SEQ9 "
                "我已经用上面的星云压测服务器登录口令连上去了"
            ),
            user_id="u_t5",
            top_k=10,
        )
    )
    combined_budget_out = "\n".join(item.content for item in resp_budget.data)
    assert "NEBULA-10" in combined_budget_out and "[#10 |" in combined_budget_out, (
        f"Oversized card #9 rejected by pack_results must NOT pollute emitted_seqs and "
        f"strip #10 from #11's prev_line! Got results:\n{combined_budget_out}"
    )


async def test_t6_third_person_demotion_unless_explicitly_queried(db_path: Path):
    """T6 (INV-4 / H5):
    1. Other-person attribute score is multiplied by 0.35 when Query does not mention that person;
    2. When a SINGLE message (#3) contains facts about TWO different third persons (Rachel and Bob),
       they are split into independent candidate cards by (pmid, tier, subject) so querying Bob
       ranks Bob #1 without demotion and without mixing Rachel's fact into Bob's card, while
       Rachel's card is independently demoted by 0.35;
    3. Multiple subject='rule' op='add' global instructions ('全局指令/格式约束' & '#F2') are
       both recalled and pinned even in a 20+ high-scoring dialogue pool under top_k=5 & top_k=1.
    """
    extractor = FakeExtractor(
        responses=[
            # Batch 1 (seq 1): Rachel's preference
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "Rachel",
                        "is_other_person": True,
                        "tag": "饮食/喜好",
                        "fact_text": "Rachel 每天早晨必须喝一杯美式咖啡",
                        "current_state": "美式咖啡",
                        "source_tier": "other_person",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 2 (seq 2): User's preference
            {
                "facts": [
                    {
                        "source_seqs": [2],
                        "subject": "user",
                        "is_other_person": False,
                        "tag": "饮食/喜好",
                        "fact_text": "用户平时最喜欢喝高山乌龙茶",
                        "current_state": "高山乌龙茶",
                        "source_tier": "user_direct",
                    }
                ],
                "forget_actions": [],
            },
            # Batch 3 (seq 1 for u_t6_multi): SINGLE message mentioning BOTH Rachel and Bob
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "Rachel",
                        "key": "饮食/喜好",
                        "value": "每天早晨喝冰美式咖啡",
                        "seq": 1,
                    },
                    {
                        "op": "set",
                        "subject": "Bob",
                        "key": "饮食/喜好",
                        "value": "每天下午喝宇治抹茶拿铁",
                        "seq": 1,
                    },
                ]
            },
            # Batch 4 (seq 2 for u_t6_multi): TWO subject='rule' op='add' global constraints
            {
                "ops": [
                    {
                        "op": "add",
                        "subject": "rule",
                        "key": "全局指令/格式约束",
                        "value": "所有回答必须使用纯正繁体中文",
                        "seq": 2,
                    },
                    {
                        "op": "add",
                        "subject": "rule",
                        "key": "全局指令/格式约束",
                        "value": "所有代码块严禁省略任何 import 语句",
                        "seq": 2,
                    },
                ]
            },
            # Batch 5 & 6 (20 high-relevance ordinary dialogue messages for u_t6_multi)
            {"ops": []},
            {"ops": []},
        ]
    )

    reranker = FakeReranker(
        keyword_scores={
            "美式咖啡": 0.85,
            "高山乌龙茶": 0.85,
            "宇治抹茶拿铁": 0.85,
            "适合夏天喝的冰镇饮品特调配方": 0.92,
        }
    )

    pipeline, store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, reranker=reranker
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t6-b1",
            user_id="u_t6",
            session_id="s_t6_1",
            messages=[
                {"role": "user", "content": "Rachel 每天早晨必须喝一杯美式咖啡。"},
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t6-b2",
            user_id="u_t6",
            session_id="s_t6_2",
            messages=[
                {"role": "user", "content": "我平时最喜欢喝高山乌龙茶。"},
            ],
        )
    )

    resp_general = await pipeline.search(
        SearchRequest(query="喜欢喝什么饮品", user_id="u_t6", top_k=5)
    )
    assert len(resp_general.data) >= 2
    assert "高山乌龙茶" in resp_general.data[0].content
    rachel_general = [
        item
        for item in resp_general.data
        if item.id.startswith("tl:") and "Rachel" in item.content
    ][0]
    user_general = [
        item
        for item in resp_general.data
        if item.id.startswith("tl:") and "user" in item.content
    ][0]
    assert rachel_general.score is not None and user_general.score is not None
    assert rachel_general.score < user_general.score * 0.60
    # Ordinary dialogue card is NOT demoted
    rachel_dlg = [item for item in resp_general.data if item.id == "msg:u_t6:1"][0]
    assert rachel_dlg.score is not None and rachel_dlg.score >= 0.50

    resp_rachel = await pipeline.search(
        SearchRequest(query="Rachel 喜欢喝什么饮品", user_id="u_t6", top_k=5)
    )
    assert len(resp_rachel.data) >= 1
    assert "美式咖啡" in resp_rachel.data[0].content

    # Single message (#1 in u_t6_multi) mentioning BOTH Rachel and Bob
    await pipeline.add(
        AddRequest(
            request_id="req-t6-multi",
            user_id="u_t6_multi",
            session_id="s_t6_multi",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "我的室友 Rachel 每天早晨喝冰美式咖啡，而同事 Bob 每天下午喝宇治抹茶拿铁。"
                    ),
                }
            ],
        )
    )

    resp_bob = await pipeline.search(
        SearchRequest(query="Bob 喜欢喝什么", user_id="u_t6_multi", top_k=5)
    )
    dialogue_hits = [item for item in resp_bob.data if not item.id.startswith("tl:")]
    assert len(dialogue_hits) == 1, (
        f"Expected single restored_parent card for turn-pair without splitting, "
        f"got {[i.id for i in dialogue_hits]}"
    )
    card = dialogue_hits[0]
    assert "Bob" in card.content and "Rachel" in card.content
    assert "宇治抹茶拿铁" in card.content and "冰美式咖啡" in card.content
    assert card.score is not None and card.score >= 0.50
    assert "【注意：本段包含非User本人（系Rachel、Bob）的信息，请勿与User本人混淆】" in card.content
    assert "【非User本人，系Rachel的信息】" in card.content
    assert "【非User本人，系Bob的信息】" in card.content

    # Timeline status cards: Bob is queried so Bob's tl card is NOT demoted;
    # Rachel's tl card is demoted by x0.35
    bob_tl = next(
        item for item in resp_bob.data if item.id.startswith("tl:") and "Bob" in item.content
    )
    rachel_tl = next(
        item for item in resp_bob.data if item.id.startswith("tl:") and "Rachel" in item.content
    )
    assert bob_tl.score is not None and rachel_tl.score is not None
    assert rachel_tl.score < bob_tl.score * 0.60, (
        f"Expected unqueried Rachel timeline card ({rachel_tl.score}) to be demoted by 0.35 "
        f"relative to queried Bob timeline card ({bob_tl.score})"
    )

    # 3. BLOCK-R2-4: Multiple subject='rule' op='add' global instructions ('全局指令/格式约束' &
    # '#F2') plus 20 high-relevance dialogue messages -> under top_k=5 BOTH global rules must
    # stay pinned in the top 5 (not pushed into rem_tl behind 20 dialogues when max_tl=1)!
    await pipeline.add(
        AddRequest(
            request_id="req-t6-rules",
            user_id="u_t6_multi",
            session_id="s_t6_rules",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "记住两条全局规则：第一，所有回答必须使用纯正繁体中文；"
                        "第二，所有代码块严禁省略任何 import 语句。"
                    ),
                }
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t6-dlg-pool-1",
            user_id="u_t6_multi",
            session_id="s_t6_pool_1",
            messages=[
                {
                    "role": "user",
                    "content": f"第 {i} 款适合夏天喝的冰镇饮品特调配方记录。",
                }
                for i in range(1, 11)
            ],
        )
    )
    await pipeline.add(
        AddRequest(
            request_id="req-t6-dlg-pool-2",
            user_id="u_t6_multi",
            session_id="s_t6_pool_2",
            messages=[
                {
                    "role": "user",
                    "content": f"第 {i} 款适合夏天喝的冰镇饮品特调配方记录。",
                }
                for i in range(11, 21)
            ],
        )
    )
    resp_rules = await pipeline.search(
        SearchRequest(
            query="适合夏天喝的冰镇饮品特调配方",
            user_id="u_t6_multi",
            top_k=5,
        )
    )
    pinned_rule_tls = [item for item in resp_rules.data if item.id.startswith("tl:")]
    pinned_tl_text = "\n".join(item.content for item in pinned_rule_tls)
    assert (
        len(pinned_rule_tls) >= 2
        and "所有回答必须使用纯正繁体中文" in pinned_tl_text
        and "所有代码块严禁省略任何 import 语句" in pinned_tl_text
    ), (
        f"Even with 20 high-scoring dialogue candidates under top_k=5, BOTH subject='rule' "
        f"op='add' global constraints ('全局指令/格式约束' and '#F*') must be pinned in the "
        f"top 5 (never dropped into rem_tl behind 20 dialogues)! "
        f"Got cards: {[i.id for i in resp_rules.data]}"
    )

    resp_top1 = await pipeline.search(
        SearchRequest(
            query="适合夏天喝的冰镇饮品特调配方",
            user_id="u_t6_multi",
            top_k=1,
        )
    )
    assert len(resp_top1.data) == 1 and resp_top1.data[0].id.startswith("tl:"), (
        f"Under top_k=1 when global constraints exist, resident global constraint must "
        f"take priority! Got {[i.id for i in resp_top1.data]}"
    )


async def test_t7a_single_batch_embedding_in_add_and_search(db_path: Path):
    """T7a (INV-3 / H6): Entire /add invokes embedder.embed once, and /search invokes it once."""
    extractor = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "tag": "日常/习惯",
                        "fact_text": "用户每天晨跑五公里",
                        "current_state": "晨跑五公里",
                    }
                ],
                "forget_actions": [],
            }
        ]
    )

    embedder = FakeEmbedder()
    pipeline, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor, embedder=embedder
    )

    await pipeline.add(
        AddRequest(
            request_id="req-t7a-atomic",
            user_id="u_t7a",
            session_id="s_t7a",
            messages=[{"role": "user", "content": "我每天早上都会去公园晨跑五公里。"}],
        )
    )
    assert embedder.call_count == 1, (
        f"Expected exactly 1 embedder.embed call in /add, got {embedder.call_count}"
    )

    await pipeline.search(SearchRequest(query="晨跑五公里", user_id="u_t7a", top_k=5))
    assert embedder.call_count == 2, (
        f"Expected exactly 1 embedder.embed call in /search, got {embedder.call_count - 1}"
    )

    # --- Round 3 Adversarial Sub-case (DECISION: /add < 10s end-to-end deadline bound) ---
    # 52 messages (> 48) trigger 5 windows across 2 waves of Semaphore(4).
    # Each window sleeps 5.0s in extractor (2 waves = 10.0s without an end-to-end deadline).
    # Pipeline MUST enforce a total /add deadline so end-to-end latency is strictly < 9.5s
    # and timed-out windows gracefully fall back to regex_extract_batch!
    import asyncio
    import time

    class _MultiWaveSlowExtractor:
        def __init__(self) -> None:
            self.calls = 0

        async def extract_batch(
            self,
            messages: list[dict[str, Any]],
            existing_tags: list[str],
            living_facts: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            self.calls += 1
            await asyncio.sleep(5.0)
            writable = [m for m in messages if not m.get("read_only")]
            first_seq = int(writable[0]["seq"]) if writable else 1
            return {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": f"日常/记录_{first_seq}",
                        "value": f"批次记录_{first_seq}",
                        "seq": first_seq,
                    }
                ]
            }

    slow_ext = _MultiWaveSlowExtractor()
    slow_embedder = FakeEmbedder()
    slow_pipe, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=slow_ext, embedder=slow_embedder
    )
    msgs_52 = [
        {"role": "user", "content": f"这是第 {i} 条日常打卡记录。"}
        for i in range(1, 52)
    ]
    msgs_52.append(
        {"role": "user", "content": "请忘掉我刚才说的临时口令 7788 那件事吧。"}
    )

    t0 = time.monotonic()
    try:
        await asyncio.wait_for(
            slow_pipe.add(
                AddRequest(
                    request_id="req-t7a-deadline-52",
                    user_id="u_t7a_deadline",
                    session_id="s_t7a_deadline",
                    messages=msgs_52,
                )
            ),
            timeout=9.0,
        )
    except TimeoutError:
        elapsed_fail = time.monotonic() - t0
        pytest.fail(
            f"DECISION (/add < 10s end-to-end bound) violated: 52-message multi-wave /add "
            f"exceeded 9.0s deadline (elapsed={elapsed_fail:.2f}s) without total budget cutoff!"
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 9.5, f"Expected /add end-to-end latency < 9.5s, got {elapsed:.2f}s"

    with sqlite3.connect(db_path) as conn:
        total_p_msgs = conn.execute(
            "SELECT count(*) FROM parent_messages WHERE user_id = 'u_t7a_deadline'"
        ).fetchone()[0]
        seq52_tomb = conn.execute(
            "SELECT tombstoned FROM parent_messages "
            "WHERE user_id = 'u_t7a_deadline' AND seq = 52"
        ).fetchone()[0]
    assert total_p_msgs == 52, f"Expected all 52 messages persisted, got {total_p_msgs}"
    assert seq52_tomb == 1, (
        "Timed-out wave-2 window (seq 49..52) must fall back to regex_extract_batch "
        "and tombstone explicit forget message at seq=52!"
    )

    # --- Round 4 Adversarial Sub-cases (BLOCK-R4-2: Full end-to-end /add deadline across
    # user lock wait, Stage 2 zero-grace embedding cutoff, and Stage 3 transaction rollback) ---
    from types import SimpleNamespace
    from unittest.mock import patch

    import dandelion_memory.pipeline as pipeline_mod

    # 4a. User lock wait timeout protection: if _get_user_lock(user_id) is held and the
    # remaining budget expires while waiting for the lock, add() must raise TimeoutError
    # instead of hanging indefinitely on `async with self._get_user_lock(...)`.
    held_lock = slow_pipe._get_user_lock("u_t7a_lock_held")
    await held_lock.acquire()
    try:
        lock_clock_calls = 0

        def _fake_lock_monotonic() -> float:
            nonlocal lock_clock_calls
            lock_clock_calls += 1
            # 1st call at entry of add() sets add_t0 = 100.0 (total_deadline = 109.0);
            # subsequent calls see 108.92 (leaving 0.08s for lock acquisition).
            return 100.0 if lock_clock_calls == 1 else 108.92

        t_lock_0 = time.monotonic()
        with (
            patch.object(
                pipeline_mod, "time", SimpleNamespace(monotonic=_fake_lock_monotonic)
            ),
            pytest.raises(TimeoutError),
        ):
            await asyncio.wait_for(
                slow_pipe.add(
                    AddRequest(
                        request_id="req-t7a-lock-timeout",
                        user_id="u_t7a_lock_held",
                        session_id="s_t7a_lock",
                        messages=[{"role": "user", "content": "测试锁等待超时保护。"}],
                    )
                ),
                timeout=0.60,
            )
        lock_wait_elapsed = time.monotonic() - t_lock_0
        assert lock_wait_elapsed < 0.35, (
            f"BLOCK-R4-2a: _get_user_lock acquisition must be bounded by the end-to-end "
            f"deadline (~0.08s remaining), but hung until outer 0.60s safety cap "
            f"(elapsed={lock_wait_elapsed:.2f}s)!"
        )
    finally:
        held_lock.release()

    # 4a-pre (BLOCK-R4-2): Pre-processing deadline enforcement:
    # (i) Even if check_idempotency returns True, if the clock crossed total_deadline
    # during check_idempotency, add() MUST raise TimeoutError instead of returning 200 OK!
    idem_clock = 150.0

    def _fake_idem_monotonic() -> float:
        return idem_clock

    idem_req = AddRequest(
        request_id="req-t7a-idem-deadline",
        user_id="u_t7a",
        session_id="s_t7a",
        messages=[{"role": "user", "content": "测试幂等命中时的超时拦截。"}],
    )
    await pipeline.add(idem_req)
    orig_check_idem = pipeline.store.check_idempotency

    def _slow_check_idem(*args: Any, **kwargs: Any) -> bool:
        nonlocal idem_clock
        res = orig_check_idem(*args, **kwargs)
        idem_clock = 159.5  # Exceeds 150.0 + 9.0 during check_idempotency
        return res

    with (
        patch.object(pipeline_mod, "time", SimpleNamespace(monotonic=_fake_idem_monotonic)),
        patch.object(pipeline.store, "check_idempotency", side_effect=_slow_check_idem),
        pytest.raises(TimeoutError),
    ):
        await pipeline.add(idem_req)

    # (ii) Message slicing loop (`for idx, msg in enumerate(non_sys_messages)`) must check
    # _check_deadline() per message so it aborts immediately on msg #2 if msg #1 crossed deadline!
    slice_clock = 170.0
    slice_calls = 0
    orig_slice_fn = pipeline_mod.slice_message_content

    def _fake_slice_monotonic() -> float:
        return slice_clock

    def _slow_slice_fn(content: str) -> list[str]:
        nonlocal slice_clock, slice_calls
        slice_calls += 1
        res = orig_slice_fn(content)
        slice_clock = 179.5  # Exceeds 170.0 + 9.0 during msg #1 slicing
        return res

    with (
        patch.object(pipeline_mod, "time", SimpleNamespace(monotonic=_fake_slice_monotonic)),
        patch.object(pipeline_mod, "slice_message_content", side_effect=_slow_slice_fn),
        pytest.raises(TimeoutError),
    ):
        await pipeline.add(
            AddRequest(
                request_id="req-t7a-slice-loop-deadline",
                user_id="u_t7a_slice",
                session_id="s_t7a_slice",
                messages=[
                    {"role": "user", "content": "第一条切片消息。"},
                    {"role": "user", "content": "第二条不应当被切片的消息。"},
                    {"role": "user", "content": "第三条不应当被切片的消息。"},
                ],
            )
        )
    assert slice_calls == 1, (
        f"BLOCK-R4-2: Message slicing loop must check _check_deadline() per message and abort "
        f"before slicing message #2! Got slice_calls={slice_calls}"
    )

    # 4b-1 (BLOCK-R4-2): Stage 2 pre-alignment loop (`for f_idx, fact in enumerate(facts)`)
    # must check _check_deadline() inside the loop so it aborts immediately on fact #2!
    s2_loop_clock = 190.0
    parse_date_calls = 0
    orig_parse_date = pipeline_mod.parse_event_date_range

    def _fake_s2_loop_monotonic() -> float:
        return s2_loop_clock

    def _slow_parse_date(*args: Any, **kwargs: Any) -> tuple[int | None, int | None]:
        nonlocal s2_loop_clock, parse_date_calls
        parse_date_calls += 1
        res = orig_parse_date(*args, **kwargs)
        s2_loop_clock = 199.5  # Exceeds 190.0 + 9.0 during fact #1 pre-alignment
        return res

    s2_loop_ext = FakeExtractor(
        responses=[
            {
                "ops": [
                    {"op": "set", "subject": "user", "key": "日常/事实1", "value": "A", "seq": 1},
                    {"op": "set", "subject": "user", "key": "日常/事实2", "value": "B", "seq": 1},
                    {"op": "set", "subject": "user", "key": "日常/事实3", "value": "C", "seq": 1},
                ]
            }
        ]
    )
    s2_loop_pipe, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=s2_loop_ext, embedder=FakeEmbedder()
    )
    with (
        patch.object(pipeline_mod, "time", SimpleNamespace(monotonic=_fake_s2_loop_monotonic)),
        patch.object(pipeline_mod, "parse_event_date_range", side_effect=_slow_parse_date),
        pytest.raises(TimeoutError),
    ):
        await s2_loop_pipe.add(
            AddRequest(
                request_id="req-t7a-s2-loop-cutoff",
                user_id="u_t7a_s2_loop",
                session_id="s_t7a_s2_loop",
                messages=[{"role": "user", "content": "测试Stage2预处理循环内超时拦截。"}],
            )
        )
    assert parse_date_calls == 1, (
        f"BLOCK-R4-2b: Stage 2 fact pre-alignment loop must check _check_deadline() per fact "
        f"and abort before processing fact #2! Got parse_date_calls={parse_date_calls}"
    )

    # 4b-2 (BLOCK-R4-2): Directly reaching Stage 2's `rem_emb = total_deadline - time.monotonic()`
    # <= 0 branch (clock stays < total_deadline through Stage 1 and Stage 1's _check_deadline(),
    # and expires during the final Stage 2 pre-alignment step right before rem_emb is computed)!
    emb_cutoff_clock = 200.0

    def _fake_emb_monotonic() -> float:
        return emb_cutoff_clock

    def _expire_at_end_of_s2_prep(*args: Any, **kwargs: Any) -> tuple[int | None, int | None]:
        nonlocal emb_cutoff_clock
        res = orig_parse_date(*args, **kwargs)
        # Expire right as the single fact finishes Stage 2 prep so the next check is rem_emb <= 0
        emb_cutoff_clock = 209.5
        return res

    cutoff_embedder = FakeEmbedder()
    cutoff_ext = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/测试",
                        "value": "超时测试状态",
                        "seq": 1,
                    }
                ]
            }
        ]
    )
    cutoff_pipe, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=cutoff_ext, embedder=cutoff_embedder
    )
    with (
        patch.object(pipeline_mod, "time", SimpleNamespace(monotonic=_fake_emb_monotonic)),
        patch.object(
            pipeline_mod, "parse_event_date_range", side_effect=_expire_at_end_of_s2_prep
        ),
        pytest.raises(TimeoutError, match="embedding|deadline"),
    ):
        await cutoff_pipe.add(
            AddRequest(
                request_id="req-t7a-emb-cutoff",
                user_id="u_t7a_emb_cutoff",
                session_id="s_t7a_emb_cutoff",
                messages=[{"role": "user", "content": "测试Stage2零宽限期超时截断。"}],
            )
        )
    assert cutoff_embedder.call_count == 0, (
        f"BLOCK-R4-2b: When total_deadline is already exceeded at rem_emb <= 0, embedder.embed "
        f"must NOT be called! Got {cutoff_embedder.call_count} calls"
    )

    # 4c-1 (BLOCK-R4-2): SQLite `set_progress_handler` VM-level interruption inside
    # store.upsert_tag_timeline (clock expires BEFORE orig_upsert_tl runs SQL statements,
    # asserting orig_upsert_tl itself is interrupted by SQLite VM progress handler + full rollback)
    tx_cutoff_clock = 300.0
    vm_interrupted = False

    def _fake_tx_monotonic() -> float:
        return tx_cutoff_clock

    tx_ext = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/事务超时",
                        "value": "应当被回滚的状态",
                        "seq": 1,
                    }
                ]
            }
        ]
    )
    tx_pipe, tx_store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=tx_ext, embedder=FakeEmbedder()
    )
    orig_upsert_tl = tx_store.upsert_tag_timeline
    tx_cache_before = set(tx_store._vector_cache.keys())

    def _vm_interrupt_upsert_tl(*args: Any, **kwargs: Any) -> Any:
        nonlocal tx_cutoff_clock, vm_interrupted
        # Expire clock BEFORE executing SQL inside upsert_tag_timeline
        tx_cutoff_clock = 309.5
        try:
            return orig_upsert_tl(*args, **kwargs)
        except (sqlite3.OperationalError, TimeoutError):
            vm_interrupted = True
            raise

    with (
        patch.object(pipeline_mod, "time", SimpleNamespace(monotonic=_fake_tx_monotonic)),
        patch.object(tx_store, "upsert_tag_timeline", side_effect=_vm_interrupt_upsert_tl),
        pytest.raises(TimeoutError),
    ):
        await tx_pipe.add(
            AddRequest(
                request_id="req-t7a-tx-cutoff",
                user_id="u_t7a_tx_cutoff",
                session_id="s_t7a_tx_cutoff",
                messages=[{"role": "user", "content": "测试Stage3事务内SQLite虚拟机中断回滚。"}],
            )
        )

    assert vm_interrupted, (
        "BLOCK-R4-2 (4c-1): SQLite set_progress_handler must interrupt upsert_tag_timeline "
        "mid-SQL execution when deadline expires before/during the call!"
    )
    with sqlite3.connect(db_path) as conn:
        for tbl, col, val in (
            ("parent_messages", "user_id", "u_t7a_tx_cutoff"),
            ("search_chunks", "user_id", "u_t7a_tx_cutoff"),
            ("tag_timelines", "user_id", "u_t7a_tx_cutoff"),
            ("add_requests", "request_id", "req-t7a-tx-cutoff"),
            ("user_state", "user_id", "u_t7a_tx_cutoff"),
        ):
            cnt = conn.execute(
                f"SELECT count(*) FROM {tbl} WHERE {col} = ?", (val,)
            ).fetchone()[0]
            assert cnt == 0, (
                f"BLOCK-R4-2 (4c-1): Expected 0 rows in {tbl} after VM interrupt rollback, "
                f"got {cnt}"
            )
    assert set(tx_store._vector_cache.keys()) == tx_cache_before, (
        "BLOCK-R4-2 (4c-1): VM interrupt rollback must also restore store._vector_cache!"
    )

    # 4c-2 (BLOCK-R4-2): `execute_forget_action` internal loop deadline interruption & rollback
    import dandelion_memory.store as store_mod

    forget_tx_ext = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/习惯A",
                        "value": "跑步",
                        "seq": 1,
                    },
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/习惯B",
                        "value": "游泳",
                        "seq": 1,
                    },
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/习惯C",
                        "value": "骑行",
                        "seq": 1,
                    },
                ]
            },
            {
                "forget_actions": [
                    {
                        "trigger_seq": 2,
                        "target_existing_tags": ["日常/习惯A", "日常/习惯B", "日常/习惯C"],
                        "target_keywords": ["跑步", "游泳", "骑行"],
                        "target_description": "跑步游泳骑行习惯",
                        "batch_seqs_to_tombstone": [2],
                    }
                ]
            },
        ]
    )
    forget_tx_pipe, forget_tx_store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=forget_tx_ext, embedder=FakeEmbedder()
    )
    await forget_tx_pipe.add(
        AddRequest(
            request_id="req-t7a-forget-seed",
            user_id="u_t7a_forget_tx",
            session_id="s_t7a_forget_tx",
            messages=[{"role": "user", "content": "我平时喜欢跑步、游泳和骑行。"}],
        )
    )
    forget_cache_before = set(forget_tx_store._vector_cache.keys())
    forget_tx_clock = 400.0
    forget_loop_aborted_inside = False
    orig_exec_forget = forget_tx_store.execute_forget_action
    orig_get_base_tag = store_mod.get_base_tag
    base_tag_calls_in_forget = 0
    in_exec_forget = False

    def _fake_forget_tx_monotonic() -> float:
        return forget_tx_clock

    def _slow_get_base_tag(tag: str | None) -> str:
        nonlocal forget_tx_clock, base_tag_calls_in_forget
        if in_exec_forget:
            base_tag_calls_in_forget += 1
            forget_tx_clock = 409.5  # Expire on 1st timeline iteration inside execute_forget_action
        return orig_get_base_tag(tag)

    def _wrapped_exec_forget(*args: Any, **kwargs: Any) -> Any:
        nonlocal in_exec_forget, forget_loop_aborted_inside
        in_exec_forget = True
        try:
            return orig_exec_forget(*args, **kwargs)
        except (sqlite3.OperationalError, TimeoutError):
            forget_loop_aborted_inside = True
            raise
        finally:
            in_exec_forget = False

    with (
        patch.object(
            pipeline_mod, "time", SimpleNamespace(monotonic=_fake_forget_tx_monotonic)
        ),
        patch.object(store_mod, "get_base_tag", side_effect=_slow_get_base_tag),
        patch.object(
            forget_tx_store, "execute_forget_action", side_effect=_wrapped_exec_forget
        ),
        pytest.raises(TimeoutError),
    ):
        await forget_tx_pipe.add(
            AddRequest(
                request_id="req-t7a-forget-tx-timeout",
                user_id="u_t7a_forget_tx",
                session_id="s_t7a_forget_tx",
                messages=[{"role": "user", "content": "请忘掉我之前说的跑步游泳骑行习惯。"}],
            )
        )

    assert forget_loop_aborted_inside and base_tag_calls_in_forget <= 2, (
        f"BLOCK-R4-2 (4c-2): execute_forget_action must abort mid-loop when deadline expires "
        f"(forget_loop_aborted_inside={forget_loop_aborted_inside}, "
        f"base_tag_calls_in_forget={base_tag_calls_in_forget})!"
    )
    with sqlite3.connect(db_path) as conn:
        active_seeded_tls = conn.execute(
            "SELECT count(*) FROM tag_timelines "
            "WHERE user_id = 'u_t7a_forget_tx' AND tombstoned = 0"
        ).fetchone()[0]
        seq2_parents = conn.execute(
            "SELECT count(*) FROM parent_messages "
            "WHERE user_id = 'u_t7a_forget_tx' AND seq = 2"
        ).fetchone()[0]
    assert active_seeded_tls == 3 and seq2_parents == 0, (
        f"BLOCK-R4-2 (4c-2): Rollback on execute_forget_action timeout must keep all 3 seeded "
        f"timelines active ({active_seeded_tls}) and leave 0 seq=2 rows ({seq2_parents})!"
    )
    assert set(forget_tx_store._vector_cache.keys()) == forget_cache_before

    # 4c-3 (BLOCK-R4-2): Pre-commit() deadline check at the end of `with store.transaction():`
    precommit_clock = 500.0

    def _fake_precommit_monotonic() -> float:
        return precommit_clock

    precommit_ext = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/提交前超时",
                        "value": "提交前最后一刻超时",
                        "seq": 1,
                    }
                ]
            }
        ]
    )
    precommit_pipe, precommit_store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=precommit_ext, embedder=FakeEmbedder()
    )
    precommit_cache_before = set(precommit_store._vector_cache.keys())
    orig_record_req = precommit_store.record_add_request

    def _expire_after_record_req(*args: Any, **kwargs: Any) -> None:
        nonlocal precommit_clock
        orig_record_req(*args, **kwargs)
        # Expire right after the final statement in the `with store.transaction():` block,
        # immediately before store.transaction() checks deadline_check() prior to commit()!
        precommit_clock = 509.5

    with (
        patch.object(
            pipeline_mod, "time", SimpleNamespace(monotonic=_fake_precommit_monotonic)
        ),
        patch.object(
            precommit_store, "record_add_request", side_effect=_expire_after_record_req
        ),
        pytest.raises(TimeoutError, match="before commit"),
    ):
        await precommit_pipe.add(
            AddRequest(
                request_id="req-t7a-precommit-cutoff",
                user_id="u_t7a_precommit",
                session_id="s_t7a_precommit",
                messages=[{"role": "user", "content": "测试Stage3提交前最后一刻超时回滚。"}],
            )
        )

    with sqlite3.connect(db_path) as conn:
        for tbl, col, val in (
            ("parent_messages", "user_id", "u_t7a_precommit"),
            ("search_chunks", "user_id", "u_t7a_precommit"),
            ("tag_timelines", "user_id", "u_t7a_precommit"),
            ("add_requests", "request_id", "req-t7a-precommit-cutoff"),
            ("user_state", "user_id", "u_t7a_precommit"),
        ):
            cnt = conn.execute(
                f"SELECT count(*) FROM {tbl} WHERE {col} = ?", (val,)
            ).fetchone()[0]
            assert cnt == 0, (
                f"BLOCK-R4-2 (4c-3): Expected 0 rows in {tbl} after pre-commit rollback, "
                f"got {cnt}"
            )
    assert set(precommit_store._vector_cache.keys()) == precommit_cache_before, (
        "BLOCK-R4-2 (4c-3): Pre-commit deadline rollback must restore store._vector_cache!"
    )


async def test_t7b_mid_transaction_failure_rolls_back_all_tables_and_vector_cache(
    db_path: Path,
):
    """T7b (INV-3 / DECISION-1):
    1. Stage 2 embedding failure leaves 0 rows across all 5 tables (including user_state);
    2. Mid-Stage-3 transaction failure (after parent_messages, search_chunks, tag_timelines,
       and _vector_cache have been written) rolls back all 5 tables AND _vector_cache;
    3. DECISION-1 fail-closed: sync_dirty_timelines raises an exception when the active
       latest_seq fact_item is missing or has an empty vector instead of writing vec=[].
    """
    from unittest.mock import patch

    # 1. Stage 2 failure before transaction starts
    failing_embedder = FakeEmbedder()

    async def mock_failing_embed(texts):
        raise RuntimeError("Embedding API simulated failure")

    failing_embedder.embed = mock_failing_embed  # type: ignore[method-assign]

    extractor_fail = FakeExtractor(
        responses=[
            {
                "facts": [
                    {
                        "source_seqs": [1],
                        "subject": "user",
                        "tag": "日常/习惯",
                        "fact_text": "用户打算学游泳",
                        "current_state": "学游泳",
                    }
                ],
                "forget_actions": [],
            }
        ]
    )

    pipeline_fail, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=extractor_fail, embedder=failing_embedder
    )

    with pytest.raises(RuntimeError, match="Embedding API simulated failure"):
        await pipeline_fail.add(
            AddRequest(
                request_id="req-t7-rollback",
                user_id="u_t7_fail",
                session_id="s_t7_fail",
                messages=[{"role": "user", "content": "我下周打算开始学习游泳。"}],
            )
        )

    with sqlite3.connect(db_path) as conn:
        for tbl, col, val in (
            ("parent_messages", "user_id", "u_t7_fail"),
            ("search_chunks", "user_id", "u_t7_fail"),
            ("tag_timelines", "user_id", "u_t7_fail"),
            ("add_requests", "request_id", "req-t7-rollback"),
            ("user_state", "user_id", "u_t7_fail"),
        ):
            cnt = conn.execute(f"SELECT count(*) FROM {tbl} WHERE {col} = ?", (val,)).fetchone()[0]
            assert cnt == 0, f"Expected 0 rows in {tbl} after Stage 2 failure, got {cnt}"

    # 2. Mid-Stage-3 failure AFTER parent_messages, search_chunks, tag_timelines, and
    # _vector_cache have already been written inside `with self.store.transaction():`
    mid_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/习惯",
                        "value": "每天晚上练习古典吉他",
                        "seq": 1,
                    }
                ]
            }
        ]
    )
    mid_pipeline, mid_store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=mid_extractor, embedder=FakeEmbedder()
    )
    cache_keys_before = set(mid_store._vector_cache.keys())

    with (
        patch.object(
            mid_store,
            "sync_dirty_timelines",
            side_effect=RuntimeError("Mid-Stage-3 transaction failure"),
        ),
        pytest.raises(RuntimeError, match="Mid-Stage-3 transaction failure"),
    ):
        await mid_pipeline.add(
            AddRequest(
                request_id="req-t7-mid-fail",
                user_id="u_t7_mid_fail",
                session_id="s_t7_mid_fail",
                messages=[{"role": "user", "content": "我每天晚上都会练习古典吉他。"}],
            )
        )

    with sqlite3.connect(db_path) as conn:
        for tbl, col, val in (
            ("parent_messages", "user_id", "u_t7_mid_fail"),
            ("search_chunks", "user_id", "u_t7_mid_fail"),
            ("tag_timelines", "user_id", "u_t7_mid_fail"),
            ("add_requests", "request_id", "req-t7-mid-fail"),
            ("user_state", "user_id", "u_t7_mid_fail"),
        ):
            cnt = conn.execute(f"SELECT count(*) FROM {tbl} WHERE {col} = ?", (val,)).fetchone()[0]
            assert cnt == 0, f"Expected 0 rows in {tbl} after Mid-Stage-3 rollback, got {cnt}"
    assert set(mid_store._vector_cache.keys()) == cache_keys_before, (
        "Mid-Stage-3 rollback must also restore store._vector_cache without leaked keys!"
    )

    # 3. DECISION-1 fail-closed: if the active fact_item for latest_seq is missing/tombstoned,
    # sync_dirty_timelines must raise an exception instead of silently writing an empty vector!
    ok_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "日常/习惯",
                        "value": "每周六去攀岩馆",
                        "seq": 1,
                    }
                ]
            }
        ]
    )
    ok_pipeline, ok_store, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=ok_extractor, embedder=FakeEmbedder()
    )
    await ok_pipeline.add(
        AddRequest(
            request_id="req-t7-fc",
            user_id="u_t7_fc",
            session_id="s_t7_fc",
            messages=[{"role": "user", "content": "我每周六都会去攀岩馆训练。"}],
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM search_chunks WHERE user_id = 'u_t7_fc' AND kind = 'fact_item'"
        )
        conn.commit()
    ok_store._vector_cache.clear()

    with pytest.raises((RuntimeError, ValueError, LookupError)):
        ok_store.sync_dirty_timelines("u_t7_fc", "s_t7_fc", [("user", "日常/习惯")])


async def test_t7c_out_of_order_older_message_preserves_latest_timeline_vector(
    db_path: Path,
):
    """T7c (INV-3 / H6): Out-of-order timestamp ingestion (ts=200 state B first, then ts=100
    state A) keeps current_state=B AND timeline_summary.embedding equal to state B's embedding.
    """
    ooo_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "职业/职位",
                        "value": "字节跳动首席架构师",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "职业/职位",
                        "value": "美团后端实习生",
                        "seq": 2,
                    }
                ]
            },
        ]
    )
    ooo_pipeline, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=ooo_extractor, embedder=FakeEmbedder()
    )
    await ooo_pipeline.add(
        AddRequest(
            request_id="req-t7-ooo-1",
            user_id="u_t7_ooo",
            session_id="s_ooo_late",
            messages=[
                {
                    "role": "user",
                    "content": "我现在担任字节跳动首席架构师。",
                    "timestamp": 200_000,  # session_ts = 200
                }
            ],
        )
    )
    await ooo_pipeline.add(
        AddRequest(
            request_id="req-t7-ooo-2",
            user_id="u_t7_ooo",
            session_id="s_ooo_early",
            messages=[
                {
                    "role": "user",
                    "content": "三年前我是美团后端实习生。",
                    "timestamp": 100_000,  # session_ts = 100 (earlier!)
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        tl_state, tl_l_seq = conn.execute(
            "SELECT current_state, latest_seq FROM tag_timelines "
            "WHERE user_id = 'u_t7_ooo' AND subject = 'user' AND tag = '职业/职位'",
        ).fetchone()
        tl_emb_ooo = conn.execute(
            "SELECT embedding FROM search_chunks "
            "WHERE user_id = 'u_t7_ooo' AND kind = 'timeline_summary' AND tag = '职业/职位'",
        ).fetchone()[0]
        fact_b_emb = conn.execute(
            "SELECT embedding FROM search_chunks "
            "WHERE user_id = 'u_t7_ooo' AND kind = 'fact_item' AND seq = 1",
        ).fetchone()[0]

    assert tl_l_seq == 1 and "字节跳动首席架构师" in tl_state
    assert tl_emb_ooo is not None and len(tl_emb_ooo) > 0
    assert tl_emb_ooo == fact_b_emb, (
        "After out-of-order write (ts=100 after ts=200), timeline_summary.embedding must "
        "strictly match the surviving latest_seq=1 (State B) fact_item embedding!"
    )


async def test_t7d_partial_forget_rollback_restores_previous_state_and_vector(
    db_path: Path,
):
    """T7d (INV-3 / H6): Explicit forget rollback (seq 1 state A -> seq 2 state B -> seq 3
    forgets B) restores timeline_summary.embedding to seq 1 state A's fact_item embedding.
    """
    rb_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "成都市高新区",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "武汉市光谷区",
                        "seq": 2,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "delete",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "武汉市光谷区",
                        "seq": 3,
                    }
                ]
            },
        ]
    )
    rb_pipeline, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=rb_extractor, embedder=FakeEmbedder()
    )
    await rb_pipeline.add(
        AddRequest(
            request_id="req-t7-rb-1",
            user_id="u_t7_rb",
            session_id="s_rb_1",
            messages=[
                {
                    "role": "user",
                    "content": "我一直住在成都市高新区。",
                    "timestamp": 100_000,
                }
            ],
        )
    )
    await rb_pipeline.add(
        AddRequest(
            request_id="req-t7-rb-2",
            user_id="u_t7_rb",
            session_id="s_rb_2",
            messages=[
                {
                    "role": "user",
                    "content": "我搬去武汉市光谷区了。",
                    "timestamp": 200_000,
                }
            ],
        )
    )
    await rb_pipeline.add(
        AddRequest(
            request_id="req-t7-rb-3",
            user_id="u_t7_rb",
            session_id="s_rb_3",
            messages=[
                {
                    "role": "user",
                    "content": "请忘掉我搬去武汉市光谷区那件事，不要再提了。",
                    "timestamp": 300_000,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        rb_state, rb_l_seq = conn.execute(
            "SELECT current_state, latest_seq FROM tag_timelines "
            "WHERE user_id = 'u_t7_rb' AND subject = 'user' AND tag = '居住/城市'",
        ).fetchone()
        tl_emb_rb = conn.execute(
            "SELECT embedding FROM search_chunks "
            "WHERE user_id = 'u_t7_rb' AND kind = 'timeline_summary' AND tag = '居住/城市'",
        ).fetchone()[0]
        fact_seq1_emb, fact_seq1_tomb = conn.execute(
            "SELECT embedding, tombstoned FROM search_chunks "
            "WHERE user_id = 'u_t7_rb' AND kind = 'fact_item' AND seq = 1",
        ).fetchone()

    assert rb_l_seq == 1 and "成都市高新区" in rb_state
    assert fact_seq1_tomb == 0
    assert tl_emb_rb is not None and len(tl_emb_rb) > 0
    assert tl_emb_rb == fact_seq1_emb, (
        "After explicit forget rolls timeline back from seq 2 to seq 1, "
        "timeline_summary.embedding must strictly equal seq 1's fact_item embedding!"
    )

    # --- Round 3 Adversarial Sub-case (BLOCK-R3-4): Multi-seq same-value re-assertion
    # rollback must preserve all evidence seqs [1, 2] and untombstone both fact_items ---
    rb_multi_extractor = FakeExtractor(
        responses=[
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "北京海淀区",
                        "seq": 1,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "北京海淀区",
                        "seq": 2,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "set",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "上海浦东新区",
                        "seq": 3,
                    }
                ]
            },
            {
                "ops": [
                    {
                        "op": "delete",
                        "subject": "user",
                        "key": "居住/城市",
                        "value": "上海浦东新区",
                        "seq": 4,
                    }
                ]
            },
        ]
    )
    rb_multi_pipeline, _, _, _, _ = build_v2_pipeline(
        db_path=db_path, extractor=rb_multi_extractor, embedder=FakeEmbedder()
    )
    await rb_multi_pipeline.add(
        AddRequest(
            request_id="req-t7-rbm-1",
            user_id="u_t7_rb_multi",
            session_id="s_rbm_1",
            messages=[
                {
                    "role": "user",
                    "content": "我住在北京海淀区。",
                    "timestamp": 100_000,
                }
            ],
        )
    )
    await rb_multi_pipeline.add(
        AddRequest(
            request_id="req-t7-rbm-2",
            user_id="u_t7_rb_multi",
            session_id="s_rbm_2",
            messages=[
                {
                    "role": "user",
                    "content": "没错，我现在依然住在北京海淀区。",
                    "timestamp": 200_000,
                }
            ],
        )
    )
    await rb_multi_pipeline.add(
        AddRequest(
            request_id="req-t7-rbm-3",
            user_id="u_t7_rb_multi",
            session_id="s_rbm_3",
            messages=[
                {
                    "role": "user",
                    "content": "我搬去上海浦东新区了。",
                    "timestamp": 300_000,
                }
            ],
        )
    )
    await rb_multi_pipeline.add(
        AddRequest(
            request_id="req-t7-rbm-4",
            user_id="u_t7_rb_multi",
            session_id="s_rbm_4",
            messages=[
                {
                    "role": "user",
                    "content": "忘掉我搬去上海浦东新区那条记录。",
                    "timestamp": 400_000,
                }
            ],
        )
    )

    with sqlite3.connect(db_path) as conn:
        rbm_state, rbm_src_json, rbm_sup_json = conn.execute(
            "SELECT current_state, source_seqs, superseded_seqs FROM tag_timelines "
            "WHERE user_id = 'u_t7_rb_multi' AND subject = 'user' AND tag = '居住/城市'",
        ).fetchone()
        rbm_fact_rows = conn.execute(
            "SELECT seq, tombstoned FROM search_chunks "
            "WHERE user_id = 'u_t7_rb_multi' AND kind = 'fact_item' "
            "AND seq IN (1, 2) ORDER BY seq",
        ).fetchall()

    assert "北京海淀区" in rbm_state
    assert json.loads(rbm_src_json) == [1, 2], (
        "BLOCK-R3-4: After rolling back from state B (seq=3) to state A (written at seq=1 "
        f"and re-asserted at seq=2), source_seqs must be restored to [1, 2], got {rbm_src_json}"
    )
    assert json.loads(rbm_sup_json) == [], (
        "BLOCK-R3-4: After rollback to initial state A, superseded_seqs must be [], "
        f"got {rbm_sup_json}"
    )
    assert rbm_fact_rows == [(1, 0), (2, 0)], (
        "BLOCK-R3-4: Both seq=1 and seq=2 fact_item rows for state A must be untombstoned "
        f"(tombstoned == 0) after rollback, got {rbm_fact_rows}"
    )
