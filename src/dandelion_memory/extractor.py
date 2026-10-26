from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .text import (
    extract_forget_target,
    extract_keywords_from_target,
    has_explicit_forget_keyword,
    regex_extract_batch,
)

EXTRACTOR_MSG_MAX_CHARS = 3000
EXTRACTOR_ASSISTANT_MAX_CHARS = 600
_PLACEHOLDER_STRINGS = frozenset({"null", "n/a", "unknown", "none", "undefined", "nil"})

_SYSTEM_PROMPT = """你是 Dandelion Memory v2.3 的写入侧记忆治理提取器（Module B）。
请结合输入的【当前活跃事实账本 (living_facts)】与待提取对话消息，
输出纯 6 字段结构化状态操作指令（ops），严格输出合法 JSON 对象：
{
  "ops": [
    {
      "op": "set",
      "target_id": "F1",
      "subject": "user",
      "key": "一级分类/二级标签",
      "value": "包含核心实体与数值的最新事实状态陈述",
      "seq": 29,
      "date": "YYYY-MM-DD"
    }
  ]
}
核心规则：
1. 操作类型 `op` 判定准则（极其关键）：
   - `"set"`（单值状态创建或原地覆写）：用于互斥单值属性（如当前居住城市、当前雇主/职位、
     房贷/薪资总额、某项赛事个人最佳成绩 PB、已尝试某类餐厅的总家数、拥有某类物品的总数量、
     单次特定事件改期、某位具体人物如朋友 Rachel 或妈妈的最新状态变化、或显式声明“改成/换成/
     不再是 A 而是 B”）。若覆写【当前活跃事实账本 (living_facts)】中已有的条目，必须将
     `target_id` 填为该条目的短编号（如 `"F1"`），否则填 `null`。
   - `"add"`（多值并列事实新增）：凡属用户本人可并存的多值属性——包括多只宠物、多个爱好/娱乐、
     多项饮食喜好/禁忌、多位不同朋友/同事/家人、多项技能、多段旅行/经历、多件不同资产/物品、
     多条全局指令——当用户提到新的并列项而非更新已有项的总数或状态时，必须使用 `"op": "add"`
     且 `"target_id": null`，严禁误用 `"set"` 覆写同分类下的已有并列事实！
   - `"delete"`（显式遗忘/撤销）：仅当用户原话包含明确遗忘/删除/作废词根（忘掉/忘记/删除/
     作废/不算数/撤回/别再记得/forget/delete/erase/remove/ignore/don't remember/
     never mention）时才输出 `"op": "delete"`，并将 `target_id` 指向账本中对应的 `"F1"`
     （若无则 `null`），`value` 填入需遗忘的关键词描述；普通改口、卖掉旧物或搬家只用 `"set"`。
2. `"seq"` 字段严格对齐消息编号（极其关键）：
   - 每条输入消息头部格式为 `[#seq | speaker | time]`（如 `[#29 | user | ...]`）。
   - 每条 `op` 的 `"seq"` **必须填写陈述该事实的那条具体消息的 `#seq` 整数编号**（如 `29`）！
   - **绝对禁止**把 `"seq"` 当成 `1, 2, 3` 的数组序号，**绝对禁止**把多条不同 `#seq` 消息中的事实
     合并进同一条 `op` 后只填首条或末条 `#seq`，也**禁止**把用户在 `[#seq | user]` 中说的事实挂到
     `assistant` 消息的 `#seq` 上（仅当某项历史偏好只在 `assistant` 消息中被提及、如
     `Since you enjoy...` 时才填该 `assistant` 的 `#seq`）。
3. 标准二级标签词表 `key`（优先从以下 18 个标签中选择）：
   身份/姓名、身份/年龄与生日、身份/教育背景、居住/地点、职业/工作与头衔、职业/技能与技术栈、
   社交/家庭与伴侣、社交/朋友与同事、宠物/动物、健康/医疗与过敏、健康/运动与作息、
   饮食/喜好、饮食/禁忌、爱好/娱乐与文艺、资产/物品与车辆、行程/旅行与计划、
   观点/价值观与目标、全局指令/格式约束。
   若均不匹配，才使用 `其他/<2-4字核心名词>`（禁止带形容词或动词）。
4. 主体 `subject` 取值准则：
   - `"user"`：用户本人的真实经历、状态、偏好、日常习惯或工作职责；
   - `"rule"`：用户提出的全局指令或格式约束（`key` 设为 `"全局指令/格式约束"`）；
   - `"draft"`：用户发来让 AI 润色/改写/翻译的小说、剧本、虚构角色设定或他人信件草稿（注意：
     若草稿或润色请求中体现了用户本人的真实军衔/职位/通勤方式/家庭等背景，应同时提取
     `"subject": "user"`）；
   - `"<具体人物称呼>"`：属于他人（如妈妈、Rachel、人物A、人物B）的独立事实，`subject` 写具体称呼。
   注意：若同一条消息中既有文稿草稿又有用户真实近况，必须拆为两条独立 op，分别设
   `"subject": "draft"` 与 `"subject": "user"`；若同一条消息提及多位他人，必须按姓名拆为独立 op。
5. 若输入中包含【前序只读上文】，它们仅供指代消解参考，严禁为只读上文的 seq 输出任何 op。
6. 请以每条消息头部的 session_time_str 作为基准时间，将消息中的昨天/前天/上周/去年/
   yesterday/last week/two days ago 等相对时间折算为 YYYY-MM-DD 填入可选字段 `date`。
7. 语言保持一致：`key` 统一使用上述中文标准二级标签；但 `value` 必须与原对话消息保持相同语言
   （若原对话为英文，必须用英文输出并保留原始专有名词、数值与表述，切勿翻译成中文；若原对话为中文则用中文）。
8. 逐条高密提取：逐条检查窗口内每条消息，凡包含个人状态、偏好、关系、关键数值、习惯或时间变更事实，
   按各自 `#seq` 独立提取（每个窗口最多提取 8 条 `op`），仅忽略纯百科问答与通用科普。
"""


@dataclass(slots=True)
class ExtractedFact:
    source_seqs: list[int]
    subject: str = "user"
    is_other_person: bool = False
    source_tier: str = "user_direct"
    tag: str = "常规/事实"
    fact_text: str = ""
    prev_state: str = ""
    current_state: str = ""
    update_reason: str = ""
    style_note: str = ""
    event_date_start: str | None = None
    event_date_end: str | None = None
    op: str = "set"
    target_id: str | None = None


@dataclass(slots=True)
class ExtractedForgetAction:
    trigger_seq: int
    target_existing_tags: list[str] = field(default_factory=list)
    target_keywords: list[str] = field(default_factory=list)
    target_description: str = ""
    batch_seqs_to_tombstone: list[int] = field(default_factory=list)
    target_id: str | None = None


class Extractor(Protocol):
    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


class RegexExtractor:
    """Deterministic regex fallback extractor."""

    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _ = (existing_tags, living_facts)
        return regex_extract_batch(messages)

    async def extract_batch(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags, living_facts=living_facts)

    async def __call__(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags, living_facts=living_facts)


NoOpExtractor = RegexExtractor


class DashScopeExtractor:
    """Module B LLM extractor calling OpenAI-compatible /chat/completions with 8.0s timeout."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.5-flash",
        timeout: float = 8.0,
        max_retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._client is None
            or self._client.is_closed
            or (
                self._client_loop is not None
                and (self._client_loop is not current_loop or self._client_loop.is_closed())
            )
        ):
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._client_loop = current_loop
        elif self._client_loop is None:
            self._client_loop = current_loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            with contextlib.suppress(RuntimeError):
                await self._client.aclose()
        self._client = None
        self._client_loop = None

    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        writable_msgs = [m for m in messages if not m.get("read_only")]
        if not writable_msgs or not self.api_key:
            return regex_extract_batch(writable_msgs)

        lines: list[str] = []
        if living_facts:
            lines.append("【当前活跃事实账本 (living_facts)】：")
            for lf in living_facts:
                sid = str(lf.get("short_id") or "")
                subj = str(lf.get("subject") or "user")
                key = str(lf.get("key") or lf.get("tag") or "")
                st = str(lf.get("current_state") or "")
                l_seq = lf.get("latest_seq")
                seq_part = f" (seq=#{l_seq})" if l_seq is not None else ""
                lines.append(f"[{sid}] {subj} | {key}: {st}{seq_part}")
        if existing_tags:
            tags_str = ", ".join(f"({subj}, {tag})" for subj, tag in existing_tags)
            lines.append(f"当前已存在的 (subject, tag) 列表: [{tags_str}]")

        ro_msgs = [m for m in messages if m.get("read_only")]
        if ro_msgs:
            lines.append("【前序只读上文（仅供指代消解，严禁为其输出任何 op）】：")
            for m in ro_msgs:
                seq = m.get("seq")
                speaker = m.get("speaker") or m.get("role")
                t_str = m.get("session_time_str") or "unknown_time"
                content = str(m.get("content") or "")
                max_c = (
                    EXTRACTOR_ASSISTANT_MAX_CHARS
                    if m.get("role") == "assistant"
                    else EXTRACTOR_MSG_MAX_CHARS
                )
                if len(content) > max_c:
                    content = content[:max_c] + "...[truncated]"
                lines.append(f"[#{seq} | {speaker} | {t_str}] {content}")
            lines.append("【当前待提取窗口（仅对以下 seq 提取）】：")
        else:
            lines.append("本批次对话消息：")

        for m in writable_msgs:
            seq = m.get("seq")
            speaker = m.get("speaker") or m.get("role")
            t_str = m.get("session_time_str") or "unknown_time"
            content = str(m.get("content") or "")
            max_c = (
                EXTRACTOR_ASSISTANT_MAX_CHARS
                if m.get("role") == "assistant"
                else EXTRACTOR_MSG_MAX_CHARS
            )
            if len(content) > max_c:
                content = content[:max_c] + "...[truncated]"
            lines.append(f"[#{seq} | {speaker} | {t_str}] {content}")

        all_text = " ".join(str(m.get("content") or "") for m in writable_msgs)
        cjk_count = sum(1 for ch in all_text if "\u4e00" <= ch <= "\u9fff")
        ascii_alpha = sum(1 for ch in all_text if "a" <= ch.lower() <= "z")
        if ascii_alpha > cjk_count * 2:
            lines.append(
                "\n[IMPORTANT: Since the dialogue above is in English, you MUST write "
                "value in ENGLISH, preserving exact proper nouns, entities, and numbers "
                "(e.g. '<exact amount>', '<proper noun>'). "
                "CRITICAL FOR 'seq': Each op's 'seq' MUST be the exact integer #seq from the "
                "message header [#seq | ...] where that specific fact was stated (NEVER number "
                "ops 1, 2, 3 sequentially, NEVER merge facts from multiple #seq turns into one "
                "op, and NEVER default all ops to the first or last #seq of the window). "
                "Use op='add' (target_id=null) for parallel multi-valued facts (multiple pets, "
                "hobbies, foods, friends, trips, items) and op='set' (with target_id='F1' "
                "when updating an existing living_facts entry) for mutually exclusive "
                "single-value state overwrites, counts/totals, personal bests, or updates to "
                "a specific person's status. Only 'key' uses the Chinese category.]"
            )

        user_prompt = "\n".join(lines)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "enable_thinking": False,
        }

        for _ in range(self.max_retries + 1):
            client = self._get_client()
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                if resp.status_code == 400:
                    break
                resp.raise_for_status()
                data = resp.json()
                content_str = data["choices"][0]["message"]["content"]
                parsed = json.loads(content_str)
                if isinstance(parsed, dict):
                    return parsed
            except httpx.TimeoutException:
                # Never retry on timeout so single call bound stays <= 8.0s
                break
            except RuntimeError as exc:
                if "Event loop is closed" in str(exc):
                    self._client = None
                    self._client_loop = None
                    continue
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    break
                continue
            except Exception:
                continue

        return regex_extract_batch(writable_msgs)

    async def extract_batch(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
        living_facts: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags, living_facts=living_facts)


def _normalize_subject_and_tier(
    raw_subject: str,
    raw_tier: str,
    raw_is_other: Any,
    raw_key: str,
) -> tuple[str, bool, str, str]:
    subj = raw_subject.strip() or "user"
    subj_lower = subj.casefold()
    tier = raw_tier.strip()
    key = raw_key.strip()

    if subj_lower in ("draft", "fiction", "creative_draft") or tier == "user_draft_fiction":
        return "user", False, "user_draft_fiction", key or "其他/草稿小说"

    if subj_lower == "rule":
        if not key or key.casefold() in ("常规/事实", "rule", "规则"):
            key = "全局指令/格式约束"
        return "user", False, "user_direct", key

    is_other = bool(raw_is_other) if raw_is_other is not None else (subj_lower != "user")
    if is_other or subj_lower != "user" or tier == "other_person":
        return subj, True, "other_person", key or "常规/事实"

    return "user", False, "user_direct", key or "常规/事实"


def sanitize_extraction_result(
    raw: Any,
    batch_messages: Sequence[dict[str, Any]],
) -> tuple[list[ExtractedFact], list[ExtractedForgetAction]]:
    """Validate LLM extraction dict item-by-item; merge regex fallback if schema is broken."""
    writable_msgs = [m for m in batch_messages if not m.get("read_only")]
    regex_fallback = regex_extract_batch(writable_msgs)
    if not isinstance(raw, dict):
        raw = regex_fallback
        schema_broken = True
    else:
        schema_broken = False

    raw_ops = raw.get("ops")
    raw_tag_updates = raw.get("tag_updates")
    raw_facts = raw.get("facts")
    raw_forgets = raw.get("forget_actions")
    top_containers = (raw_ops, raw_tag_updates, raw_facts, raw_forgets)
    has_any_valid_top = any(isinstance(v, list) for v in top_containers)
    has_any_invalid_top = any(
        v is not None and not isinstance(v, list) for v in top_containers
    )
    if not has_any_valid_top or has_any_invalid_top:
        schema_broken = True

    writable_seqs: set[int] = set()
    seq_to_content: dict[int, str] = {}
    seq_to_role: dict[int, str] = {}
    for m in writable_msgs:
        if "seq" not in m:
            continue
        raw_m_seq = m["seq"]
        if isinstance(raw_m_seq, bool):
            continue
        with contextlib.suppress(TypeError, ValueError):
            m_seq = int(raw_m_seq)
            if m_seq > 0:
                writable_seqs.add(m_seq)
                seq_to_content[m_seq] = str(m.get("content") or "")
                seq_to_role[m_seq] = str(m.get("role") or "user")
    default_seq = max(writable_seqs) if writable_seqs else 1

    _align_stop = frozenset({
        "the", "and", "for", "with", "that", "this", "from", "have", "has",
        "had", "was", "were", "are", "been", "being", "into", "about", "user",
        "currently", "recently", "planning", "interested", "using", "used",
        "new", "old", "also", "some", "more", "very", "just", "like", "prefers",
        "enjoys", "wants", "working", "including", "considered", "considering",
    })

    def _content_toks(text: str) -> set[str]:
        low = text.casefold()
        toks = {
            w
            for w in re.findall(r"[a-z0-9]+(?::[0-9]+)?", low)
            if (len(w) >= 3 or any(ch.isdigit() for ch in w)) and w not in _align_stop
        }
        for cjk in re.findall(r"[\u4e00-\u9fff]{2,}", low):
            for i in range(len(cjk) - 1):
                toks.add(cjk[i : i + 2])
        return toks

    def _num_toks(text: str) -> set[str]:
        return {
            n.replace(",", "")
            for n in re.findall(r"\d+(?:[.,:]\d+)+|\b\d+\b", text.casefold())
        }

    seq_to_toks = {s: _content_toks(c) for s, c in seq_to_content.items()}
    seq_to_nums = {s: _num_toks(c) for s, c in seq_to_content.items()}

    facts: list[ExtractedFact] = []
    seen_fact_keys: set[tuple[tuple[int, ...], str, str, str]] = set()
    forgets: list[ExtractedForgetAction] = []
    seen_forget_keys: set[tuple[int, str | None, tuple[str, ...], tuple[str, ...], str]] = set()

    def _append_fact(f: ExtractedFact) -> None:
        dedup_key = (
            tuple(f.source_seqs),
            f.subject.strip().casefold(),
            f.tag.strip().casefold(),
            f.current_state.strip().casefold(),
        )
        if dedup_key not in seen_fact_keys:
            seen_fact_keys.add(dedup_key)
            facts.append(f)

    def _append_forget(fa: ExtractedForgetAction) -> None:
        f_key = (
            int(fa.trigger_seq),
            fa.target_id,
            tuple(fa.target_existing_tags),
            tuple(fa.target_keywords),
            fa.target_description.strip().casefold(),
        )
        if f_key not in seen_forget_keys:
            seen_forget_keys.add(f_key)
            forgets.append(fa)

    def _parse_canonical_items(items: Sequence[Any], *, mark_broken: bool) -> None:
        nonlocal schema_broken
        for item in items:
            if not isinstance(item, dict):
                if mark_broken:
                    schema_broken = True
                continue
            op_type = str(item.get("op") or "set").strip().lower()
            if op_type not in ("set", "add", "delete"):
                if mark_broken:
                    schema_broken = True
                continue
            target_id_raw = item.get("target_id")
            target_id = str(target_id_raw).strip() if target_id_raw else None
            raw_subject = str(item.get("subject") or "user")
            raw_key = str(item.get("key") or item.get("tag") or "")
            val = str(item.get("value") or item.get("current_state") or "").strip()
            fact_text = str(item.get("fact_text") or "").strip()

            raw_seq = item.get("seq")
            raw_seqs = item.get("source_seqs")
            seqs: list[int] = []
            has_single_explicit_seq = False
            if raw_seq is not None:
                if (
                    not isinstance(raw_seq, bool)
                    and isinstance(raw_seq, int | str)
                    and str(raw_seq).isdigit()
                ):
                    s_val = int(raw_seq)
                    if s_val in writable_seqs:
                        seqs = [s_val]
                        has_single_explicit_seq = True
                    else:
                        if mark_broken:
                            schema_broken = True
                        continue
                else:
                    if mark_broken:
                        schema_broken = True
                    continue
            elif raw_seqs is not None:
                if isinstance(raw_seqs, list) and raw_seqs:
                    parsed_seqs = [
                        int(s)
                        for s in raw_seqs
                        if not isinstance(s, bool)
                        and isinstance(s, int | str)
                        and str(s).isdigit()
                    ]
                    seqs = [s for s in parsed_seqs if s in writable_seqs]
                    if not seqs or len(seqs) < len(raw_seqs):
                        if mark_broken:
                            schema_broken = True
                        if not seqs:
                            continue
                    if len(raw_seqs) == 1 and len(seqs) == 1:
                        has_single_explicit_seq = True
                else:
                    if mark_broken:
                        schema_broken = True
                    continue
            else:
                if op_type == "delete":
                    if mark_broken:
                        schema_broken = True
                    continue
                if not writable_seqs:
                    continue
                seqs = [default_seq]

            if op_type == "delete":
                if not has_single_explicit_seq:
                    if mark_broken:
                        schema_broken = True
                    continue
                trig = seqs[0]
                msg_text = seq_to_content.get(trig, "").strip()
                if bool(msg_text) and has_explicit_forget_keyword(msg_text):
                    key_clean = raw_key.strip()
                    tags = [key_clean] if key_clean else []
                    if val and (" " in val or len(val) > 12):
                        kws = extract_keywords_from_target(val)
                        if not kws:
                            kws = [val]
                    else:
                        kws = [val] if val else []
                    msg_target = extract_forget_target(msg_text)
                    if msg_target:
                        for m_kw in extract_keywords_from_target(msg_target):
                            if m_kw not in kws:
                                kws.append(m_kw)
                    desc = f"{key_clean} {val}".strip() or str(target_id or "")
                    _append_forget(
                        ExtractedForgetAction(
                            trigger_seq=trig,
                            target_existing_tags=tags,
                            target_keywords=kws,
                            target_description=desc,
                            batch_seqs_to_tombstone=[trig],
                            target_id=target_id,
                        )
                    )
                    continue
                # Downgrade hallucinated delete (with explicit single seq but no forget keyword)
                op_type = "set"
                if msg_text:
                    if not val or (
                        val.casefold() in msg_text.casefold() and len(msg_text) > len(val)
                    ):
                        val = msg_text
                    elif msg_text.casefold() not in val.casefold():
                        val = f"{val} ({msg_text})"
                    fact_text = val

            if not val and not fact_text:
                if mark_broken:
                    schema_broken = True
                continue

            if has_single_explicit_seq and len(writable_seqs) > 1:
                s0 = seqs[0]
                v_toks = _content_toks(f"{val} {fact_text}")
                v_nums = _num_toks(f"{val} {fact_text}")
                if v_toks:
                    cur_ov = len(v_toks & seq_to_toks.get(s0, set()))
                    if cur_ov == 0 or (seq_to_role.get(s0) == "assistant" and cur_ov <= 1):
                        best_s = s0
                        best_score = (cur_ov, 1 if seq_to_role.get(s0) == "user" else 0)
                        for cand_s in sorted(writable_seqs):
                            cand_ov = len(v_toks & seq_to_toks.get(cand_s, set()))
                            cand_score = (
                                cand_ov,
                                1 if seq_to_role.get(cand_s) == "user" else 0,
                            )
                            if cand_ov >= 2 and cand_score > best_score:
                                best_score = cand_score
                                best_s = cand_s
                        if best_s != s0:
                            seqs = [best_s]
                            s0 = best_s
                    s0_toks = seq_to_toks.get(s0, set())
                    s0_nums = seq_to_nums.get(s0, set())
                    rem_toks = v_toks - s0_toks
                    rem_nums = v_nums - s0_nums
                    if rem_toks:
                        extra_seqs: list[int] = []
                        for cand_s in sorted(writable_seqs):
                            if cand_s == s0 or seq_to_role.get(cand_s) != "user":
                                continue
                            c_toks = seq_to_toks.get(cand_s, set())
                            c_nums = seq_to_nums.get(cand_s, set())
                            if (
                                rem_nums
                                and (rem_nums & c_nums)
                                and len(rem_toks & c_toks) >= 2
                            ) or len(rem_toks & c_toks) >= 3:
                                extra_seqs.append(cand_s)
                        if extra_seqs:
                            seqs = sorted(set(seqs) | set(extra_seqs))
            subject, is_other, source_tier, key = _normalize_subject_and_tier(
                raw_subject,
                str(item.get("source_tier") or ""),
                item.get("is_other_person"),
                raw_key,
            )
            if not val:
                val = fact_text
            if not fact_text:
                fact_text = val
            prev_state = str(item.get("prev_state") or "").strip()
            update_reason = str(item.get("update_reason") or "").strip()
            style_note = str(item.get("style_note") or "").strip()
            ev_start = item.get("event_date_start") or item.get("date")
            ev_end = item.get("event_date_end") or item.get("date")
            _append_fact(
                ExtractedFact(
                    source_seqs=seqs,
                    subject=subject,
                    is_other_person=is_other,
                    source_tier=source_tier,
                    tag=key,
                    fact_text=fact_text,
                    prev_state=prev_state,
                    current_state=val,
                    update_reason=update_reason,
                    style_note=style_note,
                    event_date_start=str(ev_start) if ev_start else None,
                    event_date_end=str(ev_end) if ev_end else None,
                    op=op_type,
                    target_id=target_id,
                )
            )

    for top_list in (raw_ops, raw_tag_updates, raw_facts):
        if isinstance(top_list, list):
            _parse_canonical_items(top_list, mark_broken=True)

    def _is_valid_pos_seq(s: Any) -> bool:
        if isinstance(s, bool) or not isinstance(s, int | str):
            return False
        s_str = str(s).strip()
        if not s_str.isdigit():
            return False
        val = int(s_str)
        return val > 0 and val in writable_seqs

    if isinstance(raw_forgets, list):
        for item in raw_forgets:
            if not isinstance(item, dict) or "trigger_seq" not in item:
                schema_broken = True
                continue
            raw_trig = item["trigger_seq"]
            if not _is_valid_pos_seq(raw_trig):
                schema_broken = True
                continue
            trig = int(str(raw_trig).strip())
            raw_tags_field = item.get("target_existing_tags")
            raw_kws_field = item.get("target_keywords")
            raw_bseqs_field = item.get("batch_seqs_to_tombstone")
            if any(
                f is not None and not isinstance(f, list)
                for f in (raw_tags_field, raw_kws_field, raw_bseqs_field)
            ):
                schema_broken = True
                continue
            if raw_tags_field is not None and any(
                not isinstance(t, str)
                or not t.strip()
                or t.strip().lower() in _PLACEHOLDER_STRINGS
                for t in raw_tags_field
            ):
                schema_broken = True
                continue
            if raw_kws_field is not None and any(
                not isinstance(k, str)
                or not k.strip()
                or k.strip().lower() in _PLACEHOLDER_STRINGS
                for k in raw_kws_field
            ):
                schema_broken = True
                continue
            if raw_bseqs_field is not None and any(
                not _is_valid_pos_seq(s) for s in raw_bseqs_field
            ):
                schema_broken = True
                continue
            msg_text = seq_to_content.get(trig, "").strip()
            if not (bool(msg_text) and has_explicit_forget_keyword(msg_text)):
                continue
            tags = list(
                dict.fromkeys(
                    str(t).strip()
                    for t in (raw_tags_field or [])
                    if str(t).strip() and str(t).strip().lower() not in _PLACEHOLDER_STRINGS
                )
            )
            kws = list(
                dict.fromkeys(
                    str(k).strip()
                    for k in (raw_kws_field or [])
                    if str(k).strip() and str(k).strip().lower() not in _PLACEHOLDER_STRINGS
                )
            )
            desc = str(item.get("target_description") or "").strip()
            b_seqs_raw = raw_bseqs_field or [trig]
            b_seqs = list(
                dict.fromkeys(
                    int(str(s).strip())
                    for s in b_seqs_raw
                    if _is_valid_pos_seq(s)
                )
            )
            if trig not in b_seqs:
                b_seqs.insert(0, trig)
            _append_forget(
                ExtractedForgetAction(
                    trigger_seq=trig,
                    target_existing_tags=tags,
                    target_keywords=kws,
                    target_description=desc,
                    batch_seqs_to_tombstone=b_seqs,
                )
            )

    if schema_broken or (not facts and not forgets):
        fb_facts = regex_fallback.get("facts")
        if isinstance(fb_facts, list):
            _parse_canonical_items(fb_facts, mark_broken=False)
        for fb in regex_fallback.get("forget_actions", []):
            trig = int(fb["trigger_seq"])
            if trig <= 0 or trig not in writable_seqs:
                continue
            msg_text = seq_to_content.get(trig, "").strip()
            if not (bool(msg_text) and has_explicit_forget_keyword(msg_text)):
                continue
            fb_tags = list(
                dict.fromkeys(
                    str(t).strip()
                    for t in (fb.get("target_existing_tags") or [])
                    if str(t).strip() and str(t).strip().lower() not in _PLACEHOLDER_STRINGS
                )
            )
            fb_kws = list(
                dict.fromkeys(
                    str(k).strip()
                    for k in (fb.get("target_keywords") or [])
                    if str(k).strip() and str(k).strip().lower() not in _PLACEHOLDER_STRINGS
                )
            )
            fb_bseqs = list(
                dict.fromkeys(
                    int(str(s).strip())
                    for s in (fb.get("batch_seqs_to_tombstone") or [trig])
                    if _is_valid_pos_seq(s)
                )
            )
            if trig not in fb_bseqs:
                fb_bseqs.insert(0, trig)
            _append_forget(
                ExtractedForgetAction(
                    trigger_seq=trig,
                    target_existing_tags=fb_tags,
                    target_keywords=fb_kws,
                    target_description=str(fb.get("target_description") or ""),
                    batch_seqs_to_tombstone=fb_bseqs,
                )
            )

    return facts, forgets
