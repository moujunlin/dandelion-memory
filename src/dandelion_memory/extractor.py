from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .text import regex_extract_batch

EXTRACTOR_MSG_MAX_CHARS = 3000
EXTRACTOR_ASSISTANT_MAX_CHARS = 600

_SYSTEM_PROMPT = """你是 Dandelion Memory 的写入侧记忆治理提取器（Module B）。
请针对输入的对话消息批次，提取结构化事实（facts）与显式遗忘指令（forget_actions），
严格输出合法 JSON 对象：
{
  "facts": [
    {
      "source_seqs": [1],
      "subject": "user",
      "is_other_person": false,
      "source_tier": "user_direct",
      "tag": "一级分类/二级标签",
      "fact_text": "包含主语、状态的完整陈述句",
      "prev_state": "旧状态（若无则空字符串）",
      "current_state": "最新状态",
      "update_reason": "状态产生或变更的原因",
      "style_note": "若是文稿/设定，概括用户创作偏好（否则空字符串）",
      "event_date_start": "YYYY-MM-DD",
      "event_date_end": "YYYY-MM-DD"
    }
  ],
  "forget_actions": [
    {
      "trigger_seq": 2,
      "target_existing_tags": ["一级分类/二级标签"],
      "target_keywords": ["关键词1", "关键词2"],
      "target_description": "需要被遗忘的目标记忆描述",
      "batch_seqs_to_tombstone": [2, 3]
    }
  ]
}
规则：
1. 标准二级标签词表（优先从以下 18 个标签中选择）：
   身份/姓名、身份/年龄与生日、身份/教育背景、居住/地点、职业/工作与头衔、职业/技能与技术栈、
   社交/家庭与伴侣、社交/朋友与同事、宠物/动物、健康/医疗与过敏、健康/运动与作息、
   饮食/喜好、饮食/禁忌、爱好/娱乐与文艺、资产/物品与车辆、行程/旅行与计划、
   观点/价值观与目标、全局指令/格式约束。
   若均不匹配，才使用 `其他/<2-4字核心名词>`（禁止带形容词或动词）。
2. 来源层级 `source_tier` 枚举：
   - `"user_direct"`：用户本人的真实经历、状态、偏好或指令；
   - `"user_draft_fiction"`：用户发来让 AI 润色/改写/翻译的小说、剧本、虚构角色设定
     或文案草稿（切勿当成用户本人真实经历）；
   - `"other_person"`：属于他人（如妈妈、朋友、同事）的事实，此时 subject 写具体称呼
     且 is_other_person=true。
   注意：若用户在发虚构草稿的同一条消息中用“顺便说 / by the way / 对了”附带了真实近况，
   必须拆为两条独立 fact，分别标为 `"user_draft_fiction"` 与 `"user_direct"`。
3. 若输入中包含【前序只读上文】，它们仅供指代消解参考，严禁为只读上文的 seq 输出
   任何 facts 或 forget_actions。
4. 只有当用户原话包含明确遗忘/删除词根（忘掉/忘记/删除/别再记得/forget/delete/erase/
   remove/don't remember/never mention）时才生成 forget_actions；
   普通改口或搬家等状态变化只记录在 facts 中。
5. 请以每条消息头部的 session_time_str 作为基准时间，将消息中的昨天/前天/上周/去年/
   yesterday/last week/two days ago 等相对时间折算为 YYYY-MM-DD 填入 event_date_start/end。
6. 语言保持一致：tag 统一使用上述中文标准二级标签；但 fact_text、prev_state、current_state、
   update_reason、style_note、target_keywords、target_description 必须与原对话消息保持相同语言
   （若原对话为英文，必须用英文输出并保留原始专有名词、数值与表述，切勿翻译成中文；若原对话为中文则用中文）。
7. 精简高密提取：每个窗口最多提取 5 条最核心的个人状态、偏好、关系、关键数值或时间变更事实，
   忽略纯百科知识问答、通用建议与琐碎闲聊；每个字段简明扼要（update_reason 不超过 12 个词）。
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


@dataclass(slots=True)
class ExtractedForgetAction:
    trigger_seq: int
    target_existing_tags: list[str] = field(default_factory=list)
    target_keywords: list[str] = field(default_factory=list)
    target_description: str = ""
    batch_seqs_to_tombstone: list[int] = field(default_factory=list)


class Extractor(Protocol):
    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]: ...


class RegexExtractor:
    """Deterministic regex fallback extractor."""

    async def extract(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        _ = existing_tags
        return regex_extract_batch(messages)

    async def extract_batch(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags)

    async def __call__(
        self,
        messages: Sequence[dict[str, Any]],
        existing_tags: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags)


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
    ) -> dict[str, Any]:
        writable_msgs = [m for m in messages if not m.get("read_only")]
        if not writable_msgs or not self.api_key:
            return regex_extract_batch(writable_msgs)

        lines: list[str] = []
        if existing_tags:
            tags_str = ", ".join(f"({subj}, {tag})" for subj, tag in existing_tags)
            lines.append(f"当前已存在的 (subject, tag) 列表: [{tags_str}]")

        ro_msgs = [m for m in messages if m.get("read_only")]
        if ro_msgs:
            lines.append("【前序只读上文（仅供指代消解，严禁为其输出 facts 或 forget_actions）】：")
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
                "fact_text, prev_state, current_state, update_reason, style_note, "
                "target_keywords, and target_description in ENGLISH, preserving exact "
                "proper nouns, entities, and numbers (e.g. '$400,000', 'Wells Fargo'). "
                "Only the 'tag' field uses the canonical Chinese category.]"
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
    ) -> dict[str, Any]:
        return await self.extract(messages, existing_tags)


def sanitize_extraction_result(
    raw: Any,
    batch_messages: Sequence[dict[str, Any]],
) -> tuple[list[ExtractedFact], list[ExtractedForgetAction]]:
    """Validate LLM extraction dict item-by-item; merge regex fallback if schema is broken."""
    writable_msgs = [m for m in batch_messages if not m.get("read_only")]
    regex_fallback = regex_extract_batch(writable_msgs)
    if not isinstance(raw, dict):
        raw = regex_fallback

    raw_facts = raw.get("facts")
    raw_forgets = raw.get("forget_actions")
    schema_broken = not isinstance(raw_facts, list) or not isinstance(raw_forgets, list)

    writable_seqs = {int(m["seq"]) for m in writable_msgs if "seq" in m}
    default_seq = max(writable_seqs) if writable_seqs else 1

    facts: list[ExtractedFact] = []
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if not isinstance(item, dict):
                schema_broken = True
                continue
            fact_text = str(item.get("fact_text") or "").strip()
            current_state = str(item.get("current_state") or "").strip()
            tag = str(item.get("tag") or "").strip()
            if not fact_text and not current_state:
                schema_broken = True
                continue
            raw_seqs = item.get("source_seqs")
            if isinstance(raw_seqs, list) and raw_seqs:
                parsed_seqs = [
                    int(s) for s in raw_seqs if isinstance(s, int | str) and str(s).isdigit()
                ]
                seqs = [s for s in parsed_seqs if s in writable_seqs]
                if not seqs:
                    # Strictly discard facts attributed to read-only or out-of-window seqs
                    continue
            else:
                if not writable_seqs:
                    continue
                seqs = [default_seq]
            subject = str(item.get("subject") or "user").strip() or "user"
            is_other = bool(item.get("is_other_person", subject.casefold() != "user"))
            raw_tier = str(item.get("source_tier") or "").strip()
            if raw_tier in ("user_direct", "user_draft_fiction", "other_person"):
                source_tier = raw_tier
            elif is_other:
                source_tier = "other_person"
            else:
                source_tier = "user_direct"
            if source_tier == "other_person" and subject.casefold() != "user":
                is_other = True
            if not tag:
                tag = "常规/事实"
            if not current_state:
                current_state = fact_text
            if not fact_text:
                fact_text = current_state
            prev_state = str(item.get("prev_state") or "").strip()
            update_reason = str(item.get("update_reason") or "").strip()
            style_note = str(item.get("style_note") or "").strip()
            ev_start = item.get("event_date_start")
            ev_end = item.get("event_date_end")
            facts.append(
                ExtractedFact(
                    source_seqs=seqs,
                    subject=subject,
                    is_other_person=is_other,
                    source_tier=source_tier,
                    tag=tag,
                    fact_text=fact_text,
                    prev_state=prev_state,
                    current_state=current_state,
                    update_reason=update_reason,
                    style_note=style_note,
                    event_date_start=str(ev_start) if ev_start else None,
                    event_date_end=str(ev_end) if ev_end else None,
                )
            )

    forgets: list[ExtractedForgetAction] = []
    if isinstance(raw_forgets, list):
        for item in raw_forgets:
            if not isinstance(item, dict) or "trigger_seq" not in item:
                schema_broken = True
                continue
            try:
                trig = int(item["trigger_seq"])
            except (TypeError, ValueError):
                schema_broken = True
                continue
            if writable_seqs and trig not in writable_seqs:
                continue
            tags = [
                str(t).strip()
                for t in (item.get("target_existing_tags") or [])
                if str(t).strip()
            ]
            kws = [
                str(k).strip()
                for k in (item.get("target_keywords") or [])
                if str(k).strip()
            ]
            desc = str(item.get("target_description") or "").strip()
            b_seqs_raw = item.get("batch_seqs_to_tombstone") or [trig]
            b_seqs = [
                int(s)
                for s in b_seqs_raw
                if isinstance(s, int | str)
                and str(s).isdigit()
                and (not writable_seqs or int(s) in writable_seqs)
            ]
            if trig not in b_seqs:
                b_seqs.insert(0, trig)
            forgets.append(
                ExtractedForgetAction(
                    trigger_seq=trig,
                    target_existing_tags=tags,
                    target_keywords=kws,
                    target_description=desc,
                    batch_seqs_to_tombstone=b_seqs,
                )
            )

    if schema_broken:
        existing_trigs = {f.trigger_seq for f in forgets}
        for fb in regex_fallback.get("forget_actions", []):
            trig = int(fb["trigger_seq"])
            if writable_seqs and trig not in writable_seqs:
                continue
            if trig not in existing_trigs:
                forgets.append(
                    ExtractedForgetAction(
                        trigger_seq=trig,
                        target_existing_tags=list(fb.get("target_existing_tags") or []),
                        target_keywords=list(fb.get("target_keywords") or []),
                        target_description=str(fb.get("target_description") or ""),
                        batch_seqs_to_tombstone=list(fb.get("batch_seqs_to_tombstone") or [trig]),
                    )
                )

    return facts, forgets
