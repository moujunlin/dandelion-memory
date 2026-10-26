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

_SYSTEM_PROMPT = """你是 Dandelion Memory 的写入侧记忆治理提取器（Module B）。
请针对输入的对话消息批次，提取结构化事实（facts）与显式遗忘指令（forget_actions），
严格输出合法 JSON 对象：
{
  "facts": [
    {
      "source_seqs": [1],
      "subject": "user",
      "is_other_person": false,
      "tag": "一级分类/二级标签",
      "fact_text": "包含主语、状态的完整陈述句",
      "prev_state": "旧状态（若无则空字符串）",
      "current_state": "最新状态",
      "update_reason": "状态产生或变更的原因",
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
1. 若事实属于他人（如妈妈、朋友、同事），subject 写具体称呼且 is_other_person=true。
2. 若识别到全局回复风格或代码格式要求，固定归入 tag="全局指令/格式约束"。
3. 只有当用户原话包含明确遗忘/删除词根（忘掉/忘记/删除/别再记得/forget/delete/erase/
   remove/don't remember/never mention）时才生成 forget_actions；
   普通改口或搬家等状态变化只记录在 facts 中。
4. 请以每条消息头部的 session_time_str 作为基准时间，将消息中的昨天/前天/上周/去年/
   yesterday/last week/two days ago 等相对时间折算为 YYYY-MM-DD 填入 event_date_start/end。
"""


@dataclass(slots=True)
class ExtractedFact:
    source_seqs: list[int]
    subject: str = "user"
    is_other_person: bool = False
    tag: str = "常规/事实"
    fact_text: str = ""
    prev_state: str = ""
    current_state: str = ""
    update_reason: str = ""
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
        if not messages or not self.api_key:
            return regex_extract_batch(messages)

        lines: list[str] = []
        if existing_tags:
            tags_str = ", ".join(f"({subj}, {tag})" for subj, tag in existing_tags)
            lines.append(f"当前已存在的 (subject, tag) 列表: [{tags_str}]")
        lines.append("本批次对话消息：")
        for m in messages:
            seq = m.get("seq")
            speaker = m.get("speaker") or m.get("role")
            t_str = m.get("session_time_str") or "unknown_time"
            content = str(m.get("content") or "")
            if len(content) > EXTRACTOR_MSG_MAX_CHARS:
                content = content[:EXTRACTOR_MSG_MAX_CHARS] + "...[truncated]"
            lines.append(f"[#{seq} | {speaker} | {t_str}] {content}")

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

        return regex_extract_batch(messages)

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
    regex_fallback = regex_extract_batch(batch_messages)
    if not isinstance(raw, dict):
        raw = regex_fallback

    raw_facts = raw.get("facts")
    raw_forgets = raw.get("forget_actions")
    schema_broken = not isinstance(raw_facts, list) or not isinstance(raw_forgets, list)

    valid_seqs = {int(m["seq"]) for m in batch_messages if "seq" in m}
    default_seq = max(valid_seqs) if valid_seqs else 1

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
                seqs = [int(s) for s in raw_seqs if isinstance(s, int | str) and str(s).isdigit()]
            else:
                seqs = []
            if not seqs:
                seqs = [default_seq]
            subject = str(item.get("subject") or "user").strip() or "user"
            is_other = bool(item.get("is_other_person", subject.casefold() != "user"))
            if not tag:
                tag = "常规/事实"
            if not current_state:
                current_state = fact_text
            if not fact_text:
                fact_text = current_state
            prev_state = str(item.get("prev_state") or "").strip()
            update_reason = str(item.get("update_reason") or "").strip()
            ev_start = item.get("event_date_start")
            ev_end = item.get("event_date_end")
            facts.append(
                ExtractedFact(
                    source_seqs=seqs,
                    subject=subject,
                    is_other_person=is_other,
                    tag=tag,
                    fact_text=fact_text,
                    prev_state=prev_state,
                    current_state=current_state,
                    update_reason=update_reason,
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
                int(s) for s in b_seqs_raw if isinstance(s, int | str) and str(s).isdigit()
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

    if schema_broken or not forgets:
        existing_trigs = {f.trigger_seq for f in forgets}
        for fb in regex_fallback.get("forget_actions", []):
            trig = int(fb["trigger_seq"])
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
