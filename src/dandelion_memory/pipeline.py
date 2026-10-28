from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .embedding import Embedder
from .extractor import (
    ExtractedFact,
    ExtractedForgetAction,
    Extractor,
    RegexExtractor,
    sanitize_extraction_result,
)
from .models import (
    AddRequest,
    AddResponse,
    MemoryCandidate,
    SearchItem,
    SearchRequest,
    SearchResponse,
)
from .reranker import NoOpReranker, Reranker
from .store import MemoryStore, SQLiteStore, cosine_similarity
from .text import (
    LONG_DOC_CHAR_THRESHOLD,
    _stem_en_word,
    add_temporal_hints,
    approximate_tokens,
    build_bm25_tokens,
    extract_forget_target,
    extract_keywords_from_target,
    extract_query_time_range,
    extract_speaker,
    format_session_time_str,
    get_base_tag,
    has_explicit_forget_keyword,
    infer_user_self_names,
    is_incidental_update_text,
    normalize_timestamp_to_seconds,
    parse_event_date_range,
    parse_header_timestamp_seconds,
    regex_extract_batch,
    render_write_items,
    slice_message_content,
    temporal_hints,
    tokenize_bm25,
)

RRF_K = 60.0
W_DENSE = 1.5
W_BM25 = 1.0
W_TIME = 1.2
NEIGHBOR_MAX_CHARS = 1000
TIMELINE_PIN_MIN_SCORE = 0.40
DRAFT_FALLBACK_THRESHOLD = 0.55
DRAFT_MIN_SCORE = 0.30

_HONORIFIC_PREFIXES = frozenset({
    "lt", "col", "dr", "mr", "mrs", "ms", "prof", "my", "the", "a", "an",
    "officer", "sqn", "ldr", "wg", "cdr", "wing", "commander",
})

_ANAPHORIC_EDIT_RE = re.compile(
    r"\b(make it|shorter|revise|rewrite|trim|condense|tighten|adjust|tweak|"
    r"change that|this draft|the draft|that note|the note)\b"
    r"|(改短|修改|重写|精简|调整一下|这版|上一段)",
    re.IGNORECASE,
)


_TAG_NEUTRAL_TOPIC_FALLBACK: dict[str, str] = {
    "饮食/喜好": (
        "日常饮食与备餐计划方式（meal planning, food prep scheduling, and dietary preferences）"
    ),
    "饮食/禁忌": "日常饮食禁忌与备餐安排（dietary restrictions and meal planning preferences）",
    "饮食/口味与禁忌": (
        "日常饮食与备餐计划方式（meal planning, food prep scheduling, and dietary preferences）"
    ),
    "爱好/娱乐与文艺": (
        "日常休闲与旅途娱乐活动（leisure activities, engaging travel pastimes, "
        "and weekend entertainment events）"
    ),
    "行程/旅行与计划": (
        "日常出行与旅途休闲安排（travel routines and on-the-go leisure preferences）"
    ),
    "健康/运动与作息": (
        "日常运动健身与作息方式（exercise, fitness routines, and daily rest habits）"
    ),
    "爱好/运动与健身": (
        "日常运动健身与作息方式（exercise, fitness routines, and daily rest habits）"
    ),
    "居住/地点": "日常居住与通勤方式（living location and commute preferences）",
    "生活/居住与出行": "日常居住与通勤方式（living location and commute preferences）",
    "工具与数字偏好/软件与硬件": (
        "日常软件应用与规划工具（software apps and digital planning tools）"
    ),
    "职业/工作与头衔": "日常工作与数字化工具方式（work tools and professional preferences）",
    "职业/技能与技术栈": "日常工作与数字化工具方式（work tools, apps, and planning methods）",
    "工作与学业/职业与技能": "日常工作与学习工具方式（work and study tools or methods）",
    "资产/物品与车辆": "日常随身物品与出行工具（personal items and vehicles）",
    "消费与财务/购物与预算": "日常消费与预算方式（shopping and budgeting habits）",
}


def _sanitize_neutral_topic(
    raw_topic: str,
    target_keywords: Sequence[str],
    forget_msg_content: str,
    target_tags: Sequence[str],
) -> str:
    """Strip any forbidden forgotten keywords/stems from neutral_topic and fallback to category."""
    base_tag = get_base_tag(target_tags[0]) if target_tags else ""
    fallback = _TAG_NEUTRAL_TOPIC_FALLBACK.get(
        base_tag, "相关日常偏好（related personal preference）"
    )
    kw_sources = list(target_keywords)
    if not kw_sources and forget_msg_content:
        msg_target = extract_forget_target(forget_msg_content)
        if msg_target:
            kw_sources.extend(extract_keywords_from_target(msg_target))

    forbidden_en: set[str] = set()
    forbidden_cjk: list[str] = []
    for kw in kw_sources:
        kw_str = str(kw or "").strip()
        if not kw_str:
            continue
        en_toks = re.findall(r"[a-z0-9]+", kw_str.casefold())
        # Only add individual English tokens to forbidden_en for single-word keywords;
        # multi-word keywords (e.g., "meal-prep apps") are stripped as full phrases so
        # generic topic words ("meal", "cooking") are preserved in neutral_topic.
        if len(en_toks) == 1 and len(en_toks[0]) >= 3:
            w = en_toks[0]
            forbidden_en.add(w)
            stem = _stem_en_word(w)
            if stem and len(stem) >= 3:
                forbidden_en.add(stem)
        for cjk in re.findall(r"[\u4e00-\u9fff]{2,}", kw_str):
            forbidden_cjk.append(cjk)

    def _clean_text(text: str) -> str:
        out = text
        norm_text = text.strip(" ,;/-|()（）").casefold()
        for kw in kw_sources:
            kw_s = str(kw or "").strip()
            if kw_s.strip(" ,;/-|()（）").casefold() == norm_text:
                continue
            if len(kw_s) >= 2:
                out = re.sub(re.escape(kw_s), " ", out, flags=re.IGNORECASE)
                if kw_s.lower().endswith("s") and len(kw_s) >= 4:
                    out = re.sub(re.escape(kw_s[:-1]), " ", out, flags=re.IGNORECASE)
        for cjk in forbidden_cjk:
            out = out.replace(cjk, " ")
        if forbidden_en:
            def _repl_en(m: re.Match[str]) -> str:
                tok = m.group(0)
                low = tok.casefold()
                stem = _stem_en_word(low) or low
                if low in forbidden_en or stem in forbidden_en:
                    return ""
                return tok

            out = re.sub(r"[A-Za-z0-9]+", _repl_en, out)
        out = re.sub(r"\s{2,}", " ", out).strip(" ,;/-|()（）")
        return out

    cleaned_raw = _clean_text(str(raw_topic or "").strip())
    if len(cleaned_raw) >= 4:
        return cleaned_raw
    cleaned_fb = _clean_text(fallback)
    return cleaned_fb or "相关日常偏好（related personal preference）"


def _is_subj_in_query(
    query: str,
    person_key: str,
    aliases: Sequence[str] | None = None,
    relations: Sequence[str] | None = None,
) -> bool:
    if not query or not person_key:
        return False
    q_low = query.strip().casefold()

    candidates: set[str] = set()
    raw_terms = [person_key, *(aliases or []), *(relations or [])]
    for raw in raw_terms:
        if not raw:
            continue
        cleaned = str(raw).strip()
        if cleaned:
            candidates.add(cleaned)
        for tok in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", cleaned):
            tok_low = tok.casefold()
            if len(tok) >= 3 and tok_low not in _HONORIFIC_PREFIXES:
                candidates.add(tok)

    for term in candidates:
        t_clean = term.strip().strip(".")
        if not t_clean:
            continue
        t_low = t_clean.casefold()
        if any("\u4e00" <= ch <= "\u9fff" for ch in t_clean):
            if t_low in q_low:
                return True
        else:
            pattern = r"\b" + re.escape(t_low) + r"\b"
            if re.search(pattern, q_low):
                return True
    return False


def _literal_cold_tag_overlap(tag_a: str, tag_b: str) -> float:
    if not tag_a.startswith("其他/") or not tag_b.startswith("其他/"):
        return 0.0
    sa = tag_a[3:].strip().casefold()
    sb = tag_b[3:].strip().casefold()
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    if (len(sa) >= 2 and sa in sb) or (len(sb) >= 2 and sb in sa):
        return 0.80
    toks_a = set(tokenize_bm25(sa))
    toks_b = set(tokenize_bm25(sb))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, min(len(toks_a), len(toks_b)))


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def _compute_bm25_scores(
    chunks: Sequence[dict[str, Any]],
    query_term_weights: dict[str, float],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not chunks or not query_term_weights:
        return {}

    n_docs = len(chunks)
    doc_tokens_list: list[list[str]] = []
    doc_freqs: Counter[str] = Counter()
    total_len = 0

    for c in chunks:
        tokens = str(c.get("bm25_tokens") or "").split()
        doc_tokens_list.append(tokens)
        total_len += len(tokens)
        for t in set(tokens):
            if t in query_term_weights:
                doc_freqs[t] += 1

    avgdl = max(1.0, total_len / n_docs)
    scores: dict[str, float] = {}

    for c, tokens in zip(chunks, doc_tokens_list, strict=True):
        if not tokens:
            continue
        tf_counter = Counter(tokens)
        dl = len(tokens)
        doc_score = 0.0
        for term, q_weight in query_term_weights.items():
            tf = tf_counter.get(term, 0)
            if tf <= 0:
                continue
            df = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            doc_score += q_weight * idf * ((tf * (k1 + 1.0)) / denom)
        if doc_score > 0.0:
            scores[str(c["chunk_id"])] = doc_score

    return scores


class MemoryPipeline:
    def __init__(
        self,
        *,
        store: SQLiteStore | MemoryStore,
        embedder: Embedder,
        extractor: Extractor | Any | None = None,
        reranker: Reranker | Any | None = None,
        semantic_candidates: int = 150,
        lexical_candidates: int = 150,
        neighbor_anchors: int = 20,
        neighbor_radius: int = 1,
        token_budget: int = 100_000,
        chunk_target_chars: int = 1200,
        chunk_max_chars: int = 1800,
        add_extract_budget: float | None = None,
        add_total_budget: float | None = None,
        **extra_kwargs: Any,
    ) -> None:
        _ = extra_kwargs
        self.store = store
        self.embedder = embedder
        self.extractor = extractor if extractor is not None else RegexExtractor()
        self.reranker = reranker if reranker is not None else NoOpReranker()
        self.semantic_candidates = semantic_candidates
        self.lexical_candidates = lexical_candidates
        self.neighbor_anchors = neighbor_anchors
        self.neighbor_radius = neighbor_radius
        self.token_budget = token_budget
        self.chunk_target_chars = chunk_target_chars
        self.chunk_max_chars = chunk_max_chars
        ext_timeout = float(getattr(self.extractor, "timeout", 8.0) or 8.0)
        if add_extract_budget is not None:
            self.add_extract_budget = float(add_extract_budget)
        elif ext_timeout > 8.0:
            self.add_extract_budget = ext_timeout
        else:
            self.add_extract_budget = 7.5
        if add_total_budget is not None:
            self.add_total_budget = float(add_total_budget)
        elif self.add_extract_budget > 7.5:
            self.add_total_budget = max(75.0, self.add_extract_budget * 2.0 + 5.0)
        else:
            self.add_total_budget = 9.0
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._turns_since_state_sync: dict[str, int] = {}
        self._state_unsynced: dict[str, bool] = {}
        self._turns_since_person_reconcile: dict[str, int] = {}
        self._person_reconcile_unsynced: dict[str, bool] = {}
        self._is_subj_in_query = _is_subj_in_query

    async def _consolidate_user_state_bar(
        self, user_id: str, session_id: str, *, deadline: float
    ) -> None:
        assert isinstance(self.store, SQLiteStore)
        rem = deadline - time.monotonic()
        if rem <= 0.3 or not hasattr(self.extractor, "summarize_state_slots"):
            return
        living_facts = self.store.get_living_facts(user_id)
        self._turns_since_state_sync[user_id] = 0
        self._state_unsynced[user_id] = False
        if len(living_facts) < 2:
            return

        # BLOCKING-2: 按 subject 分组，仅对活跃槽位数 >= 2 的同一人物调用归并模型
        by_subj: dict[str, list[dict[str, Any]]] = {}
        for lf in living_facts:
            s = str(lf.get("subject") or "user").strip()
            by_subj.setdefault(s, []).append(lf)

        for subj_facts in by_subj.values():
            if len(subj_facts) < 2:
                continue
            cur_rem = deadline - time.monotonic()
            if cur_rem <= 0.3:
                break
            try:
                merges = await asyncio.wait_for(
                    self.extractor.summarize_state_slots(subj_facts),
                    timeout=cur_rem,
                )
                if merges:
                    self.store.merge_timeline_slots(user_id, session_id, merges)
            except Exception:
                pass

    async def _reconcile_person_map_task(
        self, user_id: str, session_id: str, *, deadline: float
    ) -> None:
        assert isinstance(self.store, SQLiteStore)
        rem = deadline - time.monotonic()
        if rem <= 0.3 or not hasattr(self.extractor, "reconcile_person_map"):
            return
        self._turns_since_person_reconcile[user_id] = 0
        self._person_reconcile_unsynced[user_id] = False

        living_facts = self.store.get_living_facts(user_id)
        profiles = self.store.get_person_profiles(user_id)

        subj_to_slots: dict[str, list[dict[str, Any]]] = {}
        for lf in living_facts:
            s = str(lf.get("subject") or "user").strip()
            subj_to_slots.setdefault(s, []).append({
                "fid": lf.get("slot_id"),
                "slot_id": lf.get("slot_id"),
                "tag": lf.get("tag"),
                "state": lf.get("current_state"),
            })

        all_persons = set(subj_to_slots.keys()) | set(profiles.keys())
        if len(all_persons) <= 1 and not profiles:
            return

        snapshot: list[dict[str, Any]] = []
        for p in sorted(all_persons):
            prof = profiles.get(p, {})
            snapshot.append({
                "person": p,
                "aliases": list(prof.get("aliases") or []),
                "relations": list(prof.get("relations") or []),
                "slots": subj_to_slots.get(p, []),
            })

        try:
            res = await asyncio.wait_for(
                self.extractor.reconcile_person_map(snapshot),
                timeout=rem,
            )
            if res and (
                res.get("merge_persons")
                or res.get("move_slots")
                or res.get("profile_updates")
            ):
                self.store.apply_person_map_updates(
                    user_id,
                    session_id,
                    merge_persons=res.get("merge_persons"),
                    move_slots=res.get("move_slots"),
                    profile_updates=res.get("profile_updates"),
                )
                target_persons: set[str] = set()
                for mp in (res.get("merge_persons") or []):
                    if mp.get("to"):
                        target_persons.add(str(mp["to"]))
                for mv in (res.get("move_slots") or []):
                    if mv.get("to"):
                        target_persons.add(str(mv["to"]))
                if target_persons and hasattr(self.extractor, "summarize_state_slots"):
                    await self._consolidate_user_state_bar(user_id, session_id, deadline=deadline)
        except Exception:
            pass

    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    async def _invoke_extractor(
        self,
        batch_messages: list[dict[str, Any]],
        existing_tags: list[tuple[str, str]],
        living_facts: list[dict[str, Any]] | None = None,
    ) -> Any:
        for attr in ("extract", "extract_batch"):
            if hasattr(self.extractor, attr):
                fn = getattr(self.extractor, attr)
                try:
                    return await fn(
                        batch_messages,
                        existing_tags=existing_tags,
                        living_facts=living_facts,
                    )
                except TypeError:
                    return await fn(batch_messages, existing_tags=existing_tags)
        if callable(self.extractor):
            try:
                return await self.extractor(
                    batch_messages,
                    existing_tags=existing_tags,
                    living_facts=living_facts,
                )
            except TypeError:
                return await self.extractor(batch_messages, existing_tags=existing_tags)
        return regex_extract_batch(batch_messages)

    async def add(self, request: AddRequest) -> AddResponse:
        payload_hash = hashlib.sha256(
            json.dumps(
                request.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        if isinstance(self.store, SQLiteStore):
            return await self._add_v2(request, payload_hash)
        return await self._add_v1_compat(request, payload_hash)

    async def _add_v2(self, request: AddRequest, payload_hash: str) -> AddResponse:
        add_t0 = time.monotonic()
        extract_deadline = add_t0 + self.add_extract_budget
        total_deadline = add_t0 + self.add_total_budget
        lock = self._get_user_lock(request.user_id)
        rem_lock = total_deadline - time.monotonic()
        if rem_lock <= 0.0:
            raise TimeoutError("Add request deadline exceeded before acquiring user lock")
        await asyncio.wait_for(lock.acquire(), timeout=rem_lock)
        try:
            return await self._add_v2_locked(
                request,
                payload_hash,
                extract_deadline=extract_deadline,
                total_deadline=total_deadline,
            )
        finally:
            lock.release()

    async def _add_v2_locked(
        self,
        request: AddRequest,
        payload_hash: str,
        *,
        extract_deadline: float,
        total_deadline: float,
    ) -> AddResponse:
        assert isinstance(self.store, SQLiteStore)

        def _check_deadline() -> None:
            if time.monotonic() >= total_deadline:
                raise TimeoutError("Add request deadline exceeded")

        _check_deadline()

        # Step 1: Idempotency check BEFORE any LLM or Embedding calls (F2.1)
        already_processed = self.store.check_idempotency(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_hash=payload_hash,
        )
        _check_deadline()
        if already_processed:
            return AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        # Step 2: Filter role == 'system' (F1, F5)
        non_sys_messages = [m for m in request.messages if m.role != "system"]
        if not non_sys_messages:
            _check_deadline()
            self.store.record_add_request(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                payload_hash=payload_hash,
            )
            _check_deadline()
            return AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        # Stage 1 (No DB write lock): Read state, slice in memory, and run concurrent LLM windows
        _check_deadline()
        last_seq, pending_forget = self.store.get_user_state(request.user_id)
        _check_deadline()
        inherited_ts = self.store.get_last_session_ts(request.user_id, request.session_id)
        _check_deadline()
        existing_tags = self.store.get_active_tags(request.user_id)
        _check_deadline()
        living_facts = self.store.get_living_facts(request.user_id)
        _check_deadline()

        first_msg_auto_tombstone = (
            pending_forget == 1 and non_sys_messages[0].role == "assistant"
        )

        batch_records: list[dict[str, Any]] = []
        pending_raw_chunks: list[dict[str, Any]] = []

        for idx, msg in enumerate(non_sys_messages):
            _check_deadline()
            seq = last_seq + idx + 1
            if msg.timestamp is not None:
                ts = normalize_timestamp_to_seconds(msg.timestamp)
            else:
                ts = parse_header_timestamp_seconds(msg.content)
                if ts is None:
                    ts = inherited_ts
            if ts is not None:
                inherited_ts = ts

            time_str = format_session_time_str(ts)
            speaker = extract_speaker(msg.content, msg.role)
            msg_id = f"msg:{request.user_id}:{seq}"
            is_tomb = 1 if (idx == 0 and first_msg_auto_tombstone) else 0

            rec = {
                "msg_id": msg_id,
                "seq": seq,
                "role": msg.role,
                "speaker": speaker,
                "content": msg.content,
                "session_ts": ts,
                "session_time_str": time_str,
                "tombstoned": is_tomb,
            }
            batch_records.append(rec)

            if is_tomb:
                continue

            hints = temporal_hints(msg.content, ts * 1000 if ts is not None else None)
            hint_suffix = ("\n[resolved time: " + "; ".join(hints) + "]") if hints else ""
            if (
                len(msg.content) > self.chunk_target_chars
                and self.chunk_target_chars < 1200
            ):
                step = max(100, self.chunk_target_chars - 160)
                slices = []
                start = 0
                n = len(msg.content)
                while start < n:
                    end = min(n, start + self.chunk_target_chars)
                    slices.append(msg.content[start:end])
                    if end >= n:
                        break
                    start += step
            else:
                slices = slice_message_content(msg.content)
            _check_deadline()
            for c_idx, slice_str in enumerate(slices):
                _check_deadline()
                chunk_id = f"{msg_id}:raw:{c_idx:04d}"
                header = f"[#{seq} | {speaker}" + (f" | {time_str}]" if time_str else "]")
                search_text = f"{header} {slice_str}{hint_suffix}"
                embed_input = f"{slice_str}{hint_suffix}" if hint_suffix else slice_str
                bm25_tokens = build_bm25_tokens(
                    f"{slice_str} {hint_suffix}".strip(), subject="user", tag=None
                )
                pending_raw_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "parent_msg_id": msg_id,
                        "seq": seq,
                        "session_ts": ts,
                        "kind": "raw_slice",
                        "subject": "user",
                        "tag": None,
                        "display_text": slice_str,
                        "search_text": search_text,
                        "embed_input": embed_input,
                        "bm25_tokens": bm25_tokens,
                        "event_time_start": ts,
                        "event_time_end": ts,
                        "tombstoned": 0,
                        "vector": [],
                    }
                )

        _check_deadline()

        active_batch = [r for r in batch_records if r["tombstoned"] == 0]
        facts: list[ExtractedFact] = []
        forget_actions: list[ExtractedForgetAction] = []
        use_two_pass = hasattr(self.extractor, "summarize_state_slots")
        pass1_living = None if use_two_pass else living_facts
        pass1_tags = [] if use_two_pass else existing_tags

        if active_batch:
            if len(active_batch) <= 16:
                try:
                    rem_ext = max(0.05, extract_deadline - time.monotonic())
                    raw_ext = await asyncio.wait_for(
                        self._invoke_extractor(
                            active_batch, pass1_tags, living_facts=pass1_living
                        ),
                        timeout=rem_ext,
                    )
                except Exception:
                    raw_ext = regex_extract_batch(active_batch)
                facts, forget_actions = sanitize_extraction_result(raw_ext, active_batch)
            else:
                windows: list[list[dict[str, Any]]] = []
                for w_start in range(0, len(active_batch), 12):
                    s_win = active_batch[w_start : w_start + 12]
                    s_ro = [
                        {**m, "read_only": True}
                        for m in active_batch[max(0, w_start - 2) : w_start]
                    ]
                    windows.append(s_ro + s_win)
                sem = asyncio.Semaphore(4)

                async def _run_win(win_msgs: list[dict[str, Any]]) -> Any:
                    writable_only = [
                        m for m in win_msgs if not m.get("read_only")
                    ]
                    async with sem:
                        rem_ext = extract_deadline - time.monotonic()
                        if rem_ext <= 0.05:
                            return regex_extract_batch(writable_only)
                        try:
                            return await asyncio.wait_for(
                                self._invoke_extractor(
                                    win_msgs, pass1_tags, living_facts=pass1_living
                                ),
                                timeout=rem_ext,
                            )
                        except Exception:
                            return regex_extract_batch(writable_only)

                win_results = await asyncio.gather(*[_run_win(w) for w in windows])
                for win_msgs, raw_ext in zip(windows, win_results, strict=True):
                    _check_deadline()
                    w_facts, w_forgets = sanitize_extraction_result(raw_ext, win_msgs)
                    facts.extend(w_facts)
                    forget_actions.extend(w_forgets)

        _check_deadline()

        # Stage 2 (No DB write lock): Pre-align facts & single batch embed immutable chunks
        seq_to_rec = {int(r["seq"]): r for r in batch_records}
        active_seq_set = {int(r["seq"]) for r in active_batch}

        if hasattr(self.store, "_allocate_next_slot_id"):
            curr_slot = self.store._allocate_next_slot_id(request.user_id)
        else:
            curr_slot = "F1"
        slot_counter = (
            int(curr_slot[1:])
            if curr_slot.startswith("F") and curr_slot[1:].isdigit()
            else 1
        )

        prepared_facts: list[dict[str, Any]] = []
        for f_idx, fact in enumerate(facts):
            _check_deadline()
            valid_seqs = [s for s in fact.source_seqs if s in active_seq_set]
            if not valid_seqs:
                continue
            primary_seq = max(valid_seqs)
            if fact.is_other_person or fact.subject.casefold() != "user":
                formatted_note = f"【非User本人，系{fact.subject}的信息】{fact.fact_text}"
            else:
                formatted_note = fact.fact_text
            if fact.current_state and fact.current_state not in formatted_note:
                formatted_note = f"{formatted_note}（{fact.current_state}）"
            if fact.style_note and fact.style_note not in formatted_note:
                formatted_note = f"{formatted_note}（风格偏好: {fact.style_note}）"
            if getattr(fact, "implied", ""):
                implied_str = fact.implied.strip()
                if implied_str:
                    implied_marker = f"[隐式推导 / Implied: {implied_str}]"
                    formatted_note = f"{formatted_note} {implied_marker}"

            primary_rec = seq_to_rec[primary_seq]
            primary_msg_id = str(primary_rec["msg_id"])
            primary_session_ts = primary_rec["session_ts"]
            primary_time_str = str(primary_rec["session_time_str"] or "")
            ev_start, ev_end = parse_event_date_range(
                fact.event_date_start,
                fact.event_date_end,
                fallback_ts=primary_session_ts,
            )
            fact_slot_id: str | None = None
            if fact.source_tier != "user_draft_fiction":
                clean_target = str(fact.target_id or "").strip().strip("[]#")
                if clean_target and clean_target.startswith("F") and clean_target[1:].isdigit():
                    fact_slot_id = clean_target
                    fact_chunk_id = f"fact:{request.user_id}:{fact_slot_id}"
                elif getattr(fact, "is_v24", False):
                    fact_slot_id = f"F{slot_counter}"
                    slot_counter += 1
                    fact_chunk_id = f"fact:{request.user_id}:{fact_slot_id}"
                elif fact.prev_state or fact.op in ("set", "delete"):
                    fact_slot_id = None
                    fact_chunk_id = f"fact:{request.request_id}:{f_idx:04d}"
                else:
                    fact_slot_id = f"F{slot_counter}"
                    slot_counter += 1
                    fact_chunk_id = f"fact:{request.request_id}:{f_idx:04d}"
            else:
                fact_chunk_id = f"fact:{request.request_id}:{f_idx:04d}"

            fact_search_text = (
                f"[{fact.subject} | {fact.tag}] {fact.tag} {fact.current_state} {formatted_note}"
            ).strip()
            prepared_facts.append(
                {
                    "fact": fact,
                    "slot_id": fact_slot_id,
                    "valid_seqs": valid_seqs,
                    "primary_seq": primary_seq,
                    "primary_msg_id": primary_msg_id,
                    "primary_session_ts": primary_session_ts,
                    "primary_time_str": primary_time_str,
                    "formatted_note": formatted_note,
                    "ev_start": ev_start,
                    "ev_end": ev_end,
                    "chunk_id": fact_chunk_id,
                    "embed_input": fact_search_text,
                    "vector": [],
                }
            )

        valid_forgets: list[dict[str, Any]] = []
        for fa in forget_actions:
            _check_deadline()
            trig_rec = seq_to_rec.get(fa.trigger_seq)
            if trig_rec is None or fa.trigger_seq not in active_seq_set:
                continue
            if not has_explicit_forget_keyword(str(trig_rec["content"])):
                continue
            neutral_state_text = ""
            neutral_fact_text = ""
            neutral_embed_input = ""
            neutral_tag = f"其他/偏好撤回#{fa.trigger_seq}"
            neutral_slot_id = f"FN{fa.trigger_seq}"
            if bool(fa.neutral_topic.strip()) or use_two_pass:
                clean_topic = _sanitize_neutral_topic(
                    fa.neutral_topic,
                    fa.target_keywords,
                    str(trig_rec["content"] or ""),
                    fa.target_existing_tags,
                )
                neutral_fact_text = (
                    f"用户已主动撤回并清空关于「{clean_topic}」的既有个人偏好记录 "
                    f"(PREFERENCE CLEARED on {clean_topic}: User explicitly cleared their "
                    f"previous specific preference on {clean_topic}; no specific preference is "
                    f"currently recorded on file. When answering a question about {clean_topic}, "
                    f"you MUST choose the purely general, everyday, unpersonalized option that "
                    f"does NOT assume any specific app/tool/activity and does NOT tie into the "
                    f"user's profession, duty schedule, or cultural background.)"
                )
                neutral_state_text = (
                    f"[状态槽位 {neutral_slot_id} | user | {neutral_tag}] "
                    f"当前状态: {neutral_fact_text}"
                )
                neutral_embed_input = (
                    f"[user | {neutral_tag}] {clean_topic} {neutral_fact_text}"
                )
            valid_forgets.append(
                {
                    "fa": fa,
                    "trig_rec": trig_rec,
                    "target_vec": None,
                    "neutral_slot_id": neutral_slot_id,
                    "neutral_tag": neutral_tag,
                    "neutral_state_text": neutral_state_text,
                    "neutral_fact_text": neutral_fact_text,
                    "neutral_embed_input": neutral_embed_input,
                    "neutral_vec": [],
                }
            )

        n_raw = len(pending_raw_chunks)
        all_embed_inputs: list[str] = [
            *(c["embed_input"] for c in pending_raw_chunks),
            *(pf["embed_input"] for pf in prepared_facts),
        ]
        forget_embed_indices: list[tuple[int, int]] = []
        neutral_embed_indices: list[tuple[int, int]] = []
        for vf_idx, vf in enumerate(valid_forgets):
            _check_deadline()
            desc = vf["fa"].target_description
            if desc:
                forget_embed_indices.append((vf_idx, len(all_embed_inputs)))
                all_embed_inputs.append(desc)
            n_emb = vf.get("neutral_embed_input")
            if n_emb:
                neutral_embed_indices.append((vf_idx, len(all_embed_inputs)))
                all_embed_inputs.append(str(n_emb))

        if all_embed_inputs:
            rem_emb = total_deadline - time.monotonic()
            if rem_emb <= 0.0:
                raise TimeoutError("Add request deadline exceeded before embedding")
            all_vecs = await asyncio.wait_for(
                self.embedder.embed(all_embed_inputs), timeout=rem_emb
            )
            for r_idx, c in enumerate(pending_raw_chunks):
                c["vector"] = all_vecs[r_idx]
            for pf_idx, pf in enumerate(prepared_facts):
                pf["vector"] = all_vecs[n_raw + pf_idx]
            for vf_idx, emb_idx in forget_embed_indices:
                valid_forgets[vf_idx]["target_vec"] = all_vecs[emb_idx]
            for vf_idx, emb_idx in neutral_embed_indices:
                valid_forgets[vf_idx]["neutral_vec"] = all_vecs[emb_idx]

        _check_deadline()

        # Stage 3: Short atomic SQLite transaction for state replay & persistence
        with self.store.transaction(deadline_check=lambda: time.monotonic() >= total_deadline):
            for rec in batch_records:
                _check_deadline()
                self.store.insert_parent_message(
                    msg_id=rec["msg_id"],
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    seq=rec["seq"],
                    role=rec["role"],
                    speaker=rec["speaker"],
                    content=rec["content"],
                    session_ts=rec["session_ts"],
                    session_time_str=rec["session_time_str"],
                    fact_notes="",
                    tombstoned=rec["tombstoned"],
                )

            self.store.update_user_state(
                request.user_id,
                last_seq=last_seq + len(non_sys_messages),
                pending_forget_confirm=0,
            )

            for c in pending_raw_chunks:
                _check_deadline()
                self.store.upsert_chunk(
                    chunk_id=c["chunk_id"],
                    user_id=request.user_id,
                    session_id=request.session_id,
                    parent_msg_id=c["parent_msg_id"],
                    seq=c["seq"],
                    session_ts=c["session_ts"],
                    kind=c["kind"],
                    subject=c["subject"],
                    tag=c["tag"],
                    display_text=c["display_text"],
                    search_text=c["search_text"],
                    bm25_tokens=c["bm25_tokens"],
                    event_time_start=c["event_time_start"],
                    event_time_end=c["event_time_end"],
                    embedding=c["vector"],
                    tombstoned=c["tombstoned"],
                )

            # Build cold-tag registry per subject for Scheme C (literal >= 0.75 or cosine > 0.85)
            cold_tag_registry: dict[str, dict[str, list[list[float]]]] = {}
            for pf in prepared_facts:
                _check_deadline()
                subj = pf["fact"].subject
                if subj not in cold_tag_registry:
                    db_cold = self.store.get_active_cold_tag_vectors(request.user_id, subj)
                    cold_tag_registry[subj] = {
                        tg: [item["vector"] for item in items if item.get("vector")]
                        for tg, items in db_cold.items()
                    }

            # Replay facts and forget_actions in strict causal seq order (seq = 1..N)
            facts_by_seq: dict[int, list[dict[str, Any]]] = {}
            for pf in prepared_facts:
                facts_by_seq.setdefault(int(pf["primary_seq"]), []).append(pf)

            forgets_by_seq: dict[int, list[dict[str, Any]]] = {}
            for vf in valid_forgets:
                forgets_by_seq.setdefault(int(vf["fa"].trigger_seq), []).append(vf)

            replay_seqs = sorted(
                {*active_seq_set, *facts_by_seq.keys(), *forgets_by_seq.keys()}
            )
            dirty_timeline_keys: list[tuple[str, str]] = []
            last_batch_seq = int(batch_records[-1]["seq"]) if batch_records else 0

            # Determine which seqs have ONLY user_draft_fiction facts (no non-draft facts)
            seq_all_draft_map: dict[int, bool] = {}
            for s in active_seq_set:
                s_facts = [
                    pf["fact"] for pf in prepared_facts if s in pf["valid_seqs"]
                ]
                seq_all_draft_map[s] = bool(s_facts) and all(
                    f.source_tier == "user_draft_fiction" for f in s_facts
                )

            for cur_seq in replay_seqs:
                _check_deadline()
                for pf in facts_by_seq.get(cur_seq, []):
                    _check_deadline()
                    fact: ExtractedFact = pf["fact"]
                    f_vec: list[float] = pf["vector"]
                    # Cold-tag ('其他/*') literal & semantic merging for non-draft facts
                    if (
                        fact.source_tier != "user_draft_fiction"
                        and fact.tag.startswith("其他/")
                    ):
                        subj_map = cold_tag_registry.setdefault(fact.subject, {})
                        merged_tag: str | None = None
                        for ex_tag in subj_map:
                            if _literal_cold_tag_overlap(fact.tag, ex_tag) >= 0.75:
                                merged_tag = ex_tag
                                break
                        if merged_tag is None and f_vec:
                            best_cos = 0.85
                            for ex_tag, ex_vecs in subj_map.items():
                                for ev in ex_vecs:
                                    sim = cosine_similarity(f_vec, ev)
                                    if sim > best_cos:
                                        best_cos = sim
                                        merged_tag = ex_tag
                        if merged_tag is not None:
                            fact.tag = merged_tag
                        subj_map.setdefault(fact.tag, [])
                        if f_vec:
                            subj_map[fact.tag].append(f_vec)

                    # Strict prohibition: user_draft_fiction never enters tag_timelines
                    is_fact_tombstoned = False
                    if fact.source_tier != "user_draft_fiction":
                        upsert_res = self.store.upsert_tag_timeline(
                            user_id=request.user_id,
                            subject=fact.subject,
                            tag=fact.tag,
                            seq=pf["primary_seq"],
                            session_ts=pf["primary_session_ts"],
                            time_str=pf["primary_time_str"],
                            prev_state=fact.prev_state,
                            current_state=fact.current_state,
                            update_reason=fact.update_reason,
                            fact_text=fact.fact_text,
                            op=fact.op,
                            target_id=fact.target_id,
                            fact_chunk_id=pf["chunk_id"],
                            source_seqs=pf["valid_seqs"],
                            slot_id=pf.get("slot_id"),
                        )
                        resolved_tag = upsert_res[3]
                        is_fact_tombstoned = upsert_res[4]
                        resolved_slot_id = getattr(upsert_res, "slot_id", pf.get("slot_id"))
                        _check_deadline()
                        fact.tag = resolved_tag
                        if resolved_slot_id:
                            pf["slot_id"] = resolved_slot_id
                            dirty_timeline_keys.append(resolved_slot_id)
                        dirty_timeline_keys.append((fact.subject, fact.tag))

                        if (
                            fact.subject.casefold() != "user"
                            and hasattr(self.store, "upsert_person_profile")
                        ):
                            self.store.upsert_person_profile(
                                request.user_id,
                                fact.subject,
                                aliases=getattr(fact, "aliases", None),
                                relations=getattr(fact, "relations", None),
                            )

                    for s in pf["valid_seqs"]:
                        _check_deadline()
                        self.store.append_parent_fact_note(
                            request.user_id, s, pf["formatted_note"]
                        )
                        self.store.backfill_raw_slices_for_seq(
                            request.user_id,
                            s,
                            subject=fact.subject,
                            tag=fact.tag,
                            fact_text=fact.fact_text,
                            source_tier=fact.source_tier,
                            seq_all_draft=seq_all_draft_map.get(s, False),
                        )

                    fact_search_text = f"[{fact.subject} | {fact.tag}] {pf['formatted_note']}"
                    fact_bm25 = build_bm25_tokens(
                        pf["formatted_note"], subject=fact.subject, tag=fact.tag
                    )
                    self.store.upsert_chunk(
                        chunk_id=pf["chunk_id"],
                        user_id=request.user_id,
                        session_id=request.session_id,
                        parent_msg_id=pf["primary_msg_id"],
                        slot_id=pf.get("slot_id") or "",
                        seq=pf["primary_seq"],
                        session_ts=pf["primary_session_ts"],
                        kind="fact_item",
                        subject=fact.subject,
                        tag=fact.tag,
                        source_tier=fact.source_tier,
                        display_text=pf["formatted_note"],
                        search_text=fact_search_text,
                        bm25_tokens=fact_bm25,
                        event_time_start=pf["ev_start"],
                        event_time_end=pf["ev_end"],
                        embedding=f_vec,
                        tombstoned=1 if is_fact_tombstoned else 0,
                    )

                cur_rec = seq_to_rec.get(cur_seq)
                if cur_rec is not None and cur_rec.get("role") == "assistant":
                    self.store.propagate_assistant_draft_echo(request.user_id, cur_seq)

                for vf in forgets_by_seq.get(cur_seq, []):
                    _check_deadline()
                    fa: ExtractedForgetAction = vf["fa"]
                    trig_rec = vf["trig_rec"]
                    batch_to_tomb = list(fa.batch_seqs_to_tombstone)
                    if fa.trigger_seq not in batch_to_tomb:
                        batch_to_tomb.append(fa.trigger_seq)
                    next_rec = seq_to_rec.get(fa.trigger_seq + 1)
                    if (
                        next_rec is not None
                        and next_rec["role"] == "assistant"
                        and (fa.trigger_seq + 1) not in batch_to_tomb
                    ):
                        batch_to_tomb.append(fa.trigger_seq + 1)

                    if fa.trigger_seq == last_batch_seq and trig_rec["role"] == "user":
                        self.store.update_user_state(
                            request.user_id, pending_forget_confirm=1
                        )

                    rebuilt_tls = self.store.execute_forget_action(
                        user_id=request.user_id,
                        trigger_seq=fa.trigger_seq,
                        batch_seqs_to_tombstone=batch_to_tomb,
                        target_existing_tags=fa.target_existing_tags,
                        target_keywords=fa.target_keywords,
                        target_description=fa.target_description,
                        target_vector=vf["target_vec"],
                        target_id=fa.target_id,
                    )
                    for r_subj, r_tag, _r_sum, _r_seq, _r_ts in rebuilt_tls:
                        dirty_timeline_keys.append((r_subj, r_tag))
                    # Clear tombstoned tags from cold_tag_registry
                    active_tags_now = set(self.store.get_active_tags(request.user_id))
                    for r_subj, t_map in cold_tag_registry.items():
                        for r_tag in list(t_map.keys()):
                            if (r_subj, r_tag) not in active_tags_now:
                                t_map.pop(r_tag, None)

            _check_deadline()
            self.store.sync_dirty_timelines(
                user_id=request.user_id,
                session_id=request.session_id,
                dirty_keys=dirty_timeline_keys,
            )
            _check_deadline()

            if valid_forgets:
                if hasattr(self.store, "_connect"):
                    self.store._connect().execute(
                        "UPDATE search_chunks SET tombstoned = 0 "
                        "WHERE user_id = ? AND slot_id LIKE 'FN%'",
                        (request.user_id,),
                    )
                for vf in valid_forgets:
                    _check_deadline()
                    n_state = vf.get("neutral_state_text")
                    n_fact = vf.get("neutral_fact_text")
                    n_emb = vf.get("neutral_embed_input")
                    if not (n_state and n_fact and n_emb):
                        continue
                    trig_rec = vf["trig_rec"]
                    n_slot = str(vf["neutral_slot_id"])
                    n_tag = str(vf["neutral_tag"])
                    n_vec = vf.get("neutral_vec") or []
                    n_bm25 = build_bm25_tokens(str(n_emb), subject="user", tag=n_tag)
                    self.store.upsert_chunk(
                        chunk_id=f"tl:{request.user_id}:{n_slot}",
                        user_id=request.user_id,
                        session_id=request.session_id,
                        parent_msg_id=None,
                        slot_id=n_slot,
                        seq=0,
                        session_ts=trig_rec["session_ts"],
                        kind="timeline_summary",
                        subject="user",
                        tag=n_tag,
                        source_tier="user_direct",
                        display_text=str(n_state),
                        search_text=str(n_emb),
                        bm25_tokens=n_bm25,
                        event_time_start=trig_rec["session_ts"],
                        event_time_end=trig_rec["session_ts"],
                        embedding=n_vec,
                        tombstoned=0,
                    )
                    self.store.upsert_chunk(
                        chunk_id=f"fact:{request.user_id}:{n_slot}",
                        user_id=request.user_id,
                        session_id=request.session_id,
                        parent_msg_id=None,
                        slot_id=n_slot,
                        seq=0,
                        session_ts=trig_rec["session_ts"],
                        kind="fact_item",
                        subject="user",
                        tag=n_tag,
                        source_tier="user_direct",
                        display_text=str(n_fact),
                        search_text=str(n_emb),
                        bm25_tokens=n_bm25,
                        event_time_start=trig_rec["session_ts"],
                        event_time_end=trig_rec["session_ts"],
                        embedding=n_vec,
                        tombstoned=0,
                    )

            self.store.record_add_request(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                payload_hash=payload_hash,
            )

        if use_two_pass and active_batch:
            n_rounds = max(1, sum(1 for r in active_batch if r["role"] == "user"))
            acc_rounds = self._turns_since_state_sync.get(request.user_id, 0) + n_rounds
            self._turns_since_state_sync[request.user_id] = acc_rounds
            acc_person_rounds = (
                self._turns_since_person_reconcile.get(request.user_id, 0) + n_rounds
            )
            self._turns_since_person_reconcile[request.user_id] = acc_person_rounds
            if dirty_timeline_keys:
                self._state_unsynced[request.user_id] = True
                self._person_reconcile_unsynced[request.user_id] = True

            if (
                acc_person_rounds >= 40
                and self._person_reconcile_unsynced.get(request.user_id)
            ):
                await self._reconcile_person_map_task(
                    request.user_id,
                    request.session_id,
                    deadline=total_deadline - 0.1,
                )

            if acc_rounds >= 20 and self._state_unsynced.get(request.user_id):
                await self._consolidate_user_state_bar(
                    request.user_id,
                    request.session_id,
                    deadline=total_deadline - 0.1,
                )

        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    async def _add_v1_compat(self, request: AddRequest, payload_hash: str) -> AddResponse:
        items = render_write_items(
            request, target=self.chunk_target_chars, maximum=self.chunk_max_chars
        )
        add_temporal_hints(items, request)
        if items:
            vectors = await self.embedder.embed([item.content for item in items])
            for item, vector in zip(items, vectors, strict=True):
                item.embedding = vector

        archive_ids: list[str] = []
        for message_index, message in enumerate(request.messages):
            if message.role != "user":
                continue
            target = extract_forget_target(message.content)
            if not target:
                continue
            for item in items:
                if item.message_index == message_index:
                    item.searchable = False
            [target_vector] = await self.embedder.embed([target])
            candidates = await self.store.search_candidates(  # type: ignore[union-attr]
                user_id=request.user_id,
                query=target,
                embedding=target_vector,
                semantic_limit=min(50, self.semantic_candidates),
                lexical_limit=min(50, self.lexical_candidates),
                result_limit=50,
            )
            archive_ids.extend(select_for_archive(target, candidates))

        await self.store.add_request(  # type: ignore[union-attr]
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_hash=payload_hash,
            items=items,
            archive_ids=list(dict.fromkeys(archive_ids)),
        )
        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        if isinstance(self.store, SQLiteStore):
            return await self._search_v2(request)
        return await self._search_v1_compat(request)

    async def _search_v2(self, request: SearchRequest) -> SearchResponse:
        assert isinstance(self.store, SQLiteStore)
        needs_person_flush = (
            self._person_reconcile_unsynced.get(request.user_id)
            and hasattr(self.extractor, "reconcile_person_map")
        )
        needs_state_flush = (
            self._state_unsynced.get(request.user_id)
            and hasattr(self.extractor, "summarize_state_slots")
        )
        if needs_person_flush or needs_state_flush:
            lock = self._get_user_lock(request.user_id)
            async with lock:
                if self._person_reconcile_unsynced.get(request.user_id) and hasattr(
                    self.extractor, "reconcile_person_map"
                ):
                    await self._reconcile_person_map_task(
                        request.user_id,
                        "search-flush",
                        deadline=time.monotonic() + self.add_extract_budget,
                    )
                if self._state_unsynced.get(request.user_id) and hasattr(
                    self.extractor, "summarize_state_slots"
                ):
                    await self._consolidate_user_state_bar(
                        request.user_id,
                        "search-flush",
                        deadline=time.monotonic() + self.add_extract_budget,
                    )
        chunks = self.store.get_active_chunks(request.user_id)
        if not chunks:
            return SearchResponse(data=[])

        (
            recalled_chunks,
            rrf_norm_map,
            bm25_scores,
            time_hit_cids,
        ) = await self._recall_raw_and_facts(request, chunks)
        if not recalled_chunks:
            return SearchResponse(data=[])

        self._pull_timeline_evidence(
            request.user_id, chunks, recalled_chunks, rrf_norm_map, time_hit_cids
        )
        dialogue_candidates, timeline_candidates = self._build_turn_pair_cards(
            request.user_id, recalled_chunks, rrf_norm_map, bm25_scores, time_hit_cids
        )
        all_candidates, rerank_succeeded = await self._score_and_rerank_cards(
            request.query,
            dialogue_candidates,
            timeline_candidates,
            user_id=request.user_id,
            user_self_names=getattr(self, "_last_user_self_names", None),
        )
        packed = self._pack_with_forget_guard(
            request, all_candidates, dialogue_candidates, timeline_candidates, rerank_succeeded
        )
        return SearchResponse(
            data=[
                SearchItem(
                    id=candidate.id,
                    content=candidate.content,
                    score=candidate.score,
                    created_at=candidate.created_at,
                )
                for candidate in packed
            ]
        )

    async def _recall_raw_and_facts(
        self, request: SearchRequest, chunks: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], set[str]]:
        assert isinstance(self.store, SQLiteStore)
        pure_query = request.query
        opt_texts = (
            [opt[:150] for opt in request.options if opt and opt.strip()]
            if request.options
            else []
        )

        if opt_texts:
            opt_concat = pure_query + "\n" + "\n".join(opt_texts)
            vec_q, vec_opt = await self.embedder.embed([pure_query, opt_concat])
            dim = min(len(vec_q), len(vec_opt))
            query_vec = _l2_normalize([0.7 * vec_q[i] + 0.3 * vec_opt[i] for i in range(dim)])
        else:
            [query_vec] = await self.embedder.embed([pure_query])

        term_weights: dict[str, float] = {}
        stem_tokens = tokenize_bm25(pure_query)
        if opt_texts:
            for t in stem_tokens:
                term_weights[t] = 2.0
            for opt in opt_texts:
                cleaned_opt = re.sub(r"^[A-Ha-h][.、)\s]+", "", opt)
                for t in tokenize_bm25(cleaned_opt):
                    if t not in term_weights:
                        term_weights[t] = 0.5
        else:
            for t in stem_tokens:
                term_weights[t] = 1.0

        dense_scores = {
            str(c["chunk_id"]): cosine_similarity(query_vec, c["vector"]) for c in chunks
        }
        dense_ranked = sorted(
            chunks,
            key=lambda c: (
                dense_scores[str(c["chunk_id"])],
                1 if c["kind"] == "timeline_summary" else 0,
                int(c["session_ts"] or 0),
                int(c["seq"]),
            ),
            reverse=True,
        )[: self.semantic_candidates]

        bm25_scores = _compute_bm25_scores(chunks, term_weights)
        bm25_ranked = sorted(
            [c for c in chunks if str(c["chunk_id"]) in bm25_scores],
            key=lambda c: (
                bm25_scores[str(c["chunk_id"])],
                1 if c["kind"] == "timeline_summary" else 0,
                int(c["session_ts"] or 0),
                int(c["seq"]),
            ),
            reverse=True,
        )[: self.lexical_candidates]

        anchor_ts = self.store.get_max_user_session_ts(request.user_id)
        q_time_range = extract_query_time_range(pure_query, anchor_ts)
        time_ranked: list[dict[str, Any]] = []
        if q_time_range is not None:
            q_start, q_end = q_time_range
            matched_time_seqs = self.store.get_time_matched_seqs(request.user_id, q_start, q_end)
            time_matches = [
                c
                for c in chunks
                if (
                    c["event_time_start"] is not None
                    and c["event_time_end"] is not None
                    and int(c["event_time_start"]) <= q_end
                    and int(c["event_time_end"]) >= q_start
                )
                or (c["kind"] == "raw_slice" and int(c["seq"]) in matched_time_seqs)
            ]
            time_ranked = sorted(
                time_matches,
                key=lambda c: (
                    dense_scores[str(c["chunk_id"])],
                    int(c["session_ts"] or 0),
                    int(c["seq"]),
                ),
                reverse=True,
            )[:100]

        time_hit_cids = {str(c["chunk_id"]) for c in time_ranked}
        rrf_raw: dict[str, float] = {}
        for rank_idx, c in enumerate(dense_ranked, start=1):
            cid = str(c["chunk_id"])
            rrf_raw[cid] = rrf_raw.get(cid, 0.0) + W_DENSE / (RRF_K + rank_idx)
        for rank_idx, c in enumerate(bm25_ranked, start=1):
            cid = str(c["chunk_id"])
            rrf_raw[cid] = rrf_raw.get(cid, 0.0) + W_BM25 / (RRF_K + rank_idx)
        for rank_idx, c in enumerate(time_ranked, start=1):
            cid = str(c["chunk_id"])
            rrf_raw[cid] = rrf_raw.get(cid, 0.0) + W_TIME / (RRF_K + rank_idx)

        for c in chunks:
            if (
                c["kind"] == "timeline_summary"
                and get_base_tag(str(c.get("tag") or "")) == "全局指令/格式约束"
            ):
                cid = str(c["chunk_id"])
                if cid not in rrf_raw:
                    rrf_raw[cid] = 1e-6

        if not rrf_raw:
            return [], {}, bm25_scores, time_hit_cids

        max_rrf = max(rrf_raw.values()) if rrf_raw else 1.0
        rrf_norm_map = {cid: val / max_rrf for cid, val in rrf_raw.items()}
        recalled_chunks = [c for c in chunks if str(c["chunk_id"]) in rrf_raw]
        return recalled_chunks, rrf_norm_map, bm25_scores, time_hit_cids

    def _pull_timeline_evidence(
        self,
        user_id: str,
        chunks: list[dict[str, Any]],
        recalled_chunks: list[dict[str, Any]],
        rrf_norm_map: dict[str, float],
        time_hit_cids: set[str],
    ) -> None:
        assert isinstance(self.store, SQLiteStore)
        timeline_pull_map = self.store.get_timeline_pull_map(user_id)
        recalled_cids = {str(c["chunk_id"]) for c in recalled_chunks}
        chunks_by_seq: dict[int, list[dict[str, Any]]] = {}
        for c in chunks:
            if c["kind"] == "raw_slice":
                chunks_by_seq.setdefault(int(c["seq"]), []).append(c)

        recalled_tl_chunks = sorted(
            (c for c in recalled_chunks if c["kind"] == "timeline_summary"),
            key=lambda c: rrf_norm_map.get(str(c["chunk_id"]), 0.0),
            reverse=True,
        )[:10]
        for c in recalled_tl_chunks:
            tl_cid = str(c["chunk_id"])
            tl_rrf = rrf_norm_map.get(tl_cid, 0.0)
            if tl_rrf < 0.35:
                continue
            tl_tier = str(c.get("source_tier") or "user_direct")
            for p_seq in timeline_pull_map.get(tl_cid, []):
                for raw_c in chunks_by_seq.get(p_seq, []):
                    raw_tier = str(raw_c.get("source_tier") or "user_direct")
                    if (tl_tier == "user_draft_fiction") != (raw_tier == "user_draft_fiction"):
                        continue
                    rc_id = str(raw_c["chunk_id"])
                    if time_hit_cids and rc_id not in time_hit_cids:
                        continue
                    rrf_norm_map[rc_id] = max(rrf_norm_map.get(rc_id, 0.0), 0.85 * tl_rrf)
                    if rc_id not in recalled_cids:
                        recalled_chunks.append(raw_c)
                        recalled_cids.add(rc_id)

    def _build_turn_pair_cards(
        self,
        user_id: str,
        recalled_chunks: list[dict[str, Any]],
        rrf_norm_map: dict[str, float],
        bm25_scores: dict[str, float],
        time_hit_cids: set[str],
    ) -> tuple[dict[str, MemoryCandidate], dict[str, MemoryCandidate]]:
        assert isinstance(self.store, SQLiteStore)
        latest_tag_map = self.store.get_latest_tag_map(user_id)
        active_slot_latest_seqs = self.store.get_active_slot_latest_seqs(user_id)
        superseded_map = self.store.get_superseded_seqs_map(user_id)
        parent_ids = [
            str(c["parent_msg_id"])
            for c in recalled_chunks
            if c["kind"] != "timeline_summary" and c["parent_msg_id"]
        ]
        parents_map = self.store.get_parent_messages_by_ids(user_id, parent_ids)
        parent_chunks_map = self.store.get_chunks_for_parents(user_id, parent_ids)
        parent_messages = list(parents_map.values())
        user_self_names = infer_user_self_names(parent_messages)
        user_self_names_cf = {n.casefold() for n in user_self_names}
        self._last_user_self_names = user_self_names

        timeline_candidates: dict[str, MemoryCandidate] = {}
        dialogue_candidates: dict[str, MemoryCandidate] = {}

        for c in recalled_chunks:
            cid = str(c["chunk_id"])
            kind = str(c["kind"])
            c_rrf = rrf_norm_map.get(cid, 0.0)
            subj = str(c["subject"] or "user")
            if subj.casefold() in user_self_names_cf:
                subj = "user"
            tg = str(c["tag"]) if c["tag"] else None
            source_tier = str(c.get("source_tier") or "user_direct")

            if kind == "timeline_summary":
                is_global = get_base_tag(tg) == "全局指令/格式约束"
                timeline_candidates[cid] = MemoryCandidate(
                    id=cid,
                    content=str(c["display_text"]),
                    score=c_rrf,
                    created_at=format_session_time_str(c["session_ts"]) or None,
                    session_id=str(c["session_id"]),
                    kind="timeline_summary",
                    seq=int(c["seq"]),
                    session_ts=c["session_ts"],
                    subject=subj,
                    tag=tg,
                    metadata={
                        "rrf_norm": c_rrf,
                        "is_latest_in_tag": True,
                        "is_global_constraint": is_global,
                        "source_tier": source_tier,
                        "rerank_text": str(c.get("search_text") or c["display_text"])[:600],
                    },
                )
                continue

            pmid = str(c["parent_msg_id"]) if c["parent_msg_id"] else None
            if not pmid or pmid not in parents_map:
                continue
            parent = parents_map[pmid]
            p_seq = int(parent["seq"])
            p_ts = int(parent["session_ts"]) if parent["session_ts"] is not None else None

            is_latest = False
            if (subj.strip().casefold(), p_seq) in active_slot_latest_seqs:
                is_latest = True
            elif tg and (subj, tg) in latest_tag_map:
                l_ts, l_seq = latest_tag_map[(subj, tg)]
                if p_seq == l_seq or (
                    p_ts is not None and l_ts is not None and (p_ts, p_seq) >= (l_ts, l_seq)
                ):
                    is_latest = True

            all_p_chunks = parent_chunks_map.get(pmid, [])
            raw_slices_for_parent = [ch for ch in all_p_chunks if ch["kind"] == "raw_slice"]
            all_raw_tombstoned = not raw_slices_for_parent or all(
                int(ch["tombstoned"]) == 1 for ch in raw_slices_for_parent
            )

            if int(parent["tombstoned"]) == 1 or all_raw_tombstoned:
                if kind == "fact_item":
                    t_str = str(parent["session_time_str"] or "")
                    t_header = f" | {t_str}" if t_str else ""
                    fact_body = (
                        f"[#{p_seq} | {parent['speaker']}{t_header}] "
                        f"[提炼事实: {c['display_text']}]"
                    )
                    dialogue_candidates[cid] = MemoryCandidate(
                        id=cid,
                        content=fact_body,
                        score=c_rrf,
                        created_at=t_str or None,
                        session_id=str(parent["session_id"]),
                        kind="fact_item",
                        parent_msg_id=pmid,
                        seq=p_seq,
                        session_ts=p_ts,
                        subject=subj,
                        tag=tg,
                        metadata={
                            "rrf_norm": c_rrf,
                            "is_latest_in_tag": is_latest,
                            "is_global_constraint": False,
                            "source_tier": source_tier,
                        },
                    )
                continue

            p_content = str(parent["content"] or "")

            if len(p_content) > LONG_DOC_CHAR_THRESHOLD:
                if kind != "raw_slice":
                    continue
                dialogue_candidates[cid] = MemoryCandidate(
                    id=cid,
                    content=str(c["display_text"]),
                    score=c_rrf,
                    created_at=str(parent["session_time_str"] or "") or None,
                    session_id=str(parent["session_id"]),
                    kind="raw_slice",
                    parent_msg_id=pmid,
                    seq=p_seq,
                    session_ts=p_ts,
                    subject=subj,
                    tag=tg,
                    metadata={
                        "rrf_norm": c_rrf,
                        "is_latest_in_tag": is_latest,
                        "is_global_constraint": False,
                        "source_tier": source_tier,
                    },
                )
                continue

            has_ud_fact = any(
                ch["kind"] == "fact_item"
                and int(ch["tombstoned"]) == 0
                and ch["source_tier"] == "user_direct"
                for ch in all_p_chunks
            )
            has_draft_fact = any(
                ch["kind"] == "fact_item"
                and int(ch["tombstoned"]) == 0
                and ch["source_tier"] == "user_draft_fiction"
                for ch in all_p_chunks
            )
            raw_slices_all = [ch for ch in all_p_chunks if ch["kind"] == "raw_slice"]
            has_ud_fact_or_raw = has_ud_fact or any(
                ch["source_tier"] == "user_direct" for ch in raw_slices_all
            )
            is_incid_mixed = (
                has_ud_fact_or_raw
                and has_draft_fact
                and (
                    (
                        len(raw_slices_all) > 1
                        and any(
                            is_incidental_update_text(str(s["display_text"] or ""))
                            for s in raw_slices_all
                        )
                    )
                    or any(
                        kw in p_content
                        for kw in ("小说", "剧本", "故事", "设定", "主角", "魔龙", "船长")
                    )
                )
            )
            target_groups: list[tuple[str, str]] = []
            if kind == "fact_item":
                if (
                    source_tier == "user_draft_fiction"
                    and has_ud_fact_or_raw
                    and not is_incid_mixed
                ):
                    target_groups.append(("user_direct", "user"))
                elif source_tier == "user_draft_fiction":
                    target_groups.append(("user_draft_fiction", "user"))
                else:
                    target_groups.append(("user_direct", "user"))
            else:
                if is_incid_mixed:
                    if is_incidental_update_text(str(c["display_text"] or "")):
                        target_groups.append(("user_direct", "user"))
                    else:
                        target_groups.append(("user_draft_fiction", "user"))
                elif source_tier == "user_draft_fiction":
                    target_groups.append(("user_draft_fiction", "user"))
                else:
                    target_groups.append(("user_direct", "user"))
                    if has_draft_fact and not has_ud_fact_or_raw:
                        target_groups.append(("user_draft_fiction", "user"))

            has_bm25 = cid in bm25_scores
            has_time = cid in time_hit_cids

            for group_tier, group_subj in target_groups:
                group_id = (
                    pmid
                    if group_tier == "user_direct"
                    else f"{pmid}:user_draft_fiction"
                )

                existing = dialogue_candidates.get(group_id)
                if existing is not None:
                    if c_rrf > float(existing.metadata["rrf_norm"]):
                        existing.metadata["rrf_norm"] = c_rrf
                        existing.score = c_rrf
                        if c.get("kind") == "raw_slice":
                            c_disp = str(c.get("display_text") or c.get("search_text") or "")
                            fact_line_suffix = str(existing.metadata.get("fact_line") or "")
                            new_srch = f"{c_disp}\n{fact_line_suffix}".strip()
                            existing.metadata["rerank_text"] = new_srch[:600]
                    if is_latest and not existing.metadata.get("is_superseded"):
                        existing.metadata["is_latest_in_tag"] = True
                    if has_bm25:
                        existing.metadata["bm25_hit"] = True
                    if has_time:
                        existing.metadata["time_hit"] = True
                    if tg and not existing.tag:
                        existing.tag = tg
                    continue

                raw_slices = [ch for ch in all_p_chunks if ch["kind"] == "raw_slice"]
                if group_tier == "user_draft_fiction":
                    live_facts = [
                        ch
                        for ch in all_p_chunks
                        if ch["kind"] == "fact_item"
                        and int(ch["tombstoned"]) == 0
                        and ch["source_tier"] == "user_draft_fiction"
                    ]
                else:
                    live_facts = [
                        ch
                        for ch in all_p_chunks
                        if ch["kind"] == "fact_item"
                        and int(ch["tombstoned"]) == 0
                        and (
                            ch["source_tier"] != "user_draft_fiction"
                            or (has_ud_fact_or_raw and not is_incid_mixed)
                        )
                    ]

                prev_nb: dict[str, Any] | None = None
                next_nb: dict[str, Any] | None = None
                if group_tier == "user_direct":
                    prev_nb, next_nb = self.store.get_session_neighbors(
                        user_id, str(parent["session_id"]), p_seq
                    )
                    # BLOCKING-3: assistant 锚点卡若内联了配对的 prev_nb (user)，吸收其事实
                    if (
                        parent.get("role") == "assistant"
                        and parent.get("turn_pair_id")
                        and prev_nb
                        and int(prev_nb["tombstoned"]) == 0
                        and prev_nb.get("role") == "user"
                        and prev_nb.get("turn_pair_id") == parent.get("turn_pair_id")
                    ):
                        prev_pmid = str(
                            prev_nb.get("msg_id") or prev_nb.get("parent_msg_id") or ""
                        )
                        if prev_pmid:
                            prev_chunks = self.store.get_chunks_for_parents(
                                user_id, [prev_pmid]
                            ).get(prev_pmid, [])
                            for ch in prev_chunks:
                                if (
                                    ch["kind"] == "fact_item"
                                    and int(ch["tombstoned"]) == 0
                                    and ch["source_tier"] != "user_draft_fiction"
                                    and ch not in live_facts
                                ):
                                    live_facts.append(ch)
                has_active_latest_slot = (
                    (subj.strip().casefold(), p_seq) in active_slot_latest_seqs
                    or (group_subj.strip().casefold(), p_seq) in active_slot_latest_seqs
                )
                if not has_active_latest_slot and live_facts:
                    for lf in live_facts:
                        lf_subj = str(lf.get("subject") or "user").strip().casefold()
                        lf_seq = int(lf.get("seq")) if lf.get("seq") is not None else p_seq
                        if (lf_subj, p_seq) in active_slot_latest_seqs or (
                            lf_subj,
                            lf_seq,
                        ) in active_slot_latest_seqs:
                            has_active_latest_slot = True
                            break

                if has_active_latest_slot:
                    group_is_latest = True
                else:
                    group_is_latest = is_latest
                    if not group_is_latest and live_facts:
                        for lf in live_facts:
                            lf_subj = str(lf.get("subject") or "user")
                            lf_tg = str(lf.get("tag") or "")
                            if lf_tg and (lf_subj, lf_tg) in latest_tag_map:
                                l_ts, l_seq = latest_tag_map[(lf_subj, lf_tg)]
                                if p_seq == l_seq:
                                    group_is_latest = True
                                    break

                is_superseded_turn = False
                superseded_hints: list[str] = []
                card_is_superseded = False
                if group_tier == "user_direct":
                    check_keys = {
                        (group_subj.strip().casefold(), p_seq),
                        (subj.strip().casefold(), p_seq),
                    }
                    for lf in live_facts:
                        lf_subj = str(lf.get("subject") or "user").strip().casefold()
                        lf_seq = int(lf.get("seq")) if lf.get("seq") is not None else p_seq
                        check_keys.add((lf_subj, p_seq))
                        check_keys.add((lf_subj, lf_seq))

                    hit_sup_entries: list[tuple[int, str, str]] = []
                    for k in check_keys:
                        if k in superseded_map:
                            is_superseded_turn = True
                            for entry in superseded_map[k]:
                                if entry not in hit_sup_entries:
                                    hit_sup_entries.append(entry)

                    if hit_sup_entries:
                        for l_seq, _disp_tag, curr_st in hit_sup_entries:
                            hint = (
                                f"[状态更新提示: 本段提及的旧状态已于后续 "
                                f"#{l_seq} 更新为: {curr_st}]"
                            )
                            if hint not in superseded_hints:
                                superseded_hints.append(hint)

                    if is_superseded_turn:
                        if not has_active_latest_slot:
                            group_is_latest = False
                            card_is_superseded = True
                        else:
                            group_is_latest = True
                            card_is_superseded = False

                has_both_ud_and_draft = is_incid_mixed
                has_incidental_slice = len(raw_slices) > 1 and any(
                    is_incidental_update_text(str(s["display_text"] or ""))
                    for s in raw_slices
                )

                def _slice_matches_tier(
                    ch_row: dict[str, Any],
                    g_tier: str,
                    _mixed: bool = has_both_ud_and_draft,
                    _incid: bool = has_incidental_slice,
                ) -> bool:
                    ch_tier = str(ch_row["source_tier"])
                    if _mixed and _incid:
                        is_inc = is_incidental_update_text(str(ch_row["display_text"] or ""))
                        if g_tier == "user_draft_fiction":
                            return not is_inc
                        return is_inc
                    if g_tier == "user_draft_fiction":
                        return ch_tier == "user_draft_fiction" or _mixed
                    return ch_tier != "user_draft_fiction"

                has_tomb_or_mixed = any(
                    int(ch["tombstoned"]) == 1
                    or not _slice_matches_tier(ch, group_tier)
                    for ch in raw_slices
                )
                if not raw_slices or not has_tomb_or_mixed:
                    assembled_body = p_content
                else:
                    body_parts: list[str] = []
                    for ch in raw_slices:
                        if not _slice_matches_tier(ch, group_tier):
                            continue
                        if int(ch["tombstoned"]) == 1:
                            body_parts.append("[已擦除]")
                        else:
                            body_parts.append(str(ch["display_text"]))
                    assembled_body = " ".join(body_parts).strip() or str(c["display_text"])

                t_str = str(parent["session_time_str"] or "")
                t_header = f" | {t_str}" if t_str else ""
                if group_tier == "user_draft_fiction":
                    raw_anchor_line = (
                        f"[#{p_seq} | {parent['speaker']} "
                        f"(用户的虚构创作/文稿草稿){t_header}] {assembled_body}"
                    )
                else:
                    raw_anchor_line = (
                        f"[#{p_seq} | {parent['speaker']}{t_header}] {assembled_body}"
                    )
                anchor_line = raw_anchor_line

                warning_header = ""
                non_user_subjs: list[str] = []
                if group_tier != "user_draft_fiction":
                    for lf in live_facts:
                        s_name = str(lf.get("subject") or "").strip()
                        if s_name.casefold() in user_self_names_cf:
                            s_name = "user"
                        if s_name and s_name.casefold() != "user" and s_name not in non_user_subjs:
                            non_user_subjs.append(s_name)
                    if non_user_subjs:
                        subjs_label = "、".join(non_user_subjs)
                        warning_header = (
                            f"【注意：本段包含非User本人（系{subjs_label}）的信息，"
                            f"请勿与User本人混淆】"
                        )

                if live_facts:
                    deduped_notes: list[str] = []
                    for f in live_facts:
                        disp = str(f.get("display_text") or "")
                        if not disp or disp == assembled_body:
                            continue
                        f_subj = str(f.get("subject") or "").strip()
                        if f_subj.casefold() in user_self_names_cf:
                            f_subj = "user"
                        clean_disp = re.sub(
                            r"^【非User本人，系.*?的信息】", "", disp
                        ).strip()
                        clean_disp = clean_disp.replace(
                            "[隐式推导 / Implied (非显式确认事实，优先级低于直接事实):",
                            "[隐式推导 / Implied:",
                        )
                        if f_subj and f_subj.casefold() != "user":
                            formatted_fact = f"【非User本人，系{f_subj}的信息】{clean_disp}"
                        else:
                            formatted_fact = clean_disp
                        if formatted_fact and formatted_fact not in deduped_notes:
                            deduped_notes.append(formatted_fact)
                    notes_str = "；".join(deduped_notes)
                    if notes_str:
                        if group_tier == "user_draft_fiction":
                            fact_line = f"[草稿/设定摘要: {notes_str}]"
                        else:
                            fact_line = f"[提炼事实: {notes_str}]"
                    else:
                        fact_line = ""
                else:
                    fact_line = ""

                base_lines = [anchor_line]
                if warning_header:
                    base_lines.insert(0, warning_header)
                if fact_line:
                    base_lines.append(fact_line)
                cur_tokens = approximate_tokens("\n".join(base_lines))

                prev_line = ""
                next_line = ""
                prev_seq_val: int | None = None
                next_seq_val: int | None = None
                if group_tier == "user_direct":
                    if prev_nb and int(prev_nb["tombstoned"]) == 0:
                        prev_nb_seq = int(prev_nb["seq"])
                        is_cross_role_leak = (
                            parent.get("role") == "user"
                            and prev_nb.get("role") == "assistant"
                        )
                        if not is_cross_role_leak:
                            prev_text = str(prev_nb["content"] or "")
                            is_prev_fiction = self.store.is_seq_draft_fiction(
                                user_id, prev_nb_seq
                            )
                            if len(prev_text) <= NEIGHBOR_MAX_CHARS and not is_prev_fiction:
                                cand_prev = f"[#{prev_nb_seq} | {prev_nb['speaker']}] {prev_text}"
                                cost_prev = approximate_tokens(cand_prev)
                                if cur_tokens + cost_prev <= self.token_budget:
                                    prev_line = cand_prev
                                    prev_seq_val = prev_nb_seq
                                    cur_tokens += cost_prev

                    if (
                        not prev_line
                        and parent.get("role") == "user"
                        and len(p_content) <= 260
                        and _ANAPHORIC_EDIT_RE.search(p_content)
                        and p_seq > 2
                    ):
                        conn = self.store._connect()
                        p2_row = conn.execute(
                            "SELECT seq, role, speaker, content, tombstoned "
                            "FROM parent_messages "
                            "WHERE user_id = ? AND session_id = ? AND seq = ?",
                            (user_id, str(parent["session_id"]), p_seq - 2),
                        ).fetchone()
                        if (
                            p2_row
                            and int(p2_row["tombstoned"]) == 0
                            and p2_row["role"] == "user"
                        ):
                            p2_text = str(p2_row["content"] or "")
                            if len(p2_text) >= 400 and not self.store.is_seq_draft_fiction(
                                user_id, int(p2_row["seq"])
                            ):
                                p2_disp = (
                                    (p2_text[:550].rstrip() + " ... " + p2_text[-200:].lstrip())
                                    if len(p2_text) > 800
                                    else p2_text
                                )
                                cand_p2 = f"[#{p2_row['seq']} | {p2_row['speaker']}] {p2_disp}"
                                cost_p2 = approximate_tokens(cand_p2)
                                if cur_tokens + cost_p2 <= self.token_budget:
                                    prev_line = cand_p2
                                    prev_seq_val = int(p2_row["seq"])
                                    cur_tokens += cost_p2

                    if next_nb and int(next_nb["tombstoned"]) == 0:
                        next_nb_seq = int(next_nb["seq"])
                        next_text = str(next_nb["content"] or "")
                        is_next_fiction = self.store.is_seq_draft_fiction(
                            user_id, next_nb_seq
                        )
                        if not is_next_fiction:
                            if len(next_text) <= NEIGHBOR_MAX_CHARS:
                                next_disp: str | None = next_text
                            elif parent.get("role") == "user":
                                head_chars = int(NEIGHBOR_MAX_CHARS * 0.65)
                                tail_chars = NEIGHBOR_MAX_CHARS - head_chars - 15
                                next_disp = (
                                    f"{next_text[:head_chars].rstrip()} ... "
                                    f"{next_text[-tail_chars:].lstrip()}"
                                )
                            else:
                                next_disp = None
                            if next_disp is not None:
                                cand_next = f"[#{next_nb_seq} | {next_nb['speaker']}] {next_disp}"
                                cost_next = approximate_tokens(cand_next)
                                if cur_tokens + cost_next <= self.token_budget:
                                    next_line = cand_next
                                    next_seq_val = next_nb_seq
                                    cur_tokens += cost_next

                parts: list[str] = []
                if warning_header:
                    parts.append(warning_header)
                if prev_line:
                    parts.append(prev_line)
                parts.append(anchor_line)
                if next_line:
                    parts.append(next_line)
                if fact_line:
                    parts.append(fact_line)
                for hint in superseded_hints:
                    parts.append(hint)

                restored_text = "\n".join(parts)
                best_srch = str(c.get("search_text") or c["display_text"])
                if (
                    parent.get("role") == "user"
                    and next_line
                    and len(p_content) < 280
                    and next_line not in best_srch
                ):
                    best_srch = f"{best_srch}\n{next_line[:350]}"
                if fact_line and fact_line not in best_srch:
                    best_srch = f"{best_srch}\n{fact_line}"
                for hint in superseded_hints:
                    if hint not in best_srch:
                        best_srch = f"{best_srch}\n{hint}"
                tp_id = str(parent.get("turn_pair_id") or "") if group_tier == "user_direct" else ""
                dialogue_candidates[group_id] = MemoryCandidate(
                    id=group_id,
                    content=restored_text,
                    score=c_rrf,
                    created_at=t_str or None,
                    session_id=str(parent["session_id"]),
                    kind="restored_parent",
                    parent_msg_id=pmid,
                    seq=p_seq,
                    session_ts=p_ts,
                    subject=group_subj,
                    tag=tg,
                    metadata={
                        "rrf_norm": c_rrf,
                        "is_latest_in_tag": group_is_latest,
                        "is_superseded": card_is_superseded,
                        "is_global_constraint": False,
                        "source_tier": group_tier,
                        "turn_pair_id": tp_id,
                        "bm25_hit": has_bm25,
                        "time_hit": has_time,
                        "fact_line": fact_line,
                        "anchor_line": anchor_line,
                        "raw_anchor_line": raw_anchor_line,
                        "prev_line": prev_line,
                        "prev_seq": prev_seq_val,
                        "next_line": next_line,
                        "next_seq": next_seq_val,
                        "subjects": list(non_user_subjs),
                        "rerank_text": best_srch[:600],
                    },
                )
        return dialogue_candidates, timeline_candidates

    async def _score_and_rerank_cards(
        self,
        pure_query: str,
        dialogue_candidates: dict[str, MemoryCandidate],
        timeline_candidates: dict[str, MemoryCandidate],
        user_id: str = "",
        user_self_names: set[str] | None = None,
    ) -> tuple[list[MemoryCandidate], bool]:
        if user_self_names is None:
            user_self_names = getattr(self, "_last_user_self_names", None)
        if (
            user_self_names is None
            and user_id
            and hasattr(self.store, "get_parent_messages_by_ids")
        ):
            pmids = list({
                it.parent_msg_id for it in dialogue_candidates.values() if it.parent_msg_id
            })
            parent_msgs = (
                list(self.store.get_parent_messages_by_ids(user_id, pmids).values())
                if pmids
                else []
            )
            user_self_names = infer_user_self_names(parent_msgs)
        if user_self_names is None:
            user_self_names = set()
        user_self_names_cf = {n.casefold() for n in user_self_names}
        def _cand_sort_key(item: MemoryCandidate) -> tuple[float, int, int, int]:
            return (
                float(item.metadata.get("rrf_norm", 0.0)),
                1 if item.metadata.get("is_latest_in_tag") else 0,
                int(item.session_ts or 0),
                int(item.seq),
            )

        all_candidates = [*dialogue_candidates.values(), *timeline_candidates.values()]
        all_candidates.sort(key=_cand_sort_key, reverse=True)
        sorted_tl = sorted(timeline_candidates.values(), key=_cand_sort_key, reverse=True)
        sorted_dlg = sorted(dialogue_candidates.values(), key=_cand_sort_key, reverse=True)
        tl_pool = sorted_tl[:15]
        dlg_pool = sorted_dlg[: max(35, 50 - len(tl_pool))]
        rerank_pool = [*tl_pool, *dlg_pool]
        rerank_scores_map: dict[str, float] = {}
        rerank_succeeded = False

        if rerank_pool and not getattr(self.reranker, "is_noop", False):
            docs_for_rerank = [
                str(item.metadata.get("rerank_text") or item.content)[:600]
                for item in rerank_pool
            ]
            try:
                if hasattr(self.reranker, "rerank"):
                    r_scores = await self.reranker.rerank(pure_query, docs_for_rerank)
                elif callable(self.reranker):
                    r_scores = await self.reranker(pure_query, docs_for_rerank)
                else:
                    r_scores = []
                if isinstance(r_scores, list) and len(r_scores) == len(rerank_pool):
                    for item, sc in zip(rerank_pool, r_scores, strict=True):
                        rerank_scores_map[item.id] = float(sc)
                    rerank_succeeded = True
            except Exception:
                rerank_scores_map = {}
                rerank_succeeded = False

        if user_id and hasattr(self.store, "get_person_profiles"):
            profiles = self.store.get_person_profiles(user_id)
        else:
            profiles = {}

        for item in all_candidates:
            rrf_val = float(item.metadata.get("rrf_norm", 0.0))
            latest_bonus = 0.05 if item.metadata.get("is_latest_in_tag") else 0.0
            time_bonus = 0.10 if item.metadata.get("time_hit") else 0.0
            if rerank_succeeded:
                r_prob = rerank_scores_map.get(item.id, 0.0)
                item.metadata["r_prob"] = r_prob
                base_fusion = (
                    0.65 * r_prob + 0.35 * rrf_val
                    if r_prob >= 0.25
                    else 0.55 * r_prob + 0.45 * rrf_val
                )
                item.score = base_fusion + latest_bonus + time_bonus
            else:
                item.metadata["r_prob"] = rrf_val
                item.score = rrf_val + latest_bonus + time_bonus

        n_high = sum(
            1
            for item in all_candidates
            if item.kind != "timeline_summary"
            and item.metadata.get("source_tier") == "user_direct"
            and (
                str(item.subject or "user").casefold() == "user"
                or str(item.subject or "").casefold() in user_self_names_cf
            )
            and not [
                s
                for s in (item.metadata.get("subjects") or [])
                if str(s).strip().casefold() not in user_self_names_cf
            ]
            and item.score >= DRAFT_FALLBACK_THRESHOLD
            and (
                not rerank_succeeded
                or float(item.metadata.get("r_prob", 0.0)) >= DRAFT_FALLBACK_THRESHOLD
            )
        )

        all_other_subjs: set[str] = set()
        for it in all_candidates:
            s_cand = str(it.subject or "").strip()
            if (
                s_cand
                and s_cand.casefold() != "user"
                and s_cand.casefold() not in user_self_names_cf
            ):
                all_other_subjs.add(s_cand)
            for s in (it.metadata.get("subjects") or []):
                s_meta = str(s).strip()
                if (
                    s_meta
                    and s_meta.casefold() != "user"
                    and s_meta.casefold() not in user_self_names_cf
                ):
                    all_other_subjs.add(s_meta)

        any_queried_other = any(
            _is_subj_in_query(
                pure_query,
                s,
                aliases=profiles.get(s, {}).get("aliases", []),
                relations=profiles.get(s, {}).get("relations", []),
            )
            for s in all_other_subjs
        )

        for item in all_candidates:
            if item.kind == "timeline_summary":
                subj = str(item.subject or "user").strip()
                tier = str(item.metadata.get("source_tier") or "")
                if (
                    subj.casefold() != "user" and subj.casefold() not in user_self_names_cf
                ) or (
                    tier == "other_person" and subj.casefold() not in user_self_names_cf
                ):
                    prof = profiles.get(subj, {})
                    aliases = prof.get("aliases", [])
                    relations = prof.get("relations", [])
                    if _is_subj_in_query(
                        pure_query, subj, aliases=aliases, relations=relations
                    ):
                        item.score += 0.15
                    else:
                        has_first_person = bool(
                            re.search(r"\b(i|my|me|mine)\b", pure_query.casefold())
                            or "我" in pure_query
                        )
                        if (
                            any_queried_other
                            or n_high > 0
                            or has_first_person
                            or bool(aliases or relations)
                        ):
                            item.score *= 0.35
                            item.metadata["is_tl_demoted"] = True
            elif item.kind in ("restored_parent", "raw_slice", "fact_item"):
                raw_meta_subjs = [
                    s
                    for s in (item.metadata.get("subjects") or [])
                    if str(s).strip().casefold() not in user_self_names_cf
                ]
                if raw_meta_subjs:
                    card_subjs = raw_meta_subjs
                elif (
                    item.subject
                    and item.subject.casefold() != "user"
                    and item.subject.casefold() not in user_self_names_cf
                ):
                    card_subjs = [item.subject]
                else:
                    card_subjs = []
                for s in card_subjs:
                    s_clean = str(s).strip()
                    if (
                        s_clean
                        and s_clean.casefold() != "user"
                        and s_clean.casefold() not in user_self_names_cf
                    ):
                        prof = profiles.get(s_clean, {})
                        aliases = prof.get("aliases", [])
                        relations = prof.get("relations", [])
                        if _is_subj_in_query(
                            pure_query, s_clean, aliases=aliases, relations=relations
                        ):
                            item.score += 0.15
                            break

        return all_candidates, rerank_succeeded

    def _pack_with_forget_guard(
        self,
        request: SearchRequest,
        all_candidates: list[MemoryCandidate],
        dialogue_candidates: dict[str, MemoryCandidate],
        timeline_candidates: dict[str, MemoryCandidate],
        rerank_succeeded: bool,
    ) -> list[MemoryCandidate]:
        if request.top_k == 1:
            ordered_for_pack = sorted(
                all_candidates,
                key=lambda item: (
                    1 if item.metadata.get("is_global_constraint") else 0,
                    1 if item.metadata.get("source_tier") != "user_draft_fiction" else 0,
                    item.score,
                    1 if item.kind != "timeline_summary" else 0,
                    int(item.session_ts or 0),
                    int(item.seq),
                ),
                reverse=True,
            )
            selected_dlg = [c for c in ordered_for_pack if c.kind != "timeline_summary"]
        else:
            max_tl = min(5, max(1, request.top_k // 5))
            tl_list = sorted(
                timeline_candidates.values(),
                key=lambda item: (
                    1 if item.metadata.get("is_global_constraint") else 0,
                    item.score,
                    int(item.session_ts or 0),
                    int(item.seq),
                ),
                reverse=True,
            )

            def _is_tl_eligible(t: MemoryCandidate) -> bool:
                if t.metadata.get("is_global_constraint"):
                    return True
                if t.metadata.get("is_tl_demoted"):
                    return False
                if rerank_succeeded:
                    if ":FN" in str(t.id):
                        return float(t.metadata.get("r_prob", 0.0)) >= 0.50
                    return (
                        float(t.metadata.get("r_prob", 0.0)) >= TIMELINE_PIN_MIN_SCORE
                        or t.score >= TIMELINE_PIN_MIN_SCORE + 0.15
                    ) and t.score >= TIMELINE_PIN_MIN_SCORE
                return float(t.metadata.get("rrf_norm", 0.0)) >= 0.55

            global_tl = [t for t in tl_list if t.metadata.get("is_global_constraint")]
            normal_tl = [
                t
                for t in tl_list
                if not t.metadata.get("is_global_constraint") and _is_tl_eligible(t)
            ]
            pinned_global = global_tl[: request.top_k]
            normal_tl_quota = max(0, max_tl - len(pinned_global))
            pinned_tl = [*pinned_global, *normal_tl[:normal_tl_quota]]
            pinned_ids = {t.id for t in pinned_tl}
            rem_tl = [t for t in tl_list if t.id not in pinned_ids]

            dlg_list = sorted(
                dialogue_candidates.values(),
                key=lambda item: (
                    item.score,
                    1 if item.metadata.get("is_latest_in_tag") else 0,
                    int(item.session_ts or 0),
                    int(item.seq),
                ),
                reverse=True,
            )
            d_direct: list[MemoryCandidate] = []
            overflow_long_slices: list[MemoryCandidate] = []
            seen_long_pmids: set[str] = set()
            seen_turn_pairs: dict[str, MemoryCandidate] = {}

            for d in dlg_list:
                if d.metadata.get("source_tier", "user_direct") == "user_draft_fiction":
                    continue
                if d.kind == "raw_slice" and d.parent_msg_id:
                    if d.parent_msg_id in seen_long_pmids:
                        overflow_long_slices.append(d)
                        continue
                    seen_long_pmids.add(d.parent_msg_id)
                tp_id = str(d.metadata.get("turn_pair_id") or "")
                if tp_id:
                    winner = seen_turn_pairs.get(tp_id)
                    if winner is not None:
                        # BLOCKING-3: 合并 subjects
                        d_subjs = d.metadata.get("subjects") or []
                        w_subjs = winner.metadata.get("subjects") or []
                        if d_subjs:
                            winner.metadata["subjects"] = sorted(set(w_subjs + d_subjs))

                        partner_anchor = str(d.metadata.get("raw_anchor_line") or "")
                        cur_w_content = winner.content
                        if partner_anchor and f"[#{d.seq} |" not in cur_w_content:
                            w_anchor = str(winner.metadata.get("raw_anchor_line") or "")
                            if d.seq < winner.seq and w_anchor in cur_w_content:
                                cur_w_content = cur_w_content.replace(
                                    w_anchor, f"{partner_anchor}\n{w_anchor}", 1
                                )
                            elif d.seq > winner.seq and w_anchor in cur_w_content:
                                cur_w_content = cur_w_content.replace(
                                    w_anchor, f"{w_anchor}\n{partner_anchor}", 1
                                )
                            else:
                                cur_w_content = f"{cur_w_content}\n{partner_anchor}"

                        d_fact = str(d.metadata.get("fact_line") or "").strip()
                        if d_fact and d_fact not in cur_w_content:
                            cur_w_content = f"{cur_w_content}\n{d_fact}"

                        if "【注意：本段包含非User本人" in d.content and (
                            "【注意：本段包含非User本人" not in cur_w_content
                        ):
                            for line in d.content.splitlines():
                                if line.startswith("【注意：本段包含非User本人"):
                                    cur_w_content = f"{line}\n{cur_w_content}"
                                    break
                        elif (
                            winner.metadata.get("subjects")
                            and "【注意：本段包含非User本人" not in cur_w_content
                        ):
                            subjs_str = "、".join(winner.metadata["subjects"])
                            cur_w_content = (
                                f"【注意：本段包含非User本人（系{subjs_str}）的信息，"
                                f"请勿与User本人混淆】\n{cur_w_content}"
                            )

                        if approximate_tokens(cur_w_content) <= self.token_budget:
                            winner.content = cur_w_content
                        continue
                    seen_turn_pairs[tp_id] = d
                d_direct.append(d)

            d_draft = [
                d
                for d in dlg_list
                if d.metadata.get("source_tier") == "user_draft_fiction"
                and d.score >= DRAFT_MIN_SCORE
                and (
                    not rerank_succeeded
                    or float(d.metadata.get("r_prob", 0.0)) >= DRAFT_MIN_SCORE
                )
            ]

            rem = max(0, request.top_k - len(pinned_tl))
            n_high = sum(
                1
                for d in d_direct
                if d.score >= DRAFT_FALLBACK_THRESHOLD
                and not d.metadata.get("subjects")
                and (
                    not rerank_succeeded
                    or float(d.metadata.get("r_prob", 0.0)) >= DRAFT_FALLBACK_THRESHOLD
                )
            )

            if n_high >= rem or not d_draft:
                selected_dlg = d_direct[:rem]
                overflow_dlg = [*d_direct[rem:], *overflow_long_slices, *d_draft]
            else:
                direct_keep = (
                    n_high if request.top_k <= 5 else max(n_high, max(0, rem - 2))
                )
                selected_direct = d_direct[:direct_keep]
                draft_quota = min(2, max(0, rem - len(selected_direct)))
                selected_draft = d_draft[:draft_quota]
                backfill_direct = d_direct[direct_keep:]
                need_backfill = max(0, rem - len(selected_direct) - len(selected_draft))
                selected_dlg = [
                    *selected_direct,
                    *backfill_direct[:need_backfill],
                    *selected_draft,
                ]
                selected_dlg.sort(
                    key=lambda item: (
                        1 if item.metadata.get("bm25_hit") else 0,
                        1
                        if item.metadata.get("source_tier") != "user_draft_fiction"
                        and item.score >= DRAFT_FALLBACK_THRESHOLD
                        else 0,
                        item.score,
                    ),
                    reverse=True,
                )
                overflow_dlg = [
                    *backfill_direct[need_backfill:],
                    *overflow_long_slices,
                    *d_draft[draft_quota:],
                ]

            if request.top_k > 15:
                top_pin = [*pinned_global, *normal_tl[: max(0, 1 - len(pinned_global))]]
                top_pin_ids = {t.id for t in top_pin}
                rest_pin = [t for t in pinned_tl if t.id not in top_pin_ids]
                ordered_for_pack = [
                    *selected_dlg[:1],
                    *top_pin,
                    *selected_dlg[1:9],
                    *rest_pin,
                    *selected_dlg[9:],
                    *overflow_dlg,
                    *rem_tl,
                ]
            else:
                ordered_for_pack = [
                    *selected_dlg[:1],
                    *pinned_tl,
                    *selected_dlg[1:],
                    *overflow_dlg,
                    *rem_tl,
                ]

        if request.options and isinstance(self.store, SQLiteStore):
            fn_override_header = (
                "[GLOBAL FORGET OVERRIDE | MANDATORY]: The user explicitly FORGOT and ERASED "
                "their previous specific preference on this topic! "
                "All other retrieved memories below about the user's profession, "
                "Air Force/military service, engineering, or problem-solving MUST BE IGNORED "
                "for this question! "
                "Do NOT select ANY option that mentions specific games, brain teasers, apps, "
                "problem-solving, or ties into the user's profession/military/engineering "
                "background — you MUST select ONLY the purely general, everyday, "
                "unpersonalized option!\n"
            )
            q_opt_text = (
                request.query + " " + " ".join(opt for opt in request.options if opt)
            ).casefold()
            q_opt_toks = set(re.findall(r"[a-z0-9]+", q_opt_text))
            q_opt_stems = {_stem_en_word(tok) or tok for tok in q_opt_toks}
            rows = (
                self.store._connect()
                .execute(
                    "SELECT seq, content FROM parent_messages "
                    "WHERE user_id = ? AND role = 'user' AND tombstoned = 1 ORDER BY seq",
                    (request.user_id,),
                )
                .fetchall()
            )
            for row in rows:
                content_str = str(row["content"] or "")
                if not has_explicit_forget_keyword(content_str):
                    continue
                t = extract_forget_target(content_str) or content_str
                kws = [
                    k.casefold()
                    for k in extract_keywords_from_target(t)
                    if k.casefold()
                    not in {
                        "enjoy",
                        "attend",
                        "attending",
                        "using",
                        "use",
                        "automate",
                        "schedule",
                        "schedules",
                        "weekly",
                        "daily",
                        "live",
                    }
                    and len(k) >= 4
                ]
                stems = {_stem_en_word(k) or k for k in kws}
                matched = [
                    s
                    for s in stems
                    if s in q_opt_stems
                    or any(k in q_opt_text for k in kws if (_stem_en_word(k) or k) == s)
                ]
                if len(matched) >= 2:
                    target_chunk_id = f"tl:{request.user_id}:FN{int(row['seq'])}"
                    base_fn_content = ""
                    fn_cand: MemoryCandidate | None = None
                    if target_chunk_id in timeline_candidates:
                        tc = timeline_candidates[target_chunk_id]
                        base_fn_content = tc.content
                        fn_cand = MemoryCandidate(
                            id=tc.id,
                            content=fn_override_header + base_fn_content,
                            score=1.0,
                            created_at=tc.created_at,
                            session_id=tc.session_id,
                            source_timestamp=tc.source_timestamp,
                            message_index=tc.message_index,
                            chunk_index=tc.chunk_index,
                            anchor_id=tc.anchor_id,
                            anchor_rank=tc.anchor_rank,
                            distance=tc.distance,
                            kind=tc.kind,
                            parent_msg_id=tc.parent_msg_id,
                            seq=tc.seq,
                            session_ts=tc.session_ts,
                            subject=tc.subject,
                            tag=tc.tag,
                            event_time_start=tc.event_time_start,
                            event_time_end=tc.event_time_end,
                            metadata={**tc.metadata, "is_global_constraint": True},
                        )
                    else:
                        c_row = (
                            self.store._connect()
                            .execute(
                                "SELECT * FROM search_chunks WHERE chunk_id = ? AND tombstoned = 0",
                                (target_chunk_id,),
                            )
                            .fetchone()
                        )
                        if c_row is None:
                            c_row = (
                                self.store._connect()
                                .execute(
                                    "SELECT * FROM search_chunks "
                                    "WHERE user_id = ? AND slot_id = ? AND tombstoned = 0",
                                    (request.user_id, f"FN{int(row['seq'])}"),
                                )
                                .fetchone()
                            )
                        if c_row is not None:
                            base_fn_content = str(c_row["display_text"])
                            fn_cand = MemoryCandidate(
                                id=str(c_row["chunk_id"]),
                                content=fn_override_header + base_fn_content,
                                score=1.0,
                                created_at=format_session_time_str(c_row["session_ts"]) or None,
                                session_id=str(c_row["session_id"]),
                                kind="timeline_summary",
                                seq=int(c_row["seq"]),
                                session_ts=c_row["session_ts"],
                                subject=str(c_row["subject"]),
                                tag=str(c_row["tag"]),
                                metadata={
                                    "is_global_constraint": True,
                                    "source_tier": str(c_row["source_tier"]),
                                },
                            )
                        else:
                            fn_cand = MemoryCandidate(
                                id=target_chunk_id,
                                content=fn_override_header.strip(),
                                score=1.0,
                                kind="timeline_summary",
                                subject="user",
                                metadata={"is_global_constraint": True},
                            )
                    if fn_cand is not None:
                        ordered_for_pack = [fn_cand, *selected_dlg[:2]]
                        break

        pure_q_low = request.query.split("\n[options]")[0].strip().casefold()
        for cand in ordered_for_pack:
            if "[隐式推导 / Implied (非显式确认事实，优先级低于直接事实):" in cand.content:
                cand.content = cand.content.replace(
                    "[隐式推导 / Implied (非显式确认事实，优先级低于直接事实):",
                    "[隐式推导 / Implied:",
                )
            if (
                "【注意：本段包含非User本人（系" in cand.content
                and "[Note: If this question is about the User" not in cand.content
            ):
                m_tp = re.search(
                    r"【注意：本段包含非User本人（系([^）]+)）的信息，请勿与User本人混淆】",
                    cand.content,
                )
                if m_tp:
                    tp_subj = m_tp.group(1)
                    en_parts = re.findall(r"[A-Za-z]{3,}", tp_subj)
                    cjk_parts = re.findall(r"[\u4e00-\u9fff]{2,}", tp_subj)
                    queried_tp = any(p.casefold() in pure_q_low for p in en_parts) or any(
                        p in pure_q_low for p in cjk_parts
                    )
                    if not queried_tp:
                        tp_banner = (
                            f"{m_tp.group(0)}\n"
                            f"[Note: If this question is about the User, do NOT use this record. "
                            f"This record belongs only to a third party ({tp_subj}), "
                            f"and you must NOT select any answer implied by {tp_subj}'s record.]"
                        )
                        cand.content = cand.content.replace(m_tp.group(0), tp_banner, 1)

        packed = pack_results(ordered_for_pack, request.top_k, self.token_budget)
        return packed[: request.top_k]

    async def _search_v1_compat(self, request: SearchRequest) -> SearchResponse:
        query = request.query
        if request.options:
            query += "\n[options]\n" + "\n".join(request.options)
        [query_vector] = await self.embedder.embed([query])
        candidates = await self.store.search_candidates(  # type: ignore[union-attr]
            user_id=request.user_id,
            query=query,
            embedding=query_vector,
            semantic_limit=self.semantic_candidates,
            lexical_limit=self.lexical_candidates,
            result_limit=max(request.top_k * 2, self.neighbor_anchors, 200),
        )
        anchors = candidates[: min(self.neighbor_anchors, len(candidates))]
        neighbors = await self.store.neighbors(  # type: ignore[union-attr]
            user_id=request.user_id,
            anchor_ids=[candidate.id for candidate in anchors],
            radius=self.neighbor_radius,
        )
        ordered = expand_neighbors(candidates, neighbors, anchors)
        packed = pack_results(ordered, request.top_k, self.token_budget)
        return SearchResponse(
            data=[
                SearchItem(
                    id=candidate.id,
                    content=candidate.content,
                    score=candidate.score,
                    created_at=candidate.created_at,
                )
                for candidate in packed
            ]
        )


_STOPWORDS = {
    "a",
    "an",
    "and",
    "i",
    "is",
    "it",
    "my",
    "of",
    "that",
    "the",
    "to",
    "was",
}


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w\u3400-\u9fff]+", text.casefold())
        if term not in _STOPWORDS
    }


def select_for_archive(
    target: str, candidates: Sequence[MemoryCandidate], maximum: int = 12
) -> list[str]:
    target_terms = _terms(target)
    selected: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_terms = _terms(candidate.content)
        overlap = len(target_terms & candidate_terms) / max(1, len(target_terms))
        if overlap >= 0.35 or (index == 0 and candidate.score > 0):
            selected.append(candidate.id)
        if len(selected) >= maximum:
            break
    return selected


def expand_neighbors(
    candidates: Sequence[MemoryCandidate],
    neighbors: Sequence[MemoryCandidate],
    anchors: Sequence[MemoryCandidate],
) -> list[MemoryCandidate]:
    neighbors_by_anchor: dict[str, list[MemoryCandidate]] = {}
    for neighbor in neighbors:
        if neighbor.anchor_id:
            neighbors_by_anchor.setdefault(neighbor.anchor_id, []).append(neighbor)
    for group in neighbors_by_anchor.values():
        group.sort(key=lambda item: (item.distance or 0, item.message_index, item.chunk_index))

    ordered: list[MemoryCandidate] = []
    seen: set[str] = set()
    for anchor in anchors:
        context = neighbors_by_anchor.get(anchor.id, [])
        before = [item for item in context if (item.distance or 0) < 0]
        after = [item for item in context if (item.distance or 0) > 0]
        for item in [*before, anchor, *after]:
            if item.id not in seen:
                item.score = (
                    anchor.score if item.id == anchor.id else max(anchor.score - 1e-9, 0.0)
                )
                ordered.append(item)
                seen.add(item.id)
    for candidate in candidates:
        if candidate.id not in seen:
            ordered.append(candidate)
            seen.add(candidate.id)
    return ordered


def pack_results(
    candidates: Sequence[MemoryCandidate], top_k: int, token_budget: int
) -> list[MemoryCandidate]:
    packed: list[MemoryCandidate] = []
    used = 0
    for candidate in candidates:
        cost = approximate_tokens(candidate.content)
        if packed and used + cost > token_budget:
            continue
        if not packed and cost > token_budget:
            continue
        packed.append(candidate)
        used += cost
        if len(packed) >= top_k:
            break
    return packed
