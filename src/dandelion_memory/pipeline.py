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
    add_temporal_hints,
    approximate_tokens,
    build_bm25_tokens,
    extract_forget_target,
    extract_query_time_range,
    extract_speaker,
    format_session_time_str,
    get_base_tag,
    has_explicit_forget_keyword,
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
        self._user_locks: dict[str, asyncio.Lock] = {}

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
        extract_deadline = add_t0 + 7.5
        total_deadline = add_t0 + 9.0
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
        if active_batch:
            if len(active_batch) <= 16:
                try:
                    rem_ext = max(0.05, extract_deadline - time.monotonic())
                    raw_ext = await asyncio.wait_for(
                        self._invoke_extractor(
                            active_batch, existing_tags, living_facts=living_facts
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
                                    win_msgs, existing_tags, living_facts=living_facts
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

            primary_rec = seq_to_rec[primary_seq]
            primary_msg_id = str(primary_rec["msg_id"])
            primary_session_ts = primary_rec["session_ts"]
            primary_time_str = str(primary_rec["session_time_str"] or "")
            ev_start, ev_end = parse_event_date_range(
                fact.event_date_start,
                fact.event_date_end,
                fallback_ts=primary_session_ts,
            )
            fact_chunk_id = f"fact:{request.request_id}:{f_idx:04d}"
            fact_search_text = (
                f"[{fact.subject} | {fact.tag}] {fact.tag} {fact.current_state} {formatted_note}"
            ).strip()
            prepared_facts.append(
                {
                    "fact": fact,
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
            valid_forgets.append(
                {
                    "fa": fa,
                    "trig_rec": trig_rec,
                    "target_vec": None,
                }
            )

        n_raw = len(pending_raw_chunks)
        all_embed_inputs: list[str] = [
            *(c["embed_input"] for c in pending_raw_chunks),
            *(pf["embed_input"] for pf in prepared_facts),
        ]
        forget_embed_indices: list[tuple[int, int]] = []
        for vf_idx, vf in enumerate(valid_forgets):
            _check_deadline()
            desc = vf["fa"].target_description
            if desc:
                forget_embed_indices.append((vf_idx, len(all_embed_inputs)))
                all_embed_inputs.append(desc)

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
                        rendered_summary, tl_seq, tl_ts, resolved_tag, is_fact_tombstoned = (
                            self.store.upsert_tag_timeline(
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
                            )
                        )
                        _check_deadline()
                        fact.tag = resolved_tag
                        dirty_timeline_keys.append((fact.subject, fact.tag))

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

            self.store.record_add_request(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
                payload_hash=payload_hash,
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
        chunks = self.store.get_active_chunks(request.user_id)
        if not chunks:
            return SearchResponse(data=[])

        # 1. Query embedding & Multiple-choice options 0.7 / 0.3 synthesis (F13, P1-3)
        pure_query = request.query
        if request.options:
            opt_texts = [opt[:150] for opt in request.options if opt and opt.strip()]
        else:
            opt_texts = []

        if opt_texts:
            opt_concat = pure_query + "\n" + "\n".join(opt_texts)
            vec_q, vec_opt = await self.embedder.embed([pure_query, opt_concat])
            dim = min(len(vec_q), len(vec_opt))
            combined_vec = [0.7 * vec_q[i] + 0.3 * vec_opt[i] for i in range(dim)]
            query_vec = _l2_normalize(combined_vec)
        else:
            [query_vec] = await self.embedder.embed([pure_query])

        # 2. BM25 query term weights (2.0 for query stem, 0.5 for options)
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

        # 3. Three-channel retrieval + Weighted RRF (K=60)
        dense_scores: dict[str, float] = {}
        for c in chunks:
            cid = str(c["chunk_id"])
            dense_scores[cid] = cosine_similarity(query_vec, c["vector"])

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

        # P0-1: Only include recalled chunks from the 3 channels + resident global constraints
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
            return SearchResponse(data=[])

        max_rrf = max(rrf_raw.values()) if rrf_raw else 1.0
        rrf_norm_map = {cid: val / max_rrf for cid, val in rrf_raw.items()}
        recalled_chunks = [c for c in chunks if str(c["chunk_id"]) in rrf_raw]

        # Index-to-Raw Pull (v2.3 INV-4 / T4): pull active raw dialogue for recalled timelines
        timeline_pull_map = self.store.get_timeline_pull_map(request.user_id)
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

        # 4. Module D: Homogeneous (parent_msg_id, source_tier) grouping & assembly (v2.2/v2.3)
        latest_tag_map = self.store.get_latest_tag_map(request.user_id)
        superseded_seqs_map = self.store.get_superseded_seqs_map(request.user_id)
        parent_ids = [
            str(c["parent_msg_id"])
            for c in recalled_chunks
            if c["kind"] != "timeline_summary" and c["parent_msg_id"]
        ]
        parents_map = self.store.get_parent_messages_by_ids(request.user_id, parent_ids)
        parent_chunks_map = self.store.get_chunks_for_parents(request.user_id, parent_ids)

        timeline_candidates: dict[str, MemoryCandidate] = {}
        dialogue_candidates: dict[str, MemoryCandidate] = {}
        long_doc_slice_counts: dict[str, int] = {}

        for c in recalled_chunks:
            cid = str(c["chunk_id"])
            kind = str(c["kind"])
            c_rrf = rrf_norm_map.get(cid, 0.0)
            subj = str(c["subject"] or "user")
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
            if tg and (subj, tg) in latest_tag_map:
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
                # If raw parent or all its raw_slices were tombstoned,
                # never expose raw_slice or reconstruct msg:..., but allow surviving fact_item
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

            # Long document (>3000 chars): return raw_slice without parent/neighbor expansion (F3)
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

            # Normal message (<=3000 chars): group by homogeneous (pmid, source_tier, subject)
            all_p_chunks = parent_chunks_map.get(pmid, [])
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
                else:
                    g_subj = subj if source_tier == "other_person" else "user"
                    target_groups.append((source_tier, g_subj))
            else:
                if source_tier == "user_draft_fiction":
                    target_groups.append(("user_draft_fiction", "user"))
                elif source_tier == "other_person":
                    target_groups.append(("other_person", subj))
                else:
                    op_subjs = list(
                        dict.fromkeys(
                            str(ch.get("subject") or "user")
                            for ch in all_p_chunks
                            if ch["kind"] == "fact_item"
                            and int(ch["tombstoned"]) == 0
                            and ch["source_tier"] == "other_person"
                        )
                    )
                    if is_incid_mixed:
                        if is_incidental_update_text(str(c["display_text"] or "")):
                            target_groups.append(("user_direct", "user"))
                        else:
                            target_groups.append(("user_draft_fiction", "user"))
                    else:
                        if not has_ud_fact and op_subjs:
                            for op_s in op_subjs:
                                target_groups.append(("other_person", op_s))
                        else:
                            target_groups.append(("user_direct", "user"))
                            for op_s in op_subjs:
                                target_groups.append(("other_person", op_s))
                        if has_draft_fact and not has_ud_fact_or_raw:
                            target_groups.append(("user_draft_fiction", "user"))

            has_bm25 = cid in bm25_scores
            has_time = cid in time_hit_cids

            for group_tier, group_subj in target_groups:
                if group_tier == "user_direct":
                    group_id = pmid
                elif group_tier == "user_draft_fiction":
                    group_id = f"{pmid}:user_draft_fiction"
                else:
                    group_id = f"{pmid}:other_person:{group_subj}"

                existing = dialogue_candidates.get(group_id)
                if existing is not None:
                    if c_rrf > float(existing.metadata["rrf_norm"]):
                        existing.metadata["rrf_norm"] = c_rrf
                        existing.score = c_rrf
                    if is_latest and (
                        group_tier != "other_person"
                        or subj.casefold() == group_subj.casefold()
                    ):
                        existing.metadata["is_latest_in_tag"] = True
                    if has_bm25:
                        existing.metadata["bm25_hit"] = True
                    if has_time:
                        existing.metadata["time_hit"] = True
                    if tg and not existing.tag:
                        existing.tag = tg
                    continue

                # On-demand assembly from child raw_slices & live fact_items of group
                raw_slices = [ch for ch in all_p_chunks if ch["kind"] == "raw_slice"]
                live_facts = [
                    ch
                    for ch in all_p_chunks
                    if ch["kind"] == "fact_item"
                    and int(ch["tombstoned"]) == 0
                    and (
                        ch["source_tier"] == group_tier
                        or (
                            group_tier == "user_direct"
                            and ch["source_tier"] == "user_draft_fiction"
                            and has_ud_fact_or_raw
                            and not is_incid_mixed
                        )
                    )
                    and (
                        group_tier != "other_person"
                        or str(ch.get("subject") or "").casefold() == group_subj.casefold()
                    )
                ]
                group_is_latest = (
                    is_latest
                    if (
                        group_tier != "other_person"
                        or subj.casefold() == group_subj.casefold()
                    )
                    else False
                )
                if not group_is_latest and live_facts:
                    for lf in live_facts:
                        lf_subj = str(lf.get("subject") or "user")
                        lf_tg = str(lf.get("tag") or "")
                        if lf_tg and (lf_subj, lf_tg) in latest_tag_map:
                            l_ts, l_seq = latest_tag_map[(lf_subj, lf_tg)]
                            if p_seq == l_seq:
                                group_is_latest = True
                                break

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

                if live_facts:
                    deduped_notes = list(
                        dict.fromkeys(
                            str(f["display_text"])
                            for f in live_facts
                            if f.get("display_text") and str(f["display_text"]) != assembled_body
                        )
                    )
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
                if fact_line:
                    base_lines.append(fact_line)
                cur_tokens = approximate_tokens("\n".join(base_lines))

                prev_line = ""
                next_line = ""
                prev_seq_val: int | None = None
                next_seq_val: int | None = None
                # Only inline session neighbors when group_tier == 'user_direct' (§5.4.3)
                if group_tier == "user_direct":
                    prev_nb, next_nb = self.store.get_session_neighbors(
                        request.user_id, str(parent["session_id"]), p_seq
                    )
                    if prev_nb and int(prev_nb["tombstoned"]) == 0:
                        prev_nb_seq = int(prev_nb["seq"])
                        prev_text = str(prev_nb["content"] or "")
                        if len(prev_text) <= NEIGHBOR_MAX_CHARS and not self.store.is_seq_draft_fiction(
                            request.user_id, prev_nb_seq
                        ):
                            cand_prev = f"[#{prev_nb_seq} | {prev_nb['speaker']}] {prev_text}"
                            cost_prev = approximate_tokens(cand_prev)
                            if cur_tokens + cost_prev <= self.token_budget:
                                prev_line = cand_prev
                                prev_seq_val = prev_nb_seq
                                cur_tokens += cost_prev

                    if next_nb and int(next_nb["tombstoned"]) == 0:
                        next_nb_seq = int(next_nb["seq"])
                        next_text = str(next_nb["content"] or "")
                        if len(next_text) <= NEIGHBOR_MAX_CHARS and not self.store.is_seq_draft_fiction(
                            request.user_id, next_nb_seq
                        ):
                            cand_next = f"[#{next_nb_seq} | {next_nb['speaker']}] {next_text}"
                            cost_next = approximate_tokens(cand_next)
                            if cur_tokens + cost_next <= self.token_budget:
                                next_line = cand_next
                                next_seq_val = next_nb_seq
                                cur_tokens += cost_next

                parts: list[str] = []
                if prev_line:
                    parts.append(prev_line)
                parts.append(anchor_line)
                if next_line:
                    parts.append(next_line)
                if fact_line:
                    parts.append(fact_line)

                restored_text = "\n".join(parts)
                best_srch = str(c.get("search_text") or c["display_text"])
                if fact_line and fact_line not in best_srch:
                    best_srch = f"{best_srch}\n{fact_line}"
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
                        "is_global_constraint": False,
                        "source_tier": group_tier,
                        "bm25_hit": has_bm25,
                        "time_hit": has_time,
                        "fact_line": fact_line,
                        "anchor_line": anchor_line,
                        "raw_anchor_line": raw_anchor_line,
                        "prev_line": prev_line,
                        "prev_seq": prev_seq_val,
                        "next_line": next_line,
                        "next_seq": next_seq_val,
                        "rerank_text": best_srch[:600],
                    },
                )

        # 5. Rerank top 50 candidates with pure query stem (F12.2, F13, F16, P1-4)
        all_candidates = [*dialogue_candidates.values(), *timeline_candidates.values()]
        all_candidates.sort(
            key=lambda item: (
                float(item.metadata.get("rrf_norm", 0.0)),
                1 if item.metadata.get("is_latest_in_tag") else 0,
                int(item.session_ts or 0),
                int(item.seq),
            ),
            reverse=True,
        )
        rerank_pool = all_candidates[:50]
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

        # DESIGN.md:152 & 154 scoring formulas + v2.3 T1 superseded & T6 third-person gating
        pure_query_lower = pure_query.casefold()
        history_compare_kws = (
            "before",
            "previously",
            "used to",
            "previous",
            "former",
            "originally",
            "change",
            "changed",
            "difference",
            "以前",
            "之前",
            "原来",
            "原本",
            "变化",
            "变动",
        )
        is_history_query = any(kw in pure_query_lower for kw in history_compare_kws)

        def _is_subj_in_query(subj_low: str) -> bool:
            if not subj_low or subj_low == "user":
                return False
            if subj_low in pure_query_lower:
                return True
            cleaned = re.sub(
                r"\b(my|the|a|an|friend|colleague|sister|brother|coworker|roommate)\b"
                r"|我的|朋友|同事|同学|闺蜜|室友",
                " ",
                subj_low,
            ).strip()
            for tok in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", cleaned):
                if len(tok) >= 2 and tok in pure_query_lower:
                    return True
                if (
                    len(tok) == 1
                    and tok in "妈爸姐哥弟妹婆爷奶妻夫姨叔舅姑嫂"
                    and tok in pure_query_lower
                ):
                    return True
            for ch in subj_low:
                if ch in "妈爸姐哥弟妹婆爷奶妻夫姨叔舅姑嫂" and ch in pure_query_lower:
                    return True
            return False

        active_dlg_rrf_by_seq: dict[int, float] = {}
        for d_item in dialogue_candidates.values():
            if d_item.metadata.get("source_tier") != "user_draft_fiction":
                s_no = int(d_item.seq)
                r_val = float(d_item.metadata.get("rrf_norm", 0.0))
                if r_val > active_dlg_rrf_by_seq.get(s_no, 0.0):
                    active_dlg_rrf_by_seq[s_no] = r_val

        for item in all_candidates:
            rrf_val = float(item.metadata.get("rrf_norm", 0.0))
            latest_bonus = 0.05 if item.metadata.get("is_latest_in_tag") else 0.0
            time_bonus = 0.10 if item.metadata.get("time_hit") else 0.0
            if rerank_succeeded:
                r_prob = rerank_scores_map.get(item.id, 0.0)
                item.metadata["r_prob"] = r_prob
                item.score = 0.80 * r_prob + 0.20 * rrf_val + latest_bonus + time_bonus
            else:
                item.metadata["r_prob"] = rrf_val
                item.score = rrf_val + latest_bonus + time_bonus

            subj_lower = str(item.subject or "user").strip().casefold()
            if (
                subj_lower != "user"
                or item.metadata.get("source_tier") == "other_person"
            ):
                if _is_subj_in_query(subj_lower):
                    item.score += 0.10
                else:
                    item.score *= 0.35

        # 6. Dynamic timeline relevance gating & D_direct / D_draft bucketing (v2.2/v2.3)
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
                if rerank_succeeded:
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
            for d in dlg_list:
                if d.metadata.get("source_tier", "user_direct") == "user_draft_fiction":
                    continue
                if d.kind == "raw_slice" and d.parent_msg_id:
                    if d.parent_msg_id in seen_long_pmids:
                        overflow_long_slices.append(d)
                        continue
                    seen_long_pmids.add(d.parent_msg_id)
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
                and (
                    not rerank_succeeded
                    or float(d.metadata.get("r_prob", 0.0)) >= DRAFT_FALLBACK_THRESHOLD
                )
            )

            if n_high >= rem or not d_draft:
                selected_dlg = d_direct[:rem]
                overflow_dlg = [*d_direct[rem:], *overflow_long_slices, *d_draft]
            else:
                direct_keep = max(n_high, max(0, rem - 2))
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

        packed = pack_results(ordered_for_pack, request.top_k, self.token_budget)
        packed = packed[: request.top_k]

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
