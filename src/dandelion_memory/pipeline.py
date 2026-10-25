from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .embedding import Embedder
from .extractor import Extractor, RegexExtractor, sanitize_extraction_result
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
    has_explicit_forget_keyword,
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

    async def _invoke_extractor(
        self,
        batch_messages: list[dict[str, Any]],
        existing_tags: list[tuple[str, str]],
    ) -> Any:
        if hasattr(self.extractor, "extract"):
            return await self.extractor.extract(batch_messages, existing_tags=existing_tags)
        if hasattr(self.extractor, "extract_batch"):
            return await self.extractor.extract_batch(batch_messages, existing_tags=existing_tags)
        if callable(self.extractor):
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
        assert isinstance(self.store, SQLiteStore)

        # Step 1: Idempotency check BEFORE any LLM or Embedding calls (F2.1)
        already_processed = self.store.check_idempotency(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_hash=payload_hash,
        )
        if already_processed:
            return AddResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        # Step 2: Filter role == 'system' (F1, F5)
        non_sys_messages = [m for m in request.messages if m.role != "system"]
        if not non_sys_messages:
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

        # Step 3: Wrap all state mutations & external calls in an atomic transaction (P0-2)
        with self.store.transaction():
            last_seq, pending_forget = self.store.get_user_state(request.user_id)
            inherited_ts = self.store.get_last_session_ts(request.user_id, request.session_id)

            first_msg_auto_tombstone = False
            if pending_forget == 1:
                if non_sys_messages[0].role == "assistant":
                    first_msg_auto_tombstone = True
                self.store.update_user_state(request.user_id, pending_forget_confirm=0)

            batch_records: list[dict[str, Any]] = []
            pending_raw_chunks: list[dict[str, Any]] = []

            for idx, msg in enumerate(non_sys_messages):
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

                self.store.insert_parent_message(
                    msg_id=msg_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                    seq=seq,
                    role=msg.role,
                    speaker=speaker,
                    content=msg.content,
                    session_ts=ts,
                    session_time_str=time_str,
                    fact_notes="",
                    tombstoned=is_tomb,
                )

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
                for c_idx, slice_str in enumerate(slices):
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
                        }
                    )

            self.store.update_user_state(
                request.user_id,
                last_seq=last_seq + len(non_sys_messages),
                pending_forget_confirm=0,
            )

            if pending_raw_chunks:
                raw_vecs = await self.embedder.embed(
                    [c["embed_input"] for c in pending_raw_chunks]
                )
                for c, vec in zip(pending_raw_chunks, raw_vecs, strict=True):
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
                        embedding=vec,
                        tombstoned=c["tombstoned"],
                    )

            # Step 4: Module B - Batch LLM Governance Extraction with Regex Fallback (F2.2)
            active_batch = [r for r in batch_records if r["tombstoned"] == 0]
            existing_tags = self.store.get_active_tags(request.user_id)
            if active_batch:
                try:
                    raw_ext = await self._invoke_extractor(active_batch, existing_tags)
                except Exception:
                    raw_ext = regex_extract_batch(active_batch)
                facts, forget_actions = sanitize_extraction_result(raw_ext, active_batch)
            else:
                facts, forget_actions = [], []

            seq_to_rec = {int(r["seq"]): r for r in batch_records}
            pending_derived_chunks: list[dict[str, Any]] = []

            for f_idx, fact in enumerate(facts):
                valid_seqs = [s for s in fact.source_seqs if s in seq_to_rec]
                if not valid_seqs and active_batch:
                    valid_seqs = [int(active_batch[-1]["seq"])]
                primary_seq = max(valid_seqs) if valid_seqs else (last_seq + 1)

                if fact.is_other_person or fact.subject.casefold() != "user":
                    formatted_note = f"【非User本人，系{fact.subject}的信息】{fact.fact_text}"
                else:
                    formatted_note = fact.fact_text

                primary_msg_id: str | None = None
                primary_session_ts: int | None = None
                primary_time_str = ""

                for s in valid_seqs:
                    info = self.store.append_parent_fact_note(request.user_id, s, formatted_note)
                    if info is not None and s == primary_seq:
                        primary_msg_id, _, primary_session_ts = info
                    if s in seq_to_rec and s == primary_seq:
                        primary_msg_id = str(seq_to_rec[s]["msg_id"])
                        primary_session_ts = seq_to_rec[s]["session_ts"]
                        primary_time_str = str(seq_to_rec[s]["session_time_str"] or "")

                ev_start, ev_end = parse_event_date_range(
                    fact.event_date_start,
                    fact.event_date_end,
                    fallback_ts=primary_session_ts,
                )

                for s in valid_seqs:
                    self.store.backfill_raw_slices_for_seq(
                        request.user_id,
                        s,
                        subject=fact.subject,
                        tag=fact.tag,
                        event_time_start=ev_start,
                        event_time_end=ev_end,
                        fact_text=fact.fact_text,
                    )

                fact_chunk_id = f"fact:{request.request_id}:{f_idx:04d}"
                fact_search_text = f"[{fact.subject} | {fact.tag}] {formatted_note}"
                fact_bm25 = build_bm25_tokens(
                    formatted_note, subject=fact.subject, tag=fact.tag
                )
                pending_derived_chunks.append(
                    {
                        "chunk_id": fact_chunk_id,
                        "parent_msg_id": primary_msg_id,
                        "seq": primary_seq,
                        "session_ts": primary_session_ts,
                        "kind": "fact_item",
                        "subject": fact.subject,
                        "tag": fact.tag,
                        "display_text": formatted_note,
                        "search_text": fact_search_text,
                        "embed_input": fact_search_text,
                        "bm25_tokens": fact_bm25,
                        "event_time_start": ev_start,
                        "event_time_end": ev_end,
                    }
                )

                rendered_summary, tl_seq, tl_ts = self.store.upsert_tag_timeline(
                    user_id=request.user_id,
                    subject=fact.subject,
                    tag=fact.tag,
                    seq=primary_seq,
                    session_ts=primary_session_ts,
                    time_str=primary_time_str,
                    prev_state=fact.prev_state,
                    current_state=fact.current_state,
                    update_reason=fact.update_reason,
                    fact_text=fact.fact_text,
                )
                tl_chunk_id = f"tl:{request.user_id}:{fact.subject}:{fact.tag}"
                tl_bm25 = build_bm25_tokens(
                    rendered_summary, subject=fact.subject, tag=fact.tag
                )
                pending_derived_chunks.append(
                    {
                        "chunk_id": tl_chunk_id,
                        "parent_msg_id": None,
                        "seq": tl_seq,
                        "session_ts": tl_ts,
                        "kind": "timeline_summary",
                        "subject": fact.subject,
                        "tag": fact.tag,
                        "display_text": rendered_summary,
                        "search_text": rendered_summary,
                        "embed_input": rendered_summary,
                        "bm25_tokens": tl_bm25,
                        "event_time_start": tl_ts,
                        "event_time_end": tl_ts,
                    }
                )

            if pending_derived_chunks:
                derived_vecs = await self.embedder.embed(
                    [c["embed_input"] for c in pending_derived_chunks]
                )
                for c, vec in zip(pending_derived_chunks, derived_vecs, strict=True):
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
                        embedding=vec,
                        tombstoned=0,
                    )

            # Step 5: Execute validated forget_actions (F6, F8.1, F10.2)
            last_batch_seq = int(batch_records[-1]["seq"]) if batch_records else 0
            for fa in forget_actions:
                trig_rec = seq_to_rec.get(fa.trigger_seq)
                if trig_rec is None:
                    continue
                # F8.1 Deterministic explicit forget root validation
                if not has_explicit_forget_keyword(str(trig_rec["content"])):
                    continue

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
                    self.store.update_user_state(request.user_id, pending_forget_confirm=1)

                target_vec: list[float] | None = None
                if fa.target_description:
                    [target_vec] = await self.embedder.embed([fa.target_description])

                self.store.execute_forget_action(
                    user_id=request.user_id,
                    trigger_seq=fa.trigger_seq,
                    batch_seqs_to_tombstone=batch_to_tomb,
                    target_existing_tags=fa.target_existing_tags,
                    target_keywords=fa.target_keywords,
                    target_description=fa.target_description,
                    target_vector=target_vec,
                )

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
            time_matches = [
                c
                for c in chunks
                if c["event_time_start"] is not None
                and c["event_time_end"] is not None
                and int(c["event_time_start"]) <= q_end
                and int(c["event_time_end"]) >= q_start
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
            if c["kind"] == "timeline_summary" and c["tag"] == "全局指令/格式约束":
                cid = str(c["chunk_id"])
                if cid not in rrf_raw:
                    rrf_raw[cid] = 1e-6

        if not rrf_raw:
            return SearchResponse(data=[])

        max_rrf = max(rrf_raw.values()) if rrf_raw else 1.0
        rrf_norm_map = {cid: val / max_rrf for cid, val in rrf_raw.items()}
        recalled_chunks = [c for c in chunks if str(c["chunk_id"]) in rrf_raw]

        # 4. Module D: Separate timelines vs dialogue/long-doc chunks & restore parents (P0-1)
        latest_tag_map = self.store.get_latest_tag_map(request.user_id)
        parent_ids = [
            str(c["parent_msg_id"])
            for c in recalled_chunks
            if c["kind"] != "timeline_summary" and c["parent_msg_id"]
        ]
        parents_map = self.store.get_parent_messages_by_ids(request.user_id, parent_ids)

        timeline_candidates: dict[str, MemoryCandidate] = {}
        dialogue_candidates: dict[str, MemoryCandidate] = {}

        for c in recalled_chunks:
            cid = str(c["chunk_id"])
            kind = str(c["kind"])
            c_rrf = rrf_norm_map.get(cid, 0.0)
            subj = str(c["subject"] or "user")
            tg = str(c["tag"]) if c["tag"] else None

            if kind == "timeline_summary":
                is_global = tg == "全局指令/格式约束"
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
                    },
                )
                continue

            pmid = str(c["parent_msg_id"]) if c["parent_msg_id"] else None
            if not pmid or pmid not in parents_map:
                continue
            parent = parents_map[pmid]
            if int(parent["tombstoned"]) == 1:
                continue

            p_content = str(parent["content"] or "")
            p_seq = int(parent["seq"])
            p_ts = int(parent["session_ts"]) if parent["session_ts"] is not None else None

            is_latest = False
            if tg and (subj, tg) in latest_tag_map:
                l_ts, l_seq = latest_tag_map[(subj, tg)]
                if p_seq == l_seq or (
                    p_ts is not None and l_ts is not None and (p_ts, p_seq) >= (l_ts, l_seq)
                ):
                    is_latest = True

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
                    },
                )
                continue

            # Normal message (<=3000 chars): deduplicate by parent_msg_id and restore neighbors
            existing = dialogue_candidates.get(pmid)
            if existing is not None:
                if c_rrf > float(existing.metadata["rrf_norm"]):
                    existing.metadata["rrf_norm"] = c_rrf
                    existing.score = c_rrf
                if is_latest:
                    existing.metadata["is_latest_in_tag"] = True
                if tg and not existing.tag:
                    existing.tag = tg
                continue

            prev_nb, next_nb = self.store.get_session_neighbors(
                request.user_id, str(parent["session_id"]), p_seq
            )
            t_str = str(parent["session_time_str"] or "")
            t_header = f" | {t_str}" if t_str else ""
            anchor_line = f"[#{p_seq} | {parent['speaker']}{t_header}] {p_content}"

            fact_notes = str(parent["fact_notes"] or "").strip()
            fact_line = f"[提炼事实: {fact_notes}]" if fact_notes else ""

            base_lines = [anchor_line]
            if fact_line:
                base_lines.append(fact_line)
            cur_tokens = approximate_tokens("\n".join(base_lines))

            prev_line = ""
            if prev_nb and int(prev_nb["tombstoned"]) == 0:
                prev_text = str(prev_nb["content"] or "")
                if len(prev_text) <= NEIGHBOR_MAX_CHARS:
                    cand_prev = f"[#{prev_nb['seq']} | {prev_nb['speaker']}] {prev_text}"
                    cost_prev = approximate_tokens(cand_prev)
                    if cur_tokens + cost_prev <= self.token_budget:
                        prev_line = cand_prev
                        cur_tokens += cost_prev

            next_line = ""
            if next_nb and int(next_nb["tombstoned"]) == 0:
                next_text = str(next_nb["content"] or "")
                if len(next_text) <= NEIGHBOR_MAX_CHARS:
                    cand_next = f"[#{next_nb['seq']} | {next_nb['speaker']}] {next_text}"
                    cost_next = approximate_tokens(cand_next)
                    if cur_tokens + cost_next <= self.token_budget:
                        next_line = cand_next
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
            dialogue_candidates[pmid] = MemoryCandidate(
                id=pmid,
                content=restored_text,
                score=c_rrf,
                created_at=t_str or None,
                session_id=str(parent["session_id"]),
                kind="restored_parent",
                parent_msg_id=pmid,
                seq=p_seq,
                session_ts=p_ts,
                subject=subj,
                tag=tg,
                metadata={
                    "rrf_norm": c_rrf,
                    "is_latest_in_tag": is_latest,
                    "is_global_constraint": False,
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
            docs_for_rerank = [item.content[:2400] for item in rerank_pool]
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

        # DESIGN.md:152 & 154 scoring formulas
        for item in all_candidates:
            rrf_val = float(item.metadata.get("rrf_norm", 0.0))
            latest_bonus = 0.05 if item.metadata.get("is_latest_in_tag") else 0.0
            if rerank_succeeded:
                r_prob = rerank_scores_map.get(item.id, 0.0)
                item.score = 0.80 * r_prob + 0.20 * rrf_val + latest_bonus
            else:
                item.score = rrf_val + latest_bonus

        # 6. Dynamic timeline pinning & top_k / token_budget packing (F4, DESIGN.md:28, 156)
        if request.top_k == 1:
            ordered_for_pack = sorted(
                all_candidates,
                key=lambda item: (
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
            pinned_tl = tl_list[:max_tl]
            pinned_ids = {t.id for t in pinned_tl}
            rem_tl = [t for t in tl_list if t.id not in pinned_ids]
            ordered_for_pack = [*pinned_tl, *dlg_list, *rem_tl]

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
