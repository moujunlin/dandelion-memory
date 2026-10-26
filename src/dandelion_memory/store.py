from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from array import array
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import MemoryCandidate, WriteItem
from .text import (
    LONG_DOC_CHAR_THRESHOLD,
    build_bm25_tokens,
    has_explicit_forget_keyword,
    is_incidental_update_text,
    tokenize_bm25,
)


class StoreConflictError(RuntimeError):
    pass


IdempotencyConflictError = StoreConflictError


def vector_to_blob(vec: Sequence[float]) -> bytes:
    arr = array("f", (float(x) for x in vec))
    return arr.tobytes()


def blob_to_vector(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    arr = array("f")
    arr.frombytes(blob)
    return arr.tolist()


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    if not v1 or not v2:
        return 0.0
    n = min(len(v1), len(v2))
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for i in range(n):
        a = v1[i]
        b = v2[i]
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    denom = math.sqrt(norm1) * math.sqrt(norm2)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


class MemoryStore(Protocol):
    async def add_request(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
        items: Sequence[WriteItem],
        archive_ids: Sequence[str],
    ) -> bool: ...

    async def search_candidates(
        self,
        *,
        user_id: str,
        query: str,
        embedding: Sequence[float],
        semantic_limit: int,
        lexical_limit: int,
        result_limit: int,
    ) -> list[MemoryCandidate]: ...

    async def neighbors(
        self, *, user_id: str, anchor_ids: Sequence[str], radius: int
    ) -> list[MemoryCandidate]: ...


class SQLiteStore:
    """Single-stack SQLite-WAL + float32 vector cache + BM25 store (DESIGN.md v2.1)."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._in_tx = False
        self._vector_cache: dict[str, list[float]] = {}
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def _maybe_commit(self) -> None:
        if not self._in_tx:
            self._conn.commit()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self._in_tx:
            yield self._conn
            return
        self._in_tx = True
        cache_snapshot = dict(self._vector_cache)
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            self._vector_cache = cache_snapshot
            raise
        finally:
            self._in_tx = False

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS add_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                last_seq INTEGER NOT NULL DEFAULT 0,
                pending_forget_confirm INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS parent_messages (
                msg_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                speaker TEXT NOT NULL,
                content TEXT NOT NULL,
                session_ts INTEGER,
                session_time_str TEXT DEFAULT '',
                fact_notes TEXT DEFAULT '',
                tombstoned INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_parent_user_seq
                ON parent_messages (user_id, seq);
            CREATE INDEX IF NOT EXISTS idx_parent_user_session_seq
                ON parent_messages (user_id, session_id, seq);

            CREATE TABLE IF NOT EXISTS search_chunks (
                chunk_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                parent_msg_id TEXT,
                seq INTEGER NOT NULL,
                session_ts INTEGER,
                kind TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT 'user',
                tag TEXT,
                source_tier TEXT NOT NULL DEFAULT 'user_direct',
                context_prefix TEXT DEFAULT '',
                display_text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                bm25_tokens TEXT NOT NULL,
                event_time_start INTEGER,
                event_time_end INTEGER,
                embedding BLOB,
                tombstoned INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_user_tomb
                ON search_chunks (user_id, tombstoned);
            CREATE INDEX IF NOT EXISTS idx_chunks_user_seq
                ON search_chunks (user_id, seq);
            CREATE INDEX IF NOT EXISTS idx_chunks_parent
                ON search_chunks (parent_msg_id);

            CREATE TABLE IF NOT EXISTS tag_timelines (
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                tag TEXT NOT NULL,
                latest_seq INTEGER NOT NULL,
                latest_session_ts INTEGER,
                current_state TEXT NOT NULL,
                history_chain TEXT NOT NULL,
                rendered_summary TEXT NOT NULL,
                tombstoned INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, subject, tag)
            );
            """
        )
        cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(search_chunks)").fetchall()
        }
        if "source_tier" not in cols:
            conn.execute(
                "ALTER TABLE search_chunks "
                "ADD COLUMN source_tier TEXT NOT NULL DEFAULT 'user_direct'"
            )
        if "context_prefix" not in cols:
            conn.execute(
                "ALTER TABLE search_chunks ADD COLUMN context_prefix TEXT DEFAULT ''"
            )
        conn.commit()

    def check_idempotency(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
    ) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT user_id, session_id, payload_hash FROM add_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return False
        if (
            row["user_id"] == user_id
            and row["session_id"] == session_id
            and row["payload_hash"] == payload_hash
        ):
            return True
        raise StoreConflictError(
            f"request_id {request_id!r} already exists with different payload or scope"
        )

    def record_add_request(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
    ) -> None:
        now_iso = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO add_requests
                (request_id, user_id, session_id, payload_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request_id, user_id, session_id, payload_hash, now_iso),
        )
        self._maybe_commit()

    def get_user_state(self, user_id: str) -> tuple[int, int]:
        conn = self._connect()
        row = conn.execute(
            "SELECT last_seq, pending_forget_confirm FROM user_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO user_state (user_id, last_seq, pending_forget_confirm) "
                "VALUES (?, 0, 0)",
                (user_id,),
            )
            self._maybe_commit()
            return 0, 0
        return int(row["last_seq"]), int(row["pending_forget_confirm"])

    def update_user_state(
        self,
        user_id: str,
        *,
        last_seq: int | None = None,
        pending_forget_confirm: int | None = None,
    ) -> None:
        cur_seq, cur_pending = self.get_user_state(user_id)
        new_seq = cur_seq if last_seq is None else int(last_seq)
        new_pending = cur_pending if pending_forget_confirm is None else int(pending_forget_confirm)
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO user_state (user_id, last_seq, pending_forget_confirm)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_seq = excluded.last_seq,
                pending_forget_confirm = excluded.pending_forget_confirm
            """,
            (user_id, new_seq, new_pending),
        )
        self._maybe_commit()

    def get_last_session_ts(self, user_id: str, session_id: str) -> int | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT session_ts FROM parent_messages
            WHERE user_id = ? AND session_id = ? AND session_ts IS NOT NULL
            ORDER BY seq DESC LIMIT 1
            """,
            (user_id, session_id),
        ).fetchone()
        return int(row["session_ts"]) if row and row["session_ts"] is not None else None

    def get_max_user_session_ts(self, user_id: str) -> int | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT MAX(session_ts) AS max_ts FROM parent_messages "
            "WHERE user_id = ? AND session_ts IS NOT NULL",
            (user_id,),
        ).fetchone()
        return int(row["max_ts"]) if row and row["max_ts"] is not None else None

    def get_active_tags(self, user_id: str) -> list[tuple[str, str]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT subject, tag FROM tag_timelines WHERE user_id = ? AND tombstoned = 0 "
            "ORDER BY latest_seq DESC",
            (user_id,),
        ).fetchall()
        return [(str(r["subject"]), str(r["tag"])) for r in rows]

    def insert_parent_message(
        self,
        *,
        msg_id: str,
        user_id: str,
        session_id: str,
        request_id: str,
        seq: int,
        role: str,
        speaker: str,
        content: str,
        session_ts: int | None,
        session_time_str: str,
        fact_notes: str = "",
        tombstoned: int = 0,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO parent_messages (
                msg_id, user_id, session_id, request_id, seq,
                role, speaker, content, session_ts, session_time_str,
                fact_notes, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                user_id,
                session_id,
                request_id,
                seq,
                role,
                speaker,
                content,
                session_ts,
                session_time_str,
                fact_notes,
                tombstoned,
            ),
        )
        self._maybe_commit()

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        user_id: str,
        session_id: str,
        parent_msg_id: str | None,
        seq: int,
        session_ts: int | None,
        kind: str,
        subject: str,
        tag: str | None,
        display_text: str,
        search_text: str,
        bm25_tokens: str,
        event_time_start: int | None,
        event_time_end: int | None,
        embedding: Sequence[float],
        source_tier: str = "user_direct",
        context_prefix: str = "",
        tombstoned: int = 0,
    ) -> None:
        blob = vector_to_blob(embedding)
        self._vector_cache[chunk_id] = list(embedding)
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO search_chunks (
                chunk_id, user_id, session_id, parent_msg_id, seq,
                session_ts, kind, subject, tag, source_tier, context_prefix,
                display_text, search_text, bm25_tokens, event_time_start, event_time_end,
                embedding, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                session_id = excluded.session_id,
                parent_msg_id = excluded.parent_msg_id,
                seq = excluded.seq,
                session_ts = excluded.session_ts,
                kind = excluded.kind,
                subject = excluded.subject,
                tag = excluded.tag,
                source_tier = excluded.source_tier,
                context_prefix = excluded.context_prefix,
                display_text = excluded.display_text,
                search_text = excluded.search_text,
                bm25_tokens = excluded.bm25_tokens,
                event_time_start = excluded.event_time_start,
                event_time_end = excluded.event_time_end,
                embedding = excluded.embedding,
                tombstoned = excluded.tombstoned
            """,
            (
                chunk_id,
                user_id,
                session_id,
                parent_msg_id,
                seq,
                session_ts,
                kind,
                subject,
                tag,
                source_tier,
                context_prefix,
                display_text,
                search_text,
                bm25_tokens,
                event_time_start,
                event_time_end,
                blob,
                tombstoned,
            ),
        )
        self._maybe_commit()

    def append_parent_fact_note(
        self, user_id: str, seq: int, formatted_note: str
    ) -> tuple[str, str, int | None] | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT msg_id, session_id, session_ts, fact_notes FROM parent_messages "
            "WHERE user_id = ? AND seq = ?",
            (user_id, seq),
        ).fetchone()
        if row is None:
            return None
        existing = str(row["fact_notes"] or "").strip()
        if formatted_note not in existing:
            updated = f"{existing}；{formatted_note}" if existing else formatted_note
            conn.execute(
                "UPDATE parent_messages SET fact_notes = ? WHERE msg_id = ?",
                (updated, row["msg_id"]),
            )
            self._maybe_commit()
        return (
            str(row["msg_id"]),
            str(row["session_id"]),
            int(row["session_ts"]) if row["session_ts"] is not None else None,
        )

    def backfill_raw_slices_for_seq(
        self,
        user_id: str,
        seq: int,
        *,
        subject: str,
        tag: str,
        event_time_start: int | None,
        event_time_end: int | None,
        fact_text: str = "",
        source_tier: str = "user_direct",
    ) -> None:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, display_text, bm25_tokens, tag, source_tier
            FROM search_chunks
            WHERE user_id = ? AND seq = ? AND kind = 'raw_slice'
            ORDER BY chunk_id ASC
            """,
            (user_id, seq),
        ).fetchall()
        if not rows:
            return

        target_rows = list(rows)
        if len(rows) > 1:
            fact_tokens = set(tokenize_bm25(f"{tag} {fact_text}"))
            scored_rows: list[tuple[int, sqlite3.Row]] = []
            for r in rows:
                disp = str(r["display_text"] or "")
                if source_tier == "user_draft_fiction" and is_incidental_update_text(disp):
                    scored_rows.append((-1, r))
                    continue
                slice_tokens = set(tokenize_bm25(disp))
                overlap_cnt = len(slice_tokens & fact_tokens)
                if overlap_cnt == 0 and has_explicit_forget_keyword(disp):
                    scored_rows.append((-1, r))
                    continue
                scored_rows.append((overlap_cnt, r))
            max_ov = max((sc for sc, _ in scored_rows), default=0)
            if max_ov > 0:
                target_rows = [r for sc, r in scored_rows if sc == max_ov]
            else:
                target_rows = [r for sc, r in scored_rows if sc >= 0]

        for r in target_rows:
            cid = str(r["chunk_id"])
            disp = str(r["display_text"])
            existing_bm25 = str(r["bm25_tokens"] or "").strip()
            existing_tag = r["tag"]
            resolved_tier = (
                "user_direct" if is_incidental_update_text(disp) else source_tier
            )
            extra_bm25 = build_bm25_tokens(fact_text, subject=subject, tag=tag)
            if existing_tag:
                # Merge additional fact tokens without overwriting earlier facts on the same slice
                bm25 = f"{existing_bm25} {extra_bm25}".strip()
            else:
                bm25 = build_bm25_tokens(
                    f"{disp} {fact_text}".strip(), subject=subject, tag=tag
                )
            conn.execute(
                """
                UPDATE search_chunks
                SET subject = ?, tag = ?, source_tier = ?,
                    event_time_start = ?, event_time_end = ?, bm25_tokens = ?
                WHERE chunk_id = ?
                """,
                (subject, tag, resolved_tier, event_time_start, event_time_end, bm25, cid),
            )
        self._maybe_commit()

    def propagate_assistant_draft_echo(self, user_id: str, seq: int) -> None:
        """If #(seq-1) is a user message whose raw_slices are all 'user_draft_fiction'
        and #seq (assistant) has no 'user_direct' fact_items, mark #seq raw_slices as
        'user_draft_fiction' (DESIGN.md §4.2.2.B).
        """
        if seq <= 1:
            return
        conn = self._connect()
        prev_parent = conn.execute(
            "SELECT role, tombstoned FROM parent_messages WHERE user_id = ? AND seq = ?",
            (user_id, seq - 1),
        ).fetchone()
        if (
            not prev_parent
            or str(prev_parent["role"]) != "user"
            or int(prev_parent["tombstoned"]) != 0
        ):
            return
        prev_slices = conn.execute(
            "SELECT source_tier FROM search_chunks "
            "WHERE user_id = ? AND seq = ? AND kind = 'raw_slice' AND tombstoned = 0",
            (user_id, seq - 1),
        ).fetchall()
        if not prev_slices or not all(
            str(r["source_tier"] or "user_direct") == "user_draft_fiction" for r in prev_slices
        ):
            return
        direct_fact = conn.execute(
            "SELECT 1 FROM search_chunks "
            "WHERE user_id = ? AND seq = ? AND kind = 'fact_item' "
            "AND source_tier = 'user_direct' AND tombstoned = 0 LIMIT 1",
            (user_id, seq),
        ).fetchone()
        if direct_fact is not None:
            return
        conn.execute(
            "UPDATE search_chunks SET source_tier = 'user_draft_fiction' "
            "WHERE user_id = ? AND seq = ? AND kind = 'raw_slice'",
            (user_id, seq),
        )
        self._maybe_commit()

    def upsert_tag_timeline(
        self,
        *,
        user_id: str,
        subject: str,
        tag: str,
        seq: int,
        session_ts: int | None,
        time_str: str,
        prev_state: str,
        current_state: str,
        update_reason: str,
        fact_text: str,
    ) -> tuple[str, int, int | None]:
        conn = self._connect()
        row = conn.execute(
            "SELECT history_chain, tombstoned FROM tag_timelines "
            "WHERE user_id = ? AND subject = ? AND tag = ?",
            (user_id, subject, tag),
        ).fetchone()
        chain: list[dict[str, Any]] = []
        # P1-1: Only inherit history_chain if existing timeline is NOT tombstoned
        if row and int(row["tombstoned"]) == 0 and row["history_chain"]:
            try:
                loaded = json.loads(str(row["history_chain"]))
                if isinstance(loaded, list):
                    chain = loaded
            except Exception:
                chain = []

        entry: dict[str, Any] = {
            "seq": seq,
            "session_ts": session_ts,
            "time_str": time_str,
            "state": current_state,
            "prev_state": prev_state,
            "reason": update_reason,
            "fact_text": fact_text,
        }
        replaced = False
        for idx, existing in enumerate(chain):
            if int(existing.get("seq", -1)) == seq and existing.get("state") == current_state:
                chain[idx] = entry
                replaced = True
                break
        if not replaced:
            chain.append(entry)

        def _sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
            s_ts = item.get("session_ts")
            s_seq = int(item.get("seq") or 0)
            if s_ts is not None:
                return (0, int(s_ts), s_seq)
            return (1, s_seq, s_seq)

        chain.sort(key=_sort_key)
        latest = chain[-1]
        latest_seq = int(latest.get("seq") or seq)
        latest_ts = (
            int(latest["session_ts"]) if latest.get("session_ts") is not None else None
        )
        resolved_current_state = str(latest.get("state") or current_state)

        history_parts: list[str] = []
        for idx, item in enumerate(chain):
            st = str(item.get("state") or "")
            pr = str(item.get("prev_state") or "")
            rsn = str(item.get("reason") or "")
            ft = str(item.get("fact_text") or "")
            t_label = str(item.get("time_str") or f"#{item.get('seq')}")
            if idx == 0 and pr and pr not in st:
                history_parts.append(f"[早期状态] {pr}")
            detail = f"[{t_label}] {st}"
            if rsn:
                detail += f"（原因: {rsn}）"
            if ft and ft not in detail:
                detail += f" - {ft}"
            history_parts.append(detail)

        chain_str = " -> ".join(history_parts)
        rendered_summary = (
            f"【状态时间线 | 主体: {subject} | 标签: {tag}】"
            f"当前最新状态: {resolved_current_state} | 历史演化: {chain_str}"
        )
        history_json = json.dumps(chain, ensure_ascii=False)

        conn.execute(
            """
            INSERT INTO tag_timelines (
                user_id, subject, tag, latest_seq, latest_session_ts,
                current_state, history_chain, rendered_summary, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(user_id, subject, tag) DO UPDATE SET
                latest_seq = excluded.latest_seq,
                latest_session_ts = excluded.latest_session_ts,
                current_state = excluded.current_state,
                history_chain = excluded.history_chain,
                rendered_summary = excluded.rendered_summary,
                tombstoned = 0
            """,
            (
                user_id,
                subject,
                tag,
                latest_seq,
                latest_ts,
                resolved_current_state,
                history_json,
                rendered_summary,
            ),
        )
        self._maybe_commit()
        return rendered_summary, latest_seq, latest_ts

    def tombstone_seqs(self, user_id: str, seqs: Sequence[int]) -> None:
        seq_list = [int(s) for s in seqs if s is not None]
        if not seq_list:
            return
        placeholders = ",".join("?" for _ in seq_list)
        conn = self._connect()
        conn.execute(
            f"UPDATE parent_messages SET tombstoned = 1 "
            f"WHERE user_id = ? AND seq IN ({placeholders})",
            [user_id, *seq_list],
        )
        conn.execute(
            f"UPDATE search_chunks SET tombstoned = 1 "
            f"WHERE user_id = ? AND kind = 'raw_slice' AND seq IN ({placeholders})",
            [user_id, *seq_list],
        )
        self._maybe_commit()

    def execute_forget_action(
        self,
        *,
        user_id: str,
        trigger_seq: int,
        batch_seqs_to_tombstone: Sequence[int],
        target_existing_tags: Sequence[str],
        target_keywords: Sequence[str],
        target_description: str,
        target_vector: Sequence[float] | None,
    ) -> list[tuple[str, str, str, int, int | None]]:
        all_batch_seqs = [
            int(s)
            for s in dict.fromkeys([trigger_seq, *batch_seqs_to_tombstone])
            if s is not None and int(s) <= trigger_seq + 1
        ]
        self.tombstone_seqs(user_id, all_batch_seqs)

        tag_set = {t.strip() for t in target_existing_tags if t and t.strip()}
        kw_list = [k.strip() for k in target_keywords if k and len(k.strip()) >= 2]
        # P1-2: Prefer target_keywords as denominator; fallback to target_description if empty
        kw_terms = set(tokenize_bm25(" ".join(kw_list)))
        desc_terms = set(tokenize_bm25(target_description))
        target_terms = kw_terms if kw_terms else desc_terms

        cjk_kws = [
            k
            for k in kw_list
            if sum(1 for ch in k if "\u3400" <= ch <= "\u9fff")
            >= (2 if len(kw_list) == 1 else 3)
        ]

        def _kw_hit(kw: str, lower_text: str, c_terms: set[str]) -> bool:
            if kw.casefold() in lower_text:
                return True
            parts = kw.split()
            if not parts:
                return False
            for part in parts:
                p_toks = set(tokenize_bm25(part))
                if not p_toks or not (p_toks & c_terms):
                    return False
            return True

        def _lexical_forget_match(text_combined: str) -> bool:
            if not text_combined:
                return False
            lower_text = text_combined.casefold()
            if cjk_kws and any(kw.casefold() in lower_text for kw in cjk_kws):
                return True
            c_terms = set(tokenize_bm25(text_combined))
            if kw_list:
                kw_hits = sum(1 for kw in kw_list if _kw_hit(kw, lower_text, c_terms))
                if len(kw_list) <= 2:
                    return kw_hits == len(kw_list)
                if (kw_hits / len(kw_list)) >= 0.35:
                    return True
            elif target_terms:
                matched_terms = target_terms & c_terms
                if len(target_terms) <= 2:
                    return len(matched_terms) == len(target_terms)
                if (len(matched_terms) / len(target_terms)) >= 0.35:
                    return True
            return False

        conn = self._connect()
        # 1. Check tag_timelines (respecting seq <= trigger_seq causal boundary)
        tl_rows = conn.execute(
            "SELECT subject, tag, current_state, history_chain, rendered_summary "
            "FROM tag_timelines WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        tombstoned_tags: set[tuple[str, str]] = set()
        reconstructed_tl_cids: set[str] = set()
        rebuilt_timelines: list[tuple[str, str, str, int, int | None]] = []
        for r in tl_rows:
            subj = str(r["subject"])
            tg = str(r["tag"])
            is_full_tag_hit = tg in tag_set
            text_combined = f"{tg} {r['current_state']} {r['rendered_summary']}"
            matched = is_full_tag_hit or _lexical_forget_match(text_combined)
            if matched:
                chain: list[dict[str, Any]] = []
                if r["history_chain"]:
                    with contextlib.suppress(Exception):
                        loaded = json.loads(str(r["history_chain"]))
                        if isinstance(loaded, list):
                            chain = loaded
                if is_full_tag_hit:
                    surviving_chain = [
                        item for item in chain if int(item.get("seq") or 0) > trigger_seq
                    ]
                else:
                    surviving_chain = []
                    for item in chain:
                        item_seq = int(item.get("seq") or 0)
                        if item_seq > trigger_seq:
                            surviving_chain.append(item)
                            continue
                        node_text = (
                            f"{item.get('state', '')} {item.get('prev_state', '')} "
                            f"{item.get('reason', '')} {item.get('fact_text', '')}"
                        )
                        if item_seq in all_batch_seqs or _lexical_forget_match(node_text):
                            continue
                        surviving_chain.append(item)
                if not surviving_chain:
                    tombstoned_tags.add((subj, tg))
                    conn.execute(
                        "UPDATE tag_timelines SET tombstoned = 1 "
                        "WHERE user_id = ? AND subject = ? AND tag = ?",
                        (user_id, subj, tg),
                    )
                    conn.execute(
                        "UPDATE search_chunks SET tombstoned = 1 "
                        "WHERE user_id = ? AND kind = 'timeline_summary' "
                        "AND subject = ? AND tag = ?",
                        (user_id, subj, tg),
                    )
                else:
                    if is_full_tag_hit:
                        tombstoned_tags.add((subj, tg))
                    latest = surviving_chain[-1]
                    l_seq = int(latest.get("seq") or trigger_seq)
                    l_ts = (
                        int(latest["session_ts"])
                        if latest.get("session_ts") is not None
                        else None
                    )
                    res_curr = str(latest.get("state") or "")
                    h_parts: list[str] = []
                    for idx, item in enumerate(surviving_chain):
                        st = str(item.get("state") or "")
                        pr = str(item.get("prev_state") or "")
                        rsn = str(item.get("reason") or "")
                        ft = str(item.get("fact_text") or "")
                        t_label = str(item.get("time_str") or f"#{item.get('seq')}")
                        if idx == 0 and pr and pr not in st:
                            h_parts.append(f"[早期状态] {pr}")
                        detail = f"[{t_label}] {st}"
                        if rsn:
                            detail += f"（原因: {rsn}）"
                        if ft and ft not in detail:
                            detail += f" - {ft}"
                        h_parts.append(detail)
                    c_str = " -> ".join(h_parts)
                    r_sum = (
                        f"【状态时间线 | 主体: {subj} | 标签: {tg}】"
                        f"当前最新状态: {res_curr} | 历史演化: {c_str}"
                    )
                    conn.execute(
                        "UPDATE tag_timelines SET latest_seq = ?, latest_session_ts = ?, "
                        "current_state = ?, history_chain = ?, rendered_summary = ?, "
                        "tombstoned = 0 WHERE user_id = ? AND subject = ? AND tag = ?",
                        (
                            l_seq,
                            l_ts,
                            res_curr,
                            json.dumps(surviving_chain, ensure_ascii=False),
                            r_sum,
                            user_id,
                            subj,
                            tg,
                        ),
                    )
                    conn.execute(
                        "UPDATE search_chunks SET seq = ?, session_ts = ?, display_text = ?, "
                        "search_text = ?, bm25_tokens = ?, tombstoned = 0 "
                        "WHERE user_id = ? AND kind = 'timeline_summary' "
                        "AND subject = ? AND tag = ?",
                        (
                            l_seq,
                            l_ts,
                            r_sum,
                            r_sum,
                            build_bm25_tokens(r_sum, subject=subj, tag=tg),
                            user_id,
                            subj,
                            tg,
                        ),
                    )
                    reconstructed_tl_cids.add(f"tl:{user_id}:{subj}:{tg}")
                    rebuilt_timelines.append((subj, tg, r_sum, l_seq, l_ts))

        # 2. Check search_chunks with causal boundary seq <= trigger_seq
        chunk_rows = conn.execute(
            """
            SELECT chunk_id, parent_msg_id, seq, kind, subject, tag,
                   display_text, search_text, embedding
            FROM search_chunks
            WHERE user_id = ? AND tombstoned = 0 AND seq <= ?
            """,
            (user_id, trigger_seq),
        ).fetchall()

        matched_chunk_ids: set[str] = set()
        affected_parent_ids: dict[str, list[str]] = {}

        for r in chunk_rows:
            cid = str(r["chunk_id"])
            if cid in reconstructed_tl_cids:
                continue
            pmid = str(r["parent_msg_id"]) if r["parent_msg_id"] else None
            kind = str(r["kind"])
            tg = str(r["tag"]) if r["tag"] else ""
            subj = str(r["subject"] or "user")
            disp = str(r["display_text"] or "")
            srch = str(r["search_text"] or "")
            combined = f"{disp}\n{srch}"

            hit = False
            if tg and (tg in tag_set or (subj, tg) in tombstoned_tags):
                hit = True
            if not hit and _lexical_forget_match(combined):
                hit = True
            # P1-2: Pure cosine >= 0.58 is an independent OR condition;
            # for >=3 keywords, >=1 keyword hit + cosine >= 0.42 confirms hybrid match
            if not hit and target_vector:
                vec = self._vector_cache.get(cid) or blob_to_vector(r["embedding"])
                cos_sim = cosine_similarity(target_vector, vec)
                if cos_sim >= 0.58:
                    hit = True
                elif cos_sim >= 0.42 and len(kw_list) >= 3:
                    comb_lower = combined.casefold()
                    comb_terms = set(tokenize_bm25(combined))
                    if any(_kw_hit(kw, comb_lower, comb_terms) for kw in kw_list):
                        hit = True

            if hit:
                matched_chunk_ids.add(cid)
                if pmid:
                    affected_parent_ids.setdefault(pmid, []).append(disp)
                if kind == "timeline_summary" and tg and (subj, tg) not in tombstoned_tags:
                    tombstoned_tags.add((subj, tg))
                    conn.execute(
                        "UPDATE tag_timelines SET tombstoned = 1 "
                        "WHERE user_id = ? AND subject = ? AND tag = ?",
                        (user_id, subj, tg),
                    )

        for cid in matched_chunk_ids:
            conn.execute(
                "UPDATE search_chunks SET tombstoned = 1 WHERE chunk_id = ?",
                (cid,),
            )

        # 3. Also check active parent_messages with seq <= trigger_seq
        parent_rows = conn.execute(
            "SELECT msg_id, seq, role, session_id, content FROM parent_messages "
            "WHERE user_id = ? AND tombstoned = 0 AND seq <= ? ORDER BY seq ASC",
            (user_id, trigger_seq),
        ).fetchall()

        tombstoned_user_seqs: set[tuple[str, int]] = set()
        for pr in parent_rows:
            pmid = str(pr["msg_id"])
            p_seq = int(pr["seq"])
            p_role = str(pr["role"] or "")
            p_sid = str(pr["session_id"] or "")
            p_content = str(pr["content"] or "")
            p_lower = p_content.casefold()
            p_terms = set(tokenize_bm25(p_content))
            direct_kw_hit = _lexical_forget_match(p_content)
            paired_assistant_hit = (
                p_role == "assistant"
                and (p_sid, p_seq - 1) in tombstoned_user_seqs
                and bool(kw_list and any(_kw_hit(kw, p_lower, p_terms) for kw in kw_list))
            )
            if pmid not in affected_parent_ids and not direct_kw_hit and not paired_assistant_hit:
                continue

            if len(p_content) > LONG_DOC_CHAR_THRESHOLD:
                # Partial masking for >3000 char document: replace slice / lines with [已擦除]
                masked = p_content
                for slice_disp in affected_parent_ids.get(pmid, []):
                    if slice_disp and slice_disp in masked:
                        masked = masked.replace(slice_disp, "[已擦除]")
                if kw_list:
                    new_lines: list[str] = []
                    for line in masked.splitlines(keepends=True):
                        if _lexical_forget_match(line) or any(
                            kw.casefold() in line.casefold() for kw in kw_list
                        ):
                            suffix = "\n" if line.endswith("\n") else ""
                            new_lines.append(f"[已擦除]{suffix}")
                        else:
                            new_lines.append(line)
                    masked = "".join(new_lines)
                if "[已擦除]" not in masked:
                    masked += "\n[已擦除]"
                conn.execute(
                    "UPDATE parent_messages SET content = ? WHERE msg_id = ?",
                    (masked, pmid),
                )
            else:
                # Normal parent message <= 3000 chars: check slice-level surgical tombstoning
                raw_slices = conn.execute(
                    "SELECT chunk_id, display_text, tag, tombstoned FROM search_chunks "
                    "WHERE parent_msg_id = ? AND kind = 'raw_slice' ORDER BY chunk_id ASC",
                    (pmid,),
                ).fetchall()
                distinct_tags = {
                    str(r["tag"])
                    for r in conn.execute(
                        "SELECT DISTINCT tag FROM search_chunks "
                        "WHERE parent_msg_id = ? AND tag IS NOT NULL AND tag != ''",
                        (pmid,),
                    ).fetchall()
                }
                is_multi_topic = len(distinct_tags) >= 2 or any(
                    is_incidental_update_text(str(rs["display_text"] or ""))
                    for rs in raw_slices
                )
                if len(raw_slices) <= 1 or not is_multi_topic:
                    conn.execute(
                        "UPDATE parent_messages SET tombstoned = 1 WHERE msg_id = ?",
                        (pmid,),
                    )
                    conn.execute(
                        "UPDATE search_chunks SET tombstoned = 1 "
                        "WHERE parent_msg_id = ? AND kind = 'raw_slice'",
                        (pmid,),
                    )
                    if p_role == "user":
                        tombstoned_user_seqs.add((p_sid, p_seq))
                else:
                    for rs in raw_slices:
                        if int(rs["tombstoned"]) == 0 and _lexical_forget_match(
                            str(rs["display_text"] or "")
                        ):
                            conn.execute(
                                "UPDATE search_chunks SET tombstoned = 1 WHERE chunk_id = ?",
                                (rs["chunk_id"],),
                            )
                    counts_row = conn.execute(
                        "SELECT "
                        "SUM(CASE WHEN tombstoned = 0 THEN 1 ELSE 0 END) AS active_cnt, "
                        "SUM(CASE WHEN tombstoned = 1 THEN 1 ELSE 0 END) AS tomb_cnt "
                        "FROM search_chunks "
                        "WHERE parent_msg_id = ? AND kind = 'raw_slice'",
                        (pmid,),
                    ).fetchone()
                    active_slice_cnt = int(counts_row["active_cnt"] or 0)
                    tomb_slice_cnt = int(counts_row["tomb_cnt"] or 0)
                    if active_slice_cnt == 0 or (direct_kw_hit and tomb_slice_cnt == 0):
                        conn.execute(
                            "UPDATE parent_messages SET tombstoned = 1 WHERE msg_id = ?",
                            (pmid,),
                        )
                        conn.execute(
                            "UPDATE search_chunks SET tombstoned = 1 "
                            "WHERE parent_msg_id = ? AND kind = 'raw_slice'",
                            (pmid,),
                        )
                        if p_role == "user":
                            tombstoned_user_seqs.add((p_sid, p_seq))

        self._maybe_commit()
        return rebuilt_timelines

    def get_active_chunks(self, user_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, user_id, session_id, parent_msg_id, seq, session_ts,
                   kind, subject, tag, source_tier, context_prefix,
                   display_text, search_text, bm25_tokens,
                   event_time_start, event_time_end, embedding
            FROM search_chunks
            WHERE user_id = ? AND tombstoned = 0
            ORDER BY seq ASC, chunk_id ASC
            """,
            (user_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            cid = str(r["chunk_id"])
            vec = self._vector_cache.get(cid)
            if vec is None:
                vec = blob_to_vector(r["embedding"])
                if vec:
                    self._vector_cache[cid] = vec
            result.append(
                {
                    "chunk_id": cid,
                    "user_id": str(r["user_id"]),
                    "session_id": str(r["session_id"]),
                    "parent_msg_id": str(r["parent_msg_id"]) if r["parent_msg_id"] else None,
                    "seq": int(r["seq"]),
                    "session_ts": (
                        int(r["session_ts"]) if r["session_ts"] is not None else None
                    ),
                    "kind": str(r["kind"]),
                    "subject": str(r["subject"] or "user"),
                    "tag": str(r["tag"]) if r["tag"] else None,
                    "source_tier": str(r["source_tier"] or "user_direct"),
                    "context_prefix": str(r["context_prefix"] or ""),
                    "display_text": str(r["display_text"]),
                    "search_text": str(r["search_text"]),
                    "bm25_tokens": str(r["bm25_tokens"]),
                    "event_time_start": (
                        int(r["event_time_start"])
                        if r["event_time_start"] is not None
                        else None
                    ),
                    "event_time_end": (
                        int(r["event_time_end"])
                        if r["event_time_end"] is not None
                        else None
                    ),
                    "vector": vec,
                }
            )
        return result

    def get_chunks_for_parents(
        self, user_id: str, parent_msg_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        id_list = list(dict.fromkeys(m for m in parent_msg_ids if m))
        if not id_list:
            return {}
        conn = self._connect()
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(id_list), 500):
            batch_ids = id_list[start : start + 500]
            placeholders = ",".join("?" for _ in batch_ids)
            rows = conn.execute(
                f"""
                SELECT chunk_id, parent_msg_id, seq, kind, subject, tag,
                       source_tier, display_text, tombstoned
                FROM search_chunks
                WHERE user_id = ? AND parent_msg_id IN ({placeholders})
                  AND kind IN ('raw_slice', 'fact_item')
                ORDER BY chunk_id ASC
                """,
                [user_id, *batch_ids],
            ).fetchall()
            for r in rows:
                pmid = str(r["parent_msg_id"])
                out.setdefault(pmid, []).append(
                    {
                        "chunk_id": str(r["chunk_id"]),
                        "parent_msg_id": pmid,
                        "seq": int(r["seq"]),
                        "kind": str(r["kind"]),
                        "subject": str(r["subject"] or "user"),
                        "tag": str(r["tag"]) if r["tag"] else None,
                        "source_tier": str(r["source_tier"] or "user_direct"),
                        "display_text": str(r["display_text"]),
                        "tombstoned": int(r["tombstoned"]),
                    }
                )
        return out

    def get_active_cold_tag_vectors(
        self, user_id: str, subject: str
    ) -> dict[str, list[dict[str, Any]]]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, tag, kind, display_text, embedding
            FROM search_chunks
            WHERE user_id = ? AND subject = ? AND tombstoned = 0
              AND tag LIKE '其他/%' AND kind IN ('fact_item', 'timeline_summary')
            ORDER BY seq ASC
            """,
            (user_id, subject),
        ).fetchall()
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            tg = str(r["tag"] or "")
            if not tg.startswith("其他/"):
                continue
            cid = str(r["chunk_id"])
            vec = self._vector_cache.get(cid)
            if vec is None:
                vec = blob_to_vector(r["embedding"])
                if vec:
                    self._vector_cache[cid] = vec
            out.setdefault(tg, []).append(
                {
                    "chunk_id": cid,
                    "kind": str(r["kind"]),
                    "display_text": str(r["display_text"]),
                    "vector": vec,
                }
            )
        return out

    def get_parent_messages_by_ids(
        self, user_id: str, msg_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        id_list = list(dict.fromkeys(m for m in msg_ids if m))
        if not id_list:
            return {}
        conn = self._connect()
        out: dict[str, dict[str, Any]] = {}
        # Batch in chunks of 500 to stay well below SQLite's 999 parameter limit
        for start in range(0, len(id_list), 500):
            batch_ids = id_list[start : start + 500]
            placeholders = ",".join("?" for _ in batch_ids)
            rows = conn.execute(
                f"""
                SELECT msg_id, user_id, session_id, request_id, seq, role, speaker,
                       content, session_ts, session_time_str, fact_notes, tombstoned
                FROM parent_messages
                WHERE user_id = ? AND msg_id IN ({placeholders})
                """,
                [user_id, *batch_ids],
            ).fetchall()
            for r in rows:
                out[str(r["msg_id"])] = dict(r)
        return out

    def get_session_neighbors(
        self, user_id: str, session_id: str, seq: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT msg_id, seq, role, speaker, content, session_ts,
                   session_time_str, fact_notes, tombstoned
            FROM parent_messages
            WHERE user_id = ? AND session_id = ? AND seq IN (?, ?) AND tombstoned = 0
            """,
            (user_id, session_id, seq - 1, seq + 1),
        ).fetchall()
        prev_row: dict[str, Any] | None = None
        next_row: dict[str, Any] | None = None
        for r in rows:
            d = dict(r)
            if int(d["seq"]) == seq - 1:
                prev_row = d
            elif int(d["seq"]) == seq + 1:
                next_row = d
        return prev_row, next_row

    def get_latest_tag_map(self, user_id: str) -> dict[tuple[str, str], tuple[int | None, int]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT subject, tag, latest_session_ts, latest_seq FROM tag_timelines "
            "WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        out: dict[tuple[str, str], tuple[int | None, int]] = {}
        for r in rows:
            l_ts = (
                int(r["latest_session_ts"])
                if r["latest_session_ts"] is not None
                else None
            )
            out[(str(r["subject"]), str(r["tag"]))] = (l_ts, int(r["latest_seq"]))
        return out


class SupabaseStore:
    def __init__(
        self, *, base_url: str, api_key: str, storage_token: str, timeout: float = 120.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.storage_token = storage_token
        self.timeout = timeout

    async def _rpc(self, name: str, payload: dict[str, object]) -> object:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/rest/v1/rpc/{name}",
                headers={
                    "apikey": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code == 409 or "AML_IDEMPOTENCY_CONFLICT" in response.text:
            raise StoreConflictError("request_id already exists with a different payload")
        response.raise_for_status()
        return response.json()

    async def add_request(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
        items: Sequence[WriteItem],
        archive_ids: Sequence[str],
    ) -> bool:
        result = await self._rpc(
            "aml_add_request",
            {
                "p_request_id": request_id,
                "p_user_id": user_id,
                "p_session_id": session_id,
                "p_payload_hash": payload_hash,
                "p_items": [item.as_rpc_dict() for item in items],
                "p_archive_ids": list(archive_ids),
                "p_storage_token": self.storage_token,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("aml_add_request returned an invalid result")
        return bool(result.get("inserted"))

    async def search_candidates(
        self,
        *,
        user_id: str,
        query: str,
        embedding: Sequence[float],
        semantic_limit: int,
        lexical_limit: int,
        result_limit: int,
    ) -> list[MemoryCandidate]:
        rows = await self._rpc(
            "aml_search_candidates",
            {
                "p_user_id": user_id,
                "p_query_text": query,
                "p_query_embedding": list(embedding),
                "p_semantic_limit": semantic_limit,
                "p_lexical_limit": lexical_limit,
                "p_result_limit": result_limit,
                "p_storage_token": self.storage_token,
            },
        )
        if not isinstance(rows, list):
            raise RuntimeError("aml_search_candidates returned an invalid result")
        return [self._candidate(row) for row in rows]

    async def neighbors(
        self, *, user_id: str, anchor_ids: Sequence[str], radius: int
    ) -> list[MemoryCandidate]:
        if not anchor_ids or radius <= 0:
            return []
        rows = await self._rpc(
            "aml_neighbor_memories",
            {
                "p_user_id": user_id,
                "p_anchor_ids": list(anchor_ids),
                "p_radius": radius,
                "p_storage_token": self.storage_token,
            },
        )
        if not isinstance(rows, list):
            raise RuntimeError("aml_neighbor_memories returned an invalid result")
        return [self._candidate(row) for row in rows]

    @staticmethod
    def _candidate(row: dict[str, object]) -> MemoryCandidate:
        return MemoryCandidate(
            id=str(row["id"]),
            content=str(row["content"]),
            score=float(row.get("score") or 0.0),
            created_at=str(row["created_at"]) if row.get("created_at") else None,
            session_id=str(row["session_id"]) if row.get("session_id") else None,
            source_timestamp=str(row["source_timestamp"]) if row.get("source_timestamp") else None,
            message_index=int(row.get("message_index") or 0),
            chunk_index=int(row.get("chunk_index") or 0),
            anchor_id=str(row["anchor_id"]) if row.get("anchor_id") else None,
            anchor_rank=int(row["anchor_rank"]) if row.get("anchor_rank") else None,
            distance=int(row["distance"]) if row.get("distance") is not None else None,
        )
