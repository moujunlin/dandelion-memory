from __future__ import annotations

import contextlib
import json
import math
import re
import sqlite3
from array import array
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import MemoryCandidate, WriteItem
from .text import LONG_DOC_CHAR_THRESHOLD, build_bm25_tokens, tokenize_bm25


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
        tombstoned: int = 0,
    ) -> None:
        blob = vector_to_blob(embedding)
        self._vector_cache[chunk_id] = list(embedding)
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO search_chunks (
                chunk_id, user_id, session_id, parent_msg_id, seq,
                session_ts, kind, subject, tag, display_text,
                search_text, bm25_tokens, event_time_start, event_time_end,
                embedding, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                session_id = excluded.session_id,
                parent_msg_id = excluded.parent_msg_id,
                seq = excluded.seq,
                session_ts = excluded.session_ts,
                kind = excluded.kind,
                subject = excluded.subject,
                tag = excluded.tag,
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
    ) -> None:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, display_text, bm25_tokens, tag
            FROM search_chunks
            WHERE user_id = ? AND seq = ? AND kind = 'raw_slice'
            """,
            (user_id, seq),
        ).fetchall()
        for r in rows:
            cid = str(r["chunk_id"])
            disp = str(r["display_text"])
            existing_bm25 = str(r["bm25_tokens"] or "").strip()
            existing_tag = r["tag"]
            extra_bm25 = build_bm25_tokens(fact_text, subject=subject, tag=tag)
            if existing_tag:
                # Merge additional fact tokens without overwriting earlier facts on the same seq
                bm25 = f"{existing_bm25} {extra_bm25}".strip()
            else:
                bm25 = build_bm25_tokens(
                    f"{disp} {fact_text}".strip(), subject=subject, tag=tag
                )
            conn.execute(
                """
                UPDATE search_chunks
                SET subject = ?, tag = ?, event_time_start = ?, event_time_end = ?,
                    bm25_tokens = ?
                WHERE chunk_id = ?
                """,
                (subject, tag, event_time_start, event_time_end, bm25, cid),
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
    ) -> None:
        all_batch_seqs = list(dict.fromkeys([trigger_seq, *batch_seqs_to_tombstone]))
        self.tombstone_seqs(user_id, all_batch_seqs)

        tag_set = {t.strip() for t in target_existing_tags if t and t.strip()}
        kw_list = [k.strip() for k in target_keywords if k and len(k.strip()) >= 2]
        cjk_kws = [k for k in kw_list if re.search(r"[\u3400-\u9fff]{2,}", k)]
        # P1-2: Prefer target_keywords as denominator; fallback to target_description if empty
        kw_terms = set(tokenize_bm25(" ".join(kw_list)))
        desc_terms = set(tokenize_bm25(target_description))
        target_terms = kw_terms if kw_terms else desc_terms

        conn = self._connect()
        # 1. Check tag_timelines
        tl_rows = conn.execute(
            "SELECT subject, tag, current_state, rendered_summary FROM tag_timelines "
            "WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        tombstoned_tags: set[tuple[str, str]] = set()
        for r in tl_rows:
            subj = str(r["subject"])
            tg = str(r["tag"])
            text_combined = f"{tg} {r['current_state']} {r['rendered_summary']}"
            matched = tg in tag_set or any(
                kw.casefold() in text_combined.casefold() for kw in cjk_kws
            )
            if not matched and kw_list:
                kw_hits = sum(
                    1 for kw in kw_list if kw.casefold() in text_combined.casefold()
                )
                if (kw_hits / len(kw_list)) >= 0.35:
                    matched = True
            if not matched and target_terms:
                cand_terms = set(tokenize_bm25(text_combined))
                overlap = len(target_terms & cand_terms) / max(1, len(target_terms))
                if overlap >= 0.35:
                    matched = True
            if matched:
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

        # 2. Check search_chunks
        chunk_rows = conn.execute(
            """
            SELECT chunk_id, parent_msg_id, seq, kind, subject, tag,
                   display_text, search_text, embedding
            FROM search_chunks
            WHERE user_id = ? AND tombstoned = 0
            """,
            (user_id,),
        ).fetchall()

        matched_chunk_ids: set[str] = set()
        affected_parent_ids: dict[str, list[str]] = {}

        for r in chunk_rows:
            cid = str(r["chunk_id"])
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
            if not hit and cjk_kws and any(
                kw.casefold() in combined.casefold() for kw in cjk_kws
            ):
                hit = True
            if not hit and kw_list:
                kw_hits = sum(
                    1 for kw in kw_list if kw.casefold() in combined.casefold()
                )
                if (kw_hits / len(kw_list)) >= 0.35:
                    hit = True
            if not hit and target_terms:
                c_terms = set(tokenize_bm25(combined))
                overlap = len(target_terms & c_terms) / max(1, len(target_terms))
                if overlap >= 0.35:
                    hit = True
            # P1-2: Pure cosine >= 0.58 is an independent OR condition even when overlap == 0
            if not hit and target_vector:
                vec = self._vector_cache.get(cid) or blob_to_vector(r["embedding"])
                if cosine_similarity(target_vector, vec) >= 0.58:
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

        # 3. Also check active parent_messages directly
        parent_rows = conn.execute(
            "SELECT msg_id, seq, content FROM parent_messages "
            "WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()

        for pr in parent_rows:
            pmid = str(pr["msg_id"])
            p_content = str(pr["content"] or "")
            direct_kw_hit = bool(
                kw_list and any(kw.casefold() in p_content.casefold() for kw in kw_list)
            )
            if pmid not in affected_parent_ids and not direct_kw_hit:
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
                        if any(kw.casefold() in line.casefold() for kw in kw_list):
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
                # Normal parent message <= 3000 chars: tombstone parent and its raw_slice chunks
                conn.execute(
                    "UPDATE parent_messages SET tombstoned = 1 WHERE msg_id = ?",
                    (pmid,),
                )
                conn.execute(
                    "UPDATE search_chunks SET tombstoned = 1 "
                    "WHERE parent_msg_id = ? AND kind = 'raw_slice'",
                    (pmid,),
                )

        self._maybe_commit()

    def get_active_chunks(self, user_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, user_id, session_id, parent_msg_id, seq, session_ts,
                   kind, subject, tag, display_text, search_text, bm25_tokens,
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
