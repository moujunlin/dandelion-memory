from __future__ import annotations

import contextlib
import json
import math
import re
import sqlite3
from array import array
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import MemoryCandidate, WriteItem
from .text import (
    LONG_DOC_CHAR_THRESHOLD,
    build_bm25_tokens,
    get_base_tag,
    has_explicit_forget_keyword,
    is_incidental_update_text,
    is_naturally_multivalued,
    tokenize_bm25,
)

_GENERIC_ENTITY_STOP_TOKENS = {
    "用户", "用", "户", "我", "我的", "那位", "那个", "这只", "那只", "这位",
    "一个", "一只", "一辆", "一条", "一位", "一家", "一套", "一部", "一台",
    "一", "个", "只", "辆", "条", "位", "家", "套", "部", "台",
    "养了", "领养", "买了", "新买", "有一", "拥有", "饲养", "名叫", "叫", "名字", "名为",
    "养", "领", "买", "新", "有", "名", "在", "做", "是", "的", "了",
    "今年", "最近", "目前", "现在", "已经", "刚", "又", "还",
    "user", "my", "the", "a", "an",
}

_NAMED_ID_RES = (
    re.compile(r"(?:名叫|名为|叫)\s*([A-Za-z0-9\u4e00-\u9fff]{1,6}?)(?:的|[，,。！!？?\s（(]|$)"),
    re.compile(
        r"(?:同学|朋友|同事|校友|室友|闺蜜|导师|老师|搭档)"
        r"(?!(?:在|是|有|做|的|都|也|今年|最近|目前|现在|发了|升职))"
        r"([A-Za-z\u4e00-\u9fff]{2,4}?)(?:在|于|[，,。！!？?\s（(]|$)"
    ),
)


def _extract_named_identifiers(text: str) -> set[str]:
    names: set[str] = set()
    for pat in _NAMED_ID_RES:
        for m in pat.finditer(text):
            val = m.group(1).strip().casefold()
            if val and val not in _GENERIC_ENTITY_STOP_TOKENS:
                names.add(val)
    return names


def _extract_entity_tokens(text: str, base_tag: str) -> set[str]:
    tag_toks = set(tokenize_bm25(base_tag))
    all_toks = {
        t
        for t in tokenize_bm25(text)
        if t not in _GENERIC_ENTITY_STOP_TOKENS and t not in tag_toks
    }
    multi_char = {t for t in all_toks if len(t) >= 2}
    return multi_char if multi_char else all_toks


class UpsertTimelineResult(tuple):
    slot_id: str

    def __new__(
        cls,
        rendered_summary: str,
        latest_seq: int,
        latest_ts: int | None,
        resolved_tag: str,
        is_fact_tombstoned: bool,
        slot_id: str,
    ) -> UpsertTimelineResult:
        obj = super().__new__(
            cls,
            (rendered_summary, latest_seq, latest_ts, resolved_tag, is_fact_tombstoned),
        )
        obj.slot_id = slot_id
        return obj


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
        self._active_deadline_check: Callable[[], bool] | None = None
        self._vector_cache: dict[str, list[float]] = {}
        self._init_schema()

    def _check_deadline(self) -> None:
        if self._active_deadline_check is not None and self._active_deadline_check():
            raise TimeoutError("Stage 3 operation deadline exceeded")

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def _maybe_commit(self) -> None:
        if not self._in_tx:
            self._conn.commit()

    @contextlib.contextmanager
    def transaction(
        self, *, deadline_check: Callable[[], bool] | None = None
    ) -> Iterator[sqlite3.Connection]:
        if self._in_tx:
            yield self._conn
            return
        self._in_tx = True
        prev_deadline_check = self._active_deadline_check
        self._active_deadline_check = deadline_check
        cache_snapshot = dict(self._vector_cache)
        orig_connect = sqlite3.connect
        store_conn = self._conn
        target_db = self.db_path
        if deadline_check is not None:
            self._conn.set_progress_handler(lambda: 1 if deadline_check() else 0, 1)

        class _TxConnProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(store_conn, name)

            def commit(self) -> None:
                pass

            def close(self) -> None:
                pass

            def __enter__(self) -> _TxConnProxy:
                return self

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        def _tx_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
            if str(database) == target_db:
                return _TxConnProxy()
            return orig_connect(database, *args, **kwargs)

        sqlite3.connect = _tx_connect  # type: ignore[assignment]
        try:
            yield self._conn
            if deadline_check is not None and deadline_check():
                raise TimeoutError("Stage 3 transaction deadline exceeded before commit")
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            if deadline_check is not None:
                self._conn.set_progress_handler(None, 0)
            self._conn.rollback()
            self._vector_cache = cache_snapshot
            if "interrupted" in str(exc).lower() or (
                deadline_check is not None and deadline_check()
            ):
                raise TimeoutError(
                    "Stage 3 SQLite transaction interrupted by deadline"
                ) from exc
            raise
        except Exception:
            if deadline_check is not None:
                self._conn.set_progress_handler(None, 0)
            self._conn.rollback()
            self._vector_cache = cache_snapshot
            raise
        finally:
            self._active_deadline_check = prev_deadline_check
            if deadline_check is not None:
                self._conn.set_progress_handler(None, 0)
            sqlite3.connect = orig_connect  # type: ignore[assignment]
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
                turn_pair_id TEXT DEFAULT '',
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
                slot_id TEXT NOT NULL DEFAULT '',
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
            CREATE INDEX IF NOT EXISTS idx_chunks_user_slot
                ON search_chunks (user_id, slot_id);

            CREATE TABLE IF NOT EXISTS tag_timelines (
                user_id TEXT NOT NULL,
                slot_id TEXT NOT NULL DEFAULT (hex(randomblob(8))),
                short_id TEXT NOT NULL DEFAULT '',
                subject TEXT NOT NULL DEFAULT 'user',
                tag TEXT NOT NULL,
                latest_seq INTEGER NOT NULL DEFAULT 0,
                latest_session_ts INTEGER,
                current_state TEXT NOT NULL DEFAULT '',
                history_chain TEXT NOT NULL DEFAULT '',
                rendered_summary TEXT NOT NULL DEFAULT '',
                superseded_seqs TEXT NOT NULL DEFAULT '[]',
                source_seqs TEXT NOT NULL DEFAULT '[]',
                tombstoned INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, slot_id)
            );
            CREATE TRIGGER IF NOT EXISTS trg_tag_timelines_slot_id_sync
            AFTER INSERT ON tag_timelines
            BEGIN
                UPDATE tag_timelines
                SET slot_id = CASE WHEN NEW.short_id != '' THEN NEW.short_id ELSE slot_id END,
                    short_id = CASE
                        WHEN NEW.slot_id != '' AND NEW.short_id = '' THEN NEW.slot_id
                        ELSE short_id
                    END
                WHERE rowid = NEW.rowid;
            END;
            CREATE INDEX IF NOT EXISTS idx_tl_user_subj_active
                ON tag_timelines (user_id, subject, tombstoned);
            CREATE INDEX IF NOT EXISTS idx_tl_user_tag
                ON tag_timelines (user_id, tag);

            CREATE TABLE IF NOT EXISTS person_profiles (
                user_id TEXT NOT NULL,
                person TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                relations TEXT NOT NULL DEFAULT '[]',
                updated_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, person)
            );
            """
        )
        pm_cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(parent_messages)").fetchall()
        }
        if "turn_pair_id" not in pm_cols:
            conn.execute(
                "ALTER TABLE parent_messages ADD COLUMN turn_pair_id TEXT DEFAULT ''"
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
        if "slot_id" not in cols:
            conn.execute(
                "ALTER TABLE search_chunks ADD COLUMN slot_id TEXT NOT NULL DEFAULT ''"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_user_slot ON search_chunks(user_id, slot_id)"
        )
        tl_cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(tag_timelines)").fetchall()
        }
        if "slot_id" not in tl_cols:
            conn.execute(
                "ALTER TABLE tag_timelines ADD COLUMN slot_id TEXT NOT NULL DEFAULT ''"
            )
            conn.execute("UPDATE tag_timelines SET slot_id = short_id WHERE slot_id = ''")
        if "short_id" not in tl_cols:
            conn.execute(
                "ALTER TABLE tag_timelines ADD COLUMN short_id TEXT NOT NULL DEFAULT ''"
            )
        if "superseded_seqs" not in tl_cols:
            conn.execute(
                "ALTER TABLE tag_timelines "
                "ADD COLUMN superseded_seqs TEXT NOT NULL DEFAULT '[]'"
            )
        if "source_seqs" not in tl_cols:
            conn.execute(
                "ALTER TABLE tag_timelines "
                "ADD COLUMN source_seqs TEXT NOT NULL DEFAULT '[]'"
            )
        if "updated_at" not in tl_cols:
            conn.execute(
                "ALTER TABLE tag_timelines ADD COLUMN updated_at INTEGER DEFAULT 0"
            )
        conn.execute("DROP VIEW IF EXISTS living_facts")
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS living_facts AS
            SELECT
                user_id,
                slot_id,
                short_id,
                subject,
                tag AS key,
                tag,
                current_state,
                latest_seq,
                latest_session_ts,
                superseded_seqs,
                source_seqs
            FROM tag_timelines
            WHERE tombstoned = 0
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

    def _allocate_next_slot_id(self, user_id: str) -> str:
        conn = self._connect()
        rows = conn.execute(
            "SELECT slot_id, short_id FROM tag_timelines WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        max_idx = 0
        for r in rows:
            for col in ("slot_id", "short_id"):
                sid = str(r[col] or "").strip()
                if sid.startswith("F") and sid[1:].isdigit():
                    max_idx = max(max_idx, int(sid[1:]))
        return f"F{max_idx + 1}"

    _allocate_next_short_id = _allocate_next_slot_id

    def upsert_person_profile(
        self,
        user_id: str,
        person: str,
        aliases: Sequence[str] | None = None,
        relations: Sequence[str] | None = None,
        updated_at: int = 0,
    ) -> None:
        person_clean = str(person).strip()
        if not person_clean or person_clean.casefold() == "user":
            return
        conn = self._connect()
        row = conn.execute(
            "SELECT aliases, relations FROM person_profiles WHERE user_id = ? AND person = ?",
            (user_id, person_clean),
        ).fetchone()
        existing_aliases: list[str] = []
        existing_relations: list[str] = []
        if row is not None:
            with contextlib.suppress(Exception):
                existing_aliases = json.loads(str(row["aliases"] or "[]"))
            with contextlib.suppress(Exception):
                existing_relations = json.loads(str(row["relations"] or "[]"))

        merged_aliases = list(
            dict.fromkeys(
                [a for a in existing_aliases if a]
                + [str(a).strip() for a in (aliases or []) if str(a).strip()]
            )
        )
        merged_relations = list(
            dict.fromkeys(
                [r for r in existing_relations if r]
                + [str(r).strip() for r in (relations or []) if str(r).strip()]
            )
        )
        now_ts = updated_at or int(datetime.now(tz=UTC).timestamp())
        conn.execute(
            """
            INSERT INTO person_profiles (user_id, person, aliases, relations, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, person) DO UPDATE SET
                aliases = excluded.aliases,
                relations = excluded.relations,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                person_clean,
                json.dumps(merged_aliases, ensure_ascii=False),
                json.dumps(merged_relations, ensure_ascii=False),
                now_ts,
            ),
        )
        self._maybe_commit()

    def get_person_profiles(self, user_id: str) -> dict[str, dict[str, list[str]]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT person, aliases, relations FROM person_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        result: dict[str, dict[str, list[str]]] = {}
        for r in rows:
            p = str(r["person"])
            a_list: list[str] = []
            r_list: list[str] = []
            with contextlib.suppress(Exception):
                a_list = json.loads(str(r["aliases"] or "[]"))
            with contextlib.suppress(Exception):
                r_list = json.loads(str(r["relations"] or "[]"))
            result[p] = {"aliases": a_list, "relations": r_list}
        return result

    def get_living_facts(self, user_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT slot_id, short_id, subject, tag, current_state, latest_seq, latest_session_ts "
            "FROM tag_timelines WHERE user_id = ? AND tombstoned = 0 "
            "ORDER BY latest_seq ASC",
            (user_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            raw_tag = str(r["tag"])
            base_key = get_base_tag(raw_tag)
            sid = str(r["slot_id"] or r["short_id"] or "")
            out.append(
                {
                    "slot_id": sid,
                    "short_id": sid,
                    "fid": sid,
                    "subject": str(r["subject"]),
                    "key": base_key,
                    "tag": raw_tag,
                    "current_state": str(r["current_state"]),
                    "latest_seq": int(r["latest_seq"]),
                    "latest_session_ts": (
                        int(r["latest_session_ts"])
                        if r["latest_session_ts"] is not None
                        else None
                    ),
                }
            )
        return out

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
        turn_pair_id: str | None = None,
        tombstoned: int = 0,
    ) -> None:
        conn = self._connect()
        if turn_pair_id is None:
            if tombstoned != 0 or len(content) > LONG_DOC_CHAR_THRESHOLD:
                turn_pair_id = ""
            elif role == "user":
                turn_pair_id = f"{session_id}:tp:{seq}"
            elif role == "assistant" and seq > 1:
                prev_row = conn.execute(
                    "SELECT role, tombstoned, turn_pair_id FROM parent_messages "
                    "WHERE user_id = ? AND session_id = ? AND seq = ?",
                    (user_id, session_id, seq - 1),
                ).fetchone()
                if (
                    prev_row is not None
                    and str(prev_row["role"]) == "user"
                    and int(prev_row["tombstoned"]) == 0
                    and str(prev_row["turn_pair_id"] or "")
                ):
                    turn_pair_id = str(prev_row["turn_pair_id"])
                else:
                    turn_pair_id = ""
            else:
                turn_pair_id = ""

        conn.execute(
            """
            INSERT OR REPLACE INTO parent_messages (
                msg_id, user_id, session_id, request_id, seq,
                role, speaker, content, session_ts, session_time_str,
                fact_notes, turn_pair_id, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                turn_pair_id,
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
        slot_id: str = "",
    ) -> None:
        if not slot_id:
            parts = chunk_id.split(":")
            if len(parts) >= 3 and parts[-1].startswith("F") and parts[-1][1:].isdigit():
                slot_id = parts[-1]
        blob = vector_to_blob(embedding)
        self._vector_cache[chunk_id] = list(embedding)
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO search_chunks (
                chunk_id, user_id, session_id, parent_msg_id, slot_id, seq,
                session_ts, kind, subject, tag, source_tier, context_prefix,
                display_text, search_text, bm25_tokens, event_time_start, event_time_end,
                embedding, tombstoned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                session_id = excluded.session_id,
                parent_msg_id = excluded.parent_msg_id,
                slot_id = excluded.slot_id,
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
                slot_id,
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
        fact_text: str = "",
        source_tier: str = "user_direct",
        seq_all_draft: bool = False,
    ) -> None:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT chunk_id, display_text, bm25_tokens
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

        target_cids = {str(r["chunk_id"]) for r in target_rows}
        resolved_tier = "user_draft_fiction" if seq_all_draft else "user_direct"
        extra_bm25 = build_bm25_tokens(fact_text, subject=subject, tag=tag)
        for r in rows:
            cid = str(r["chunk_id"])
            if cid in target_cids:
                existing_bm25 = str(r["bm25_tokens"] or "").strip()
                bm25 = f"{existing_bm25} {extra_bm25}".strip()
                conn.execute(
                    """
                    UPDATE search_chunks
                    SET source_tier = ?, bm25_tokens = ?
                    WHERE chunk_id = ?
                    """,
                    (resolved_tier, bm25, cid),
                )
            else:
                conn.execute(
                    """
                    UPDATE search_chunks
                    SET source_tier = ?
                    WHERE chunk_id = ?
                    """,
                    (resolved_tier, cid),
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
        conn.execute(
            "UPDATE parent_messages SET turn_pair_id = '' "
            "WHERE user_id = ? AND seq IN (?, ?)",
            (user_id, seq - 1, seq),
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
        op: str = "set",
        target_id: str | None = None,
        fact_chunk_id: str | None = None,
        source_seqs: Sequence[int] | None = None,
        slot_id: str | None = None,
    ) -> UpsertTimelineResult:
        def _norm_st(s: str) -> str:
            return " ".join(s.strip().split()).casefold()

        conn = self._connect()
        resolved_subject = subject
        base_tag = get_base_tag(tag)
        resolved_tag = base_tag
        row = None
        norm_curr = _norm_st(current_state)

        lookup_sid = slot_id or target_id
        if lookup_sid:
            clean_sid = str(lookup_sid).strip().strip("[]#")
            row = conn.execute(
                "SELECT slot_id, short_id, subject, tag, history_chain, superseded_seqs, "
                "source_seqs, current_state, tombstoned FROM tag_timelines "
                "WHERE user_id = ? AND (slot_id = ? OR short_id = ?) AND tombstoned = 0",
                (user_id, clean_sid, clean_sid),
            ).fetchone()
            if row is not None:
                resolved_subject = str(row["subject"])
                resolved_tag = get_base_tag(str(row["tag"]))

        if row is None and not slot_id:
            sibling_rows = conn.execute(
                "SELECT slot_id, short_id, subject, tag, history_chain, superseded_seqs, "
                "source_seqs, current_state, tombstoned FROM tag_timelines "
                "WHERE user_id = ? AND subject = ? "
                "AND (tag = ? OR tag LIKE ? || '#F%') "
                "ORDER BY latest_seq ASC",
                (user_id, resolved_subject, base_tag, base_tag),
            ).fetchall()
            active_siblings = [r for r in sibling_rows if int(r["tombstoned"]) == 0]

            exact_match = next(
                (
                    r
                    for r in active_siblings
                    if _norm_st(str(r["current_state"] or "")) == norm_curr
                ),
                None,
            )
            if exact_match is not None:
                row = exact_match
            elif op == "add" and not (
                resolved_subject.casefold() != "user"
                and active_siblings
                and base_tag
                in (
                    "社交/朋友与同事",
                    "社交/家庭与伴侣",
                    "居住/地点",
                    "职业/工作与头衔",
                )
            ):
                base_active = next(
                    (r for r in active_siblings if str(r["tag"]) == base_tag), None
                )
                if base_active is None:
                    row = next(
                        (r for r in sibling_rows if str(r["tag"]) == base_tag), None
                    )
                    resolved_tag = base_tag
                else:
                    row = None
            else:
                if not active_siblings:
                    row = next(
                        (r for r in sibling_rows if str(r["tag"]) == base_tag), None
                    )
                    resolved_tag = base_tag
                else:
                    best_row = None
                    if prev_state:
                        p_low = prev_state.casefold()
                        for r in active_siblings:
                            c_low = str(r["current_state"] or "").casefold()
                            if c_low and (p_low in c_low or c_low in p_low):
                                best_row = r
                                break
                    q_text_full = f"{prev_state} {current_state} {fact_text}"
                    q_ent_toks = _extract_entity_tokens(q_text_full, base_tag)
                    q_all_toks = set(tokenize_bm25(q_text_full))
                    q_names = _extract_named_identifiers(q_text_full)
                    sib_ent_tok_sets = [
                        _extract_entity_tokens(str(r["current_state"] or ""), base_tag)
                        for r in active_siblings
                    ]
                    common_sib_toks = (
                        set.intersection(*sib_ent_tok_sets)
                        if len(sib_ent_tok_sets) >= 2 and all(sib_ent_tok_sets)
                        else set()
                    )
                    q_disc_toks = q_ent_toks - common_sib_toks

                    scored_candidates: list[tuple[float, int, sqlite3.Row]] = []
                    for idx_sib, r in enumerate(active_siblings):
                        r_text = str(r["current_state"] or "")
                        r_ent_toks = sib_ent_tok_sets[idx_sib]
                        r_names = _extract_named_identifiers(r_text)
                        if q_names and r_names:
                            ratio = 1.0 if (q_names & r_names) else 0.0
                        elif not q_names and r_names and any(
                            nm in q_text_full.casefold() for nm in r_names
                        ):
                            ratio = 1.0
                        elif q_disc_toks and r_ent_toks:
                            r_disc_toks = r_ent_toks - common_sib_toks
                            if not q_names and r_names:
                                name_chars = {ch for nm in r_names for ch in nm}
                                stripped_disc = {
                                    t
                                    for t in r_disc_toks
                                    if not any(ch in t for ch in name_chars)
                                }
                                if stripped_disc:
                                    r_disc_toks = stripped_disc
                            if r_disc_toks:
                                ov_ent = len(q_disc_toks & r_disc_toks)
                                ratio = ov_ent / max(1, min(len(q_disc_toks), len(r_disc_toks)))
                            else:
                                ratio = 0.0
                        else:
                            r_all_toks = set(tokenize_bm25(r_text))
                            ov_all = len(q_all_toks & r_all_toks)
                            ratio = (
                                0.0
                                if is_naturally_multivalued(base_tag)
                                else ov_all / max(1, min(len(q_all_toks), len(r_all_toks)))
                            )
                        scored_candidates.append((ratio, -idx_sib, r))
                    scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    best_ov_ratio = scored_candidates[0][0] if scored_candidates else 0.0
                    second_ov_ratio = (
                        scored_candidates[1][0] if len(scored_candidates) >= 2 else 0.0
                    )
                    if best_row is None and scored_candidates and best_ov_ratio > 0.0:
                        best_row = scored_candidates[0][2]
                    is_ambiguous = (
                        len(scored_candidates) >= 2
                        and (best_ov_ratio - second_ov_ratio) < 0.20
                    )
                    if (
                        not prev_state
                        and is_naturally_multivalued(base_tag)
                        and (best_ov_ratio < 0.45 or is_ambiguous)
                    ):
                        row = None
                    else:
                        if best_row is not None:
                            row = best_row
                        elif active_siblings and not is_naturally_multivalued(base_tag):
                            row = active_siblings[-1]
                        else:
                            row = None

        if row is not None:
            resolved_slot_id = str(row["slot_id"] or row["short_id"])
        elif lookup_sid:
            resolved_slot_id = str(lookup_sid).strip().strip("[]#")
        else:
            resolved_slot_id = self._allocate_next_slot_id(user_id)

        short_id = resolved_slot_id

        chain: list[dict[str, Any]] = []
        old_source_seqs: list[int] = []
        old_sup_seqs: list[int] = []
        if row and int(row["tombstoned"]) == 0:
            if row["history_chain"]:
                with contextlib.suppress(Exception):
                    loaded = json.loads(str(row["history_chain"]))
                    if isinstance(loaded, list):
                        chain = loaded
            if row["source_seqs"]:
                with contextlib.suppress(Exception):
                    loaded_src = json.loads(str(row["source_seqs"]))
                    if isinstance(loaded_src, list):
                        old_source_seqs = [int(x) for x in loaded_src if x is not None]
            if row["superseded_seqs"]:
                with contextlib.suppress(Exception):
                    loaded_sup = json.loads(str(row["superseded_seqs"]))
                    if isinstance(loaded_sup, list):
                        old_sup_seqs = [int(x) for x in loaded_sup if x is not None]

        is_idempotent_repeat = bool(
            row is not None
            and int(row["tombstoned"]) == 0
            and chain
            and _norm_st(str(row["current_state"] or "")) == norm_curr
        )

        entry_src_seqs = sorted(
            {int(s) for s in (source_seqs or [seq]) if s is not None} | {seq}
        )
        entry: dict[str, Any] = {
            "seq": seq,
            "session_ts": session_ts,
            "time_str": time_str,
            "state": current_state,
            "prev_state": prev_state,
            "reason": update_reason,
            "fact_text": fact_text,
            "source_seqs": entry_src_seqs,
        }
        replaced = False
        for idx, existing in enumerate(chain):
            if int(existing.get("seq", -1)) == seq:
                prev_n_seqs = [
                    int(s)
                    for s in (
                        existing.get("source_seqs")
                        or ([existing.get("seq")] if existing.get("seq") is not None else [])
                    )
                    if s is not None
                ]
                entry["source_seqs"] = sorted(set(prev_n_seqs) | set(entry_src_seqs))
                chain[idx] = entry
                replaced = True
                break
        if not replaced and not is_idempotent_repeat:
            chain.append(entry)

        def _sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
            s_ts = item.get("session_ts")
            s_seq = int(item.get("seq") or 0)
            if s_ts is not None:
                return (0, int(s_ts), s_seq)
            return (1, s_seq, s_seq)

        chain.sort(key=_sort_key)
        latest = chain[-1]
        if is_idempotent_repeat:
            prev_l_seqs = [
                int(s)
                for s in (
                    latest.get("source_seqs")
                    or ([latest.get("seq")] if latest.get("seq") is not None else [])
                )
                if s is not None
            ]
            latest["source_seqs"] = sorted(set(prev_l_seqs) | set(old_source_seqs) | {seq})
        elif not latest.get("source_seqs"):
            latest["source_seqs"] = [int(latest.get("seq") or seq)]

        latest_seq = int(latest.get("seq") or seq)
        latest_ts = (
            int(latest["session_ts"]) if latest.get("session_ts") is not None else None
        )
        resolved_current_state = str(latest.get("state") or current_state)

        chain_seqs: list[int] = []
        for item in chain:
            for s in item.get("source_seqs") or (
                [item.get("seq")] if item.get("seq") is not None else []
            ):
                if s is not None:
                    chain_seqs.append(int(s))
        all_known_seqs = sorted(
            set(old_sup_seqs) | set(old_source_seqs) | set(chain_seqs) | {seq}
        )
        is_generic_bucket = (
            resolved_tag in ("日常/事实", "日常/生活", "常规/事实") and not prev_state
        )
        if is_generic_bucket:
            sup_seqs = []
            src_seqs = all_known_seqs or [latest_seq]
        elif is_idempotent_repeat or latest_seq == seq:
            src_seqs = sorted(set(latest["source_seqs"]))
            sup_seqs = sorted(s for s in all_known_seqs if s not in src_seqs)
        else:
            latest["source_seqs"] = sorted(
                set(latest["source_seqs"]) | set(old_source_seqs) | {latest_seq}
            )
            src_seqs = latest["source_seqs"]
            sup_seqs = sorted(s for s in all_known_seqs if s not in src_seqs)

        history_parts: list[str] = []
        prev_norm_rendered: str | None = None
        for idx, item in enumerate(chain):
            st = str(item.get("state") or "")
            cur_norm = _norm_st(st)
            if prev_norm_rendered is not None and cur_norm == prev_norm_rendered:
                continue
            prev_norm_rendered = cur_norm
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

        display_tag = get_base_tag(resolved_tag)
        chain_str = " -> ".join(history_parts)
        rendered_summary = (
            f"【状态时间线 | 主体: {resolved_subject} | 标签: {display_tag}】"
            f"当前最新状态: {resolved_current_state} | 历史演化: {chain_str}"
        )
        history_json = json.dumps(chain, ensure_ascii=False)
        sup_json = json.dumps(sup_seqs, ensure_ascii=False)
        src_json = json.dumps(src_seqs, ensure_ascii=False)

        now_ts = int(datetime.now(tz=UTC).timestamp())
        conn.execute(
            """
            INSERT INTO tag_timelines (
                user_id, slot_id, short_id, subject, tag, latest_seq, latest_session_ts,
                current_state, history_chain, rendered_summary,
                superseded_seqs, source_seqs, tombstoned, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(user_id, slot_id) DO UPDATE SET
                short_id = excluded.short_id,
                subject = excluded.subject,
                tag = excluded.tag,
                latest_seq = excluded.latest_seq,
                latest_session_ts = excluded.latest_session_ts,
                current_state = excluded.current_state,
                history_chain = excluded.history_chain,
                rendered_summary = excluded.rendered_summary,
                superseded_seqs = excluded.superseded_seqs,
                source_seqs = excluded.source_seqs,
                tombstoned = 0,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                resolved_slot_id,
                short_id,
                resolved_subject,
                resolved_tag,
                latest_seq,
                latest_ts,
                resolved_current_state,
                history_json,
                rendered_summary,
                sup_json,
                src_json,
                now_ts,
            ),
        )
        self._maybe_commit()
        is_fact_tombstoned = seq in sup_seqs
        return UpsertTimelineResult(
            rendered_summary,
            latest_seq,
            latest_ts,
            resolved_tag,
            is_fact_tombstoned,
            resolved_slot_id,
        )

    def merge_timeline_slots(
        self,
        user_id: str,
        session_id: str,
        merges: Sequence[dict[str, Any]],
    ) -> int:
        if not merges:
            return 0
        conn = self._connect()
        merged_count = 0

        def _norm_st(s: str) -> str:
            return " ".join(s.strip().split()).casefold()

        def _sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
            s_ts = item.get("session_ts")
            s_seq = int(item.get("seq") or 0)
            if s_ts is not None:
                return (0, int(s_ts), s_seq)
            return (1, s_seq, s_seq)

        redirect_map: dict[str, str] = {}
        for m in merges:
            older_id = str(
                m.get("older_fid")
                or m.get("drop_id")
                or m.get("older_id")
                or ""
            ).strip().strip("[]#")
            newer_id = str(
                m.get("newer_fid")
                or m.get("keep_id")
                or m.get("newer_id")
                or ""
            ).strip().strip("[]#")
            latest_state = str(
                m.get("latest_state")
                or m.get("updated_state")
                or m.get("merged_state")
                or ""
            ).strip()

            ls_low = latest_state.casefold()
            if (
                ls_low.startswith(("no merge", "none"))
                or "no merge required" in ls_low
                or "unrelated facts" in ls_low
                or "无需合并" in ls_low
            ):
                continue

            while older_id in redirect_map:
                older_id = redirect_map[older_id]
            while newer_id in redirect_map:
                newer_id = redirect_map[newer_id]

            if not older_id or not newer_id or older_id == newer_id:
                continue

            older_row = conn.execute(
                "SELECT * FROM tag_timelines WHERE user_id = ? "
                "AND (slot_id = ? OR short_id = ?) AND tombstoned = 0",
                (user_id, older_id, older_id),
            ).fetchone()
            newer_row = conn.execute(
                "SELECT * FROM tag_timelines WHERE user_id = ? "
                "AND (slot_id = ? OR short_id = ?) AND tombstoned = 0",
                (user_id, newer_id, newer_id),
            ).fetchone()

            if older_row is None or newer_row is None:
                continue

            # BLOCKING-2 同人校验：严禁跨人物合并状态槽位
            older_subj = str(older_row["subject"]).strip().casefold()
            newer_subj = str(newer_row["subject"]).strip().casefold()
            if older_subj != newer_subj:
                continue

            explicit_surviving_id = str(m.get("newer_fid") or m.get("to") or "").strip()
            explicit_tombstone_id = str(m.get("older_fid") or m.get("from") or "").strip()

            if int(older_row["latest_seq"]) > int(newer_row["latest_seq"]):
                older_row, newer_row = newer_row, older_row
                older_id, newer_id = newer_id, older_id

            if explicit_surviving_id and explicit_tombstone_id:
                surviving_slot_id = explicit_surviving_id
                tombstone_slot_id = explicit_tombstone_id
            else:
                surviving_slot_id = str(newer_row["slot_id"] or newer_row["short_id"])
                tombstone_slot_id = str(older_row["slot_id"] or older_row["short_id"])

            chain_by_seq: dict[int, dict[str, Any]] = {}
            for r in (older_row, newer_row):
                if r["history_chain"]:
                    with contextlib.suppress(Exception):
                        loaded = json.loads(str(r["history_chain"]))
                        if isinstance(loaded, list):
                            for item in loaded:
                                if isinstance(item, dict) and item.get("seq") is not None:
                                    chain_by_seq[int(item["seq"])] = dict(item)
            chain = sorted(chain_by_seq.values(), key=_sort_key)
            if not chain:
                continue

            latest = chain[-1]
            latest_seq = int(latest.get("seq") or newer_row["latest_seq"])
            latest_ts = (
                int(latest["session_ts"])
                if latest.get("session_ts") is not None
                else None
            )

            if latest_state:
                latest["state"] = latest_state
                resolved_current_state = latest_state
            else:
                resolved_current_state = str(latest.get("state") or newer_row["current_state"])

            all_known_seqs: set[int] = set()
            for r in (older_row, newer_row):
                for col in ("superseded_seqs", "source_seqs"):
                    if r[col]:
                        with contextlib.suppress(Exception):
                            arr = json.loads(str(r[col]))
                            if isinstance(arr, list):
                                all_known_seqs.update(int(x) for x in arr if x is not None)
            for item in chain:
                src_list = item.get("source_seqs")
                if not src_list and item.get("seq") is not None:
                    src_list = [item.get("seq")]
                for s in src_list or []:
                    if s is not None:
                        all_known_seqs.add(int(s))

            older_src: list[int] = []
            if older_row["source_seqs"]:
                with contextlib.suppress(Exception):
                    loaded_osrc = json.loads(str(older_row["source_seqs"]))
                    if isinstance(loaded_osrc, list):
                        older_src = [int(x) for x in loaded_osrc if x is not None]

            src_seqs = sorted(
                set(
                    int(s)
                    for s in (latest.get("source_seqs") or [latest_seq])
                    if s is not None
                )
            )
            sup_seqs = sorted(set(s for s in all_known_seqs if s not in src_seqs) | set(older_src))

            history_parts: list[str] = []
            prev_norm_rendered: str | None = None
            for idx, item in enumerate(chain):
                st = str(item.get("state") or "")
                cur_norm = _norm_st(st)
                if prev_norm_rendered is not None and cur_norm == prev_norm_rendered:
                    continue
                prev_norm_rendered = cur_norm
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

            surviving_subj = str(newer_row["subject"])
            display_tag = get_base_tag(str(newer_row["tag"]))
            chain_str = " -> ".join(history_parts)
            rendered_summary = (
                f"【状态时间线 | 主体: {surviving_subj} | 标签: {display_tag}】"
                f"当前最新状态: {resolved_current_state} | 历史演化: {chain_str}"
            )

            conn.execute(
                """
                UPDATE tag_timelines SET
                    latest_seq = ?, latest_session_ts = ?,
                    current_state = ?, history_chain = ?, rendered_summary = ?,
                    superseded_seqs = ?, source_seqs = ?, tombstoned = 0
                WHERE user_id = ? AND slot_id = ?
                """,
                (
                    latest_seq,
                    latest_ts,
                    resolved_current_state,
                    json.dumps(chain, ensure_ascii=False),
                    rendered_summary,
                    json.dumps(sup_seqs, ensure_ascii=False),
                    json.dumps(src_seqs, ensure_ascii=False),
                    user_id,
                    surviving_slot_id,
                ),
            )

            conn.execute(
                "UPDATE tag_timelines SET tombstoned = 1 WHERE user_id = ? AND slot_id = ?",
                (user_id, tombstone_slot_id),
            )

            # INV-1 铁律：只 tombstone 旧状态卡，绝不 tombstone fact_item！
            conn.execute(
                """
                UPDATE search_chunks SET tombstoned = 1
                WHERE user_id = ? AND kind = 'timeline_summary'
                AND (slot_id = ? OR chunk_id IN (?, ?))
                """,
                (
                    user_id,
                    tombstone_slot_id,
                    f"tl:{user_id}:{tombstone_slot_id}",
                    f"tl:{user_id}:{older_row['subject']}:{older_row['tag']}",
                ),
            )

            tl_search_text = f"{display_tag} {resolved_current_state}\n{rendered_summary}".strip()
            tl_bm25 = build_bm25_tokens(tl_search_text, subject=surviving_subj, tag=display_tag)
            conn.execute(
                """
                UPDATE search_chunks SET
                    display_text = ?, search_text = ?, bm25_tokens = ?, tombstoned = 0
                WHERE user_id = ? AND kind = 'timeline_summary'
                AND (slot_id = ? OR chunk_id IN (?, ?))
                """,
                (
                    rendered_summary,
                    tl_search_text,
                    tl_bm25,
                    user_id,
                    surviving_slot_id,
                    f"tl:{user_id}:{surviving_slot_id}",
                    f"tl:{user_id}:{surviving_subj}:{newer_row['tag']}",
                ),
            )

            if surviving_subj.casefold() != "user":
                fact_display_text = (
                    f"【非User本人，系{surviving_subj}的信息】{resolved_current_state}"
                )
            else:
                fact_display_text = resolved_current_state

            # 查询存活槽位原有的 fact_item 的 display_text，提取并保留 [隐式推导 / Implied ...]
            slots_to_sync = {surviving_slot_id}
            if explicit_surviving_id:
                slots_to_sync.add(explicit_surviving_id)

            for s_slot in slots_to_sync:
                old_chunk = conn.execute(
                    """
                    SELECT display_text FROM search_chunks
                    WHERE user_id = ? AND kind = 'fact_item'
                    AND (slot_id = ? OR chunk_id = ?)
                    """,
                    (user_id, s_slot, f"fact:{user_id}:{s_slot}"),
                ).fetchone()
                s_fact_display = fact_display_text
                if old_chunk and old_chunk["display_text"]:
                    old_disp = str(old_chunk["display_text"])
                    m_implied = re.search(r"(\[隐式推导 / Implied[^\]]*\])", old_disp)
                    if m_implied:
                        implied_tag = m_implied.group(1)
                        if implied_tag not in s_fact_display:
                            s_fact_display = f"{s_fact_display} {implied_tag}"

                s_fact_search = f"[{surviving_subj} | {newer_row['tag']}] {s_fact_display}"
                s_fact_bm25 = build_bm25_tokens(
                    s_fact_display, subject=surviving_subj, tag=str(newer_row["tag"])
                )
                conn.execute(
                    """
                    UPDATE search_chunks SET
                        display_text = ?, search_text = ?, bm25_tokens = ?
                    WHERE user_id = ? AND kind = 'fact_item'
                    AND (slot_id = ? OR chunk_id = ?) AND tombstoned = 0
                    """,
                    (
                        s_fact_display,
                        s_fact_search,
                        s_fact_bm25,
                        user_id,
                        s_slot,
                        f"fact:{user_id}:{s_slot}",
                    ),
                )
            redirect_map[tombstone_slot_id] = surviving_slot_id
            merged_count += 1

        self._maybe_commit()
        return merged_count

    def apply_person_map_updates(
        self,
        user_id: str,
        session_id: str,
        merge_persons: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        move_slots: Sequence[dict[str, Any]] | None = None,
        profile_updates: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        if isinstance(merge_persons, dict):
            payload = merge_persons
            profile_updates = payload.get("persons") or payload.get("profile_updates")
            move_slots = payload.get("move_slots")
            merge_persons = payload.get("merge_persons")

        conn = self._connect()
        results = {"merged_persons": 0, "moved_slots": 0, "profiles_updated": 0}

        for pu in (profile_updates or []):
            p_name = str(pu.get("person") or pu.get("name") or "").strip()
            if p_name:
                self.upsert_person_profile(
                    user_id,
                    p_name,
                    aliases=pu.get("aliases"),
                    relations=pu.get("relations"),
                )
                results["profiles_updated"] += 1

        _honorifics = {
            "lt", "col", "dr", "mr", "mrs", "ms", "prof",
            "my", "the", "a", "an", "officer", "sqn", "ldr",
        }
        for mv in (move_slots or []):
            fid = str(mv.get("fid") or mv.get("slot_id") or "").strip()
            to_person = str(mv.get("to") or "").strip()
            if not fid or not to_person:
                continue
            tl_pre = conn.execute(
                "SELECT slot_id, current_state, rendered_summary, latest_seq "
                "FROM tag_timelines WHERE user_id = ? AND (slot_id = ? OR short_id = ?) "
                "AND tombstoned = 0",
                (user_id, fid, fid),
            ).fetchone()
            if to_person.casefold() != "user" and tl_pre:
                name_toks = {
                    t.casefold()
                    for t in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", to_person)
                    if (
                        (any("\u4e00" <= ch <= "\u9fff" for ch in t) and len(t) >= 2)
                        or (len(t) >= 3 and t.casefold() not in _honorifics)
                    )
                }
                if name_toks:
                    ev_text = (
                        f"{tl_pre['current_state'] or ''} {tl_pre['rendered_summary'] or ''}"
                    ).casefold()
                    l_seq = tl_pre["latest_seq"]
                    if l_seq is not None:
                        p_row = conn.execute(
                            "SELECT content FROM parent_messages WHERE user_id = ? AND seq = ?",
                            (user_id, int(l_seq)),
                        ).fetchone()
                        if p_row and p_row["content"]:
                            ev_text += " " + str(p_row["content"]).casefold()
                    if not any(nt in ev_text for nt in name_toks):
                        continue
            conn.execute(
                "UPDATE tag_timelines SET subject = ? "
                "WHERE user_id = ? AND (slot_id = ? OR short_id = ?)",
                (to_person, user_id, fid, fid),
            )
            tier = "user_direct" if to_person.lower() == "user" else "other_person"
            conn.execute(
                "UPDATE search_chunks SET subject = ?, source_tier = ? "
                "WHERE user_id = ? AND (slot_id = ? OR chunk_id IN (?, ?))",
                (to_person, tier, user_id, fid, f"fact:{user_id}:{fid}", f"tl:{user_id}:{fid}"),
            )
            if to_person.lower() != "user":
                self.upsert_person_profile(
                    user_id,
                    to_person,
                    aliases=mv.get("aliases"),
                    relations=mv.get("relations"),
                )

            # BLOCKING-2: move_slots 同步刷新 rendered_summary 与 search_chunks
            tl_row = conn.execute(
                "SELECT slot_id, current_state, rendered_summary, tag "
                "FROM tag_timelines WHERE user_id = ? AND (slot_id = ? OR short_id = ?) "
                "AND tombstoned = 0",
                (user_id, fid, fid),
            ).fetchone()
            if tl_row:
                s_id = str(tl_row["slot_id"])
                d_tag = get_base_tag(str(tl_row["tag"]))
                c_st = str(tl_row["current_state"] or "")
                old_sum = str(tl_row["rendered_summary"] or "")
                hist_suffix = (
                    (" | 历史演化:" + old_sum.split("| 历史演化:", 1)[1])
                    if "| 历史演化:" in old_sum
                    else ""
                )
                new_sum = (
                    f"【状态时间线 | 主体: {to_person} | 标签: {d_tag}】"
                    f"当前最新状态: {c_st}{hist_suffix}"
                )
                conn.execute(
                    "UPDATE tag_timelines SET rendered_summary = ? "
                    "WHERE user_id = ? AND slot_id = ?",
                    (new_sum, user_id, s_id),
                )
                prof = self.get_person_profiles(user_id).get(to_person, {})
                aliases = list(prof.get("aliases") or [])
                relations = list(prof.get("relations") or [])
                extra_terms = " ".join([*aliases, *relations]).strip()
                tl_st = f"{d_tag} {c_st}\n{new_sum}"
                if extra_terms:
                    tl_st = f"{tl_st}\n{extra_terms}"
                tl_st = tl_st.strip()
                tl_bm25 = build_bm25_tokens(tl_st, subject=to_person, tag=d_tag)
                conn.execute(
                    "UPDATE search_chunks SET display_text = ?, search_text = ?, "
                    "bm25_tokens = ? WHERE user_id = ? AND kind = 'timeline_summary' "
                    "AND (slot_id = ? OR chunk_id IN (?, ?))",
                    (
                        new_sum,
                        tl_st,
                        tl_bm25,
                        user_id,
                        s_id,
                        f"tl:{user_id}:{s_id}",
                        f"tl:{user_id}:{fid}",
                    ),
                )
            results["moved_slots"] += 1

        generic_terms = {
            "a friend", "my friend", "friend", "a colleague", "my colleague",
            "colleague", "someone", "a buddy", "some friend", "friends", "colleagues"
        }
        for mp in (merge_persons or []):
            p_from = str(mp.get("from") or "").strip()
            p_to = str(mp.get("to") or "").strip()
            if not p_to:
                continue

            raw_aliases = mp.get("aliases") or []
            source_names = [
                n.strip()
                for n in [p_from, *raw_aliases]
                if n and str(n).strip()
                and str(n).strip().casefold() != p_to.casefold()
                and str(n).strip().casefold() != "user"
            ]
            source_names = list(dict.fromkeys(source_names))
            if not source_names:
                continue

            tier = "user_direct" if p_to.casefold() == "user" else "other_person"

            for src_name in source_names:
                if src_name.casefold() == "user":
                    continue
                # NON-BLOCKING-1: 禁止把具名人物并入匿名泛称（如 Alice -> a friend）
                if src_name.lower() not in generic_terms and p_to.lower() in generic_terms:
                    continue

                if src_name.lower() in generic_terms:
                    all_profs = self.get_person_profiles(user_id)
                    target_relation = "friend" if "friend" in src_name.lower() else "colleague"
                    all_named_persons = {
                        str(r["subject"])
                        for r in conn.execute(
                            "SELECT DISTINCT subject FROM tag_timelines "
                            "WHERE user_id = ? AND tombstoned = 0",
                            (user_id,),
                        ).fetchall()
                        if str(r["subject"]).lower() not in generic_terms
                        and str(r["subject"]).lower() != "user"
                    } | {
                        k for k in all_profs
                        if k.lower() not in generic_terms and k.lower() != "user"
                    }
                    matching_candidates = 0
                    for name in all_named_persons:
                        p_data = all_profs.get(name, {})
                        rels = [r.lower() for r in p_data.get("relations", [])]
                        has_match = not rels or target_relation in rels or any(
                            target_relation in r for r in rels
                        )
                        if has_match:
                            matching_candidates += 1

                    if matching_candidates >= 2:
                        from_states = " ".join(
                            str(r["current_state"])
                            for r in conn.execute(
                                "SELECT current_state FROM tag_timelines "
                                "WHERE user_id = ? AND subject = ?",
                                (user_id, src_name),
                            ).fetchall()
                        )
                        to_states = " ".join(
                            str(r["current_state"])
                            for r in conn.execute(
                                "SELECT current_state FROM tag_timelines "
                                "WHERE user_id = ? AND subject = ?",
                                (user_id, p_to),
                            ).fetchall()
                        )
                        from_toks = set(tokenize_bm25(from_states)) - _GENERIC_ENTITY_STOP_TOKENS
                        to_toks = set(tokenize_bm25(to_states)) - _GENERIC_ENTITY_STOP_TOKENS
                        if not (from_toks & to_toks):
                            continue

                conn.execute(
                    "UPDATE tag_timelines SET subject = ? WHERE user_id = ? AND subject = ?",
                    (p_to, user_id, src_name),
                )
                conn.execute(
                    "UPDATE search_chunks SET subject = ?, source_tier = ? "
                    "WHERE user_id = ? AND subject = ? "
                    "AND kind IN ('timeline_summary', 'fact_item')",
                    (p_to, tier, user_id, src_name),
                )

                if p_to.casefold() == "user":
                    pm_rows = conn.execute(
                        "SELECT DISTINCT parent_msg_id FROM search_chunks "
                        "WHERE user_id = ? AND subject = ? AND parent_msg_id IS NOT NULL",
                        (user_id, p_to),
                    ).fetchall()
                    for pmr in pm_rows:
                        pmid = pmr["parent_msg_id"]
                        conn.execute(
                            "UPDATE search_chunks SET subject = 'user', "
                            "source_tier = 'user_direct' "
                            "WHERE user_id = ? AND parent_msg_id = ? "
                            "AND source_tier = 'other_person'",
                            (user_id, pmid),
                        )

                    prefix_pat = re.compile(rf"【非User本人，系{re.escape(src_name)}的信息】")
                    sc_rows = conn.execute(
                        "SELECT chunk_id, display_text, search_text "
                        "FROM search_chunks "
                        "WHERE user_id = ? AND display_text LIKE '%【非User本人%'",
                        (user_id,),
                    ).fetchall()
                    for scr in sc_rows:
                        disp = str(scr["display_text"] or "")
                        srch = str(scr["search_text"] or "")
                        new_disp = prefix_pat.sub("", disp)
                        new_srch = prefix_pat.sub("", srch)
                        if new_disp != disp or new_srch != srch:
                            conn.execute(
                                "UPDATE search_chunks SET display_text = ?, search_text = ? "
                                "WHERE user_id = ? AND chunk_id = ?",
                                (new_disp, new_srch, user_id, scr["chunk_id"]),
                            )

                    pm_notes_rows = conn.execute(
                        "SELECT msg_id, fact_notes FROM parent_messages "
                        "WHERE user_id = ? AND fact_notes LIKE '%【非User本人%'",
                        (user_id,),
                    ).fetchall()
                    for pmn in pm_notes_rows:
                        fnotes = str(pmn["fact_notes"] or "")
                        new_fnotes = prefix_pat.sub("", fnotes)
                        if new_fnotes != fnotes:
                            conn.execute(
                                "UPDATE parent_messages SET fact_notes = ? "
                                "WHERE user_id = ? AND msg_id = ?",
                                (new_fnotes, user_id, pmn["msg_id"]),
                            )
                else:
                    prefix_pat = re.compile(rf"【非User本人，系{re.escape(src_name)}的信息】")
                    replacement_prefix = f"【非User本人，系{p_to}的信息】"
                    sc_rows = conn.execute(
                        "SELECT chunk_id, display_text, search_text "
                        "FROM search_chunks "
                        "WHERE user_id = ? AND display_text LIKE '%【非User本人%'",
                        (user_id,),
                    ).fetchall()
                    for scr in sc_rows:
                        disp = str(scr["display_text"] or "")
                        srch = str(scr["search_text"] or "")
                        new_disp = prefix_pat.sub(replacement_prefix, disp)
                        new_srch = prefix_pat.sub(replacement_prefix, srch)
                        if new_disp != disp or new_srch != srch:
                            conn.execute(
                                "UPDATE search_chunks SET display_text = ?, search_text = ? "
                                "WHERE user_id = ? AND chunk_id = ?",
                                (new_disp, new_srch, user_id, scr["chunk_id"]),
                            )

                    pm_notes_rows = conn.execute(
                        "SELECT msg_id, fact_notes FROM parent_messages "
                        "WHERE user_id = ? AND fact_notes LIKE '%【非User本人%'",
                        (user_id,),
                    ).fetchall()
                    for pmn in pm_notes_rows:
                        fnotes = str(pmn["fact_notes"] or "")
                        new_fnotes = prefix_pat.sub(replacement_prefix, fnotes)
                        if new_fnotes != fnotes:
                            conn.execute(
                                "UPDATE parent_messages SET fact_notes = ? "
                                "WHERE user_id = ? AND msg_id = ?",
                                (new_fnotes, user_id, pmn["msg_id"]),
                            )

                aliases_to_merge = list(raw_aliases) + [src_name]
                self.upsert_person_profile(
                    user_id,
                    p_to,
                    aliases=aliases_to_merge,
                    relations=mp.get("relations"),
                )
                conn.execute(
                    "DELETE FROM person_profiles WHERE user_id = ? AND person = ?",
                    (user_id, src_name),
                )

            # BLOCKING-2: merge_persons 刷新状态卡时保留历史演化，拼入 aliases / relations
            prof = self.get_person_profiles(user_id).get(p_to, {})
            aliases = list(prof.get("aliases") or [])
            relations = list(prof.get("relations") or [])
            extra_terms = " ".join([*aliases, *relations]).strip()

            to_tl_rows = conn.execute(
                "SELECT slot_id, current_state, rendered_summary, tag "
                "FROM tag_timelines WHERE user_id = ? AND subject = ? AND tombstoned = 0",
                (user_id, p_to),
            ).fetchall()
            for tr in to_tl_rows:
                s_id = str(tr["slot_id"])
                d_tag = get_base_tag(str(tr["tag"]))
                c_st = str(tr["current_state"] or "")
                old_sum = str(tr["rendered_summary"] or "")
                hist_suffix = (
                    (" | 历史演化:" + old_sum.split("| 历史演化:", 1)[1])
                    if "| 历史演化:" in old_sum
                    else ""
                )
                new_sum = (
                    f"【状态时间线 | 主体: {p_to} | 标签: {d_tag}】"
                    f"当前最新状态: {c_st}{hist_suffix}"
                )
                conn.execute(
                    "UPDATE tag_timelines SET rendered_summary = ? "
                    "WHERE user_id = ? AND slot_id = ?",
                    (new_sum, user_id, s_id),
                )
                tl_st = f"{d_tag} {c_st}\n{new_sum}"
                if extra_terms:
                    tl_st = f"{tl_st}\n{extra_terms}"
                tl_st = tl_st.strip()
                tl_bm25 = build_bm25_tokens(tl_st, subject=p_to, tag=d_tag)
                conn.execute(
                    "UPDATE search_chunks SET display_text = ?, search_text = ?, "
                    "bm25_tokens = ? WHERE user_id = ? AND kind = 'timeline_summary' "
                    "AND slot_id = ?",
                    (new_sum, tl_st, tl_bm25, user_id, s_id),
                )

            results["merged_persons"] += 1

        self._maybe_commit()
        return results

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
        target_id: str | None = None,
    ) -> list[tuple[str, str, str, int, int | None]]:
        self._check_deadline()
        all_batch_seqs = [
            int(s)
            for s in dict.fromkeys([trigger_seq, *batch_seqs_to_tombstone])
            if s is not None and int(s) <= trigger_seq + 1
        ]
        self.tombstone_seqs(user_id, all_batch_seqs)
        self._check_deadline()

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
                if part.casefold() in lower_text:
                    continue
                if any("\u3400" <= ch <= "\u9fff" for ch in part):
                    return False
                p_toks = set(tokenize_bm25(part))
                if not p_toks or not (p_toks & c_terms):
                    return False
            return True

        _generic_cat_words = {
            "饮食", "喜好", "偏好", "食物", "口味", "diet", "food", "meal", "meals",
            "preference", "preferences", "兴趣", "日常", "习惯", "爱好", "hobby",
            "hobbies", "habit", "habits", "leisure", "工作", "职业", "技能", "work",
            "job", "career", "居住", "城市", "住址", "健康", "医疗", "禁忌", "运动",
            "财务", "预算", "资产", "宠物", "旅行", "出行", "常规", "事实", "其他",
        }

        def _is_high_spec_kw(kw: str) -> bool:
            clean = kw.strip()
            if not clean:
                return False
            if " " in clean or "-" in clean or "_" in clean:
                return True
            if any(c.isupper() for c in clean[1:]):
                return True
            cjk_count = sum(1 for c in clean if "\u4e00" <= c <= "\u9fff")
            if cjk_count >= 3:
                return True
            return len(clean) >= 8 and clean.casefold() not in _generic_cat_words

        def _lexical_forget_match(text_combined: str) -> bool:
            if not text_combined:
                return False
            lower_text = text_combined.casefold()
            if cjk_kws and any(kw.casefold() in lower_text for kw in cjk_kws):
                return True
            c_terms = set(tokenize_bm25(text_combined))
            if kw_list:
                for kw in kw_list:
                    if _is_high_spec_kw(kw) and _kw_hit(kw, lower_text, c_terms):
                        return True
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
        tag_tokens = {t.casefold() for tg_item in tag_set for t in tokenize_bm25(tg_item)}
        specific_kw_tokens = {
            t.casefold()
            for kw in kw_list
            for t in tokenize_bm25(kw)
            if t.casefold() not in _generic_cat_words and t.casefold() not in tag_tokens
        }
        has_specific_kws = bool(specific_kw_tokens) and not bool(target_id)

        conn = self._connect()
        # 1. Check tag_timelines (respecting seq <= trigger_seq causal boundary)
        tl_rows = conn.execute(
            "SELECT subject, tag, short_id, current_state, history_chain, rendered_summary "
            "FROM tag_timelines WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        tombstoned_tags: set[tuple[str, str]] = set()
        tombstoned_slots: set[str] = set()
        slot_rolled_back_surviving_seqs: dict[str, set[int]] = {}
        rolled_back_surviving_seqs: dict[tuple[str, str], set[int]] = {}
        reconstructed_tl_cids: set[str] = set()
        rebuilt_timelines: list[tuple[str, str, str, int, int | None]] = []
        for r in tl_rows:
            self._check_deadline()
            subj = str(r["subject"])
            tg = str(r["tag"])
            sid = str(r["short_id"] or "")
            base_tg = get_base_tag(tg)
            self._check_deadline()
            is_target_id_hit = bool(target_id and sid == target_id)
            is_full_tag_hit = (
                is_target_id_hit
                or (tg in tag_set)
                or (base_tg in tag_set and not target_id)
            )
            text_combined = f"{base_tg} {r['current_state']} {r['rendered_summary']}"
            matched = is_full_tag_hit or _lexical_forget_match(text_combined)
            if matched:
                chain: list[dict[str, Any]] = []
                if r["history_chain"]:
                    with contextlib.suppress(Exception):
                        loaded = json.loads(str(r["history_chain"]))
                        if isinstance(loaded, list):
                            chain = loaded
                node_filtered_chain: list[dict[str, Any]] = []
                any_node_matched = False
                for item in chain:
                    self._check_deadline()
                    raw_n_seqs = item.get("source_seqs") or (
                        [item.get("seq")] if item.get("seq") is not None else []
                    )
                    node_seqs = [int(s) for s in raw_n_seqs if s is not None]
                    item_seq = int(item.get("seq") or (node_seqs[-1] if node_seqs else 0))
                    if item_seq > trigger_seq and all(s > trigger_seq for s in node_seqs):
                        node_filtered_chain.append(item)
                        continue
                    node_text = (
                        f"{item.get('state', '')} {item.get('prev_state', '')} "
                        f"{item.get('reason', '')} {item.get('fact_text', '')}"
                    )
                    if _lexical_forget_match(node_text):
                        any_node_matched = True
                        continue
                    rem_node_seqs = [
                        s for s in node_seqs if not (s in all_batch_seqs and s <= trigger_seq)
                    ]
                    if len(rem_node_seqs) < len(node_seqs):
                        any_node_matched = True
                    if not rem_node_seqs:
                        continue
                    updated_item = dict(item)
                    updated_item["source_seqs"] = rem_node_seqs
                    updated_item["seq"] = max(rem_node_seqs)
                    node_filtered_chain.append(updated_item)

                if is_full_tag_hit and not any_node_matched:
                    if has_specific_kws:
                        continue
                    surviving_chain = [
                        item for item in chain if int(item.get("seq") or 0) > trigger_seq
                    ]
                    partial_rollback = False
                else:
                    surviving_chain = node_filtered_chain
                    partial_rollback = any(
                        int(item.get("seq") or 0) <= trigger_seq for item in surviving_chain
                    )

                if not surviving_chain:
                    if sid:
                        tombstoned_slots.add(sid)
                        conn.execute(
                            "UPDATE tag_timelines SET tombstoned = 1 "
                            "WHERE user_id = ? AND (short_id = ? OR slot_id = ?)",
                            (user_id, sid, sid),
                        )
                        conn.execute(
                            "UPDATE search_chunks SET tombstoned = 1 "
                            "WHERE user_id = ? AND (slot_id = ? OR chunk_id IN (?, ?))",
                            (user_id, sid, f"tl:{user_id}:{sid}", f"fact:{user_id}:{sid}"),
                        )
                    elif has_specific_kws:
                        conn.execute(
                            "UPDATE tag_timelines SET tombstoned = 1 "
                            "WHERE user_id = ? AND subject = ? AND tag = ?",
                            (user_id, subj, tg),
                        )
                    if not has_specific_kws:
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
                    if is_full_tag_hit and not partial_rollback and not has_specific_kws:
                        tombstoned_tags.add((subj, tg))
                    latest = surviving_chain[-1]
                    src_seqs = sorted(
                        {
                            int(s)
                            for s in (
                                latest.get("source_seqs")
                                or (
                                    [latest.get("seq")]
                                    if latest.get("seq") is not None
                                    else []
                                )
                            )
                            if s is not None
                        }
                    )
                    l_seq = (
                        max(src_seqs)
                        if src_seqs
                        else int(latest.get("seq") or trigger_seq)
                    )
                    if not src_seqs:
                        src_seqs = [l_seq]
                    l_ts = (
                        int(latest["session_ts"])
                        if latest.get("session_ts") is not None
                        else None
                    )
                    res_curr = str(latest.get("state") or "")
                    surv_seqs: list[int] = []
                    for item in surviving_chain:
                        for s in item.get("source_seqs") or (
                            [item.get("seq")] if item.get("seq") is not None else []
                        ):
                            if s is not None and int(s) not in surv_seqs:
                                surv_seqs.append(int(s))
                    if partial_rollback:
                        if sid:
                            slot_rolled_back_surviving_seqs[sid] = set(surv_seqs)
                        else:
                            rolled_back_surviving_seqs[(subj, tg)] = set(surv_seqs)
                    sup_seqs = sorted(s for s in surv_seqs if s not in src_seqs)
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
                        f"【状态时间线 | 主体: {subj} | 标签: {base_tg}】"
                        f"当前最新状态: {res_curr} | 历史演化: {c_str}"
                    )
                    if sid:
                        conn.execute(
                            "UPDATE tag_timelines SET latest_seq = ?, latest_session_ts = ?, "
                            "current_state = ?, history_chain = ?, rendered_summary = ?, "
                            "superseded_seqs = ?, source_seqs = ?, "
                            "tombstoned = 0 WHERE user_id = ? AND (short_id = ? OR slot_id = ?)",
                            (
                                l_seq,
                                l_ts,
                                res_curr,
                                json.dumps(surviving_chain, ensure_ascii=False),
                                r_sum,
                                json.dumps(sup_seqs, ensure_ascii=False),
                                json.dumps(src_seqs, ensure_ascii=False),
                                user_id,
                                sid,
                                sid,
                            ),
                        )
                    else:
                        conn.execute(
                            "UPDATE tag_timelines SET latest_seq = ?, latest_session_ts = ?, "
                            "current_state = ?, history_chain = ?, rendered_summary = ?, "
                            "superseded_seqs = ?, source_seqs = ?, "
                            "tombstoned = 0 WHERE user_id = ? AND subject = ? AND tag = ?",
                            (
                                l_seq,
                                l_ts,
                                res_curr,
                                json.dumps(surviving_chain, ensure_ascii=False),
                                r_sum,
                                json.dumps(sup_seqs, ensure_ascii=False),
                                json.dumps(src_seqs, ensure_ascii=False),
                                user_id,
                                subj,
                                tg,
                            ),
                        )
                    src_placeholders = ",".join("?" for _ in src_seqs)
                    if sid:
                        conn.execute(
                            f"UPDATE search_chunks SET tombstoned = 0 "
                            f"WHERE user_id = ? AND kind = 'fact_item' "
                            f"AND (slot_id = ? OR chunk_id = ?) AND seq IN ({src_placeholders})",
                            [user_id, sid, f"fact:{user_id}:{sid}", *src_seqs],
                        )
                        conn.execute(
                            "UPDATE search_chunks SET seq = ?, session_ts = ?, "
                            "display_text = ?, search_text = ?, bm25_tokens = ?, tombstoned = 0 "
                            "WHERE user_id = ? AND kind = 'timeline_summary' "
                            "AND (slot_id = ? OR chunk_id = ?)",
                            (
                                l_seq,
                                l_ts,
                                r_sum,
                                r_sum,
                                build_bm25_tokens(r_sum, subject=subj, tag=base_tg),
                                user_id,
                                sid,
                                f"tl:{user_id}:{sid}",
                            ),
                        )
                        reconstructed_tl_cids.add(f"tl:{user_id}:{sid}")
                    else:
                        conn.execute(
                            f"UPDATE search_chunks SET tombstoned = 0 "
                            f"WHERE user_id = ? AND kind = 'fact_item' "
                            f"AND subject = ? AND tag = ? AND seq IN ({src_placeholders})",
                            [user_id, subj, tg, *src_seqs],
                        )
                        conn.execute(
                            "UPDATE search_chunks SET seq = ?, session_ts = ?, "
                            "display_text = ?, search_text = ?, bm25_tokens = ?, tombstoned = 0 "
                            "WHERE user_id = ? AND kind = 'timeline_summary' "
                            "AND subject = ? AND tag = ?",
                            (
                                l_seq,
                                l_ts,
                                r_sum,
                                r_sum,
                                build_bm25_tokens(r_sum, subject=subj, tag=base_tg),
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
            SELECT chunk_id, parent_msg_id, slot_id, seq, kind, subject, tag,
                   display_text, search_text, embedding
            FROM search_chunks
            WHERE user_id = ? AND tombstoned = 0 AND seq <= ?
            """,
            (user_id, trigger_seq),
        ).fetchall()

        matched_chunk_ids: set[str] = set()
        affected_parent_ids: dict[str, list[str]] = {}

        for r in chunk_rows:
            self._check_deadline()
            cid = str(r["chunk_id"])
            if cid in reconstructed_tl_cids:
                continue
            pmid = str(r["parent_msg_id"]) if r["parent_msg_id"] else None
            c_slot = str(r["slot_id"] or "")
            c_seq = int(r["seq"])
            kind = str(r["kind"])
            tg = str(r["tag"]) if r["tag"] else ""
            base_tg = get_base_tag(tg)
            self._check_deadline()
            subj = str(r["subject"] or "user")
            disp = str(r["display_text"] or "")
            srch = str(r["search_text"] or "")
            combined = f"{disp}\n{srch}"

            if (
                c_slot
                and c_slot in slot_rolled_back_surviving_seqs
                and c_seq in slot_rolled_back_surviving_seqs[c_slot]
            ):
                continue
            if (
                not c_slot
                and (subj, tg) in rolled_back_surviving_seqs
                and c_seq in rolled_back_surviving_seqs[(subj, tg)]
            ):
                continue

            hit = False
            if c_slot and c_slot in tombstoned_slots:
                hit = True
            elif c_slot and c_slot in slot_rolled_back_surviving_seqs:
                if c_seq not in slot_rolled_back_surviving_seqs[c_slot]:
                    hit = True
            elif not c_slot and (subj, tg) in rolled_back_surviving_seqs:
                if c_seq not in rolled_back_surviving_seqs[(subj, tg)]:
                    hit = True
            elif (
                (subj, tg) in tombstoned_tags
                or (
                    not has_specific_kws
                    and tg
                    and (tg in tag_set or (base_tg in tag_set and not target_id))
                )
            ):
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
                if kind == "timeline_summary":
                    if c_slot:
                        conn.execute(
                            "UPDATE tag_timelines SET tombstoned = 1 "
                            "WHERE user_id = ? AND (short_id = ? OR slot_id = ?)",
                            (user_id, c_slot, c_slot),
                        )
                    elif not has_specific_kws and tg and (subj, tg) not in tombstoned_tags:
                        tombstoned_tags.add((subj, tg))
                        conn.execute(
                            "UPDATE tag_timelines SET tombstoned = 1 "
                            "WHERE user_id = ? AND subject = ? AND tag = ?",
                            (user_id, subj, tg),
                        )

        for cid in matched_chunk_ids:
            self._check_deadline()
            conn.execute(
                "UPDATE search_chunks SET tombstoned = 1 WHERE chunk_id = ?",
                (cid,),
            )

        # 3. Also check active parent_messages with seq <= trigger_seq
        parent_rows = conn.execute(
            "SELECT msg_id, seq, role, session_id, turn_pair_id, content FROM parent_messages "
            "WHERE user_id = ? AND tombstoned = 0 AND seq <= ? ORDER BY seq ASC",
            (user_id, trigger_seq),
        ).fetchall()

        tombstoned_user_seqs: set[tuple[str, int]] = set()
        tombstoned_turn_pairs: set[str] = set()
        for pr in parent_rows:
            self._check_deadline()
            pmid = str(pr["msg_id"])
            p_seq = int(pr["seq"])
            p_role = str(pr["role"] or "")
            p_sid = str(pr["session_id"] or "")
            p_tp = str(pr["turn_pair_id"] or "")
            p_content = str(pr["content"] or "")
            p_lower = p_content.casefold()
            p_terms = set(tokenize_bm25(p_content))
            direct_kw_hit = _lexical_forget_match(p_content)
            is_paired_with_tombstoned_user = p_role == "assistant" and (
                (p_sid, p_seq - 1) in tombstoned_user_seqs
                or (p_tp and p_tp in tombstoned_turn_pairs)
            )
            if is_paired_with_tombstoned_user and p_tp:
                conn.execute(
                    "UPDATE parent_messages SET turn_pair_id = '' WHERE msg_id = ?",
                    (pmid,),
                )
            paired_assistant_hit = is_paired_with_tombstoned_user and bool(
                kw_list and any(_kw_hit(kw, p_lower, p_terms) for kw in kw_list)
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
                        if p_tp:
                            tombstoned_turn_pairs.add(p_tp)
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
                            if p_tp:
                                tombstoned_turn_pairs.add(p_tp)

        self._maybe_commit()
        return rebuilt_timelines

    def sync_dirty_timelines(
        self,
        user_id: str,
        session_id: str,
        dirty_keys: Sequence[Any],
    ) -> None:
        if not dirty_keys:
            return
        conn = self._connect()
        for key_item in dict.fromkeys(dirty_keys):
            if isinstance(key_item, str):
                tl_row = conn.execute(
                    "SELECT slot_id, short_id, subject, tag, latest_seq, "
                    "latest_session_ts, current_state, rendered_summary "
                    "FROM tag_timelines "
                    "WHERE user_id = ? AND (slot_id = ? OR short_id = ?) AND tombstoned = 0",
                    (user_id, key_item, key_item),
                ).fetchone()
            elif isinstance(key_item, tuple | list) and len(key_item) == 2:
                subj, tag = key_item
                clean_tag = get_base_tag(tag)
                tl_row = conn.execute(
                    "SELECT slot_id, short_id, subject, tag, latest_seq, "
                    "latest_session_ts, current_state, rendered_summary "
                    "FROM tag_timelines "
                    "WHERE user_id = ? AND subject = ? AND (tag = ? OR tag = ?) "
                    "AND tombstoned = 0",
                    (user_id, subj, tag, clean_tag),
                ).fetchone()
            else:
                continue

            if tl_row is None:
                continue
            slot_id = str(tl_row["slot_id"] or tl_row["short_id"])
            subj = str(tl_row["subject"])
            tag = str(tl_row["tag"])
            tl_seq = int(tl_row["latest_seq"])
            tl_ts = (
                int(tl_row["latest_session_ts"])
                if tl_row["latest_session_ts"] is not None
                else None
            )
            curr_st = str(tl_row["current_state"] or "")
            rendered_summary = str(tl_row["rendered_summary"] or "")

            display_tag = get_base_tag(tag)
            fact_row = conn.execute(
                "SELECT chunk_id, embedding FROM search_chunks "
                "WHERE user_id = ? AND tombstoned = 0 "
                "AND (chunk_id = ? OR (kind = 'fact_item' AND slot_id = ?) "
                "OR (kind = 'fact_item' AND subject = ? AND (tag = ? OR tag = ?) AND seq = ?)) "
                "ORDER BY chunk_id DESC LIMIT 1",
                (user_id, f"fact:{user_id}:{slot_id}", slot_id, subj, tag, display_tag, tl_seq),
            ).fetchone()
            if fact_row is None:
                fact_row = conn.execute(
                    "SELECT chunk_id, embedding FROM search_chunks "
                    "WHERE user_id = ? AND tombstoned = 0 "
                    "AND kind = 'fact_item' AND subject = ? AND seq = ? "
                    "ORDER BY chunk_id DESC LIMIT 1",
                    (user_id, subj, tl_seq),
                ).fetchone()
            vec: list[float] = []
            if fact_row is not None:
                vec = blob_to_vector(fact_row["embedding"])
                if not vec:
                    vec = list(self._vector_cache.get(str(fact_row["chunk_id"])) or [])
            if not vec:
                vec = list(self._vector_cache.get(f"fact:{user_id}:{slot_id}") or [])
            if not vec:
                # 兼容旧式 fact chunk 命名查找
                vec = list(self._vector_cache.get(f"tl:{user_id}:{subj}:{tag}") or [])
            if fact_row is None and not vec:
                raise RuntimeError(
                    f"Fail-closed: missing active fact_item or embedding for timeline "
                    f"({user_id}, {subj}, {tag}) at seq={tl_seq}"
                )
            if not vec:
                vec = [0.0] * 1024

            tl_chunk_id = f"tl:{user_id}:{slot_id}"
            display_tag = get_base_tag(tag)
            tl_search_text = f"{display_tag} {curr_st}\n{rendered_summary}".strip()
            tl_bm25 = build_bm25_tokens(tl_search_text, subject=subj, tag=display_tag)
            tier = "other_person" if subj.casefold() != "user" else "user_direct"
            self.upsert_chunk(
                chunk_id=tl_chunk_id,
                user_id=user_id,
                session_id=session_id,
                parent_msg_id=None,
                slot_id=slot_id,
                seq=tl_seq,
                session_ts=tl_ts,
                kind="timeline_summary",
                subject=subj,
                tag=tag,
                source_tier=tier,
                display_text=rendered_summary,
                search_text=tl_search_text,
                bm25_tokens=tl_bm25,
                event_time_start=tl_ts,
                event_time_end=tl_ts,
                embedding=vec,
                tombstoned=0,
            )

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
                       content, session_ts, session_time_str, fact_notes, turn_pair_id, tombstoned
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
                   session_time_str, fact_notes, turn_pair_id, tombstoned
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

    def get_active_slot_latest_seqs(self, user_id: str) -> set[tuple[str, int]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT subject, latest_seq FROM tag_timelines WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        return {
            (str(r["subject"] or "user").strip().casefold(), int(r["latest_seq"]))
            for r in rows
        }

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

    def get_superseded_seqs_map(
        self, user_id: str
    ) -> dict[tuple[str, int], list[tuple[int, str, str]]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT subject, tag, current_state, latest_seq, superseded_seqs "
            "FROM tag_timelines WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        out: dict[tuple[str, int], list[tuple[int, str, str]]] = {}
        for r in rows:
            subj_low = str(r["subject"] or "user").strip().casefold()
            l_seq = int(r["latest_seq"])
            raw_tag = str(r["tag"] or "")
            disp_tag = get_base_tag(raw_tag)
            curr_st = str(r["current_state"] or "")
            raw_sup = str(r["superseded_seqs"] or "[]")
            with contextlib.suppress(Exception):
                sup_list = json.loads(raw_sup)
                if isinstance(sup_list, list):
                    for s in sup_list:
                        s_int = int(s)
                        if s_int != l_seq:
                            out.setdefault((subj_low, s_int), []).append(
                                (l_seq, disp_tag, curr_st)
                            )
        return out

    def get_timeline_pull_map(self, user_id: str) -> dict[str, list[int]]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT slot_id, short_id, subject, tag, latest_seq, superseded_seqs, source_seqs "
            "FROM tag_timelines WHERE user_id = ? AND tombstoned = 0",
            (user_id,),
        ).fetchall()
        out: dict[str, list[int]] = {}
        for r in rows:
            subj = str(r["subject"])
            tg = str(r["tag"])
            sid = str(r["slot_id"] or r["short_id"] or "")
            l_seq = int(r["latest_seq"])
            sup_set: set[int] = set()
            with contextlib.suppress(Exception):
                loaded_sup = json.loads(str(r["superseded_seqs"] or "[]"))
                if isinstance(loaded_sup, list):
                    sup_set = {int(x) for x in loaded_sup}
            src_list: list[int] = []
            with contextlib.suppress(Exception):
                loaded_src = json.loads(str(r["source_seqs"] or "[]"))
                if isinstance(loaded_src, list):
                    src_list = [int(x) for x in loaded_src if int(x) not in sup_set]
            if not src_list and l_seq not in sup_set:
                src_list = [l_seq]
            if sid:
                out[f"tl:{user_id}:{sid}"] = src_list
            out[f"tl:{user_id}:{subj}:{tg}"] = src_list
        return out

    def get_time_matched_seqs(self, user_id: str, start_ts: int, end_ts: int) -> set[int]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT DISTINCT sc.seq
            FROM search_chunks sc
            WHERE sc.user_id = ?
              AND sc.event_time_start IS NOT NULL
              AND sc.event_time_end IS NOT NULL
              AND sc.event_time_start <= ?
              AND sc.event_time_end >= ?
              AND (
                  sc.tombstoned = 0
                  OR (
                      sc.kind = 'fact_item'
                      AND EXISTS (
                          SELECT 1 FROM parent_messages p
                          WHERE p.msg_id = sc.parent_msg_id AND p.tombstoned = 0
                      )
                  )
              )
            """,
            (user_id, end_ts, start_ts),
        ).fetchall()
        return {int(r["seq"]) for r in rows if r["seq"] is not None}

    def get_parent_messages_by_seqs(
        self, user_id: str, seqs: Sequence[int]
    ) -> dict[int, dict[str, Any]]:
        seq_list = list(dict.fromkeys(int(s) for s in seqs if s is not None))
        if not seq_list:
            return {}
        conn = self._connect()
        placeholders = ",".join("?" for _ in seq_list)
        rows = conn.execute(
            f"""
            SELECT msg_id, user_id, session_id, request_id, seq, role, speaker,
                   content, session_ts, session_time_str, fact_notes, tombstoned
            FROM parent_messages
            WHERE user_id = ? AND seq IN ({placeholders}) AND tombstoned = 0
            """,
            [user_id, *seq_list],
        ).fetchall()
        return {int(r["seq"]): dict(r) for r in rows}

    def is_seq_draft_fiction(self, user_id: str, seq: int) -> bool:
        conn = self._connect()
        rows = conn.execute(
            "SELECT source_tier FROM search_chunks "
            "WHERE user_id = ? AND seq = ? AND kind = 'raw_slice' AND tombstoned = 0",
            (user_id, seq),
        ).fetchall()
        return bool(rows) and all(
            str(r["source_tier"] or "user_direct") == "user_draft_fiction" for r in rows
        )


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
