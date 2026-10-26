from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .models import AddRequest, WriteItem

TAG_BOOST = 2
SUBJECT_BOOST = 2
LONG_DOC_CHAR_THRESHOLD = 3000
LONG_DOC_WINDOW_CHARS = 800
LONG_DOC_OVERLAP_CHARS = 160
NORMAL_SLICE_MAX_CHARS = 300
NORMAL_SLICE_SENTENCES = 2

_BREAK_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;\n])\s*")
_SPEAKER_PREFIX_RE = re.compile(r"^([A-Z][a-zA-Z0-9_ -]{1,30}):\s+")
_HEADER_DATE_RE = re.compile(
    r"(?:^|\b)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?"
)
_ISO_DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})(?:[-/月](\d{1,2})日?)?")

_EN_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "enjoy",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "like",
    "love",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "prefer",
    "preference",
    "preferences",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "use",
    "used",
    "using",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_FORGET_PATTERNS = (
    re.compile(r"(?is)\b(?:please\s+)?forget(?:\s+about|\s+that)?\s+(.+)$"),
    re.compile(
        r"(?is)\b(?:please\s+)?(?:delete|erase|remove|wipe)\s+(?:the\s+)?"
        r"(?:memory|memories|information|record|records)?\s*(?:about|of|that)?\s+(.+)$"
    ),
    re.compile(
        r"(?is)\b(?:don'?t|do\s+not|never)\s+(?:remember|mention)\s+(?:that\s+|about\s+)?(.+)$"
    ),
    re.compile(
        r"(?s)(?:请|麻烦|帮我)?(?:忘掉|忘记|删除|抹除|擦除|抹掉|别再记得|不要记得|不要再提|别再提)"
        r"(?:关于|我说过的|那件事|这件事|的记忆|的记录|：|:)?\s*(.+)$"
    ),
)

_EXPLICIT_FORGET_ROOT_RE = re.compile(
    r"(?i)(?:\b(?:forget|delete|erase|remove|wipe)\b|"
    r"\bdon'?t\s+remember\b|\bdo\s+not\s+remember\b|\bnever\s+mention\b|"
    r"忘掉|忘记|删除|抹除|擦除|抹掉|别再记得|不要记得|别记得|不再记得|不要再提|别再提)"
)

_TRAILING_FORGET_NOISE_RE = re.compile(
    r"(?:这件事|那件事|的记忆|的记录|的信息|，?不要再提了?|，?别再提了?)+$"
)


def iso_from_millis(value: int | None) -> str | None:
    if value is None:
        return None
    ts_sec = value / 1000.0 if value > 10_000_000_000 else float(value)
    return datetime.fromtimestamp(ts_sec, tz=UTC).isoformat().replace("+00:00", "Z")


def normalize_timestamp_to_seconds(value: int | None) -> int | None:
    if value is None:
        return None
    if value > 10_000_000_000:
        return int(value // 1000)
    return int(value)


def parse_header_timestamp_seconds(content: str) -> int | None:
    head = content[:160]
    match = _HEADER_DATE_RE.search(head)
    if not match:
        return None
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        second = int(match.group(6) or 0)
        dt = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
        return int(dt.timestamp())
    except ValueError:
        return None


def format_session_time_str(session_ts: int | None) -> str:
    if session_ts is None:
        return ""
    return datetime.fromtimestamp(session_ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def extract_speaker(content: str, default_role: str) -> str:
    match = _SPEAKER_PREFIX_RE.match(content)
    if match:
        return match.group(1).strip()
    return default_role


def parse_single_date_utc(date_str: str | None, *, end_of_day: bool = False) -> int | None:
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = date_str.strip()
    if not cleaned:
        return None
    match = _ISO_DATE_RE.search(cleaned)
    if not match:
        return None
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        if match.group(3) is not None:
            day = int(match.group(3))
        else:
            day = calendar.monthrange(year, month)[1] if end_of_day else 1
        if end_of_day:
            dt = datetime(year, month, day, 23, 59, 59, tzinfo=UTC)
        else:
            dt = datetime(year, month, day, 0, 0, 0, tzinfo=UTC)
        return int(dt.timestamp())
    except ValueError:
        return None


def parse_event_date_range(
    date_start: str | None,
    date_end: str | None,
    fallback_ts: int | None = None,
) -> tuple[int | None, int | None]:
    start_ts = parse_single_date_utc(date_start, end_of_day=False)
    end_ts = parse_single_date_utc(date_end, end_of_day=True)
    if start_ts is not None and end_ts is None:
        end_ts = parse_single_date_utc(date_start, end_of_day=True)
    elif end_ts is not None and start_ts is None:
        start_ts = parse_single_date_utc(date_end, end_of_day=False)
    if start_ts is None and end_ts is None:
        return fallback_ts, fallback_ts
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


def extract_query_time_range(query: str, anchor_ts: int | None = None) -> tuple[int, int] | None:
    match = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", query)
    if match:
        ds = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        s = parse_single_date_utc(ds, end_of_day=False)
        e = parse_single_date_utc(ds, end_of_day=True)
        if s is not None and e is not None:
            return s, e

    ym_match = re.search(r"(\d{4})[-/年](\d{1,2})月?", query)
    if ym_match:
        year = int(ym_match.group(1))
        month = int(ym_match.group(2))
        if 1 <= month <= 12:
            last_day = calendar.monthrange(year, month)[1]
            s = parse_single_date_utc(f"{year:04d}-{month:02d}-01", end_of_day=False)
            e = parse_single_date_utc(f"{year:04d}-{month:02d}-{last_day:02d}", end_of_day=True)
            if s is not None and e is not None:
                return s, e

    y_match = re.search(r"\b(20\d{2})\s*年|\bin\s+(20\d{2})\b", query, flags=re.IGNORECASE)
    if y_match:
        year = int(y_match.group(1) or y_match.group(2))
        s = parse_single_date_utc(f"{year:04d}-01-01", end_of_day=False)
        e = parse_single_date_utc(f"{year:04d}-12-31", end_of_day=True)
        if s is not None and e is not None:
            return s, e

    if anchor_ts is None:
        return None

    base_dt = datetime.fromtimestamp(anchor_ts, tz=UTC)
    lower = query.casefold()
    if "yesterday" in lower or "昨天" in query:
        target = (base_dt - timedelta(days=1)).date().isoformat()
        s = parse_single_date_utc(target, end_of_day=False)
        e = parse_single_date_utc(target, end_of_day=True)
        if s is not None and e is not None:
            return s, e
    if "前天" in query:
        target = (base_dt - timedelta(days=2)).date().isoformat()
        s = parse_single_date_utc(target, end_of_day=False)
        e = parse_single_date_utc(target, end_of_day=True)
        if s is not None and e is not None:
            return s, e
    if "last week" in lower or "上周" in query:
        end_dt = base_dt
        start_dt = base_dt - timedelta(days=14)
        return int(start_dt.timestamp()), int(end_dt.timestamp())
    if "last month" in lower or "上个月" in query:
        end_dt = base_dt
        start_dt = base_dt - timedelta(days=45)
        return int(start_dt.timestamp()), int(end_dt.timestamp())

    rel_en = re.search(r"(?i)\b(\d{1,3})\s+(day|week|month)s?\s+ago\b", query)
    if rel_en:
        n = int(rel_en.group(1))
        unit = rel_en.group(2).lower()
        days = n * (30 if unit == "month" else 7 if unit == "week" else 1)
        center = base_dt - timedelta(days=days)
        return int((center - timedelta(days=2)).timestamp()), int(
            (center + timedelta(days=2)).timestamp()
        )

    rel_zh = re.search(r"(\d{1,3})\s*(天|周|个月)前", query)
    if rel_zh:
        n = int(rel_zh.group(1))
        unit = rel_zh.group(2)
        days = n * (30 if unit == "个月" else 7 if unit == "周" else 1)
        center = base_dt - timedelta(days=days)
        return int((center - timedelta(days=2)).timestamp()), int(
            (center + timedelta(days=2)).timestamp()
        )

    return None


_INCIDENTAL_START_RE = re.compile(
    r"^\s*(?:顺便说(?:一句|一下)?|顺便提(?:一句|一下|一嘴)?|对了[，,：:\s]|另外提一嘴|另外顺便|"
    r"by\s+the\s+way\b|btw\b|oh\s+also\b|incidentally\b|speaking\s+of\s+which\b|"
    r"on\s+another\s+note\b)",
    re.IGNORECASE,
)

_INCIDENTAL_INLINE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?；;，,\s])\s*(?=(?:顺便说(?:一句|一下)?|顺便提(?:一句|一下|一嘴)?|"
    r"对了[，,：:]|另外提一嘴|另外顺便|"
    r"by\s+the\s+way\b|btw\b|oh\s+also\b|incidentally\b|speaking\s+of\s+which\b|"
    r"on\s+another\s+note\b))",
    re.IGNORECASE,
)


def is_incidental_update_text(text: str) -> bool:
    if not text:
        return False
    return bool(_INCIDENTAL_START_RE.match(text))


def _stem_en_word(w: str) -> str | None:
    if len(w) < 4 or not w.isalpha():
        return None
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) >= 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeioulsz":
            return base[:-1]
        if re.search(r"(?<![aeiou])[aeiou][bcdfghjklmnpqrstvz]$", base) or base.endswith(
            ("c", "v", "z", "u")
        ):
            return base + "e"
        return base if len(base) >= 3 else None
    if w.endswith("ed") and len(w) >= 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeioulsz":
            return base[:-1]
        if re.search(r"(?<![aeiou])[aeiou][bcdfghjklmnpqrstvz]$", base) or base.endswith(
            ("c", "v", "z", "u")
        ):
            return base + "e"
        return base if len(base) >= 3 else None
    if w.endswith(("ches", "shes", "xes", "zes", "sses")) and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) >= 4:
        return w[:-1]
    return None


_CJK_STOP_CHARS = {"的", "了", "吗", "呢", "吧", "啊", "呀", "着", "过", "什", "么"}


def tokenize_bm25(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.casefold()
    en_words: list[str] = []
    for w in re.findall(r"[a-z0-9_]+", lowered):
        if w in _EN_STOPWORDS or len(w) < 1:
            continue
        en_words.append(w)
        stem = _stem_en_word(w)
        if stem and stem != w and stem not in _EN_STOPWORDS:
            en_words.append(stem)
    cjk_chars = [
        ch for ch in re.findall(r"[\u3400-\u9fff]", lowered) if ch not in _CJK_STOP_CHARS
    ]
    cjk_bigrams: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]{2,}", lowered):
        for i in range(len(run) - 1):
            cjk_bigrams.append(run[i : i + 2])
    return en_words + cjk_chars + cjk_bigrams


def build_bm25_tokens(
    text: str,
    *,
    subject: str = "user",
    tag: str | None = None,
    tag_boost: int = TAG_BOOST,
    subject_boost: int = SUBJECT_BOOST,
) -> str:
    tokens = list(tokenize_bm25(text))
    if subject:
        subj_tokens = tokenize_bm25(subject)
        for part in re.split(r"[/\s,，_|-]+", subject.casefold()):
            if part and part not in subj_tokens:
                subj_tokens.append(part)
        for _ in range(max(0, subject_boost)):
            tokens.extend(subj_tokens)
    if tag:
        tag_tokens = tokenize_bm25(tag)
        for part in re.split(r"[/\s,，_|-]+", tag.casefold()):
            if part and part not in tag_tokens:
                tag_tokens.append(part)
        for _ in range(max(0, tag_boost)):
            tokens.extend(tag_tokens)
    return " ".join(tokens)


def slice_message_content(
    content: str,
    *,
    long_threshold: int = LONG_DOC_CHAR_THRESHOLD,
    window_chars: int = LONG_DOC_WINDOW_CHARS,
    overlap_chars: int = LONG_DOC_OVERLAP_CHARS,
) -> list[str]:
    text = content.strip()
    if not text:
        return []

    if len(text) > long_threshold:
        step = max(100, window_chars - overlap_chars)
        first_newline = text.find("\n")
        header_prefix = ""
        if 0 < first_newline <= 120:
            first_line = text[:first_newline].strip()
            if "|" in first_line or "\t" in first_line or "表" in first_line:
                header_prefix = first_line + "\n"

        slices: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + window_chars)
            window = text[start:end]
            if start > 0 and header_prefix and not window.startswith(header_prefix):
                window = header_prefix + window
            slices.append(window)
            if end >= n:
                break
            start += step
        return slices

    raw_sentences: list[str] = []
    for base_sent in _SENTENCE_SPLIT_RE.split(text):
        base_sent = base_sent.strip()
        if not base_sent:
            continue
        for sub_sent in _INCIDENTAL_INLINE_SPLIT_RE.split(base_sent):
            sub_sent = sub_sent.strip()
            if sub_sent:
                raw_sentences.append(sub_sent)
    if not raw_sentences:
        return [text]

    slices = []
    cur_group: list[str] = []
    cur_len = 0
    for sent in raw_sentences:
        if len(sent) > NORMAL_SLICE_MAX_CHARS:
            if cur_group:
                slices.append(" ".join(cur_group))
                cur_group = []
                cur_len = 0
            for idx in range(0, len(sent), NORMAL_SLICE_MAX_CHARS):
                slices.append(sent[idx : idx + NORMAL_SLICE_MAX_CHARS])
            continue
        if cur_group and (
            is_incidental_update_text(sent)
            or len(cur_group) >= NORMAL_SLICE_SENTENCES
            or cur_len + len(sent) > NORMAL_SLICE_MAX_CHARS
        ):
            slices.append(" ".join(cur_group))
            cur_group = [sent]
            cur_len = len(sent)
        else:
            cur_group.append(sent)
            cur_len += len(sent)
    if cur_group:
        slices.append(" ".join(cur_group))
    return slices


def has_explicit_forget_keyword(text: str) -> bool:
    if not text:
        return False
    return bool(_EXPLICIT_FORGET_ROOT_RE.search(text))


def split_text(text: str, target: int = 1200, maximum: int = 1800) -> list[str]:
    if len(text) <= maximum:
        return [text]

    pieces = [piece for piece in _BREAK_RE.split(text) if piece]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > maximum:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(piece[start : start + maximum] for start in range(0, len(piece), maximum))
            continue
        separator = "\n" if current else ""
        if current and len(current) + len(separator) + len(piece) > target:
            chunks.append(current)
            current = piece
        else:
            current += separator + piece
    if current:
        chunks.append(current)
    return chunks


def render_write_items(
    request: AddRequest, target: int = 1200, maximum: int = 1800
) -> list[WriteItem]:
    items: list[WriteItem] = []
    for message_index, message in enumerate(request.messages):
        if message.role == "system":
            continue
        chunks = split_text(message.content, target=target, maximum=maximum)
        source_timestamp = iso_from_millis(message.timestamp)
        time_label = source_timestamp or f"session:{request.session_id} message:{message_index + 1}"
        for chunk_index, chunk in enumerate(chunks):
            part_label = f" | part {chunk_index + 1}/{len(chunks)}" if len(chunks) > 1 else ""
            content = f"[{message.role} | {time_label}{part_label}] {chunk}"
            stable_name = f"{request.request_id}\0{message_index}\0{chunk_index}"
            items.append(
                WriteItem(
                    id=str(uuid5(NAMESPACE_URL, stable_name)),
                    message_index=message_index,
                    chunk_index=chunk_index,
                    role=message.role,
                    source_timestamp=source_timestamp,
                    content=content,
                )
            )
    return items


def extract_forget_target(text: str) -> str | None:
    stripped = text.strip().strip(".。!?！？")
    for pattern in _FORGET_PATTERNS:
        match = pattern.search(stripped)
        if match:
            target = match.group(1).strip().strip(".。!?！？\"'")
            target = _TRAILING_FORGET_NOISE_RE.sub("", target).strip().strip(".。!?！？\"'")
            if len(target) >= 2:
                return target
    return None


def extract_keywords_from_target(target: str) -> list[str]:
    cleaned = _TRAILING_FORGET_NOISE_RE.sub("", target).strip()
    keywords: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_-]{2,}", cleaned):
        if token.casefold() not in _EN_STOPWORDS and token not in keywords:
            keywords.append(token)
    for cjk_chunk in re.findall(r"[\u3400-\u9fff]+", cleaned):
        for seg in re.split(r"[的在是把了和与及]+", cjk_chunk):
            seg = seg.strip()
            if 2 <= len(seg) <= 12 and seg not in keywords:
                keywords.append(seg)
    if not keywords and len(cleaned) >= 2:
        keywords.append(cleaned)
    return keywords


def regex_extract_batch(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic regex fallback when Module B LLM times out or returns invalid JSON."""
    forget_actions: list[dict[str, Any]] = []
    msg_list = list(messages)
    for idx, msg in enumerate(msg_list):
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        seq = int(msg.get("seq") or 0)
        if role != "user" or not has_explicit_forget_keyword(content):
            continue
        target = extract_forget_target(content)
        if not target:
            continue
        batch_seqs = [seq]
        if idx + 1 < len(msg_list) and str(msg_list[idx + 1].get("role")) == "assistant":
            batch_seqs.append(int(msg_list[idx + 1].get("seq") or 0))
        forget_actions.append(
            {
                "trigger_seq": seq,
                "target_existing_tags": [],
                "target_keywords": extract_keywords_from_target(target),
                "target_description": target,
                "batch_seqs_to_tombstone": batch_seqs,
            }
        )
    return {"facts": [], "forget_actions": forget_actions}


def temporal_hints(text: str, timestamp_ms: int | None) -> list[str]:
    """Resolve a deliberately small, auditable set of relative-time phrases."""
    if timestamp_ms is None:
        return []
    ts_sec = timestamp_ms / 1000.0 if timestamp_ms > 10_000_000_000 else float(timestamp_ms)
    base = datetime.fromtimestamp(ts_sec, tz=UTC)
    lower = text.lower()
    hints: list[str] = []
    fixed = {
        "today": base,
        "yesterday": base - timedelta(days=1),
        "tomorrow": base + timedelta(days=1),
        "今天": base,
        "昨天": base - timedelta(days=1),
        "明天": base + timedelta(days=1),
        "前天": base - timedelta(days=2),
        "后天": base + timedelta(days=2),
    }
    for phrase, resolved in fixed.items():
        if phrase in lower:
            hints.append(f"{phrase}={resolved.date().isoformat()}")
    for match in re.finditer(r"(?i)\b(\d{1,3})\s+(day|week)s?\s+ago\b", text):
        count = int(match.group(1))
        days = count * (7 if match.group(2).lower() == "week" else 1)
        hints.append(f"{match.group(0)}={(base - timedelta(days=days)).date().isoformat()}")
    for match in re.finditer(r"(\d{1,3})\s*(天|周)前", text):
        count = int(match.group(1))
        days = count * (7 if match.group(2) == "周" else 1)
        hints.append(f"{match.group(0)}={(base - timedelta(days=days)).date().isoformat()}")
    return hints


def add_temporal_hints(items: list[WriteItem], request: AddRequest) -> None:
    by_message: dict[int, list[str]] = {}
    for index, message in enumerate(request.messages):
        hints = temporal_hints(message.content, message.timestamp)
        if hints:
            by_message[index] = hints
    for item in items:
        if hints := by_message.get(item.message_index):
            item.content += "\n[resolved time: " + "; ".join(hints) + "]"


def approximate_tokens(text: str) -> int:
    # Conservative enough for the shared 117,760-token input budget, without
    # coupling the service to the Answer model's private tokenizer.
    return max(1, (len(text.encode("utf-8")) + 2) // 3)
