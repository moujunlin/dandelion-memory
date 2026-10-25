from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from .models import AddRequest, WriteItem

_BREAK_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
_FORGET_PATTERNS = (
    re.compile(r"(?is)\b(?:please\s+)?forget(?:\s+about|\s+that)?\s+(.+)$"),
    re.compile(
        r"(?is)\bdelete\s+(?:the\s+)?(?:memory|memories|information)?\s*(?:about|that)?\s+(.+)$"
    ),
    re.compile(
        r"(?s)(?:请)?(?:忘掉|忘记|删除|别再记得|不要记得)(?:关于|我说过的|那件事|这件事|：|:)?\s*(.+)$"
    ),
)


def iso_from_millis(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


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
            if len(target) >= 2:
                return target
    return None


def temporal_hints(text: str, timestamp_ms: int | None) -> list[str]:
    """Resolve a deliberately small, auditable set of relative-time phrases."""
    if timestamp_ms is None:
        return []
    base = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
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
