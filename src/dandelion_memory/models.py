from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)
    timestamp: int | None = Field(default=None, ge=0)

    @field_validator("content")
    @classmethod
    def content_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=500)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=500)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1, max_length=500)
    top_k: int = Field(ge=1, le=1000)

    @field_validator("query")
    @classmethod
    def query_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchItem]


@dataclass(slots=True)
class WriteItem:
    id: str
    message_index: int
    chunk_index: int
    role: str
    source_timestamp: str | None
    content: str
    searchable: bool = True
    embedding: list[float] = field(default_factory=list)

    def as_rpc_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "message_index": self.message_index,
            "chunk_index": self.chunk_index,
            "role": self.role,
            "source_timestamp": self.source_timestamp,
            "content": self.content,
            "searchable": self.searchable,
            "embedding": self.embedding,
        }


@dataclass(slots=True)
class MemoryCandidate:
    id: str
    content: str
    score: float
    created_at: str | None = None
    session_id: str | None = None
    source_timestamp: str | None = None
    message_index: int = 0
    chunk_index: int = 0
    anchor_id: str | None = None
    anchor_rank: int | None = None
    distance: int | None = None
    kind: str = "raw_slice"
    parent_msg_id: str | None = None
    seq: int = 0
    session_ts: int | None = None
    subject: str = "user"
    tag: str | None = None
    event_time_start: int | None = None
    event_time_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
