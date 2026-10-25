from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from .models import MemoryCandidate, WriteItem


class StoreConflictError(RuntimeError):
    pass


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
