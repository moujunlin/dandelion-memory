from __future__ import annotations

from collections.abc import Sequence

import pytest

from dandelion_memory.config import Settings
from dandelion_memory.models import MemoryCandidate, WriteItem


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    def __init__(self) -> None:
        self.adds: list[dict[str, object]] = []
        self.searches: list[dict[str, object]] = []
        self.candidates: list[MemoryCandidate] = []
        self.neighbor_rows: list[MemoryCandidate] = []

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
        self.adds.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "payload_hash": payload_hash,
                "items": list(items),
                "archive_ids": list(archive_ids),
            }
        )
        return True

    async def search_candidates(self, **kwargs: object) -> list[MemoryCandidate]:
        self.searches.append(kwargs)
        return list(self.candidates)

    async def neighbors(self, **kwargs: object) -> list[MemoryCandidate]:
        return list(self.neighbor_rows)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        memory_api_key="test-secret-key-123456",
        supabase_url="https://example.supabase.co",
        supabase_key="anon-key",
        storage_rpc_token="storage-token-for-tests-1234567890",
        dashscope_api_key="dashscope-key",
    )
