from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class DashScopeEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        batch_size: int = 10,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch, "dimensions": self.dimensions},
                )
                response.raise_for_status()
                payload = response.json()
                ordered = sorted(payload["data"], key=lambda item: item["index"])
                batch_vectors = [item["embedding"] for item in ordered]
                if len(batch_vectors) != len(batch):
                    raise RuntimeError("embedding API returned the wrong vector count")
                if any(len(vector) != self.dimensions for vector in batch_vectors):
                    raise RuntimeError("embedding API returned the wrong vector dimension")
                vectors.extend(batch_vectors)
        return vectors
