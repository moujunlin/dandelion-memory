from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Protocol

import httpx

EMBED_MAX_INPUT_CHARS = 6000


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class DashScopeEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        batch_size: int = 10,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._client is None
            or self._client.is_closed
            or (
                self._client_loop is not None
                and (self._client_loop is not current_loop or self._client_loop.is_closed())
            )
        ):
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._client_loop = current_loop
        elif self._client_loop is None:
            self._client_loop = current_loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            with contextlib.suppress(RuntimeError):
                await self._client.aclose()
        self._client = None
        self._client_loop = None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = [
                (t[:EMBED_MAX_INPUT_CHARS] if len(t) > EMBED_MAX_INPUT_CHARS else t) or " "
                for t in texts[start : start + self.batch_size]
            ]
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                client = self._get_client()
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "input": batch,
                            "dimensions": self.dimensions,
                        },
                    )
                    if (
                        response.status_code in (429, 500, 502, 503, 504)
                        and attempt < self.max_retries
                    ):
                        await asyncio.sleep(0.05 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    ordered = sorted(payload["data"], key=lambda item: item["index"])
                    batch_vectors = [item["embedding"] for item in ordered]
                    if len(batch_vectors) != len(batch):
                        raise RuntimeError("embedding API returned the wrong vector count")
                    if any(len(vector) != self.dimensions for vector in batch_vectors):
                        raise RuntimeError("embedding API returned the wrong vector dimension")
                    vectors.extend(batch_vectors)
                    last_exc = None
                    break
                except RuntimeError as exc:
                    if "Event loop is closed" in str(exc) and attempt < self.max_retries:
                        self._client = None
                        self._client_loop = None
                        continue
                    raise
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 500
                        and exc.response.status_code != 429
                    ):
                        raise
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.05 * (attempt + 1))
                        continue
                    raise
            if last_exc is not None:
                raise last_exc
        return vectors
