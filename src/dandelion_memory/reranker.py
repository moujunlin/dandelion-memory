from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from .text import tokenize_bm25

RERANK_DOC_MAX_CHARS = 2400


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]: ...


class NoOpReranker:
    """Fallback reranker marker when remote API key is not configured."""

    is_noop: bool = True

    async def rerank(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        _ = (args, kwargs)
        doc_list = list(documents)
        if not doc_list:
            return []
        q_tokens = set(tokenize_bm25(query))
        scores: list[float] = []
        for doc in doc_list:
            d_tokens = set(tokenize_bm25(doc[:RERANK_DOC_MAX_CHARS]))
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            scores.append(min(0.99, max(0.05, overlap)))
        return scores

    async def __call__(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        return await self.rerank(query, documents, *args, **kwargs)


class DashScopeReranker:
    """Module D DashScope qwen3-rerank client with 4.0s timeout."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = (
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        ),
        model: str = "qwen3-rerank",
        timeout: float = 4.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client
        self._client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_noop(self) -> bool:
        return not bool(self.api_key)

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

    async def rerank(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        _ = (args, kwargs)
        doc_list = [doc[:RERANK_DOC_MAX_CHARS] for doc in documents]
        if not doc_list:
            return []
        if not self.api_key:
            raise RuntimeError("DashScopeReranker api_key is empty; fallback to RRF")

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": doc_list,
            },
            "parameters": {
                "return_documents": False,
                "top_n": len(doc_list),
            },
        }
        for attempt in range(2):
            client = self._get_client()
            try:
                resp = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
                break
            except RuntimeError as exc:
                if "Event loop is closed" in str(exc) and attempt == 0:
                    self._client = None
                    self._client_loop = None
                    continue
                raise

        scores = [0.0] * len(doc_list)
        results = (
            body.get("output", {}).get("results")
            or body.get("results")
            or body.get("data")
            or []
        )
        for item in results:
            idx = int(item.get("index", 0))
            if 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    async def __call__(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        return await self.rerank(query, documents, *args, **kwargs)
