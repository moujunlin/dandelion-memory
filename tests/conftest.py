from __future__ import annotations

import hashlib
import inspect
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from dandelion_memory.config import Settings
from dandelion_memory.models import MemoryCandidate, WriteItem


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]


def _tokenize_for_hash(text: str) -> list[str]:
    lowered = text.casefold()
    words = re.findall(r"[a-z0-9_]+", lowered)
    stems = [w[:5] for w in words if len(w) > 5]
    cjk_chars = re.findall(r"[\u3400-\u9fff]", lowered)
    cjk_bigrams = [
        lowered[i : i + 2]
        for i in range(len(lowered) - 1)
        if "\u3400" <= lowered[i] <= "\u9fff" and "\u3400" <= lowered[i + 1] <= "\u9fff"
    ]
    return words + stems + cjk_chars + cjk_bigrams


class FakeEmbedder:
    """Deterministic local embedder using signed feature hashing + optional custom overrides."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.custom_vectors: dict[str, list[float]] = {}
        self.substring_vectors: list[tuple[str, list[float]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def set_vector(
        self, text: str, vector: Sequence[float], *, match_substring: bool = False
    ) -> None:
        padded = list(vector) + [0.0] * max(0, self.dimensions - len(vector))
        normed = _l2_normalize(padded[: self.dimensions])
        if match_substring:
            self.substring_vectors.append((text, normed))
        else:
            self.custom_vectors[text] = normed

    def compute_vector(self, text: str) -> list[float]:
        if text in self.custom_vectors:
            return list(self.custom_vectors[text])
        for sub, vec in self.substring_vectors:
            if sub in text:
                return list(vec)

        vec = [0.0] * self.dimensions
        tokens = _tokenize_for_hash(text)
        active_dims = min(256, self.dimensions)
        if not tokens:
            vec[0] = 1.0
            return vec

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            weight = 2.0 if token.isdigit() else 1.0
            for slot in range(4):
                idx = digest[slot * 2] % active_dims
                sign = 1.0 if (digest[slot * 2 + 1] & 1) == 0 else -1.0
                vec[idx] += sign * weight
        return _l2_normalize(vec)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [self.compute_vector(text) for text in batch]


class FakeExtractor:
    """Deterministic mock for Module B batch LLM governance extractor."""

    def __init__(
        self,
        *,
        responses: list[dict[str, Any] | str | Exception] | None = None,
        handler: Callable[..., dict[str, Any]] | None = None,
        side_effect: Exception | None = None,
    ) -> None:
        self.responses: list[dict[str, Any] | str | Exception] = list(responses or [])
        self.handler = handler
        self.side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def queue_response(self, response: dict[str, Any] | str | Exception) -> None:
        self.responses.append(response)

    async def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.side_effect is not None:
            raise self.side_effect
        if self.responses:
            nxt = self.responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            if isinstance(nxt, str):
                import json

                return json.loads(nxt)
            return nxt
        if self.handler is not None:
            return self.handler(*args, **kwargs)
        return {"facts": [], "forget_actions": []}

    async def extract_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self.extract(*args, **kwargs)

    async def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self.extract(*args, **kwargs)


class FakeReranker:
    """Deterministic mock for Module D DashScope qwen3-rerank."""

    def __init__(
        self,
        *,
        score_fn: Callable[[str, str, int], float] | None = None,
        keyword_scores: dict[str, float] | None = None,
        side_effect: Exception | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.keyword_scores: dict[str, float] = dict(keyword_scores or {})
        self.side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def rerank(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        doc_list = list(documents)
        self.calls.append({"query": query, "documents": doc_list, "args": args, "kwargs": kwargs})
        if self.side_effect is not None:
            raise self.side_effect

        scores: list[float] = []
        q_tokens = set(_tokenize_for_hash(query))
        for idx, doc in enumerate(doc_list):
            if self.score_fn is not None:
                scores.append(float(self.score_fn(query, doc, idx)))
                continue
            matched_kw_score: float | None = None
            for kw, val in self.keyword_scores.items():
                if kw in doc:
                    matched_kw_score = (
                        val if matched_kw_score is None else max(matched_kw_score, val)
                    )
            if matched_kw_score is not None:
                scores.append(matched_kw_score)
                continue
            d_tokens = set(_tokenize_for_hash(doc))
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            scores.append(min(0.99, max(0.05, overlap)))
        return scores

    async def __call__(
        self, query: str, documents: Sequence[str], *args: Any, **kwargs: Any
    ) -> list[float]:
        return await self.rerank(query, documents, *args, **kwargs)


class FakeStore:
    """Legacy in-memory fake store retained for v1 unit test compatibility."""

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


def build_sqlite_store(db_path: Path) -> Any:
    import dandelion_memory.store as store_mod

    store_cls = getattr(store_mod, "SQLiteStore", None)
    if store_cls is None:
        raise RuntimeError(
            "SQLiteStore is not yet defined in dandelion_memory.store (required by DESIGN.md v2.1)"
        )
    return store_cls(db_path)


def build_v2_pipeline(
    *,
    db_path: Path,
    embedder: FakeEmbedder | None = None,
    extractor: FakeExtractor | None = None,
    reranker: FakeReranker | None = None,
    store: Any | None = None,
    **extra_kwargs: Any,
) -> tuple[Any, Any, FakeEmbedder, FakeExtractor, FakeReranker]:
    from dandelion_memory.pipeline import MemoryPipeline

    resolved_store = store if store is not None else build_sqlite_store(db_path)
    resolved_embedder = embedder or FakeEmbedder()
    resolved_extractor = extractor or FakeExtractor()
    resolved_reranker = reranker or FakeReranker()

    sig = inspect.signature(MemoryPipeline.__init__)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    init_kwargs: dict[str, Any] = {
        "store": resolved_store,
        "embedder": resolved_embedder,
    }
    for name in ("extractor", "llm_extractor", "llm"):
        if name in params or accepts_var_kw:
            init_kwargs[name] = resolved_extractor
            break
    for name in ("reranker", "rerank_client"):
        if name in params or accepts_var_kw:
            init_kwargs[name] = resolved_reranker
            break

    for k, v in extra_kwargs.items():
        if k in params or accepts_var_kw:
            init_kwargs[k] = v

    pipeline = MemoryPipeline(**init_kwargs)
    return pipeline, resolved_store, resolved_embedder, resolved_extractor, resolved_reranker


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        memory_api_key="test-secret-key-123456",
        supabase_url="https://example.supabase.co",
        supabase_key="anon-key",
        storage_rpc_token="storage-token-for-tests-1234567890",
        dashscope_api_key="dashscope-key",
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "dandelion_v2.sqlite3"
