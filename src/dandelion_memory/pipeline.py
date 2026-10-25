from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from .embedding import Embedder
from .models import (
    AddRequest,
    AddResponse,
    MemoryCandidate,
    SearchItem,
    SearchRequest,
    SearchResponse,
)
from .store import MemoryStore
from .text import add_temporal_hints, approximate_tokens, extract_forget_target, render_write_items


class MemoryPipeline:
    def __init__(
        self,
        *,
        store: MemoryStore,
        embedder: Embedder,
        semantic_candidates: int = 200,
        lexical_candidates: int = 200,
        neighbor_anchors: int = 20,
        neighbor_radius: int = 1,
        token_budget: int = 100_000,
        chunk_target_chars: int = 1200,
        chunk_max_chars: int = 1800,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.semantic_candidates = semantic_candidates
        self.lexical_candidates = lexical_candidates
        self.neighbor_anchors = neighbor_anchors
        self.neighbor_radius = neighbor_radius
        self.token_budget = token_budget
        self.chunk_target_chars = chunk_target_chars
        self.chunk_max_chars = chunk_max_chars

    async def add(self, request: AddRequest) -> AddResponse:
        items = render_write_items(
            request, target=self.chunk_target_chars, maximum=self.chunk_max_chars
        )
        add_temporal_hints(items, request)
        vectors = await self.embedder.embed([item.content for item in items])
        for item, vector in zip(items, vectors, strict=True):
            item.embedding = vector

        archive_ids: list[str] = []
        for message_index, message in enumerate(request.messages):
            if message.role != "user":
                continue
            target = extract_forget_target(message.content)
            if not target:
                continue
            for item in items:
                if item.message_index == message_index:
                    item.searchable = False
            [target_vector] = await self.embedder.embed([target])
            candidates = await self.store.search_candidates(
                user_id=request.user_id,
                query=target,
                embedding=target_vector,
                semantic_limit=min(50, self.semantic_candidates),
                lexical_limit=min(50, self.lexical_candidates),
                result_limit=50,
            )
            archive_ids.extend(select_for_archive(target, candidates))

        payload_hash = hashlib.sha256(
            json.dumps(
                request.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        await self.store.add_request(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_hash=payload_hash,
            items=items,
            archive_ids=list(dict.fromkeys(archive_ids)),
        )
        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        query = request.query
        if request.options:
            query += "\n[options]\n" + "\n".join(request.options)
        [query_vector] = await self.embedder.embed([query])
        candidates = await self.store.search_candidates(
            user_id=request.user_id,
            query=query,
            embedding=query_vector,
            semantic_limit=self.semantic_candidates,
            lexical_limit=self.lexical_candidates,
            result_limit=max(request.top_k * 2, self.neighbor_anchors, 200),
        )
        anchors = candidates[: min(self.neighbor_anchors, len(candidates))]
        neighbors = await self.store.neighbors(
            user_id=request.user_id,
            anchor_ids=[candidate.id for candidate in anchors],
            radius=self.neighbor_radius,
        )
        ordered = expand_neighbors(candidates, neighbors, anchors)
        packed = pack_results(ordered, request.top_k, self.token_budget)
        return SearchResponse(
            data=[
                SearchItem(
                    id=candidate.id,
                    content=candidate.content,
                    score=candidate.score,
                    created_at=candidate.created_at,
                )
                for candidate in packed
            ]
        )


_STOPWORDS = {
    "a",
    "an",
    "and",
    "i",
    "is",
    "it",
    "my",
    "of",
    "that",
    "the",
    "to",
    "was",
}


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w\u3400-\u9fff]+", text.casefold())
        if term not in _STOPWORDS
    }


def select_for_archive(
    target: str, candidates: Sequence[MemoryCandidate], maximum: int = 12
) -> list[str]:
    target_terms = _terms(target)
    selected: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_terms = _terms(candidate.content)
        overlap = len(target_terms & candidate_terms) / max(1, len(target_terms))
        if overlap >= 0.35 or (index == 0 and candidate.score > 0):
            selected.append(candidate.id)
        if len(selected) >= maximum:
            break
    return selected


def expand_neighbors(
    candidates: Sequence[MemoryCandidate],
    neighbors: Sequence[MemoryCandidate],
    anchors: Sequence[MemoryCandidate],
) -> list[MemoryCandidate]:
    neighbors_by_anchor: dict[str, list[MemoryCandidate]] = {}
    for neighbor in neighbors:
        if neighbor.anchor_id:
            neighbors_by_anchor.setdefault(neighbor.anchor_id, []).append(neighbor)
    for group in neighbors_by_anchor.values():
        group.sort(key=lambda item: (item.distance or 0, item.message_index, item.chunk_index))

    ordered: list[MemoryCandidate] = []
    seen: set[str] = set()
    for anchor in anchors:
        context = neighbors_by_anchor.get(anchor.id, [])
        before = [item for item in context if (item.distance or 0) < 0]
        after = [item for item in context if (item.distance or 0) > 0]
        for item in [*before, anchor, *after]:
            if item.id not in seen:
                item.score = anchor.score if item.id == anchor.id else max(anchor.score - 1e-9, 0.0)
                ordered.append(item)
                seen.add(item.id)
    for candidate in candidates:
        if candidate.id not in seen:
            ordered.append(candidate)
            seen.add(candidate.id)
    return ordered


def pack_results(
    candidates: Sequence[MemoryCandidate], top_k: int, token_budget: int
) -> list[MemoryCandidate]:
    packed: list[MemoryCandidate] = []
    used = 0
    for candidate in candidates:
        cost = approximate_tokens(candidate.content)
        if packed and used + cost > token_budget:
            continue
        if not packed and cost > token_budget:
            continue
        packed.append(candidate)
        used += cost
        if len(packed) >= top_k:
            break
    return packed
