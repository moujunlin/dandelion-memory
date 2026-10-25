from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .config import Settings, get_settings
from .embedding import DashScopeEmbedder
from .extractor import DashScopeExtractor
from .models import AddRequest, AddResponse, SearchRequest, SearchResponse
from .pipeline import MemoryPipeline
from .reranker import DashScopeReranker
from .store import SQLiteStore, StoreConflictError


def build_pipeline(settings: Settings) -> MemoryPipeline:
    store = SQLiteStore(settings.sqlite_path)
    embedder = DashScopeEmbedder(
        api_key=settings.dashscope_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )
    extractor = DashScopeExtractor(
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
    reranker = DashScopeReranker(
        api_key=settings.dashscope_api_key,
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        timeout=settings.rerank_timeout,
    )
    return MemoryPipeline(
        store=store,
        embedder=embedder,
        extractor=extractor,
        reranker=reranker,
        semantic_candidates=settings.semantic_candidates,
        lexical_candidates=settings.lexical_candidates,
        neighbor_anchors=settings.neighbor_anchors,
        neighbor_radius=settings.neighbor_radius,
        token_budget=settings.answer_token_budget,
        chunk_target_chars=settings.chunk_target_chars,
        chunk_max_chars=settings.chunk_max_chars,
    )


def create_app(
    *,
    settings: Settings | None = None,
    pipeline: MemoryPipeline | None = None,
) -> FastAPI:
    app = FastAPI(title="Dandelion Memory", version="0.2.1")
    resolved_settings = settings or get_settings()
    resolved_pipeline = pipeline or build_pipeline(resolved_settings)

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        valid = scheme.casefold() == "bearer" and hmac.compare_digest(
            supplied, resolved_settings.memory_api_key
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key"
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "dandelion-memory", "version": "0.2.1"}

    @app.post("/add", response_model=AddResponse, dependencies=[Depends(require_auth)])
    async def add(request: AddRequest) -> AddResponse:
        try:
            return await resolved_pipeline.add(request)
        except StoreConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_auth)])
    async def search(request: SearchRequest) -> SearchResponse:
        return await resolved_pipeline.search(request)

    return app
