# Submission Notes — Dandelion Memory (`v3.1.1`)

- **System**: Dandelion Memory
- **Track**: Textual Memory, Open-source Methods (Academic / Open-source Leaderboard)
- **Version**: `v3.1.1`
- **Public Repository**: `https://github.com/moujunlin/dandelion-memory` (MIT License)
- **Add / Organizer LLM**: `gpt-4o-mini` (stateless atomic fact and implied-cue extraction, imperative forget-action parsing, longitudinal state-slot consolidation, and lexically grounded person-map reconciliation)
- **Embedding Model**: Alibaba Cloud Model Studio `text-embedding-v4` (1024 dimensions)
- **Reranker Model**: Alibaba Cloud Model Studio `qwen3-rerank`
- **Storage Engine**: Embedded SQLite (`SQLiteStore`, WAL mode) with strict `user_id` isolation across `parent_messages`, `search_chunks` (`WITHOUT ROWID` binary float32 embeddings), `tag_timelines`, `person_profiles`, `user_state`, and `add_requests`
- **Retrieval Pipeline**:
  1. Tri-route recall combining dense cosine similarity, BM25 lexical scoring (with CJK bigrams and identifier tokens), and temporal range matching fused via Reciprocal Rank Fusion (RRF).
  2. Index-to-raw evidence pull (`timeline_summary` / `fact_item` -> `parent_messages`) and smooth Turn-Pair dialogue card restoration (including oversized assistant reply head/tail preservation, high-score slice `rerank_text` refresh, and prior-turn draft context recovery).
  3. Cross-encoder reranking (`qwen3-rerank`) with fine-grained evidence ladder gating (`x0.35` demotion on unqueried third-party cards when high-confidence first-person evidence exists) and bilingual third-party attribution constraints.
  4. Explicit forgetting & revocation governance: selectively tombstones revoked turns/slices, facts, and timeline slots with word-boundary and multi-topic slice guards (`Zero-Leak@100 = 100%`, `0` collateral Gold tombstoning), and prepends specific revoked-preference (`FN`) constraints when queried.
- **API Endpoints & Authentication**:
  - Base URL: `https://dandelion-aml.47-242-148-141.sslip.io`
  - Health: unauthenticated `GET /health`
  - Add: synchronous `POST /add` (`Authorization: Bearer <Memory System Key>`)
  - Search: synchronous `POST /search` (`Authorization: Bearer <Memory System Key>`, hard maximum `top_k`)
- **Capacity & Operational Measurements**:
  - Deployed on a dedicated 4 vCPU / 15 GiB RAM / 73 GiB SSD Linux instance (Alibaba Cloud Hong Kong) behind Caddy HTTPS (`response-header` timeout 35m) with 1 Uvicorn worker and persistent volume mount (`/opt/dandelion-memory/data:/data`).
  - Verified on 100-question multi-benchmark evaluation (`LongMemEval-S`, `LoCoMo`, `PersonaMem-v2` up to 128k / ~1000 turns per conversation): search latency `p50 = 1.35s` (`p95 = 3.11s`), peak process memory `VmRSS = 221.1 MiB`.
- **Data Retention**: All evaluation data is isolated per `user_id` and deleted within 30 days after final evaluation.

