# Dandelion Memory design

Dandelion keeps source turns as deterministic, speaker-labelled chunks. It does not generate answers and it does not rewrite benchmark questions.

The Search path is a configurable seven-stage pipeline:

1. **Views:** speaker, source time, exact chunk text, and a small deterministic set of relative-time hints.
2. **Multi-route recall:** `text-embedding-v4` cosine search plus PGroonga full-text search, always filtered by `user_id`.
3. **Fusion:** reciprocal-rank fusion with `k=60`.
4. **Rerank:** identity in v0.1; the stage boundary remains explicit for later measured variants.
5. **Neighbor expansion:** the top anchors pull adjacent turns from the same session.
6. **Ordering:** anchor relevance first, with local context adjacent to its anchor.
7. **Budget packing:** stable deduplication, `top_k` enforcement, and a conservative input-token budget.

## Write semantics

Add is synchronous: success is returned only after vectors and rows commit. `request_id` is a strict idempotency key. A retry with the same payload is a no-op; reuse with a different payload is a conflict.

A direct forgetting request is detected before the new turn commits. Candidate memories are retrieved inside the same `user_id`; matching rows are archived in the same transaction that writes the request. The directive itself is retained as an archived audit row, so its wording cannot reintroduce the forgotten fact through Search. Source rows remain auditable but are excluded from all retrieval paths.

## Storage and privacy

Only the dedicated `aml` schema in the Supabase DEV project is used. Tables have RLS enabled and no direct anon policies. The service calls narrowly scoped `SECURITY DEFINER` RPCs through a publishable key plus a separate random storage token; the database stores only its SHA-256 hash. No competition data enters Lori's personal memory systems. Evaluation data will be deleted within 30 days of the final run.

## Model disclosure

The first baseline uses `text-embedding-v4` and no LLM component. If an Add-time extractor is enabled for the formal version, the open-method requirement will be satisfied with `gpt-4o-mini` and the fixed commit will disclose that change.
