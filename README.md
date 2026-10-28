# Dandelion Memory

A conversational memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) (Cycle 2, Textual Memory track, Open-source Methods division).

## Architecture Overview (`v3.1.1`)

- **Immutable Verbatim Evidence + Living State Timelines**: Raw conversation turns are stored verbatim with speaker and timestamp (`parent_messages` + `raw_slice`), while `gpt-4o-mini` extracts structured atomic facts (`fact_item`), explicit vs. implied cue layers (`implied`), evolving per-slot timelines (`tag_timelines`, `#F1..#Fn`), and grounded person profiles (`person_profiles`).
- **Two-Stage Consolidation & Grounding Guards**: Periodic and search-flush organizers (`gpt-4o-mini`) consolidate longitudinal state updates and reconcile person aliases/relations with strict lexical grounding guards and first-person signature attribution.
- **Tri-Route Recall, Fine-Grained Evidence Ladder & Reranking**: Hybrid retrieval fuses dense embeddings (`text-embedding-v4`, 1024-dim), BM25 lexical matching, and temporal range matching via Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking (`qwen3-rerank`), index-to-raw evidence pull, smooth turn-pair restoration, tiered evidence prioritization, and bounded token-budget packing.
- **Memory Governance & Attribution**: Imperative forgetting directives selectively tombstone target turns/slices, facts, and timelines without collateral damage while injecting specific revoked-preference (`FN`) constraints; third-party colleague emails, case studies, and fictional drafts are isolated or annotated with bilingual attribution constraints.

See `docs/DESIGN.md` and `docs/SUBMISSION.md` for full architecture and evaluation details.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
.venv/bin/uvicorn aml_entry:app --host 127.0.0.1 --port 8000
```

Run tests with `.venv/bin/pytest`. After deployment, run `python scripts/smoke_check.py BASE_URL API_KEY` to verify synchronous Add/Search behavior.

License: MIT.
