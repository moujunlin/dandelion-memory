# Dandelion Memory

A memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) (Cycle 2, textual track, open-source methods).

Raw turns are stored verbatim with speaker and time; Search runs a configurable seven-stage pipeline (views → multi-route recall → fusion → rerank → neighbor expansion → ordering → budget packing). See `docs/DESIGN.md`.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
cp .env.example .env
.venv/bin/uvicorn aml_entry:app --host 127.0.0.1 --port 8000
```

Apply the SQL files in `migrations/` to a PostgreSQL project with pgvector, PGroonga, and pgcrypto before starting the API. Generate an independent `STORAGE_RPC_TOKEN`, then store only its SHA-256 hash in `aml.config` under `storage_token_sha256`. The service uses a publishable Supabase key plus that private token; table access stays behind three narrowly scoped RPCs.

Run tests with `.venv/bin/pytest`. After deployment, run `python scripts/smoke_check.py BASE_URL API_KEY` to verify synchronous Add/Search behavior.

License: MIT.
