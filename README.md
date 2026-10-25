# Dandelion Memory

A memory service for the [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) (Cycle 2, textual track, open-source methods).

Raw turns are stored verbatim with speaker and time; Search runs a configurable seven-stage pipeline (views → multi-route recall → fusion → rerank → neighbor expansion → ordering → budget packing). See `docs/DESIGN.md`.

License: MIT.
