# Submission notes (draft)

- System: Dandelion Memory
- Track: Textual Memory, Open-source Methods
- Version: 0.1 baseline
- Embedding: Alibaba Cloud Model Studio `text-embedding-v4`, 1024 dimensions
- Add-time LLM: none in baseline
- Retrieval: cosine + PGroonga, reciprocal-rank fusion, adjacent-turn expansion
- Authentication: Bearer
- Health: unauthenticated `GET /health`
- Add: synchronous `POST /add`
- Search: `POST /search`, hard maximum `top_k`
- Isolation: every database read and archive operation is filtered by `user_id`
- Storage: dedicated `aml` schema in an isolated development project
- Storage RPCs: publishable project key plus an independent hashed bearer secret
- Data retention: evaluation data deleted within 30 days after the final evaluation

The deployed URL, capacity measurements, fixed commit, and exact runtime configuration are added only after deployment and load testing.
