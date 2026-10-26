#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import uuid

from dandelion_memory.models import WriteItem
from dandelion_memory.store import StoreConflictError, SupabaseStore


async def run(base_url: str, api_key: str, storage_token: str) -> str:
    suffix = uuid.uuid4().hex
    request_id = f"rest-smoke-{suffix}"
    user_id = f"rest-smoke-user-{suffix}"
    session_id = f"rest-smoke-session-{suffix}"
    memory_id = str(uuid.uuid4())
    hidden_memory_id = str(uuid.uuid4())
    vector = [1.0, *([0.0] * 1023)]
    store = SupabaseStore(base_url=base_url, api_key=api_key, storage_token=storage_token)
    add_kwargs = {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "payload_hash": hashlib.sha256(request_id.encode()).hexdigest(),
        "items": [
            WriteItem(
                id=memory_id,
                message_index=0,
                chunk_index=0,
                role="user",
                source_timestamp="2026-01-01T00:00:00Z",
                content="[user] REST smoke flower is a blue iris.",
                embedding=vector,
            ),
            WriteItem(
                id=hidden_memory_id,
                message_index=1,
                chunk_index=0,
                role="user",
                source_timestamp="2026-01-01T00:01:00Z",
                content="[user] Please forget that the private flower is a red tulip.",
                searchable=False,
                embedding=vector,
            ),
        ],
        "archive_ids": [],
    }
    add_results = await asyncio.gather(*(store.add_request(**add_kwargs) for _ in range(8)))
    if sum(add_results) != 1:
        raise RuntimeError("concurrent idempotent Add did not commit exactly once")
    try:
        await store.add_request(**{**add_kwargs, "payload_hash": "f" * 64})
    except StoreConflictError:
        pass
    else:
        raise RuntimeError("request_id reuse with a different payload was not rejected")

    search_kwargs = {
        "user_id": user_id,
        "query": "blue iris",
        "embedding": vector,
        "semantic_limit": 20,
        "lexical_limit": 20,
        "result_limit": 20,
    }
    search_results = await asyncio.gather(
        *(store.search_candidates(**search_kwargs) for _ in range(32))
    )
    if not all(any(row.id == memory_id for row in rows) for rows in search_results):
        raise RuntimeError("REST smoke search did not retrieve the inserted memory")
    if any(any(row.id == hidden_memory_id for row in rows) for rows in search_results):
        raise RuntimeError("archived forgetting directive leaked through Search")
    print(request_id)
    return request_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("api_key")
    parser.add_argument("storage_token")
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.api_key, args.storage_token))


if __name__ == "__main__":
    main()
