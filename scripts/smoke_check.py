#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("api_key")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    suffix = uuid.uuid4().hex
    user_id = f"smoke-user-{suffix}"
    session_id = f"smoke-session-{suffix}"
    request_id = f"smoke-request-{suffix}"
    add_payload = {
        "request_id": request_id,
        "messages": [{"role": "user", "content": "The smoke-test flower is a blue iris."}],
        "user_id": user_id,
        "session_id": session_id,
    }
    with httpx.Client(timeout=120) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        added = client.post(f"{base}/add", headers=headers, json=add_payload)
        added.raise_for_status()
        if added.json() != {
            "success": True,
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
        }:
            raise RuntimeError("Add response contract mismatch")
        searched = client.post(
            f"{base}/search",
            headers=headers,
            json={
                "query": "What flower was used in the smoke test?",
                "user_id": user_id,
                "top_k": 100,
            },
        )
        searched.raise_for_status()
        rows = searched.json().get("data")
        if not isinstance(rows, list) or len(rows) > 100:
            raise RuntimeError("Search response contract mismatch")
        if not any("blue iris" in row.get("content", "") for row in rows):
            raise RuntimeError("synchronous Add/Search check failed")
    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
