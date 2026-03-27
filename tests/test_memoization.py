from __future__ import annotations

import pytest

from app.schemas import QueryPayload
from app.services import retrieval_wrapper


@pytest.mark.asyncio
async def test_embedding_cache_uses_same_query_text_once(monkeypatch: pytest.MonkeyPatch) -> None:
    query = QueryPayload(query_text="warehouse software", geography=["United Kingdom"])
    calls = {"embed": 0, "query": 0}

    async def fake_embed(text: str) -> list[float]:
        calls["embed"] += 1
        return [0.1, 0.2, 0.3]

    async def fake_query(embedding: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        calls["query"] += 1
        return []

    monkeypatch.setattr(retrieval_wrapper._vector_store, "embed", fake_embed)
    monkeypatch.setattr(retrieval_wrapper._vector_store, "query", fake_query)
    retrieval_wrapper._embedding_cache._data.clear()
    retrieval_wrapper._retrieval_cache._data.clear()

    await retrieval_wrapper.retrieve_with_retry(query, top_k=5, trace_id="t1")
    await retrieval_wrapper.retrieve_with_retry(query, top_k=5, trace_id="t2")

    assert calls["embed"] == 1
    assert calls["query"] == 1


@pytest.mark.asyncio
async def test_retrieval_cache_key_respects_normalized_query_order(monkeypatch: pytest.MonkeyPatch) -> None:
    query_a = QueryPayload(
        query_text="warehouse software",
        geography=["United Kingdom", "UK"],
        exclusions=["consumer", "payments"],
    )
    query_b = QueryPayload(
        query_text=" warehouse software ",
        geography=["UK", "United Kingdom"],
        exclusions=["payments", "consumer"],
    )
    calls = {"embed": 0, "query": 0}

    async def fake_embed(text: str) -> list[float]:
        calls["embed"] += 1
        return [0.4, 0.5, 0.6]

    async def fake_query(embedding: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        calls["query"] += 1
        return []

    monkeypatch.setattr(retrieval_wrapper._vector_store, "embed", fake_embed)
    monkeypatch.setattr(retrieval_wrapper._vector_store, "query", fake_query)
    retrieval_wrapper._embedding_cache._data.clear()
    retrieval_wrapper._retrieval_cache._data.clear()

    await retrieval_wrapper.retrieve_with_retry(query_a, top_k=5, trace_id="t1")
    await retrieval_wrapper.retrieve_with_retry(query_b, top_k=5, trace_id="t2")

    assert calls["query"] == 1
