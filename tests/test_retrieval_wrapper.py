from __future__ import annotations

import pytest

from app.retrieval import RetrievalError
from app.schemas import QueryPayload
from app.services import retrieval_wrapper


@pytest.mark.asyncio
async def test_retrieve_with_retry_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    query = QueryPayload(query_text="warehouse software")
    calls = {"count": 0}

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2]

    async def fake_query(embedding: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RetrievalError("boom")
        return [{"id": "1", "company_name": "A", "country": "UK", "long_offering": "x", "score": 0.8}]

    monkeypatch.setattr(retrieval_wrapper._vector_store, "embed", fake_embed)
    monkeypatch.setattr(retrieval_wrapper._vector_store, "query", fake_query)
    retrieval_wrapper._embedding_cache._data.clear()
    retrieval_wrapper._retrieval_cache._data.clear()

    results = await retrieval_wrapper.retrieve_with_retry(query, top_k=5, trace_id="t1")
    assert len(results) == 1
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_retrieve_with_retry_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    query = QueryPayload(query_text="warehouse software")

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2]

    async def fake_query(embedding: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        raise RetrievalError("always fails")

    monkeypatch.setattr(retrieval_wrapper._vector_store, "embed", fake_embed)
    monkeypatch.setattr(retrieval_wrapper._vector_store, "query", fake_query)
    retrieval_wrapper._embedding_cache._data.clear()
    retrieval_wrapper._retrieval_cache._data.clear()

    with pytest.raises(RetrievalError, match="trace_id=t2"):
        await retrieval_wrapper.retrieve_with_retry(query, top_k=5, trace_id="t2")
