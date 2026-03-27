from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic, TypeVar

from app.config import settings
from app.retrieval import CompanyResult, RetrievalError
from app.schemas import QueryPayload
from app.services.vector_store import VectorStoreClient
from app.utils.normalization import normalized_query_key

T = TypeVar("T")


class _LRUCache(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, T] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    async def set(self, key: str, value: T) -> None:
        async with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


_vector_store = VectorStoreClient()
_embedding_cache: _LRUCache[list[float]] = _LRUCache(settings.embedding_cache_size)
_retrieval_cache: _LRUCache[list[CompanyResult]] = _LRUCache(settings.retrieval_cache_size)
_retrieval_semaphore = asyncio.Semaphore(settings.retrieval_max_concurrency)


def _load_corpus_version() -> str:
    path = Path(settings.corpus_version_path)
    if not path.exists():
        return "missing"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "missing"
    return value or "missing"


def _normalize_input(
    query: QueryPayload | str,
    geography: list[str] | None = None,
) -> QueryPayload:
    if isinstance(query, QueryPayload):
        return query
    return QueryPayload(query_text=query, geography=geography or [])


def _embedding_cache_key(query: QueryPayload) -> str:
    return f"{settings.embed_model}|{query.query_text.lower().strip()}"


def _retrieval_cache_key(query: QueryPayload, top_k: int) -> str:
    return f"{_load_corpus_version()}|{normalized_query_key(query)}|{top_k}"


def _coerce_company_result(item: Any) -> CompanyResult:
    if isinstance(item, CompanyResult):
        return item
    if not isinstance(item, dict):
        raise RetrievalError(f"Unexpected vector result type: {type(item).__name__}")
    try:
        return CompanyResult(
            id=str(item["id"]),
            company_name=str(item.get("company_name", "")),
            country=str(item.get("country", "")),
            long_offering=str(item.get("long_offering", "")),
            score=float(item.get("score", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalError(f"Malformed vector result payload: {exc}") from exc


async def _retrieve_once(query: QueryPayload, top_k: int) -> list[CompanyResult]:
    cache_key = _retrieval_cache_key(query, top_k)
    cached = await _retrieval_cache.get(cache_key)
    if cached is not None:
        return cached

    embedding_key = _embedding_cache_key(query)
    embedding = await _embedding_cache.get(embedding_key)
    if embedding is None:
        embedding = await _vector_store.embed(query.query_text)
        await _embedding_cache.set(embedding_key, embedding)

    async with _retrieval_semaphore:
        raw_results = await _vector_store.query(
            embedding=embedding,
            top_k=top_k,
            filters={"geography": query.geography},
        )

    results = [_coerce_company_result(item) for item in raw_results]
    await _retrieval_cache.set(cache_key, results)
    return results


async def retrieve_with_retry(
    query: QueryPayload | str,
    top_k: int,
    trace_id: str,
    geography: list[str] | None = None,
) -> list[CompanyResult]:
    """
    Retrieval wrapper with bounded concurrency, timeout, retries, and caches.

    The public contract accepts either a QueryPayload or a raw query string so
    later phases can use the same utility without forcing a signature change.
    """
    payload = _normalize_input(query, geography=geography)

    last_error: RetrievalError | None = None
    for attempt in range(1, settings.retrieval_max_retries + 1):
        try:
            return await asyncio.wait_for(
                _retrieve_once(payload, top_k),
                timeout=settings.retrieval_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            last_error = RetrievalError(
                f"Retrieval timed out after {settings.retrieval_timeout_s:.1f}s"
            )
            last_error.__cause__ = exc
        except RetrievalError as exc:
            last_error = exc

        if attempt >= settings.retrieval_max_retries:
            break

        backoff_s = min(0.25 * (2 ** (attempt - 1)), 2.0)
        await asyncio.sleep(backoff_s)

    assert last_error is not None
    raise RetrievalError(f"{last_error} [trace_id={trace_id}]") from last_error
