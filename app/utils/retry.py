"""Retry wrapper for mock_retrieve with backoff and timeout."""

from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

from app.retrieval import CompanyResult, RetrievalError

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    backoff_ms: int = 100,
    timeout_s: float = 5.0,
) -> T:
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout_s)
        except (RetrievalError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt < max_attempts - 1:
                await asyncio.sleep((backoff_ms * (2**attempt)) / 1000.0)
    raise last_err or RuntimeError("Retry exhausted")


async def retrieve_with_retry(query: str, top_k: int) -> list[CompanyResult]:
    from app.retrieval import mock_retrieve

    return await with_retry(lambda: mock_retrieve(query, top_k), timeout_s=10.0)
