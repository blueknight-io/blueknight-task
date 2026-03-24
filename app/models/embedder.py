"""OpenAI text embeddings via HTTP (httpx) with rate limit retry."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from app.settings import get_settings

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODEL = "text-embedding-3-small"


async def embed_texts(
    texts: Sequence[str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 3,
) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    key = api_key or s.openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    m = model or s.openai_embedding_model or DEFAULT_MODEL
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": m, "input": list(texts)}

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(OPENAI_EMBEDDINGS_URL, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("retry-after", 2))
                wait_s = min(retry_after, 2 ** attempt)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_s)
                    last_error = e
                    continue
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

    raise last_error or RuntimeError("Embedding failed after retries")


async def embed_query(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 60.0,
) -> list[float]:
    vectors = await embed_texts(
        [text], model=model, api_key=api_key, timeout_s=timeout_s
    )
    return vectors[0]
