from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import settings


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def _normalize_geography_terms(values: list[str] | None) -> list[str]:
    if not values:
        return []

    aliases: dict[str, list[str]] = {
        "uk": [
            "United Kingdom",
            "UK",
            "Great Britain",
            "Britain",
            "England",
            "Scotland",
            "Wales",
            "Northern Ireland",
        ],
        "united kingdom": [
            "United Kingdom",
            "UK",
            "Great Britain",
            "Britain",
            "England",
            "Scotland",
            "Wales",
            "Northern Ireland",
        ],
        "us": ["United States", "USA", "US", "United States of America"],
        "usa": ["United States", "USA", "US", "United States of America"],
        "united states": ["United States", "USA", "US", "United States of America"],
    }

    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized:
            continue
        expanded = aliases.get(normalized, [value.strip()])
        for term in expanded:
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _coerce_point_id(value: Any) -> int | str:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(int(digest, 16))


class VectorStoreClient:
    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._client_lock = threading.Lock()
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._collection_name = settings.qdrant_collection
        self._qdrant_path = Path(settings.qdrant_path)
        self._qdrant_path.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = QdrantClient(path=str(self._qdrant_path))
        return self._client

    def _ensure_collection_sync(self) -> None:
        client = self._get_client()
        try:
            exists = client.collection_exists(collection_name=self._collection_name)
        except Exception:
            exists = False
        if exists:
            return
        client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=settings.embed_dims,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    async def ensure_collection(self) -> None:
        await asyncio.to_thread(self._ensure_collection_sync)

    async def embed(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")

        response = await self._openai.embeddings.create(
            model=settings.embed_model,
            input=texts,
        )
        vectors = [_normalize_vector(list(item.embedding)) for item in response.data]
        return vectors

    async def upsert(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return

        await self.ensure_collection()
        points = [
            qdrant_models.PointStruct(
                id=_coerce_point_id(item["id"]),
                vector=_normalize_vector(list(item["embedding"])),
                payload={
                    "id": str(item["id"]),
                    "company_name": item.get("company_name", ""),
                    "country": item.get("country", ""),
                    "long_offering": item.get("long_offering", ""),
                },
            )
            for item in items
        ]
        client = self._get_client()
        await asyncio.to_thread(
            client.upsert,
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def _build_filter(
        self, filters: dict[str, Any] | None
    ) -> qdrant_models.Filter | None:
        if not filters:
            return None

        geography = filters.get("geography") or []
        terms = _normalize_geography_terms(list(geography))
        if not terms:
            return None

        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="country",
                    match=qdrant_models.MatchAny(any=terms),
                )
            ]
        )

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        await self.ensure_collection()
        client = self._get_client()
        query_filter = self._build_filter(filters)
        search_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query": _normalize_vector(list(embedding)),
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        if query_filter is not None:
            search_kwargs["query_filter"] = query_filter

        response = await asyncio.to_thread(client.query_points, **search_kwargs)
        results = response.points
        return [
            {
                "id": str(point.id),
                "company_name": (point.payload or {}).get("company_name", ""),
                "country": (point.payload or {}).get("country", ""),
                "long_offering": (point.payload or {}).get("long_offering", ""),
                "score": float(point.score or 0.0),
            }
            for point in results
        ]
