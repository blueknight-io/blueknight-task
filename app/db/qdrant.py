from functools import lru_cache

from qdrant_client import QdrantClient

from app.settings import get_settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    s = get_settings()
    if not s.vector_db_url or not s.vector_db_api_key:
        raise EnvironmentError("VECTOR_DB_URL and VECTOR_DB_API_KEY must be set in .env")
    return QdrantClient(url=s.vector_db_url, api_key=s.vector_db_api_key)


def collection_name() -> str:
    return get_settings().vector_db_collection


def search_vectors(
    query_vector: list[float],
    limit: int,
    *,
    score_threshold: float | None = None,
) -> list[dict]:
    client = get_client()
    hits = client.query_points(
        collection_name=collection_name(),
        query=query_vector,
        limit=limit,
        score_threshold=score_threshold,
    ).points
    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "rank": hit.payload.get("rank"),
            "company_name": hit.payload.get("company_name", ""),
            "country": hit.payload.get("country", ""),
            "long_offering": hit.payload.get("long_offering", ""),
        }
        for hit in hits
    ]