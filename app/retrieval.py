from __future__ import annotations

from dataclasses import dataclass

from app.db.qdrant import search_vectors
from app.models.embedder import embed_query


class RetrievalError(RuntimeError):
    """Transient retrieval failure."""


@dataclass
class CompanyResult:
    id: str
    company_name: str
    country: str
    long_offering: str
    score: float


async def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    """
    Vector search over embedded long_offering values in Qdrant.
    
    Retrieves companies by semantic similarity of their long_offering field.
    May raise RetrievalError on transient failures.
    """
    try:
        query_vector = await embed_query(query)
        hits = search_vectors(query_vector, limit=top_k)
        
        return [
            CompanyResult(
                id=hit["id"],
                company_name=hit["company_name"],
                country=hit["country"],
                long_offering=hit["long_offering"],
                score=hit["score"],
            )
            for hit in hits
        ]
    except Exception as e:
        raise RetrievalError(f"Vector retrieval failed: {e}") from e


async def test_retrieval() -> None:
    """Test retrieval with sample queries."""
    test_queries = [
        ("logistics software", 5),
        ("Vertical SaaS for logistics operators in the UK", 3),
        ("fintech companies", 5),
    ]
    
    for query, top_k in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: '{query}' (top_k={top_k})")
        print('=' * 80)
        
        try:
            results = await mock_retrieve(query, top_k)
            print(f"Found {len(results)} results\n")
            
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.company_name} ({r.country})")
                print(f"   Score: {r.score:.4f}")
                print(f"   Offering: {r.long_offering[:150]}...")
                print()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_retrieval())
