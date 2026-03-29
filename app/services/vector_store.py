from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_index = _pc.Index(os.environ.get("PINECONE_INDEX_NAME", "blueknight-companies"))


class VectorStoreClient:
    """
    Placeholder abstraction for vector DB.

    Candidate can back this with Qdrant, Pinecone, FAISS, pgvector, etc.
    """

    async def upsert(self, items: list[dict[str, Any]]) -> None:
        # raise NotImplementedError("Implement VectorStoreClient.upsert")
        """items: list of {id, values, metadata} dicts."""
        _index.upsert(vectors=items)
    
    async def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query Pinecone and return a list of dicts with keys:
        id, score, company_name, country, long_offering.
        """
        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filters:
            kwargs["filter"] = filters

        response = _index.query(**kwargs)

        return [
            {
                "id": match["id"],
                "score": match["score"],
                "company_name": match["metadata"].get("company_name", ""),
                "country": match["metadata"].get("country", ""),
                "long_offering": match["metadata"].get("long_offering", ""),
            }
            for match in response["matches"]
        ]


# Module-level singleton — import this directly where needed
vector_store = VectorStoreClient()

