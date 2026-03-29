from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from app.services.vector_store import vector_store

load_dotenv()

_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_EMBEDDING_DIMENSION = 512


class RetrievalError(RuntimeError):
    """Transient retrieval failure."""


@dataclass
class CompanyResult:
    id: str
    company_name: str
    country: str
    long_offering: str
    score: float


def _embed(query: str) -> list[float]:
    response = _openai.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=[query],
        dimensions=_EMBEDDING_DIMENSION,
    )
    return response.data[0].embedding


def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    """
    Simulates vector search over pre-embedded long_offering values.

    Behaviour:
    - ~200-300ms mean latency with occasional spikes
    - ~5% transient failure rate (raises RetrievalError)
    - Not safe to call with unbounded concurrency
    """
    # delay_ms = random.randint(180, 320) + random.choice([0, 0, 0, 200])
    # time.sleep(delay_ms / 1000.0)

    # if random.random() < 0.05:
    #     raise RetrievalError("Transient vector index error")

    # raise NotImplementedError(
    #     "Replace mock_retrieve with real vector retrieval over data/companies.csv"
    # )
    """
    Real vector retrieval over Pinecone-indexed long_offering embeddings.
    Signature kept identical to the original stub.
    """
    try:
        embedding = _embed(query)
        hits = asyncio.run(vector_store.query(embedding, top_k))
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc

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
