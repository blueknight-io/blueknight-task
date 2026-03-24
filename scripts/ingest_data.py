"""Ingestion script: load Excel data into Qdrant with embeddings."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.models.embedder import embed_texts
from app.settings import get_settings


def load_data(data_path: Path) -> pd.DataFrame:
    print(f"Reading {data_path}...")
    df = pd.read_excel(data_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def setup_collection(client: QdrantClient, collection_name: str) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )


async def embed_all_documents(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    print(f"Embedding {len(texts)} documents...")
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} docs)...")
        
        embeddings = await embed_texts(batch_texts, max_retries=5)
        all_embeddings.extend(embeddings)
        
        if i + batch_size < len(texts):
            await asyncio.sleep(1)
    
    return all_embeddings


def upsert_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    df: pd.DataFrame,
    embeddings: list[list[float]],
    batch_size: int = 100,
) -> None:
    print(f"Upserting {len(embeddings)} documents to Qdrant...")
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Upserting batch {batch_num}/{total_batches} ({len(batch_df)} docs)...")
        
        points = [
            PointStruct(
                id=int(row["consolidated_id"]),
                vector=embeddings[i + idx],
                payload={
                    "rank": int(row["rank"]),
                    "company_name": str(row["company_name"]),
                    "country": str(row["country"]),
                    "long_offering": str(row["long_offering"]),
                },
            )
            for idx, (_, row) in enumerate(batch_df.iterrows())
        ]
        client.upsert(collection_name=collection_name, points=points)


async def main() -> None:
    settings = get_settings()
    if not settings.vector_db_url or not settings.vector_db_api_key:
        print("Error: VECTOR_DB_URL and VECTOR_DB_API_KEY must be set in .env")
        sys.exit(1)
    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY must be set in .env")
        sys.exit(1)

    data_path = Path(__file__).parent.parent / "data" / "company_1000_data.xlsx"
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    df = load_data(data_path)
    
    client = QdrantClient(url=settings.vector_db_url, api_key=settings.vector_db_api_key)
    collection_name = settings.vector_db_collection
    
    setup_collection(client, collection_name)
    
    texts = df["long_offering"].fillna("").astype(str).tolist()
    embeddings = await embed_all_documents(texts)
    
    upsert_to_qdrant(client, collection_name, df, embeddings)

    print(f"✓ Ingestion complete. {len(df)} companies indexed in {collection_name}.")


if __name__ == "__main__":
    asyncio.run(main())
