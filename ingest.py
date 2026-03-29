"""
Ingestion script: reads company CSV, embeds `long_offering` via OpenAI,
and upserts vectors into a Pinecone serverless index.

Usage:
    python ingest.py              # ingest all rows
    python ingest.py --limit 10   # ingest first N rows (for testing)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

#TODO: pydantic settings could be used here instead of module-level constants, but this is a simple script and the config is only used in a few places, so module-level is fine for now.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# text-embedding-3-small supports Matryoshka reduction;  matches the existing Pinecone index
EMBEDDING_DIMENSION = 512

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "blueknight-companies")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

CSV_PATH = Path(__file__).parent / "company_1000_data - Results.csv"
UPSERT_BATCH_SIZE = 100  # Pinecone recommends <= 100 vectors per upsert call

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

#TODO: OPENAI and Pinecone clients in util could be wrapped in classes for better abstraction and testability, but for this simple script, direct usage is fine.
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_index_exists() -> None:
    """Create the Pinecone serverless index if it does not already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait until the index is ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to become ready ...")
            time.sleep(2)
        print("  Index ready.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists, skipping creation.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI Embeddings API for a batch of texts.
    Returns a list of float vectors in the same order as input.
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSION,  # reduce from default 1536 → 512 via Matryoshka
    )
    # response.data is ordered by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def load_csv(limit: int | None) -> pd.DataFrame:
    """Load and normalise the companies CSV."""
    df = pd.read_csv(CSV_PATH)

    # Normalise column names to match the project schema
    df = df.rename(
        columns={
            "Consolidated ID": "id",
            "Company Name": "company_name",
            "Country": "country",
            "Long Offering": "long_offering",
        }
    )

    # Drop rows with missing bio — nothing to embed
    df = df.dropna(subset=["long_offering"])
    df["id"] = df["id"].astype(str)

    if limit is not None:
        df = df.head(limit)

    return df


def upsert_to_pinecone(df: pd.DataFrame) -> None:
    """
    Embed long_offering in batches then upsert to Pinecone.
    Each vector's metadata stores all fields needed by retrieval.py.
    """
    index = pc.Index(PINECONE_INDEX_NAME)
    rows = df.to_dict(orient="records")
    total = len(rows)

    print(f"Embedding and upserting {total} companies in batches of {UPSERT_BATCH_SIZE} ...")

    for batch_start in range(0, total, UPSERT_BATCH_SIZE):
        batch = rows[batch_start : batch_start + UPSERT_BATCH_SIZE]
        texts = [row["long_offering"] for row in batch]

        # --- Embed ---
        embeddings = embed_texts(texts)

        # --- Build Pinecone vectors ---
        vectors = [
            {
                # Pinecone vector id must be a string; use company id
                "id": row["id"],
                "values": embedding,
                "metadata": {
                    "company_name": str(row.get("company_name", "")),
                    "country": str(row.get("country", "")),
                    # Store truncated long_offering (Pinecone metadata limit: 40KB per vector)
                    "long_offering": row["long_offering"][:2000],
                },
            }
            for row, embedding in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors)

        end = min(batch_start + UPSERT_BATCH_SIZE, total)
        print(f"  Upserted rows {batch_start + 1}–{end} / {total}")

    print("Done. All vectors upserted to Pinecone.", vectors[0] if vectors else "No vectors to upsert.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest companies CSV into Pinecone.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N rows (omit for full dataset).",
    )
    args = parser.parse_args()

    ensure_index_exists()

    df = load_csv(limit=args.limit)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    upsert_to_pinecone(df)


if __name__ == "__main__":
    main()
