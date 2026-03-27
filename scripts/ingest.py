from __future__ import annotations

import asyncio
import csv
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.services.vector_store import VectorStoreClient


CSV_FILENAME = "company_1000_data - Results.csv"


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_corpus_version(csv_path: Path, version_path: Path) -> str:
    corpus_version = sha256(csv_path.read_bytes()).hexdigest()[:12]
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(corpus_version + "\n", encoding="utf-8")
    return corpus_version


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _row_to_item(row: dict[str, str], embedding: list[float]) -> dict[str, Any]:
    company_id = _coerce_text(row.get("Consolidated ID")) or _coerce_text(row.get("Rank"))
    return {
        "id": company_id,
        "company_name": _coerce_text(row.get("Company Name")),
        "country": _coerce_text(row.get("Country")),
        "long_offering": _coerce_text(row.get("Long Offering")),
        "embedding": embedding,
    }


async def ingest() -> None:
    csv_path = PROJECT_ROOT / CSV_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    corpus_version_path = PROJECT_ROOT / settings.corpus_version_path
    corpus_version = _write_corpus_version(csv_path, corpus_version_path)
    rows = _read_csv_rows(csv_path)

    vector_store = VectorStoreClient()
    await vector_store.ensure_collection()

    batch_size = max(1, getattr(settings, "embed_batch_size", 64))
    total = len(rows)
    print(f"Corpus version: {corpus_version}")
    print(f"Rows found: {total}")

    for index in range(0, total, batch_size):
        batch_rows = rows[index : index + batch_size]
        texts = [_coerce_text(row.get("Long Offering")) for row in batch_rows]
        embeddings = await vector_store.embed_batch(texts)
        items = [
            _row_to_item(row, embedding)
            for row, embedding in zip(batch_rows, embeddings, strict=True)
        ]
        await vector_store.upsert(items)
        batch_number = index // batch_size + 1
        print(
            f"Batch {batch_number}: upserted {len(items)} records "
            f"({min(index + batch_size, total)}/{total})"
        )


def main() -> None:
    asyncio.run(ingest())


if __name__ == "__main__":
    main()
