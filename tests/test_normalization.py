from __future__ import annotations

from app.schemas import QueryPayload
from app.utils.normalization import normalized_query_key


def test_normalized_query_key_normalizes_and_sorts_fields() -> None:
    query = QueryPayload(
        query_text="  Industrial Software  ",
        geography=["UK", " united kingdom "],
        exclusions=["Payments", " consumer "],
    )

    assert normalized_query_key(query) == "industrial software|uk,united kingdom|consumer,payments"


def test_normalized_query_key_skips_empty_values() -> None:
    query = QueryPayload(query_text="A", geography=["", " US "], exclusions=["", " fintech "])
    assert normalized_query_key(query) == "a|us|fintech"
