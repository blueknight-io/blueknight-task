from __future__ import annotations

from app.schemas import QueryPayload


def normalized_query_key(query: QueryPayload) -> str:
    return "|".join(
        [
            query.query_text.lower().strip(),
            ",".join(sorted(g.lower().strip() for g in query.geography if g.strip())),
            ",".join(sorted(e.lower().strip() for e in query.exclusions if e.strip())),
        ]
    )
