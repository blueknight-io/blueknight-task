from __future__ import annotations


class Reranker:
    """
    Re-ranks candidates using a weighted combination of signals:

    Score components (each 0.0–1.0, weighted sum):
      - vector_score  (0.50): raw cosine similarity from Pinecone
      - keyword_boost (0.30): fraction of include_keywords found in long_offering
      - geo_boost     (0.20): 1.0 if company country matches a requested geography, else 0.0

    Final score = 0.5 * vector_score + 0.3 * keyword_boost + 0.2 * geo_boost
    """

    def rerank(self, candidates: list[dict], query: dict, top_k: int) -> list[dict]:
        """
        Args:
            candidates: post-filtered list of company dicts with keys
                        id, score, company_name, country, long_offering
            query:      QueryPayload as dict (keys: geography, include_keywords, exclude_terms)
            top_k:      number of results to return

        Returns:
            Top-k candidates sorted by final_score descending,
            each augmented with a `score_components` dict.
        """
        geography: list[str] = [g.lower() for g in query.get("geography", [])]
        include_keywords: list[str] = [
            kw.lower() for kw in query.get("include_keywords", [])
        ]

        scored: list[dict] = []
        for c in candidates:
            vector_score = float(c.get("score", 0.0))
            offering_lower = c.get("long_offering", "").lower()
            country_lower = c.get("country", "").lower()

            # Keyword boost — fraction of include_keywords present in offering text
            if include_keywords:
                hits = sum(1 for kw in include_keywords if kw in offering_lower)
                keyword_boost = hits / len(include_keywords)
            else:
                keyword_boost = 1.0  # no constraint → no penalty

            # Geography boost
            if geography:
                geo_boost = 1.0 if any(g in country_lower for g in geography) else 0.0
            else:
                geo_boost = 1.0  # no constraint → no penalty

            final_score = (
                0.5 * vector_score
                + 0.3 * keyword_boost
                + 0.2 * geo_boost
            )

            scored.append({
                **c,
                "score": round(final_score, 6),
                "score_components": {
                    "vector_score": round(vector_score, 6),
                    "keyword_boost": round(keyword_boost, 6),
                    "geo_boost": round(geo_boost, 6),
                },
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


