from __future__ import annotations

import re

from app.config import settings
from app.retrieval import CompanyResult
from app.schemas import QueryPayload, SearchResult

class Reranker:
    def rerank(
        self,
        candidates: list[CompanyResult],
        query: QueryPayload,
        top_k: int,
        offset: int = 0,
    ) -> list[SearchResult]:
        if top_k <= 0 or offset < 0:
            return []

        scored_results: list[SearchResult] = []
        query_terms = self._tokenize(query.query_text)
        geography_terms = self._expand_geography_terms(query.geography)

        for candidate in candidates:
            geography_bonus = 1.0 if candidate.country.lower().strip() in geography_terms else 0.0
            keyword_density = self._keyword_density(query_terms, candidate.long_offering)
            name_match_bonus = self._name_match_bonus(query_terms, candidate.company_name)
            final_score = (
                settings.weight_vector * candidate.score
                + settings.weight_geography * geography_bonus
                + settings.weight_keyword_density * keyword_density
                + settings.weight_name_match * name_match_bonus
            )
            scored_results.append(
                SearchResult(
                    id=candidate.id,
                    company_name=candidate.company_name,
                    country=candidate.country,
                    score=round(final_score, 6),
                    score_components={
                        "vector": round(candidate.score, 6),
                        "geography_bonus": round(geography_bonus, 6),
                        "keyword_density": round(keyword_density, 6),
                        "name_match_bonus": round(name_match_bonus, 6),
                    },
                    long_offering=candidate.long_offering,
                )
            )

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[offset : offset + top_k]

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    def _keyword_density(self, query_terms: set[str], offering: str) -> float:
        if not query_terms:
            return 0.0
        offering_terms = self._tokenize(offering)
        if not offering_terms:
            return 0.0
        return len(query_terms & offering_terms) / len(query_terms)

    def _name_match_bonus(self, query_terms: set[str], company_name: str) -> float:
        if not query_terms:
            return 0.0
        name_terms = self._tokenize(company_name)
        return 1.0 if query_terms & name_terms else 0.0

    def _expand_geography_terms(self, values: list[str]) -> set[str]:
        aliases = {
            "uk": {"uk", "united kingdom", "england", "scotland", "wales", "britain"},
            "united kingdom": {"uk", "united kingdom", "england", "scotland", "wales", "britain"},
            "us": {"us", "usa", "united states", "united states of america"},
            "usa": {"us", "usa", "united states", "united states of america"},
            "united states": {"us", "usa", "united states", "united states of america"},
        }
        expanded: set[str] = set()
        for value in values:
            normalized = value.lower().strip()
            if not normalized:
                continue
            expanded.update(aliases.get(normalized, {normalized}))
        return expanded
