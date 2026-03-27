from __future__ import annotations

import re

from app.config import settings
from app.retrieval import CompanyResult


def _normalize_terms(values: list[str]) -> list[str]:
    return [value.strip().lower() for value in values if value and value.strip()]


def _candidate_exclusion_patterns(term: str) -> list[str]:
    patterns = [term]
    if term.endswith("s") and len(term) > 3:
        patterns.append(term[:-1])
    elif len(term) > 3:
        patterns.append(f"{term}s")
    return patterns


def _expand_geography_terms(values: list[str]) -> set[str]:
    aliases = {
        "uk": {"uk", "united kingdom", "england", "scotland", "wales", "britain"},
        "united kingdom": {
            "uk",
            "united kingdom",
            "england",
            "scotland",
            "wales",
            "britain",
        },
        "us": {"us", "usa", "united states", "united states of america"},
        "usa": {"us", "usa", "united states", "united states of america"},
        "united states": {"us", "usa", "united states", "united states of america"},
    }
    expanded: set[str] = set()
    for value in _normalize_terms(values):
        expanded.update(aliases.get(value, {value}))
    return expanded


class PostFilter:
    def filter(
        self,
        candidates: list[CompanyResult],
        geography: list[str],
        exclusions: list[str],
    ) -> tuple[list[CompanyResult], dict[str, int], int]:
        kept: list[CompanyResult] = []
        drop_reasons: dict[str, int] = {}
        geography_terms = _expand_geography_terms(geography)
        exclusion_terms = _normalize_terms(exclusions)

        for candidate in candidates:
            reasons: set[str] = set()
            if candidate.score < settings.score_floor:
                reasons.add("low_vector_score")
            if geography_terms and not self._matches_geography(candidate, geography_terms):
                reasons.add("geography_mismatch")
            if exclusion_terms and self._contains_exclusion(candidate.long_offering, exclusion_terms):
                reasons.add("exclude_term")

            if reasons:
                for reason in reasons:
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue
            kept.append(candidate)

        return kept, drop_reasons, len(candidates) - len(kept)

    def _matches_geography(self, candidate: CompanyResult, geography_terms: set[str]) -> bool:
        country = candidate.country.lower().strip()
        if country in geography_terms:
            return True
        offering = candidate.long_offering.lower()
        return any(term in offering for term in geography_terms)

    def _contains_exclusion(self, text: str, exclusions: list[str]) -> bool:
        lower_text = text.lower()
        return any(
            re.search(r"\b" + re.escape(pattern) + r"\b", lower_text) is not None
            for term in exclusions
            for pattern in _candidate_exclusion_patterns(term)
        )
