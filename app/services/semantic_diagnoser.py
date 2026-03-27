from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.schemas import QualityMetrics, QueryPayload, SearchResponse
from app.utils.json_contract import parse_json_contract


class _DiagnosisCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    async def set(self, key: str, value: dict[str, Any]) -> None:
        async with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)


class SemanticDiagnoser:
    def __init__(self) -> None:
        self._cache = _DiagnosisCache(settings.diagnosis_cache_size)
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def diagnose(
        self,
        *,
        message: str,
        query: QueryPayload,
        response: SearchResponse,
        metrics: QualityMetrics,
    ) -> dict[str, Any]:
        evidence = self._build_evidence_summary(query=query, response=response, metrics=metrics)
        cache_key = f"{settings.prompt_version}|{hash(evidence)}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            diagnosis = self._heuristic_diagnosis(message=message, query=query, metrics=metrics)
            await self._cache.set(cache_key, diagnosis)
            return diagnosis

        diagnosis = await self._llm_diagnosis(message=message, query=query, evidence=evidence)
        await self._cache.set(cache_key, diagnosis)
        return diagnosis

    def _build_evidence_summary(
        self,
        *,
        query: QueryPayload,
        response: SearchResponse,
        metrics: QualityMetrics,
    ) -> str:
        top_results = [
            {
                "company_name": result.company_name,
                "country": result.country,
                "score": result.score,
                "excerpt": result.long_offering[:120],
            }
            for result in response.results[:5]
        ]
        return json.dumps(
            {
                "query": query.model_dump(),
                "metrics": metrics.model_dump(),
                "drop_reasons": response.diagnostics.drop_reasons,
                "top_results": top_results,
            },
            sort_keys=True,
        )

    async def _llm_diagnosis(
        self,
        *,
        message: str,
        query: QueryPayload,
        evidence: str,
    ) -> dict[str, Any]:
        assert self._client is not None
        prompt = (
            "You diagnose search result quality for company matching. "
            "Return JSON with keys: intent_clarity, primary_failure_mode, secondary_failure_modes, "
            "observed_patterns, suggested_actions, proposed_query_text, proposed_geography, "
            "proposed_exclusions, confidence."
        )
        try:
            completion = await self._client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Original message: {message}\n"
                            f"Current query: {query.model_dump_json()}\n"
                            f"Evidence: {evidence}"
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = parse_json_contract(content)
            if "_parse_error" in parsed:
                return self._heuristic_diagnosis(message=message, query=query, metrics=QualityMetrics())
            return parsed.get("diagnosis", parsed)
        except Exception:
            return self._heuristic_diagnosis(message=message, query=query, metrics=QualityMetrics())

    def _heuristic_diagnosis(
        self,
        *,
        message: str,
        query: QueryPayload,
        metrics: QualityMetrics,
    ) -> dict[str, Any]:
        primary_failure_mode = "good"
        suggested_actions: list[str] = []

        if metrics.reranked_count == 0 or metrics.top_score < 0.30:
            primary_failure_mode = "low_signal"
            suggested_actions = ["broaden_domain"]
        elif query.geography and metrics.geo_match_rate < 0.70:
            primary_failure_mode = "geo_mismatch"
            suggested_actions = ["add_geography_constraint"]
        elif query.exclusions and metrics.exclusion_hit_rate < 0.10:
            primary_failure_mode = "exclusion_failure"
            suggested_actions = ["add_exclusion", "strengthen_exclusion"]
        elif metrics.score_spread < 0.08:
            primary_failure_mode = "too_broad"
            suggested_actions = ["narrow_domain"]

        return {
            "intent_clarity": "medium",
            "primary_failure_mode": primary_failure_mode,
            "secondary_failure_modes": [],
            "observed_patterns": [f"top_score={metrics.top_score:.3f}", f"top_k_overlap={metrics.top_k_overlap:.3f}"],
            "suggested_actions": suggested_actions or ["rewrite_query_text"],
            "proposed_query_text": query.query_text or message.strip(),
            "proposed_geography": query.geography,
            "proposed_exclusions": query.exclusions,
            "confidence": 0.55,
        }
