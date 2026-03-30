from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.schemas import QualityMetrics, QueryPayload, SearchResponse
from app.utils.json_contract import parse_json_contract
from app.utils.normalization import normalized_query_key


@dataclass
class ReactStep:
    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)
    stop: bool = False
    stop_reason: str = ""


@dataclass
class ReactState:
    original_message: str
    history: list[dict[str, Any]]
    current_query: QueryPayload
    previous_queries: list[QueryPayload] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    top_results_history: list[list[str]] = field(default_factory=list)
    improvement_history: list[float] = field(default_factory=list)
    no_improvement_counter: int = 0


class _StepCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, ReactStep] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> ReactStep | None:
        async with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    async def set(self, key: str, value: ReactStep) -> None:
        async with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)


class ReactAgent:
    VALID_ACTIONS = {"search_run", "update_query", "stop"}

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._cache = _StepCache(settings.diagnosis_cache_size)

    async def next_step(
        self,
        *,
        state: ReactState,
        last_response: SearchResponse | None,
        last_metrics: QualityMetrics | None,
        iteration: int,
    ) -> ReactStep:
        observation = self._build_observation(
            state=state,
            last_response=last_response,
            last_metrics=last_metrics,
            iteration=iteration,
        )
        cache_key = f"{settings.prompt_version}|react|{hash(observation)}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            step = self._heuristic_step(
                state=state,
                last_response=last_response,
                last_metrics=last_metrics,
                iteration=iteration,
            )
            await self._cache.set(cache_key, step)
            return step

        step = await self._llm_step(observation=observation, state=state)
        await self._cache.set(cache_key, step)
        return step

    def validate_step(
        self,
        *,
        step: ReactStep,
        state: ReactState,
        iteration: int,
    ) -> ReactStep:
        if step.action not in self.VALID_ACTIONS:
            return ReactStep(
                thought="Model produced an invalid action; falling back to update_query.",
                action="update_query",
                action_input={"query_text": state.current_query.query_text or state.original_message},
            )

        if step.stop and step.action != "stop":
            step.action = "stop"

        if iteration >= settings.max_no_improvement_iterations + 1 and step.action == "update_query":
            proposed_query = str(step.action_input.get("query_text", "")).strip()
            if proposed_query and proposed_query.lower().strip() == state.current_query.query_text.lower().strip():
                return ReactStep(
                    thought="Repeated same query without improvement; stopping deterministically.",
                    action="stop",
                    stop=True,
                    stop_reason="No meaningful improvement across repeated refinement steps.",
                )
        return step

    def apply_query_update(self, current_query: QueryPayload, action_input: dict[str, Any]) -> QueryPayload:
        updated = current_query.model_copy(deep=True)
        query_text = str(action_input.get("query_text", "")).strip()
        if query_text:
            updated.query_text = query_text

        geography = self._normalize_list(action_input.get("geography", []))
        exclusions = self._normalize_list(action_input.get("exclusions", []))
        if geography:
            updated.geography = geography
        if exclusions:
            updated.exclusions = exclusions
        return updated

    def _build_observation(
        self,
        *,
        state: ReactState,
        last_response: SearchResponse | None,
        last_metrics: QualityMetrics | None,
        iteration: int,
    ) -> str:
        top_results = []
        if last_response is not None:
            top_results = [
                {
                    "company_name": result.company_name,
                    "country": result.country,
                    "score": result.score,
                    "excerpt": result.long_offering[:120],
                }
                for result in last_response.results[:5]
            ]

        return json.dumps(
            {
                "iteration": iteration,
                "message": state.original_message,
                "current_query": state.current_query.model_dump(),
                "previous_actions": state.previous_actions[-3:],
                "previous_queries": [query.model_dump() for query in state.previous_queries[-2:]],
                "last_metrics": last_metrics.model_dump() if last_metrics else None,
                "last_drop_reasons": last_response.diagnostics.drop_reasons if last_response else {},
                "top_results": top_results,
            },
            sort_keys=True,
        )

    async def _llm_step(self, *, observation: str, state: ReactState) -> ReactStep:
        assert self._client is not None
        system_prompt = (
            "You are a ReAct-style search refinement agent. "
            "You receive the original user request, current structured query, recent search observations, "
            "and must return exactly one JSON object with keys: thought, action, action_input, stop, stop_reason. "
            "Valid actions are: search_run, update_query, stop. "
            "Use natural-language understanding of the user's message first. "
            "When rewriting, make the query more semantically specific, not just shorter. "
            "Always keep the response as valid JSON."
        )
        try:
            completion = await self._client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = parse_json_contract(content)
            if "_parse_error" in parsed:
                return self._heuristic_step(
                    state=state,
                    last_response=None,
                    last_metrics=None,
                    iteration=len(state.previous_actions) + 1,
                )
            return self._coerce_step(parsed)
        except Exception:
            return self._heuristic_step(
                state=state,
                last_response=None,
                last_metrics=None,
                iteration=len(state.previous_actions) + 1,
            )

    def _heuristic_step(
        self,
        *,
        state: ReactState,
        last_response: SearchResponse | None,
        last_metrics: QualityMetrics | None,
        iteration: int,
    ) -> ReactStep:
        if iteration == 1:
            return ReactStep(
                thought="Start by translating the natural-language request into a more retrieval-friendly operational query.",
                action="update_query",
                action_input=self._expanded_query(state.original_message, state.current_query),
            )

        if last_metrics is None:
            return ReactStep(
                thought="No prior metrics available; run search.",
                action="search_run",
            )

        if last_metrics.top_score >= 0.85 and last_metrics.score_spread >= 0.12 and last_metrics.top_k_overlap >= 0.80:
            return ReactStep(
                thought="Results look stable and relevant enough to stop.",
                action="stop",
                stop=True,
                stop_reason="Top results stabilized with acceptable score and spread.",
            )

        if state.no_improvement_counter >= settings.max_no_improvement_iterations:
            return ReactStep(
                thought="Repeated refinement is not improving the results enough.",
                action="stop",
                stop=True,
                stop_reason="No meaningful improvement across repeated refinement steps.",
            )

        if self._looks_generic_business_results(last_response):
            return ReactStep(
                thought="Results are too generic and CRM-heavy; rewrite toward the user's operational intent.",
                action="update_query",
                action_input=self._expanded_query(state.original_message, state.current_query),
            )

        return ReactStep(
            thought="Run another search pass using the current refined query.",
            action="search_run",
        )

    def _expanded_query(self, message: str, current_query: QueryPayload) -> dict[str, Any]:
        lowered = message.lower()
        query_text = current_query.query_text or message.strip()
        geography = current_query.geography
        exclusions = current_query.exclusions

        if "warehouse" in lowered or "logistics" in lowered:
            query_text = (
                "warehouse management software, warehouse operations, inventory operations, "
                "fulfillment, distribution center software, logistics operators in UK"
            )
            if not geography and ("uk" in lowered or "united kingdom" in lowered):
                geography = ["United Kingdom"]
        elif "fintech" in lowered and "payment" in lowered:
            query_text = "fintech software excluding payment processing and payments infrastructure"
            exclusions = sorted({*exclusions, "payments", "payment processing"})
        elif "frontline" in lowered or "onboarding" in lowered:
            query_text = "software solving onboarding inefficiency for frontline teams and field workers"

        return {
            "query_text": query_text,
            "geography": geography,
            "exclusions": exclusions,
        }

    def _looks_generic_business_results(self, response: SearchResponse | None) -> bool:
        if response is None or not response.results:
            return False
        generic_terms = ("crm", "sales", "marketing", "customer relationship management")
        generic_count = sum(
            1
            for result in response.results[:5]
            if any(term in result.long_offering.lower() for term in generic_terms)
        )
        return generic_count >= 3

    def _coerce_step(self, parsed: dict[str, Any]) -> ReactStep:
        return ReactStep(
            thought=str(parsed.get("thought", "")).strip() or "Proceeding with the next step.",
            action=str(parsed.get("action", "search_run")).strip(),
            action_input=parsed.get("action_input", {}) if isinstance(parsed.get("action_input", {}), dict) else {},
            stop=bool(parsed.get("stop", False)),
            stop_reason=str(parsed.get("stop_reason", "")).strip(),
        )

    def _normalize_list(self, values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        return sorted({str(value).strip() for value in values if str(value).strip()})
