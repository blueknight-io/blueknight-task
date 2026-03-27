from __future__ import annotations

import re

from app.config import settings
from app.schemas import (
    Action,
    AgentMeta,
    IterationRecord,
    QualityMetrics,
    QueryPayload,
    RefineRequest,
    RefineResponse,
    SearchRequest,
)
from app.services.controller import AgentState, Controller
from app.services.semantic_diagnoser import SemanticDiagnoser
from app.services.search_pipeline import SearchPipeline
from app.utils.normalization import normalized_query_key


_search_pipeline = SearchPipeline()
_controller = Controller()
_diagnoser = SemanticDiagnoser()


class QueryRefinerAgent:
    async def refine(self, request: RefineRequest) -> RefineResponse:
        current_query = self._starting_query(request)
        state = AgentState(
            original_message=request.message,
            history=request.history,
            current_query=current_query,
            user_supplied_geography=list(current_query.geography),
        )
        iterations: list[IterationRecord] = []
        last_result_ids: set[str] = set()
        rationale = "Max iterations reached without convergence."
        final_action = "stop_max_iterations"
        previous_metrics: QualityMetrics | None = None

        for iteration in range(1, request.max_iterations + 1):
            state.iteration = iteration
            query_before = current_query.model_copy(deep=True)

            if iteration == 1:
                current_query = self._apply_message(current_query, request.message)
                state.current_query = current_query

            search_response = await _search_pipeline.run(
                SearchRequest(
                    query=current_query,
                    top_k_raw=100,
                    top_k_final=10,
                    trace_id=request.trace_id,
                )
            )

            current_ids = {result.id for result in search_response.results}
            overlap = self._top_k_overlap(last_result_ids, current_ids)
            last_result_ids = current_ids
            metrics = QualityMetrics(
                top_score=search_response.diagnostics.top_score,
                score_spread=search_response.diagnostics.score_spread,
                geo_match_rate=search_response.diagnostics.geo_match_rate,
                exclusion_hit_rate=search_response.diagnostics.exclusion_hit_rate,
                top_k_overlap=overlap,
                raw_count=search_response.diagnostics.raw_count,
                filtered_count=search_response.diagnostics.filtered_count,
                reranked_count=search_response.diagnostics.reranked_count,
            )
            metrics.improvement_score = _controller.compute_improvement(
                previous_metrics,
                metrics,
                has_geography=bool(current_query.geography),
            )
            previous_metrics = metrics.model_copy(deep=True)
            if metrics.improvement_score >= 0.05:
                state.no_improvement_counter = 0
            elif metrics.improvement_score < 0.05:
                state.no_improvement_counter += 1

            _controller.update_best_state(state, metrics)

            if _controller.should_invoke_diagnoser(
                iteration=iteration,
                metrics=metrics,
                query=current_query,
            ):
                diagnosis = await _diagnoser.diagnose(
                    message=request.message,
                    query=current_query,
                    response=search_response,
                    metrics=metrics,
                )
            else:
                diagnosis = {
                    "primary_failure_mode": "good",
                    "suggested_actions": ["stop_success"],
                    "proposed_query_text": current_query.query_text,
                    "proposed_geography": current_query.geography,
                    "proposed_exclusions": current_query.exclusions,
                }

            state.previous_diagnostics.append(diagnosis)
            state.previous_queries.append(query_before.model_copy(deep=True))
            state.top_results_history.append(list(current_ids))

            if (
                state.no_improvement_counter >= settings.max_no_improvement_iterations
                and normalized_query_key(current_query) == normalized_query_key(query_before)
            ):
                diagnosis["primary_failure_mode"] = "stalled"

            decision = _controller.choose_action(
                state=state,
                query_before=query_before,
                metrics=metrics,
                diagnosis=diagnosis,
            )
            state.previous_actions.append(decision.action)
            state.failure_mode_history.append(decision.failure_mode)
            state.improvement_history.append(metrics.improvement_score)

            if not decision.should_stop:
                current_query = _controller.apply_action(
                    state=state,
                    decision=decision,
                    diagnosis=diagnosis,
                )
            else:
                rationale = decision.rationale
                final_action = decision.action

            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    action=decision.action,
                    failure_mode=decision.failure_mode,
                    query_before=query_before,
                    query_after=current_query.model_copy(deep=True),
                    metrics=metrics,
                    rationale=decision.rationale,
                )
            )

            if decision.should_stop:
                break

        return RefineResponse(
            refined_query=current_query,
            rationale=rationale,
            actions=default_actions(),
            iterations_used=len(iterations),
            meta=AgentMeta(
                best_iteration=state.best_state_so_far.iteration if state.best_state_so_far else 0,
                final_action=final_action,
                iterations=iterations,
            ),
        )

    def _starting_query(self, request: RefineRequest) -> QueryPayload:
        if request.base_query is not None:
            return request.base_query.model_copy(deep=True)
        return default_query_payload()

    def _apply_message(self, query: QueryPayload, message: str) -> QueryPayload:
        updated = query.model_copy(deep=True)
        stripped = message.strip()
        if stripped:
            updated.query_text = stripped
        if not updated.geography:
            updated.geography = self._extract_geography(message)
        if not updated.exclusions:
            updated.exclusions = self._extract_exclusions(message)
        return updated

    def _extract_geography(self, text: str) -> list[str]:
        lowered = text.lower()
        if "united kingdom" in lowered or re.search(r"\buk\b", lowered):
            return ["United Kingdom"]
        if "united states" in lowered or re.search(r"\busa?\b", lowered):
            return ["United States"]
        return []

    def _extract_exclusions(self, text: str) -> list[str]:
        lowered = text.lower()
        matches = re.findall(r"not focused on ([a-z\s-]+)", lowered)
        exclusions = [match.strip(" .,!") for match in matches if match.strip()]
        if "excluding " in lowered:
            exclusions.extend(
                part.strip(" .,!") for part in lowered.split("excluding ", 1)[1].split(" and ")
            )
        return sorted({value for value in exclusions if value})

    def _top_k_overlap(self, previous_ids: set[str], current_ids: set[str]) -> float:
        if not previous_ids or not current_ids:
            return 0.0
        return len(previous_ids & current_ids) / len(previous_ids | current_ids)


def default_actions() -> list[Action]:
    return [Action(id="show_results", label="Show results")]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()
