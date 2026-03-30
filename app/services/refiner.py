from __future__ import annotations

import time

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
from app.services.react_agent import ReactAgent, ReactState, ReactStep
from app.services.search_pipeline import SearchPipeline
from app.utils.logging import get_logger, log_stage
from app.utils.normalization import normalized_query_key


_search_pipeline = SearchPipeline()
_react_agent = ReactAgent()
_logger = get_logger(__name__)


class QueryRefinerAgent:
    async def refine(self, request: RefineRequest) -> RefineResponse:
        current_query = self._starting_query(request)
        state = ReactState(
            original_message=request.message,
            history=request.history,
            current_query=current_query,
        )
        iterations: list[IterationRecord] = []
        last_result_ids: set[str] = set()
        rationale = "Max iterations reached without convergence."
        final_action = "stop_max_iterations"
        previous_metrics: QualityMetrics | None = None
        last_response = None

        for iteration in range(1, request.max_iterations + 1):
            iteration_start = time.perf_counter()
            query_before = current_query.model_copy(deep=True)
            step = await _react_agent.next_step(
                state=state,
                last_response=last_response,
                last_metrics=previous_metrics,
                iteration=iteration,
            )
            step = _react_agent.validate_step(step=step, state=state, iteration=iteration)

            if step.action == "update_query":
                current_query = _react_agent.apply_query_update(current_query, step.action_input)
                state.current_query = current_query
                search_response = await _search_pipeline.run(
                    SearchRequest(
                        query=current_query,
                        top_k_raw=100,
                        top_k_final=10,
                        trace_id=request.trace_id,
                    )
                )
                last_response = search_response
            elif step.action == "search_run":
                search_response = await _search_pipeline.run(
                    SearchRequest(
                        query=current_query,
                        top_k_raw=100,
                        top_k_final=10,
                        trace_id=request.trace_id,
                    )
                )
                last_response = search_response
            else:
                search_response = last_response or await _search_pipeline.run(
                    SearchRequest(
                        query=current_query,
                        top_k_raw=100,
                        top_k_final=10,
                        trace_id=request.trace_id,
                    )
                )
                last_response = search_response

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
            metrics.improvement_score = self._compute_improvement(
                previous=previous_metrics,
                current=metrics,
                has_geography=bool(current_query.geography),
            )
            previous_metrics = metrics.model_copy(deep=True)
            if metrics.improvement_score >= 0.05:
                state.no_improvement_counter = 0
            else:
                state.no_improvement_counter += 1

            state.previous_queries.append(query_before.model_copy(deep=True))
            state.previous_actions.append(step.action)
            state.top_results_history.append(list(current_ids))
            state.improvement_history.append(metrics.improvement_score)

            failure_mode = self._failure_mode_from_step(step)
            if step.stop:
                rationale = step.stop_reason or step.thought
                final_action = step.action
            elif (
                state.no_improvement_counter >= settings.max_no_improvement_iterations
                and normalized_query_key(current_query) == normalized_query_key(query_before)
            ):
                rationale = "No meaningful improvement across repeated refinement steps."
                final_action = "stop"
                step.stop = True
                failure_mode = "stalled"

            iterations.append(
                IterationRecord(
                    iteration=iteration,
                    action=step.action,
                    failure_mode=failure_mode,
                    query_before=query_before,
                    query_after=current_query.model_copy(deep=True),
                    metrics=metrics,
                    rationale=step.stop_reason or step.thought,
                )
            )

            iteration_duration_ms = int((time.perf_counter() - iteration_start) * 1000)
            log_stage(
                _logger,
                trace_id=request.trace_id,
                stage="refiner_iteration",
                duration_ms=iteration_duration_ms,
                item_count=metrics.reranked_count,
                iteration=iteration,
                action=step.action,
                query_before=query_before.query_text,
                query_after=current_query.query_text,
                top_score=metrics.top_score,
                score_spread=metrics.score_spread,
                top_k_overlap=metrics.top_k_overlap,
                improvement_score=metrics.improvement_score,
                should_stop=step.stop,
                stop_reason=(step.stop_reason or rationale or ""),
            )

            if step.stop:
                break

        return RefineResponse(
            refined_query=current_query,
            rationale=rationale,
            actions=default_actions(),
            iterations_used=len(iterations),
            meta=AgentMeta(
                best_iteration=self._best_iteration(iterations),
                final_action=final_action,
                iterations=iterations,
            ),
        )

    def _starting_query(self, request: RefineRequest) -> QueryPayload:
        if request.base_query is not None:
            return request.base_query.model_copy(deep=True)
        return default_query_payload()

    def _top_k_overlap(self, previous_ids: set[str], current_ids: set[str]) -> float:
        if not previous_ids or not current_ids:
            return 0.0
        return len(previous_ids & current_ids) / len(previous_ids | current_ids)

    def _compute_improvement(
        self,
        *,
        previous: QualityMetrics | None,
        current: QualityMetrics,
        has_geography: bool,
    ) -> float:
        if previous is None:
            return 0.0
        top_score_delta = current.top_score - previous.top_score
        score_spread_delta = current.score_spread - previous.score_spread
        if has_geography:
            geo_delta = current.geo_match_rate - previous.geo_match_rate
            return 0.50 * top_score_delta + 0.30 * score_spread_delta + 0.20 * geo_delta
        return 0.65 * top_score_delta + 0.35 * score_spread_delta

    def _failure_mode_from_step(self, step: ReactStep) -> str:
        if step.action == "stop" and step.stop:
            return "completed"
        if step.action == "update_query":
            return "refined"
        if step.action == "search_run":
            return "search_only"
        return "partial"

    def _best_iteration(self, iterations: list[IterationRecord]) -> int:
        if not iterations:
            return 0
        best = max(iterations, key=lambda item: item.metrics.top_score)
        return best.iteration


def default_actions() -> list[Action]:
    return [Action(id="show_results", label="Show results")]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()
