from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.schemas import QualityMetrics, QueryPayload
from app.utils.normalization import normalized_query_key


@dataclass
class BestState:
    iteration: int
    query: QueryPayload
    metrics: QualityMetrics


@dataclass
class AgentState:
    original_message: str
    history: list[dict]
    current_query: QueryPayload
    iteration: int = 0
    previous_queries: list[QueryPayload] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    previous_diagnostics: list[dict] = field(default_factory=list)
    top_results_history: list[list[str]] = field(default_factory=list)
    failure_mode_history: list[str] = field(default_factory=list)
    improvement_history: list[float] = field(default_factory=list)
    user_supplied_geography: list[str] = field(default_factory=list)
    best_state_so_far: BestState | None = None
    stop_reason: str = ""
    no_improvement_counter: int = 0


@dataclass
class ControllerDecision:
    action: str
    failure_mode: str
    rationale: str
    should_stop: bool = False


class Controller:
    def should_invoke_diagnoser(self, *, iteration: int, metrics: QualityMetrics, query: QueryPayload) -> bool:
        if iteration == 1:
            return True
        if metrics.top_score < 0.78:
            return True
        if metrics.score_spread < 0.08:
            return True
        if 0.35 <= metrics.top_k_overlap < 0.80:
            return True
        if query.geography and metrics.geo_match_rate < 0.70:
            return True
        if metrics.filtered_count / max(metrics.raw_count, 1) > 0.35:
            return True
        if query.exclusions and metrics.exclusion_hit_rate < 0.10:
            return True
        return metrics.improvement_score < 0.05

    def compute_improvement(
        self,
        previous: QualityMetrics | None,
        current: QualityMetrics,
        *,
        has_geography: bool,
    ) -> float:
        if previous is None:
            return 0.0
        top_score_delta = current.top_score - previous.top_score
        score_spread_delta = current.score_spread - previous.score_spread
        if has_geography:
            geo_delta = current.geo_match_rate - previous.geo_match_rate
            return (
                0.50 * top_score_delta
                + 0.30 * score_spread_delta
                + 0.20 * geo_delta
            )
        return 0.65 * top_score_delta + 0.35 * score_spread_delta

    def update_best_state(self, state: AgentState, metrics: QualityMetrics) -> None:
        best = state.best_state_so_far
        if best is None or metrics.top_score > best.metrics.top_score:
            state.best_state_so_far = BestState(
                iteration=state.iteration,
                query=state.current_query.model_copy(deep=True),
                metrics=metrics.model_copy(deep=True),
            )

    def choose_action(
        self,
        *,
        state: AgentState,
        query_before: QueryPayload,
        metrics: QualityMetrics,
        diagnosis: dict,
    ) -> ControllerDecision:
        current_key = normalized_query_key(state.current_query)
        previous_key = normalized_query_key(query_before)
        suggested_actions = diagnosis.get("suggested_actions", [])
        failure_mode = diagnosis.get("primary_failure_mode", "partial")

        if metrics.top_score >= 0.85 and metrics.score_spread >= 0.12 and metrics.top_k_overlap >= 0.80:
            return ControllerDecision(
                action="stop_success",
                failure_mode="good",
                rationale="Results are strong and stable; stopping.",
                should_stop=True,
            )

        if state.no_improvement_counter >= settings.max_no_improvement_iterations:
            return ControllerDecision(
                action="stop_no_improvement",
                failure_mode="stalled",
                rationale="No meaningful improvement across iterations; stopping.",
                should_stop=True,
            )

        if metrics.improvement_score <= -0.03 and state.best_state_so_far is not None:
            return ControllerDecision(
                action="revert_to_best_query",
                failure_mode="regression",
                rationale="Current iteration regressed; reverting to best known query.",
            )

        if not metrics.reranked_count:
            if state.current_query.geography:
                return ControllerDecision(
                    action="remove_geography_constraint",
                    failure_mode="too_narrow",
                    rationale="No results after filtering; relaxing geography if agent-added.",
                )
            return ControllerDecision(
                action="stop_exhausted",
                failure_mode="low_signal",
                rationale="No usable results after refinement attempts.",
                should_stop=True,
            )

        if failure_mode == "geo_mismatch" and current_key == previous_key:
            return ControllerDecision(
                action="add_geography_constraint",
                failure_mode=failure_mode,
                rationale="Geography mismatch remains high; adding explicit geography.",
            )

        if failure_mode == "exclusion_failure" and current_key == previous_key:
            return ControllerDecision(
                action="add_exclusion",
                failure_mode=failure_mode,
                rationale="Exclusions are not filtering enough; strengthening them.",
            )

        if failure_mode in {"too_broad", "wrong_subsector"} and "narrow_domain" in suggested_actions:
            return ControllerDecision(
                action="narrow_domain",
                failure_mode=failure_mode,
                rationale="Results are broad; narrowing the query.",
            )

        if failure_mode in {"too_narrow", "low_signal"} and "broaden_domain" in suggested_actions:
            return ControllerDecision(
                action="broaden_domain",
                failure_mode=failure_mode,
                rationale="Signals are too narrow; broadening the query.",
            )

        return ControllerDecision(
            action="rewrite_query_text",
            failure_mode=failure_mode,
            rationale="Applying query rewrite for another retrieval pass.",
        )

    def apply_action(
        self,
        *,
        state: AgentState,
        decision: ControllerDecision,
        diagnosis: dict,
    ) -> QueryPayload:
        updated = state.current_query.model_copy(deep=True)
        proposed_query_text = str(diagnosis.get("proposed_query_text", "")).strip()
        proposed_geography = self._normalize_list(diagnosis.get("proposed_geography", []))
        proposed_exclusions = self._normalize_list(diagnosis.get("proposed_exclusions", []))

        if decision.action in {"rewrite_query_text", "narrow_domain"} and proposed_query_text:
            updated.query_text = proposed_query_text
        elif decision.action == "broaden_domain":
            if proposed_query_text:
                updated.query_text = proposed_query_text
            elif len(updated.exclusions) > 1:
                updated.exclusions = updated.exclusions[:-1]
            elif updated.geography and not self._is_user_supplied_geography(state, updated.geography):
                updated.geography = []
        elif decision.action == "add_geography_constraint" and proposed_geography:
            updated.geography = proposed_geography
        elif decision.action == "remove_geography_constraint":
            if updated.geography and not self._is_user_supplied_geography(state, updated.geography):
                updated.geography = []
        elif decision.action == "add_exclusion":
            updated.exclusions = sorted({*updated.exclusions, *proposed_exclusions})
        elif decision.action == "strengthen_exclusion":
            updated.exclusions = sorted({*updated.exclusions, *proposed_exclusions})
        elif decision.action == "revert_to_best_query" and state.best_state_so_far is not None:
            updated = state.best_state_so_far.query.model_copy(deep=True)

        state.current_query = updated
        return updated

    def _normalize_list(self, values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        return sorted({str(value).strip() for value in values if str(value).strip()})

    def _is_user_supplied_geography(self, state: AgentState, geography: list[str]) -> bool:
        return sorted(state.user_supplied_geography) == sorted(geography)
