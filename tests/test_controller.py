from __future__ import annotations

import pytest

from app.schemas import QualityMetrics, QueryPayload
from app.services.controller import AgentState, BestState, Controller


def test_controller_stops_on_strong_stable_results() -> None:
    controller = Controller()
    state = AgentState(
        original_message="warehouse software uk",
        history=[],
        current_query=QueryPayload(query_text="warehouse software", geography=["United Kingdom"]),
    )
    metrics = QualityMetrics(
        top_score=0.9,
        score_spread=0.2,
        top_k_overlap=0.85,
        reranked_count=10,
    )

    decision = controller.choose_action(
        state=state,
        query_before=state.current_query,
        metrics=metrics,
        diagnosis={"primary_failure_mode": "good", "suggested_actions": ["stop_success"]},
    )

    assert decision.action == "stop_success"
    assert decision.should_stop is True


def test_controller_stops_after_no_improvement_limit() -> None:
    controller = Controller()
    state = AgentState(
        original_message="warehouse software uk",
        history=[],
        current_query=QueryPayload(query_text="warehouse software"),
        no_improvement_counter=2,
    )
    metrics = QualityMetrics(reranked_count=5)

    decision = controller.choose_action(
        state=state,
        query_before=state.current_query,
        metrics=metrics,
        diagnosis={"primary_failure_mode": "partial", "suggested_actions": []},
    )

    assert decision.action == "stop_no_improvement"
    assert decision.should_stop is True


def test_controller_reverts_on_regression_when_best_state_exists() -> None:
    controller = Controller()
    state = AgentState(
        original_message="warehouse software uk",
        history=[],
        current_query=QueryPayload(query_text="warehouse software", geography=["United Kingdom"]),
        best_state_so_far=BestState(
            iteration=1,
            query=QueryPayload(query_text="warehouse software", geography=["United Kingdom"]),
            metrics=QualityMetrics(top_score=0.85),
        ),
    )
    metrics = QualityMetrics(improvement_score=-0.05, reranked_count=10)

    decision = controller.choose_action(
        state=state,
        query_before=state.current_query,
        metrics=metrics,
        diagnosis={"primary_failure_mode": "regression", "suggested_actions": []},
    )

    assert decision.action == "revert_to_best_query"
    assert decision.should_stop is False


def test_controller_adds_geography_constraint_on_geo_mismatch() -> None:
    controller = Controller()
    state = AgentState(
        original_message="warehouse software uk",
        history=[],
        current_query=QueryPayload(query_text="warehouse software"),
    )
    metrics = QualityMetrics(reranked_count=10, geo_match_rate=0.3)

    decision = controller.choose_action(
        state=state,
        query_before=QueryPayload(query_text="warehouse software"),
        metrics=metrics,
        diagnosis={"primary_failure_mode": "geo_mismatch", "suggested_actions": ["add_geography_constraint"]},
    )

    assert decision.action == "add_geography_constraint"


def test_controller_compute_improvement_with_geography() -> None:
    controller = Controller()
    previous = QualityMetrics(top_score=0.5, score_spread=0.1, geo_match_rate=0.4)
    current = QualityMetrics(top_score=0.6, score_spread=0.2, geo_match_rate=0.6)

    improvement = controller.compute_improvement(previous, current, has_geography=True)

    assert improvement == pytest.approx(0.5 * 0.1 + 0.3 * 0.1 + 0.2 * 0.2)


def test_controller_should_invoke_diagnoser_on_thresholds() -> None:
    controller = Controller()
    metrics = QualityMetrics(top_score=0.77, reranked_count=10)
    query = QueryPayload(query_text="warehouse software")

    assert controller.should_invoke_diagnoser(iteration=2, metrics=metrics, query=query) is True


def test_controller_apply_action_adds_exclusions() -> None:
    controller = Controller()
    state = AgentState(
        original_message="fintech not focused on payments",
        history=[],
        current_query=QueryPayload(query_text="fintech", exclusions=[]),
    )

    updated = controller.apply_action(
        state=state,
        decision=controller.choose_action(
            state=state,
            query_before=state.current_query,
            metrics=QualityMetrics(reranked_count=10),
            diagnosis={"primary_failure_mode": "exclusion_failure", "suggested_actions": []},
        ),
        diagnosis={"proposed_exclusions": ["payments"]},
    )

    assert updated.exclusions == ["payments"]
