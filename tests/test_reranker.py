from __future__ import annotations

from app.retrieval import CompanyResult
from app.schemas import QueryPayload
from app.services.reranker import Reranker


def test_reranker_returns_expected_score_component_keys() -> None:
    reranker = Reranker()
    candidates = [
        CompanyResult(
            id="1",
            company_name="WarehouseFlow",
            country="United Kingdom",
            long_offering="Industrial software for warehouse operations in the UK.",
            score=0.75,
        )
    ]

    results = reranker.rerank(
        candidates,
        QueryPayload(query_text="warehouse software", geography=["United Kingdom"]),
        top_k=1,
    )

    assert len(results) == 1
    assert set(results[0].score_components) == {
        "vector",
        "geography_bonus",
        "keyword_density",
        "name_match_bonus",
        "domain_anchor_score",
        "phrase_match_bonus",
        "generic_mismatch_penalty",
    }


def test_reranker_prefers_geography_match_when_other_scores_are_equal() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse software", geography=["United Kingdom"])
    candidates = [
        CompanyResult(
            id="uk",
            company_name="WarehouseFlow",
            country="United Kingdom",
            long_offering="Warehouse software for operators.",
            score=0.7,
        ),
        CompanyResult(
            id="us",
            company_name="WarehouseFlow US",
            country="United States",
            long_offering="Warehouse software for operators.",
            score=0.7,
        ),
    ]

    results = reranker.rerank(candidates, query, top_k=2)
    assert [result.id for result in results] == ["uk", "us"]


def test_reranker_uses_expected_score_formula() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse software", geography=["United Kingdom"])
    candidate = CompanyResult(
        id="1",
        company_name="WarehouseFlow",
        country="United Kingdom",
        long_offering="Warehouse software for operators.",
        score=0.5,
    )

    result = reranker.rerank([candidate], query, top_k=1)[0]
    expected = (
        0.60 * 0.5
        + 0.25 * 1.0
        + 0.10 * 1.0
        + 0.05 * 0.0
        + 0.12 * 1.0
        + 0.08 * 0.0
        - 0.15 * 0.0
    )
    assert result.score == round(expected, 6)


def test_reranker_applies_keyword_density() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse logistics software", geography=[])
    candidates = [
        CompanyResult(
            id="many",
            company_name="OpsFlow",
            country="United Kingdom",
            long_offering="Warehouse logistics software for industrial operators.",
            score=0.6,
        ),
        CompanyResult(
            id="few",
            company_name="OpsGeneric",
            country="United Kingdom",
            long_offering="Software for operators.",
            score=0.6,
        ),
    ]

    results = reranker.rerank(candidates, query, top_k=2)
    assert [result.id for result in results] == ["many", "few"]


def test_reranker_handles_geography_alias_match() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse software", geography=["United Kingdom"])
    candidates = [
        CompanyResult(
            id="alias",
            company_name="WarehouseFlow",
            country="UK",
            long_offering="Warehouse software for operators.",
            score=0.7,
        ),
        CompanyResult(
            id="other",
            company_name="WarehouseFlow US",
            country="United States",
            long_offering="Warehouse software for operators.",
            score=0.7,
        ),
    ]

    results = reranker.rerank(candidates, query, top_k=2)
    assert [result.id for result in results] == ["alias", "other"]


def test_reranker_respects_top_k_and_offset() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse software", geography=[])
    candidates = [
        CompanyResult(id="1", company_name="A", country="UK", long_offering="warehouse software", score=0.9),
        CompanyResult(id="2", company_name="B", country="UK", long_offering="warehouse software", score=0.8),
        CompanyResult(id="3", company_name="C", country="UK", long_offering="warehouse software", score=0.7),
    ]

    results = reranker.rerank(candidates, query, top_k=1, offset=1)
    assert [result.id for result in results] == ["2"]


def test_reranker_demotes_generic_crm_for_warehouse_query() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="industrial software for warehouse operations in UK", geography=["United Kingdom"])
    candidates = [
        CompanyResult(
            id="generic",
            company_name="Generic CRM",
            country="United Kingdom",
            long_offering="CRM and sales automation software for growing businesses and marketing teams.",
            score=0.62,
        ),
        CompanyResult(
            id="domain",
            company_name="Warehouse Ops",
            country="United Kingdom",
            long_offering="Industrial software for warehouse operations, inventory workflows, and logistics teams.",
            score=0.56,
        ),
    ]

    results = reranker.rerank(candidates, query, top_k=2)
    assert [result.id for result in results] == ["domain", "generic"]


def test_reranker_rewards_phrase_level_alignment() -> None:
    reranker = Reranker()
    query = QueryPayload(query_text="warehouse operations software", geography=[])
    candidates = [
        CompanyResult(
            id="phrase",
            company_name="Warehouse Ops",
            country="United Kingdom",
            long_offering="Warehouse operations software for logistics operators.",
            score=0.5,
        ),
        CompanyResult(
            id="terms",
            company_name="Term Match",
            country="United Kingdom",
            long_offering="Software for warehouse teams and business operations.",
            score=0.5,
        ),
    ]

    results = reranker.rerank(candidates, query, top_k=2)
    assert [result.id for result in results] == ["phrase", "terms"]
