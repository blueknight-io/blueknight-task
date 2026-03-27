from __future__ import annotations

from app.retrieval import CompanyResult
from app.services.post_filter import PostFilter


def test_post_filter_drops_low_score_and_counts_reason() -> None:
    post_filter = PostFilter()
    candidates = [
        CompanyResult(
            id="1",
            company_name="LowScore",
            country="United Kingdom",
            long_offering="Software for warehouse operators.",
            score=0.10,
        )
    ]

    kept, drop_reasons, filtered_count = post_filter.filter(candidates, geography=[], exclusions=[])

    assert kept == []
    assert filtered_count == 1
    assert drop_reasons["low_vector_score"] == 1


def test_post_filter_drops_geography_mismatch() -> None:
    post_filter = PostFilter()
    candidates = [
        CompanyResult(
            id="1",
            company_name="USCo",
            country="United States",
            long_offering="Industrial software for US logistics operators.",
            score=0.8,
        )
    ]

    kept, drop_reasons, filtered_count = post_filter.filter(
        candidates,
        geography=["United Kingdom"],
        exclusions=[],
    )

    assert kept == []
    assert filtered_count == 1
    assert drop_reasons["geography_mismatch"] == 1


def test_post_filter_exclusion_uses_word_boundaries() -> None:
    post_filter = PostFilter()
    candidates = [
        CompanyResult(
            id="1",
            company_name="PayCo",
            country="United Kingdom",
            long_offering="The company focuses on pay cycles for workers.",
            score=0.9,
        ),
        CompanyResult(
            id="2",
            company_name="PaymentsCo",
            country="United Kingdom",
            long_offering="The company provides payment processing software.",
            score=0.9,
        ),
    ]

    kept, drop_reasons, filtered_count = post_filter.filter(
        candidates,
        geography=[],
        exclusions=["payments"],
    )

    assert [candidate.id for candidate in kept] == ["1"]
    assert filtered_count == 1
    assert drop_reasons["exclude_term"] == 1


def test_post_filter_counts_additive_drop_reasons() -> None:
    post_filter = PostFilter()
    candidates = [
        CompanyResult(
            id="1",
            company_name="WeakUSPayments",
            country="United States",
            long_offering="Payment processing for US retailers.",
            score=0.1,
        )
    ]

    kept, drop_reasons, filtered_count = post_filter.filter(
        candidates,
        geography=["United Kingdom"],
        exclusions=["payments"],
    )

    assert kept == []
    assert filtered_count == 1
    assert drop_reasons["low_vector_score"] == 1
    assert drop_reasons["geography_mismatch"] == 1
    assert drop_reasons["exclude_term"] == 1


def test_post_filter_matches_geography_via_offering_text_alias() -> None:
    post_filter = PostFilter()
    candidates = [
        CompanyResult(
            id="1",
            company_name="LondonOps",
            country="Germany",
            long_offering="Industrial software for warehouse operators across London and the UK.",
            score=0.8,
        )
    ]

    kept, drop_reasons, filtered_count = post_filter.filter(
        candidates,
        geography=["United Kingdom"],
        exclusions=[],
    )

    assert [candidate.id for candidate in kept] == ["1"]
    assert filtered_count == 0
    assert drop_reasons == {}
