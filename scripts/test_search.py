"""Test script for search pipeline."""

from __future__ import annotations

import asyncio

from app.schemas import QueryPayload, SearchRequest
from app.services.search_pipeline import SearchPipeline


async def test_search_pipeline():
    """Test search pipeline with sample queries covering all features."""
    test_cases = [
        {
            "name": "Basic query",
            "query": QueryPayload(query_text="logistics software", geography=[]),
            "top_k_raw": 100,
            "top_k_final": 10,
        },
        {
            "name": "Geography filter",
            "query": QueryPayload(
                query_text="Vertical SaaS for logistics operators",
                geography=["United Kingdom"]
            ),
            "top_k_raw": 100,
            "top_k_final": 10,
        },
        {
            "name": "Exclusion query",
            "query": QueryPayload(
                query_text="fintech companies not focused on payments",
                geography=[]
            ),
            "top_k_raw": 100,
            "top_k_final": 10,
        },
    ]
    
    pipeline = SearchPipeline()
    
    for test in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test: {test['name']}")
        print(f"Query: '{test['query'].query_text}'")
        print(f"Geography: {test['query'].geography or 'None'}")
        print('=' * 80)
        
        try:
            request = SearchRequest(
                query=test["query"],
                top_k_raw=test["top_k_raw"],
                top_k_final=test["top_k_final"],
            )
            response = await pipeline.run(request)
            
            print(f"\nResults: {response.total} total, {len(response.results)} returned")
            print(f"\nDiagnostics:")
            print(f"  Raw count: {response.diagnostics.raw_count}")
            print(f"  Filtered count: {response.diagnostics.filtered_count}")
            print(f"  Final count: {response.diagnostics.reranked_count}")
            print(f"  Stage latencies:")
            for stage, ms in response.diagnostics.stage_latency_ms.items():
                print(f"    {stage}: {ms}ms")
            if response.diagnostics.drop_reasons:
                print(f"  Drop reasons:")
                for reason, count in response.diagnostics.drop_reasons.items():
                    print(f"    {reason}: {count}")
            
            print(f"\nTop 3 results:")
            for i, r in enumerate(response.results[:3], 1):
                print(f"\n{i}. {r.company_name} ({r.country})")
                print(f"   Score: {r.score:.4f}")
                print(f"   Components: {r.score_components}")
                print(f"   Offering: {r.long_offering[:120]}...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_search_pipeline())
