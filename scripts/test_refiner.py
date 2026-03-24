"""Test script for refiner agent."""

from __future__ import annotations

import asyncio

from app.schemas import RefineRequest
from app.services.refiner import QueryRefinerAgent


async def test_refiner():
    """Test refiner with sample queries."""
    test_cases = [
        "Vertical SaaS for logistics operators in the UK",
        "fintech companies not focused on payments",
        "a",
        "logistics software",
    ]
    
    agent = QueryRefinerAgent()
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: '{message}'")
        print('=' * 80)
        
        try:
            request = RefineRequest(message=message, max_iterations=3)
            response = await agent.refine(request)
            
            print(f"\nIterations used: {response.iterations_used}")
            print(f"Refined query: {response.refined_query.query_text}")
            print(f"Geography: {response.refined_query.geography}")
            print(f"Rationale: {response.rationale}")
            
            if response.meta.get("refinement_history"):
                print(f"\nRefinement history:")
                for h in response.meta["refinement_history"]:
                    print(f"  Iteration {h['iteration']}: {h['result_count']} results, top score {h['top_score']:.3f}")
                    print(f"    Query: {h['query']['query_text']}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_refiner())
