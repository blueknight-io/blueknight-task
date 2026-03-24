from __future__ import annotations

import json

import httpx

from app.models.llm import chat_completion
from app.schemas import Action, QueryPayload, RefineRequest, RefineResponse, SearchRequest
from app.settings import get_settings


class QueryRefinerAgent:
    """
    Iterative refinement agent that:
    1. Refines user message into structured QueryPayload
    2. Calls POST /search/run to evaluate results
    3. Decides whether to iterate or terminate based on result quality
    4. Returns refined query with rationale
    """

    async def refine(self, request: RefineRequest) -> RefineResponse:
        settings = get_settings()
        iterations_used = 0
        current_query = request.base_query or default_query_payload()
        refinement_history = []
        
        for iteration in range(request.max_iterations):
            iterations_used = iteration + 1
            
            refined_query = await self._refine_query(
                message=request.message,
                current_query=current_query,
                refinement_history=refinement_history,
                settings=settings,
            )
            
            search_response = await self._call_search_api(refined_query)
            
            should_continue, rationale = self._should_continue(
                refined_query=refined_query,
                search_response=search_response,
                iteration=iteration,
                max_iterations=request.max_iterations,
            )
            
            refinement_history.append({
                "iteration": iteration + 1,
                "query": refined_query.model_dump(),
                "result_count": search_response.get("total", 0),
                "top_score": search_response["results"][0]["score"] if search_response.get("results") else 0,
            })
            
            if not should_continue:
                return RefineResponse(
                    refined_query=refined_query,
                    rationale=rationale,
                    actions=default_actions(),
                    iterations_used=iterations_used,
                    meta={"refinement_history": refinement_history},
                )
            
            current_query = refined_query
        
        return RefineResponse(
            refined_query=current_query,
            rationale=f"Reached max iterations ({request.max_iterations})",
            actions=default_actions(),
            iterations_used=iterations_used,
            meta={"refinement_history": refinement_history},
        )
    
    async def _refine_query(
        self,
        message: str,
        current_query: QueryPayload,
        refinement_history: list[dict],
        settings,
    ) -> QueryPayload:
        """Use LLM to refine user message into structured QueryPayload."""
        system_prompt = """You are a query refinement agent for a company search system. Extract structured search parameters from user messages.

Output valid JSON with this schema:
{
  "query_text": "refined semantic search query",
  "geography": ["list", "of", "countries"]
}

Rules:
- query_text should be clear, specific, and optimized for semantic search over company descriptions
- Extract geography from location mentions (e.g., "UK" → "United Kingdom", "US" → "United States")
- Expand vague queries (e.g., "a" → "software companies")
- Preserve user intent and key concepts
- If no geography specified, return empty array

Examples:
- "logistics in UK" → {"query_text": "logistics software and services", "geography": ["United Kingdom"]}
- "fintech not payments" → {"query_text": "fintech companies excluding payment processing", "geography": []}
- "a" → {"query_text": "technology and software companies", "geography": []}"""

        user_content = f"User message: {message}"
        
        if refinement_history:
            user_content += f"\n\nPrevious refinement attempts:"
            for h in refinement_history:
                user_content += f"\n- Iteration {h['iteration']}: \"{h['query']['query_text']}\""
                user_content += f" → {h['result_count']} results, top score {h['top_score']:.3f}"
                if h['top_score'] < 0.5:
                    user_content += " (POOR QUALITY)"
                elif h['result_count'] < 5:
                    user_content += " (TOO FEW RESULTS)"
            user_content += "\n\nGenerate a DIFFERENT query to improve results."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        try:
            response = await chat_completion(
                messages=messages,
                model=settings.llm_model,
                temperature=0.3 if refinement_history else 0.2,
                response_format={"type": "json_object"},
                api_key=settings.openai_api_key,
            )
            parsed = json.loads(response)
            return QueryPayload(
                query_text=parsed.get("query_text", message),
                geography=parsed.get("geography", []),
            )
        except Exception:
            return QueryPayload(query_text=message, geography=[])
    
    async def _call_search_api(self, query: QueryPayload) -> dict:
        """Call POST /search/run internally via HTTP."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8000/search/run",
                    json=SearchRequest(query=query, top_k_raw=100, top_k_final=20).model_dump(),
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "results": data.get("results", []),
                    "total": data.get("total", 0),
                }
        except Exception:
            return {"results": [], "total": 0}
    
    def _should_continue(
        self,
        refined_query: QueryPayload,
        search_response: dict,
        iteration: int,
        max_iterations: int,
    ) -> tuple[bool, str]:
        """
        Determine if refinement should continue based on search result quality.
        
        Termination logic (deterministic, priority order):
        1. Query too vague → Continue (if iterations left)
        2. Good results (>=10 results, top score >=0.7) → Stop
        3. Acceptable results (>=5 results, top score >=0.5) → Stop  
        4. Poor results (<3 results or top score <0.3) → Continue (if iterations left)
        5. Moderate results → Continue (if iterations left)
        6. Max iterations reached → Stop
        """
        result_count = search_response.get("total", 0)
        results = search_response.get("results", [])
        top_score = results[0]["score"] if results else 0
        
        if not refined_query.query_text or len(refined_query.query_text.strip()) < 3:
            if iteration >= max_iterations - 1:
                return False, f"Max iterations ({max_iterations}) reached with incomplete query"
            return True, "Query too short, needs expansion"
        
        if result_count >= 10 and top_score >= 0.7:
            return False, f"Good results: {result_count} results, top score {top_score:.3f}"
        
        if result_count >= 5 and top_score >= 0.5:
            return False, f"Acceptable results: {result_count} results, top score {top_score:.3f}"
        
        if iteration >= max_iterations - 1:
            return False, f"Max iterations ({max_iterations}) reached"
        
        if result_count < 3 or top_score < 0.3:
            return True, f"Poor results ({result_count} results, top score {top_score:.3f}), refining query"
        
        return True, f"Moderate results ({result_count} results, top score {top_score:.3f}), attempting refinement"


def default_actions() -> list[Action]:
    """Starter UI action contract. Candidate may extend."""
    return [Action(id="ideas", label="Suggest more search ideas")]


def default_query_payload() -> QueryPayload:
    """Default payload used for deterministic query shape."""
    return QueryPayload()

