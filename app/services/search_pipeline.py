from __future__ import annotations

import re
import time

from app.retrieval import CompanyResult
from app.schemas import Diagnostics, QueryPayload, SearchRequest, SearchResponse, SearchResult
from app.utils.retry import retrieve_with_retry


class SearchPipeline:
    """
    Three-stage search pipeline:
    1. Vector recall - retrieve top_k_raw results from Qdrant
    2. Post-filtering - filter by geography, exclusions, domain signals
    3. Re-ranking - score and sort, return top_k_final
    """

    async def run(self, request: SearchRequest) -> SearchResponse:
        trace_id = request.trace_id
        query = request.query
        diagnostics = Diagnostics(trace_id=trace_id)
        
        start_recall = time.perf_counter()
        raw_results = await retrieve_with_retry(query.query_text, request.top_k_raw)
        diagnostics.stage_latency_ms["vector_recall"] = int((time.perf_counter() - start_recall) * 1000)
        diagnostics.raw_count = len(raw_results)
        
        start_filter = time.perf_counter()
        filtered_results = self._post_filter(raw_results, query, diagnostics)
        diagnostics.stage_latency_ms["post_filter"] = int((time.perf_counter() - start_filter) * 1000)
        diagnostics.filtered_count = len(filtered_results)
        
        start_rerank = time.perf_counter()
        reranked_results = self._rerank(filtered_results, query)
        diagnostics.stage_latency_ms["rerank"] = int((time.perf_counter() - start_rerank) * 1000)
        
        final_results = reranked_results[request.offset : request.offset + request.top_k_final]
        diagnostics.reranked_count = len(final_results)
        
        results = [
            SearchResult(
                id=r.id,
                company_name=r.company_name,
                country=r.country,
                long_offering=r.long_offering,
                score=r.score,
                score_components=getattr(r, "_score_components", {"vector_similarity": r.score}),
            )
            for r in final_results
        ]
        
        return SearchResponse(
            results=results,
            total=len(reranked_results),
            diagnostics=diagnostics,
        )
    
    def _post_filter(
        self,
        results: list[CompanyResult],
        query: QueryPayload,
        diagnostics: Diagnostics,
    ) -> list[CompanyResult]:
        """
        Post-filtering based on signals in long_offering and query.
        
        Filters applied:
        1. Geography - if query.geography specified, keep only matching countries
        2. Exclusions - detect "not X" patterns and filter out companies heavily focused on X
        3. Domain keywords - boost/filter based on domain relevance
        """
        filtered = []
        exclusion_terms = self._extract_exclusions(query.query_text)
        
        for r in results:
            if query.geography and r.country not in query.geography:
                diagnostics.drop_reasons["geography_mismatch"] = (
                    diagnostics.drop_reasons.get("geography_mismatch", 0) + 1
                )
                continue
            
            if self._has_excluded_content(r.long_offering, exclusion_terms):
                diagnostics.drop_reasons["exclude_term"] = (
                    diagnostics.drop_reasons.get("exclude_term", 0) + 1
                )
                continue
            
            filtered.append(r)
        
        return filtered
    
    def _extract_exclusions(self, query_text: str) -> list[str]:
        """Extract exclusion terms from query (e.g., 'fintech not payments' → ['payments'])."""
        patterns = [
            r"not\s+(?:focused\s+on\s+)?(\w+)",
            r"excluding\s+(\w+)",
            r"except\s+(\w+)",
            r"without\s+(\w+)",
        ]
        exclusions = []
        for pattern in patterns:
            matches = re.findall(pattern, query_text.lower())
            exclusions.extend(matches)
        return exclusions
    
    def _has_excluded_content(self, text: str, exclusions: list[str]) -> bool:
        """Check if text heavily mentions excluded terms."""
        if not exclusions:
            return False
        
        text_lower = text.lower()
        for term in exclusions:
            count = text_lower.count(term)
            if count >= 3:
                return True
        return False
    
    def _rerank(
        self,
        results: list[CompanyResult],
        query: QueryPayload,
    ) -> list[CompanyResult]:
        """
        Re-rank results using hybrid scoring.
        
        Scoring components:
        1. Vector similarity (0-1): base semantic relevance
        2. Query keyword overlap (0-0.2): boost if query terms appear in long_offering
        3. Geography bonus (0-0.1): boost if country matches query geography
        
        Final score = vector_score * 0.8 + keyword_boost + geo_boost
        """
        query_terms = set(query.query_text.lower().split())
        query_terms = {t for t in query_terms if len(t) > 2}
        
        scored_results = []
        for r in results:
            vector_score = r.score
            
            keyword_boost = 0.0
            if query_terms:
                text_lower = r.long_offering.lower()
                matches = sum(1 for term in query_terms if term in text_lower)
                keyword_boost = min(0.2, (matches / len(query_terms)) * 0.2)
            
            geo_boost = 0.0
            if query.geography and r.country in query.geography:
                geo_boost = 0.1
            
            final_score = (vector_score * 0.8) + keyword_boost + geo_boost
            
            scored_results.append({
                "result": r,
                "final_score": final_score,
                "components": {
                    "vector_similarity": vector_score,
                    "keyword_overlap": keyword_boost,
                    "geography_bonus": geo_boost,
                }
            })
        
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        reranked = []
        for item in scored_results:
            r = item["result"]
            reranked_result = CompanyResult(
                id=r.id,
                company_name=r.company_name,
                country=r.country,
                long_offering=r.long_offering,
                score=item["final_score"],
            )
            reranked_result._score_components = item["components"]
            reranked.append(reranked_result)
        
        return reranked

