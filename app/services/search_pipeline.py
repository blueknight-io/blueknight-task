from __future__ import annotations

import time

from app.retrieval import RetrievalError
from app.schemas import Diagnostics, SearchRequest, SearchResponse
from app.services.post_filter import PostFilter
from app.services.reranker import Reranker
from app.services.retrieval_wrapper import retrieve_with_retry
from app.utils.logging import get_logger, log_stage


_logger = get_logger(__name__)
_post_filter = PostFilter()
_reranker = Reranker()

class SearchPipeline:
    async def run(self, request: SearchRequest) -> SearchResponse:
        trace_id = request.trace_id
        stage_latency_ms = {"vector_recall": 0, "post_filter": 0, "rerank": 0}

        recall_start = time.perf_counter()
        try:
            recalled = await retrieve_with_retry(
                request.query,
                top_k=request.top_k_raw,
                trace_id=trace_id,
            )
        except RetrievalError:
            stage_latency_ms["vector_recall"] = int((time.perf_counter() - recall_start) * 1000)
            log_stage(
                _logger,
                trace_id=trace_id,
                stage="vector_recall",
                duration_ms=stage_latency_ms["vector_recall"],
                item_count=0,
                error="retrieval_failed",
            )
            return SearchResponse(
                results=[],
                total=0,
                diagnostics=Diagnostics(stage_latency_ms=stage_latency_ms, trace_id=trace_id),
            )

        stage_latency_ms["vector_recall"] = int((time.perf_counter() - recall_start) * 1000)
        log_stage(
            _logger,
            trace_id=trace_id,
            stage="vector_recall",
            duration_ms=stage_latency_ms["vector_recall"],
            item_count=len(recalled),
        )

        filter_start = time.perf_counter()
        filtered, drop_reasons, filtered_count = _post_filter.filter(
            recalled,
            geography=request.query.geography,
            exclusions=request.query.exclusions,
        )
        stage_latency_ms["post_filter"] = int((time.perf_counter() - filter_start) * 1000)
        log_stage(
            _logger,
            trace_id=trace_id,
            stage="post_filter",
            duration_ms=stage_latency_ms["post_filter"],
            item_count=len(filtered),
            filtered_count=filtered_count,
        )

        rerank_start = time.perf_counter()
        results = _reranker.rerank(
            filtered,
            query=request.query,
            top_k=request.top_k_final,
            offset=request.offset,
        )
        stage_latency_ms["rerank"] = int((time.perf_counter() - rerank_start) * 1000)
        log_stage(
            _logger,
            trace_id=trace_id,
            stage="rerank",
            duration_ms=stage_latency_ms["rerank"],
            item_count=len(results),
        )

        geography_match_count = sum(
            1
            for result in results
            if not request.query.geography
            or result.country.lower().strip() in {g.lower().strip() for g in request.query.geography}
        )
        top_scores = [result.score for result in results]
        top_score = top_scores[0] if top_scores else 0.0
        score_spread = max(top_scores) - min(top_scores) if len(top_scores) > 1 else top_score

        diagnostics = Diagnostics(
            raw_count=len(recalled),
            filtered_count=filtered_count,
            reranked_count=len(results),
            drop_reasons=drop_reasons,
            stage_latency_ms=stage_latency_ms,
            top_score=top_score,
            score_spread=score_spread,
            geo_match_rate=geography_match_count / max(len(results), 1),
            exclusion_hit_rate=drop_reasons.get("exclude_term", 0) / max(len(recalled), 1),
            trace_id=trace_id,
        )
        return SearchResponse(results=results, total=len(filtered), diagnostics=diagnostics)
