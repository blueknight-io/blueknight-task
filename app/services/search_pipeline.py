from __future__ import annotations

import asyncio
import time
from typing import Any

from app.retrieval import CompanyResult, RetrievalError, mock_retrieve
from app.schemas import Diagnostics, QueryPayload, SearchRequest, SearchResponse, SearchResult
from app.services.reranker import Reranker

# Max concurrent calls into mock_retrieve (spec: not safe with unbounded concurrency)
_SEMAPHORE = asyncio.Semaphore(5)
# Per-call timeout in seconds — must cover OpenAI embedding + Pinecone query (~3-4s observed)
_RETRIEVE_TIMEOUT_S = 15.0
# Retry attempts on transient RetrievalError
_MAX_RETRIES = 3


class SearchPipeline:
    """
    Three-stage pipeline:
      1. Vector recall   — mock_retrieve with retry + timeout + concurrency guard
      2. Post-filter     — drop results that violate geography or exclude_terms signals
      3. Re-rank         — score by vector + keyword + geo signals, return top_k_final
    """

    async def run(self, request: SearchRequest) -> SearchResponse:
        query = request.query
        trace_id = request.trace_id
        drop_reasons: dict[str, int] = {}
        stage_latency: dict[str, int] = {}

        # ── Stage 1: Vector recall ───────────────────────────────────────────
        t0 = time.monotonic()
        raw_results = await self._recall(query.query_text, request.top_k_raw)
        stage_latency["vector_recall"] = int((time.monotonic() - t0) * 1000)
        print(f"[PIPELINE] vector_recall retrieved={len(raw_results)} latency={stage_latency['vector_recall']}ms trace_id={trace_id}")

        # ── Stage 2: Post-filter ─────────────────────────────────────────────
        t0 = time.monotonic()
        filtered, drop_reasons = self._post_filter(raw_results, query)
        stage_latency["post_filter"] = int((time.monotonic() - t0) * 1000)
        filtered_count = len(raw_results) - len(filtered)
        print(f"[PIPELINE] post_filter kept={len(filtered)} dropped={filtered_count} reasons={drop_reasons} latency={stage_latency['post_filter']}ms trace_id={trace_id}")

        # ── Stage 3: Re-rank ─────────────────────────────────────────────────
        t0 = time.monotonic()
        reranker = Reranker()
        reranked = reranker.rerank(
            candidates=[self._to_dict(r) for r in filtered],
            query=query.model_dump(),
            top_k=request.top_k_final,
        )
        stage_latency["rerank"] = int((time.monotonic() - t0) * 1000)
        print(f"[PIPELINE] rerank top_k={len(reranked)} latency={stage_latency['rerank']}ms trace_id={trace_id}")

        # ── Assemble response ────────────────────────────────────────────────
        offset = request.offset
        page = reranked[offset : offset + request.top_k_final]

        results = [
            SearchResult(
                id=r["id"],
                company_name=r.get("company_name", ""),
                country=r.get("country", ""),
                score=r["score"],
                score_components=r.get("score_components", {}),
                long_offering=r.get("long_offering", ""),
            )
            for r in page
        ]

        diagnostics = Diagnostics(
            raw_count=len(raw_results),
            filtered_count=filtered_count,
            reranked_count=len(reranked),
            drop_reasons=drop_reasons,
            stage_latency_ms=stage_latency,
            trace_id=trace_id,
        )

        return SearchResponse(
            results=results,
            total=len(reranked),
            diagnostics=diagnostics,
        )

    # ── Stage 1 helpers ──────────────────────────────────────────────────────

    async def _recall(self, query_text: str, top_k: int) -> list[CompanyResult]:
        """Wraps mock_retrieve with semaphore, timeout, and retry logic."""
        # Graceful fallback for degenerate / empty queries
        if not query_text or not query_text.strip():
            query_text = "enterprise B2B software company"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with _SEMAPHORE:
                    results = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, mock_retrieve, query_text, top_k
                        ),
                        timeout=_RETRIEVE_TIMEOUT_S,
                    )
                return results
            except RetrievalError as exc:
                if attempt == _MAX_RETRIES:
                    print(f"[PIPELINE] vector_recall failed after {_MAX_RETRIES} retries: {exc}")
                    return []
                await asyncio.sleep(0.1 * attempt)  # brief back-off
            except asyncio.TimeoutError:
                print(f"[PIPELINE] vector_recall timeout attempt={attempt}")
                if attempt == _MAX_RETRIES:
                    return []

        return []

    # ── Stage 2 helpers ──────────────────────────────────────────────────────

    def _post_filter(
        self,
        results: list[CompanyResult],
        query: QueryPayload,
    ) -> tuple[list[CompanyResult], dict[str, int]]:
        """
        Filters results using signals extractable from long_offering text.

        Signals applied:
          - geography_mismatch: company country not in requested geographies
          - exclude_term:       a banned term appears in long_offering or company_name
        """
        geography = [g.lower() for g in query.geography]
        exclude_terms = [t.lower() for t in query.exclude_terms]

        kept: list[CompanyResult] = []
        drop_reasons: dict[str, int] = {}

        for r in results:
            offering_lower = r.long_offering.lower()
            country_lower = r.country.lower()
            name_lower = r.company_name.lower()

            # Geography filter — only apply when geography is explicitly requested
            if geography and not any(g in country_lower for g in geography):
                drop_reasons["geography_mismatch"] = (
                    drop_reasons.get("geography_mismatch", 0) + 1
                )
                continue

            # Exclusion filter — drop if any excluded term found in offering or name
            excluded_hit = next(
                (t for t in exclude_terms if t in offering_lower or t in name_lower),
                None,
            )
            if excluded_hit:
                drop_reasons["exclude_term"] = drop_reasons.get("exclude_term", 0) + 1
                continue

            kept.append(r)

        return kept, drop_reasons

    @staticmethod
    def _to_dict(r: CompanyResult) -> dict[str, Any]:
        return {
            "id": r.id,
            "score": r.score,
            "company_name": r.company_name,
            "country": r.country,
            "long_offering": r.long_offering,
        }


