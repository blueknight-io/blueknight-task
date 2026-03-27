from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Action(BaseModel):
    id: str
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class QueryPayload(BaseModel):
    query_text: str = ""
    geography: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)


class QualityMetrics(BaseModel):
    top_score: float = 0.0
    score_spread: float = 0.0
    geo_match_rate: float = 0.0
    exclusion_hit_rate: float = 0.0
    top_k_overlap: float = 0.0
    improvement_score: float = 0.0
    raw_count: int = 0
    filtered_count: int = 0
    reranked_count: int = 0


class IterationRecord(BaseModel):
    iteration: int
    action: str
    failure_mode: str
    query_before: QueryPayload
    query_after: QueryPayload
    metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    rationale: str = ""


class AgentMeta(BaseModel):
    best_iteration: int = 0
    final_action: str = ""
    iterations: list[IterationRecord] = Field(default_factory=list)


class RefineRequest(BaseModel):
    message: str
    base_query: Optional[QueryPayload] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    max_iterations: int = 3
    trace_id: str = Field(default_factory=lambda: str(uuid4()))


class RefineResponse(BaseModel):
    refined_query: QueryPayload
    rationale: str
    actions: list[Action] = Field(default_factory=list)
    iterations_used: int = 1
    meta: AgentMeta = Field(default_factory=AgentMeta)


class SearchRequest(BaseModel):
    query: QueryPayload
    top_k_raw: int = 1000
    top_k_final: int = 50
    offset: int = 0
    trace_id: str = Field(default_factory=lambda: str(uuid4()))


class SearchResult(BaseModel):
    id: str
    company_name: str = ""
    country: str = ""
    score: float
    score_components: dict[str, float] = Field(default_factory=dict)
    long_offering: str = ""


class Diagnostics(BaseModel):
    raw_count: int = 0
    filtered_count: int = 0
    reranked_count: int = 0
    drop_reasons: dict[str, int] = Field(default_factory=dict)
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    top_score: float = 0.0
    score_spread: float = 0.0
    geo_match_rate: float = 0.0
    exclusion_hit_rate: float = 0.0
    trace_id: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
    total: int = 0
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)
