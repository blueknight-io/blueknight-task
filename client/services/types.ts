export type QueryPayload = {
  query_text: string;
  geography: string[];
};

export type RefinementHistoryEntry = {
  iteration: number;
  query: QueryPayload;
  result_count: number;
  top_score: number;
};

export type RefineRequest = {
  message: string;
  base_query?: QueryPayload | null;
  history: Record<string, unknown>[];
  max_iterations?: number;
  trace_id?: string;
};

export type Action = {
  id: string;
  label: string;
  payload?: Record<string, unknown>;
};

export type RefineResponse = {
  refined_query: QueryPayload;
  rationale: string;
  actions: Action[];
  iterations_used: number;
  meta: {
    refinement_history?: RefinementHistoryEntry[];
    [key: string]: unknown;
  };
};

export type SearchRequest = {
  query: QueryPayload;
  top_k_raw?: number;
  top_k_final?: number;
  offset?: number;
  trace_id?: string;
};

export type SearchResult = {
  id: string;
  company_name: string;
  country: string;
  score: number;
  score_components?: Record<string, number>;
  long_offering: string;
};

export type Diagnostics = {
  raw_count: number;
  filtered_count: number;
  reranked_count: number;
  drop_reasons: Record<string, number>;
  stage_latency_ms: Record<string, number>;
  trace_id: string;
};

export type SearchResponse = {
  results: SearchResult[];
  total: number;
  diagnostics: Diagnostics;
};
