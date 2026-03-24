# Refiner and Reranker - Implementation

This repo implements an agentic search workflow for M&A-style company matching. Users describe what they're looking for in natural language; the system refines that into a structured query, retrieves matching companies via vector search, and re-ranks results.

---

## Setup

### 1. Backend (FastAPI + Qdrant)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, VECTOR_DB_URL, VECTOR_DB_API_KEY
```

Ingest data into Qdrant:

```bash
python scripts/ingest_data.py
```

This reads `data/company_1000_data.xlsx`, embeds the `long_offering` field using OpenAI `text-embedding-3-small`, and uploads 1,000 points to Qdrant.

Run the API server:

```bash
uvicorn app.main:app --reload --reload-dir app
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Frontend (Next.js + Tailwind)

```bash
cd client
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The UI calls `/agent/refine` and `/search/run` via the Next.js rewrite at `/api/py/*`.

---

## Architecture

### Subtask 1: Refinement Agent Loop (`POST /agent/refine`)

**Flow:**
1. User sends natural language message.
2. Agent calls OpenAI chat completion to extract structured `QueryPayload` (query text + geography).
3. Agent runs the search pipeline (`POST /search/run`).
4. Agent evaluates termination condition.
5. Either stops and returns, or refines further (up to `max_iterations`).

**Termination logic:**
- Stop if query has ≥3 tokens AND search returned ≥5 results.
- Rationale: A query with at least 3 meaningful tokens is specific enough for vector search. If that query yields ≥5 results after filtering, we have signal. Iterating further when results are already present adds latency without clear benefit. If <5 results, the agent tries again with a slightly broader or rephrased query.

**Fallback:**
- If LLM call fails, fall back to using the user message as-is + regex-based geography extraction.
- All required response fields (`refined_query`, `rationale`, `actions`, `iterations_used`, `meta`) are guaranteed present via defaults.

### Subtask 2: Search Pipeline (`POST /search/run`)

**Stage 1: Vector recall**
- Wrapped call to `mock_retrieve` (now backed by Qdrant) via retry helper.
- Retry up to 3 times with exponential backoff on `RetrievalError` or timeout.
- Logs: `search.start`, `search.recall_complete` with `trace_id`, `raw_count`, `duration_ms`.

**Stage 2: Post-filtering**
- Geography filter: if `query.geography` is set, drop results where `country` doesn't match.
- Tracks `drop_reasons["geography_mismatch"]`.
- Logs: `search.filter_complete` with `filtered_count`, `duration_ms`, `drop_reasons`.

**Stage 3: Reranking**
- Heuristic: starts with vector score, adds small boost for query token presence in `long_offering`.
- Exposes `score_components` in response: `{"vector": ..., "rerank_boost": ...}`.
- Returns top `top_k_final` after sorting by final score descending.
- Logs: `search.rerank_complete`, `search.complete` with `reranked_count`, `total_ms`.

**Diagnostics contract:**
- `raw_count`, `filtered_count`, `reranked_count`
- `drop_reasons` dict
- `stage_latency_ms` dict with `vector_recall`, `post_filter`, `rerank`, `total`
- `trace_id` propagated from request

All log entries include `trace_id` so you can filter logs by request in production.

### Subtask 3: Production Observability

**Question:** "The system serves 10,000 queries/day. Result relevance silently degrades - no errors are thrown, but users are getting poor matches. How would you detect this before users complain, and what is your first operational change?"

**Answer:**

Silent degradation means users see results but they're not helpful. Without explicit feedback (thumbs down, etc.), we need synthetic signals.

**Detection:**
1. **Track baseline metrics**: median/p95 of `final_score` in top-3 results per query. If scores drop week-over-week, embedding drift or data staleness is likely.
2. **Score distribution alerts**: if >20% of queries return all results with `score < 0.6`, the index or model changed.
3. **Geography mismatch ratio**: `drop_reasons["geography_mismatch"] / raw_count`. A spike suggests the index lost geo signal or queries shifted.
4. **Result diversity**: if top-10 all come from the same country or have near-identical offerings, ranking logic may be broken.

Dashboards aggregating these from structured logs (`trace_id`, `stage_latency_ms`, `score_components`) make degradation visible in hours, not days.

**First operational change:**
Emit a daily summary comparing current metrics vs. trailing 7-day baseline. If median top-3 score drops >10% or geography mismatch rate doubles, page on-call. This catches silent drift before user churn.

---

## Assumptions

1. **Data format**: Excel has columns `Consolidated ID`, `Company Name`, `Country`, `Long Offering`. Script lowercases and replaces spaces with underscores.
2. **Embedding model**: `text-embedding-3-small` (1536-dim) balances cost and quality for this dataset size.
3. **Qdrant**: Uses cloud or self-hosted instance. Collection recreated on each ingestion run (dev simplicity).
4. **Geography normalization**: Minimal fallback regex for "UK", "USA" variants. Production would use a gazetteer or NER model.
5. **Reranking**: Token-overlap heuristic. Production would use a cross-encoder or learned-to-rank model.
6. **Concurrency**: Retry wrapper limits to 3 attempts; no semaphore on `mock_retrieve` since Qdrant client handles connection pooling internally.

---

## Termination Condition Rationale

The refinement loop stops when:
- **Query has ≥3 tokens** (specific enough for vector search)
- **AND search returns ≥5 results** (enough signal to show the user)

**Why this works:**
- Queries like "a" trigger fallback and may iterate to add context.
- Well-formed queries like "Vertical SaaS for logistics in UK" exit after 1 iteration if results are found.
- If initial recall returns 0, the agent broadens the query (e.g., adds "software") and tries again.

This balances latency (don't iterate unnecessarily) and coverage (don't return empty results when another iteration might help).

---

## What I Would Do With More Time

1. **Exclusion filtering**: `QueryPayload.exclusions` exists but isn't used. Add negative term matching in post-filter stage.
2. **Must-include filtering**: Same for `QueryPayload.must_include`.
3. **Cross-encoder reranking**: Replace token heuristic with a sentence-transformer cross-encoder for semantic reranking.
4. **Structured logging config**: Add JSON formatter for production logs (currently using default Python logger).
5. **Rate limiting**: Semaphore on `retrieve_with_retry` to cap concurrent Qdrant calls.
6. **Caching**: LRU cache on `(query_text, top_k)` → results for repeat queries.
7. **Frontend polish**: Add loading states, pagination, action button handlers.
8. **Tests**: Unit tests for reranker scoring, integration test mocking Qdrant, end-to-end test with test collection.

---

## Running the System

**Terminal 1 (API):**
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --reload-dir app
```

**Terminal 2 (Frontend):**
```bash
cd client
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000), enter a query, click **Refine query** then **Run search**.

---

## Tech Stack

- **Backend**: FastAPI, Pydantic, Qdrant, OpenAI embeddings + chat
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS
- **Infra**: Python 3.11+, Node 20+

---

## Project Structure

```
app/
├── db/
│   └── qdrant.py          # Qdrant client + search helpers
├── models/
│   ├── embedder.py        # OpenAI embeddings via httpx
│   └── llm.py             # OpenAI chat completions via httpx
├── services/
│   ├── refiner.py         # QueryRefinerAgent with loop
│   ├── search_pipeline.py # SearchPipeline: recall, filter, rerank
│   └── reranker.py        # Heuristic reranker with score components
├── utils/
│   ├── retry.py           # Retry wrapper for retrieval
│   └── json_contract.py   # (existing util)
├── settings.py            # Centralized env config
├── schemas.py             # Pydantic models
├── retrieval.py           # mock_retrieve (now real Qdrant search)
└── main.py                # FastAPI app + CORS + endpoints

client/
├── app/
│   ├── page.tsx           # Main UI page
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Tailwind base styles
├── components/
│   ├── AppHeader.tsx      # Header component
│   └── QueryWorkbench.tsx # Query input + results display
├── services/
│   ├── types.ts           # TypeScript types matching Python schemas
│   └── api.ts             # API client functions
└── next.config.ts         # Rewrites for API proxy

scripts/
└── ingest_data.py         # Ingestion script for Excel → Qdrant

data/
└── company_1000_data.xlsx # Company dataset
```

---

## License

MIT
