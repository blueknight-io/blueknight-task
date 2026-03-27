# Refiner and Reranker - Take-Home Spec

## Context

You are building an agentic search workflow for an M&A-style company matching system.

Users describe the kind of company they are looking for in natural language. Your agent must
refine that intent into a structured query, retrieve matching companies, re-rank them, and decide
whether the results are good enough to return - or worth another iteration.

Matching is based on `**long_offering**` - a rich text field describing what each company does,
who it serves, and how it delivers value.

---

## What You Are Given

### 1. A dataset of ~1,000 companies

`data/companies.csv` contains the company corpus. Each row has:

| Column | Type | Notes |
|---|---|---|
| `id` | `str` | unique identifier |
| `company_name` | `text` | |
| `country` | `text` | |
| `long_offering` | `text` | rich text, 100-400 words - **this is the bio** |

You must ingest this CSV into a vector database of your choice (Qdrant, Pinecone, FAISS,
pgvector, etc.), embed the `long_offering` field, and make it queryable. All retrieval and
re-ranking logic should operate on `long_offering`.



### 2. A mock retrieval function

`app/retrieval.py` contains:

```python
def mock_retrieve(query: str, top_k: int) -> list[CompanyResult]:
    ...
```

This is a **placeholder** that simulates latency and transient failures. You must replace its
internals with real vector retrieval over the ingested dataset, keeping the same function
signature. Your retrieval wrapper must handle the operational characteristics described below
as if they still apply to the underlying vector DB:

- ~200-300ms mean latency with occasional spikes
- Transient failures (~5%) raising `RetrievalError`
- Not safe to call with unbounded concurrency

**You must wrap this function** - do not call it directly from business logic or endpoint
handlers.

### 3. Schema stubs

`app/schemas.py` contains placeholder models as the data contract. Do not rename existing
fields - you may add fields where needed.

---

## Example Queries

Use these to ground your design decisions.


| Query                                                             | What "good" looks like                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `"Vertical SaaS for logistics operators in the UK"`               | `long_offering` describes software product + logistics domain + UK market            |
| `"Industrial software providers with field-deployed delivery"`    | `long_offering` signals on-premise or field deployment, not cloud-only               |
| `"Companies solving onboarding inefficiency for frontline teams"` | Problem + use-case signal present in `long_offering`, not just category tags         |
| `"Fintech companies not focused on payments"`                     | Exclusion intent respected - payments-heavy `long_offering` should rank down or drop |
| `"a"`                                                             | Graceful fallback - must not crash, must return a sensible default response          |


---

## Task Structure

### Subtask 1 - Refinement Agent Loop

Implement `POST /agent/refine`.

The agent must run a **loop**, not a single pass. On each iteration it:

1. Refines the user's message into a structured `QueryPayload`
2. Calls the retrieval pipeline
3. Evaluates whether results are sufficient
4. Either iterates (up to `max_iterations`) or returns

**You define the termination condition.** Document your reasoning - this is a primary evaluation
criterion. A candidate who always exits after one iteration has not met this requirement.

**Input:**


| Field            | Type                  | Notes                             |
| ---------------- | --------------------- | --------------------------------- |
| `message`        | `str`                 | the user's natural language query |
| `base_query`     | `QueryPayload | null` | optional prior structured query   |
| `history`        | `list[dict]`          | prior turns in the session        |
| `max_iterations` | `int`                 | default `3`                       |


**Output:**


| Field             | Type           | Notes                            |
| ----------------- | -------------- | -------------------------------- |
| `refined_query`   | `QueryPayload` |                                  |
| `rationale`       | `str`          | why the loop stopped when it did |
| `actions`         | `list[Action]` |                                  |
| `iterations_used` | `int`          |                                  |
| `meta`            | `dict`         |                                  |


**Requirements:**

- Deterministic JSON contract with robust fallback if model output is malformed
- Required fields normalized via a default payload - no field should ever be missing in the response
- Loop termination logic must be explicit and deterministic

> **Note:** The agent loop in Subtask 1 must call `POST /search/run` internally rather than re-implementing retrieval logic.


---

### Subtask 2 - Retrieval and Re-ranking Pipeline

Implement `POST /search/run`.

**Three stages:**

1. **Vector recall** - call `mock_retrieve(query, top_k=top_k_raw)`, wrapped with retry and
  timeout logic
2. **Post-filtering** - filter results using signals extractable from `long_offering` text
  (e.g. geography mentions, domain keywords, explicit exclusions from the query). Document
   what signals you extract and why.
3. **Re-rank** - produce a final ordered list of `top_k_final` results. The ranking approach
  is your choice; a well-reasoned heuristic is fine. Document your scoring logic.

**Input:**


| Field         | Type           | Notes          |
| ------------- | -------------- | -------------- |
| `query`       | `QueryPayload` |                |
| `top_k_raw`   | `int`          | default `1000` |
| `top_k_final` | `int`          | default `50`   |
| `offset`      | `int`          | default `0`    |


**Output:**


| Field         | Type                 | Notes     |
| ------------- | -------------------- | --------- |
| `results`     | `list[SearchResult]` |           |
| `total`       | `int`                |           |
| `diagnostics` | `Diagnostics`        | see below |


`**Diagnostics` must include:**

- `filtered_count` - results removed in stage 2
- `drop_reasons` - why they were removed (e.g. `{"geography_mismatch": 18, "exclude_term": 7}`)
- `stage_latency_ms` - separate timings for `vector_recall`, `post_filter`, and `rerank`

Per-stage latency is **required** and must be readable in structured logs without a debugger
attached.

---

### Subtask 3 - Production Readiness *(written, ~200 words)*

No code required. Answer this in your README:

> *"The system serves 10,000 queries/day. Result relevance silently degrades - no errors are
> thrown, but users are getting poor matches. How would you detect this before users complain,
> and what is your first operational change?"*

There is no right answer. We are looking for how you reason about silent failure and
observability - not a specific solution.

---

## Observability Requirements *(cross-cutting)*

- Every request must carry a `trace_id` propagated through all stages
- Structured logs emitted at each stage boundary with at minimum:
`trace_id`, `stage`, `duration_ms`, `item_count`
- A latency spike in reranking must be distinguishable from a spike in vector recall from
logs alone

---

## API Contracts

### `POST /agent/refine`

```json
// Request
{
  "message": "industrial software providers for warehouse operations in UK",
  "base_query": null,
  "history": [],
  "max_iterations": 3
}

// Response
{
  "refined_query": {
    "query_text": "industrial software for warehouse operations in UK",
    "geography": ["United Kingdom"],
    "exclusions": []
  },
  "rationale": "Stopped after 2 iterations - second pass produced stable top-10 with sufficient score spread and low filter-drop ratio.",
  "actions": [
    { "id": "show_results", "label": "Show results", "payload": {} }
  ],
  "iterations_used": 2,
  "meta": {}
}
```

### `POST /search/run`

```json
// Request
{
  "query": {
    "query_text": "industrial software for warehouse operations in UK",
    "geography": ["United Kingdom"],
    "exclusions": []
  },
  "top_k_raw": 1000,
  "top_k_final": 50,
  "offset": 0
}

// Response
{
  "results": [
    {
      "id": "company-123",
      "company_name": "Example Co",
      "country": "United Kingdom",
      "score": 0.89,
      "score_components": {
        "vector": 0.81,
        "rerank_boost": 0.08
      },
      "long_offering": "Example Co provides warehouse management software..."
    }
  ],
  "total": 50,
  "diagnostics": {
    "raw_count": 1000,
    "filtered_count": 312,
    "reranked_count": 50,
    "drop_reasons": {
      "geography_mismatch": 200,
      "exclude_term": 57,
      "low_vector_score": 55
    },
    "stage_latency_ms": {
      "vector_recall": 238,
      "post_filter": 41,
      "rerank": 63
    }
  }
}
```

---

## Non-goals

- Perfect ranking quality
- Production-grade authentication
- Frontend UI

---

## Setup

1. Create a local environment and install dependencies:
  ```bash
  make install
  ```
2. Copy the environment template and set your OpenAI API key:
  ```bash
  cp .env.example .env
  ```
3. Ingest the company corpus into embedded Qdrant:
  ```bash
  make ingest
  ```
4. Run the API:
  ```bash
  make run
  ```
5. Run the unit tests:
  ```bash
  make test
  ```

FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Usage

Example search request:

```bash
curl -X POST http://127.0.0.1:8000/search/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "query_text": "industrial software for warehouse operations in UK",
      "geography": ["United Kingdom"],
      "exclusions": []
    },
    "top_k_raw": 50,
    "top_k_final": 10,
    "offset": 0
  }'
```

Example refine request:

```bash
curl -X POST http://127.0.0.1:8000/agent/refine \
  -H "Content-Type: application/json" \
  -d '{
    "message": "industrial software providers for warehouse operations in UK",
    "base_query": null,
    "history": [],
    "max_iterations": 3
  }'
```

---

## Submission (Fork Required)

1. Fork this repository to your own GitHub account.
2. Complete the task in your fork.
3. Commit your changes with clear messages.
4. Share your fork URL (and branch name if not `main`) for review.

---

## Deliverables

- Working API for both subtasks
- README covering:
  - Assumptions made
  - Termination condition rationale
  - Production answer (Subtask 3)
  - What you would do with more time

---

## Implementation Notes

### Assumptions made

- The actual dataset file in this repository is `company_1000_data - Results.csv`.
- The fields used for ingestion are `Consolidated ID`, `Company Name`, `Country`, and `Long Offering`.
- OpenAI embeddings are used for ingestion and live query embedding.
- Qdrant embedded mode is used for local development and take-home simplicity.

### Termination condition rationale

`POST /agent/refine` uses a controller-driven multishot loop. The controller, not the LLM, owns stop conditions. The system stops when:

- results are strong and stable enough to stop successfully
- no meaningful improvement is observed for the configured iteration window
- no usable results remain after refinement attempts
- a regression is detected and the controller reverts to the best known state
- `max_iterations` is reached

This was chosen to keep the loop deterministic, inspectable, and robust even when LLM output is unavailable or weak.

### Production readiness answer

If this system served 10,000 queries per day and relevance silently degraded, I would detect it by adding quality proxy metrics to the existing structured logs and watching their rolling baselines. The most useful signals here are `top_score`, `score_spread`, `geo_match_rate`, `filtered_count`, and the distribution of `drop_reasons`. None of these require human labels, but together they tell us whether the system is still finding strong matches or whether retrieval has drifted into broader, weaker candidates. I would also maintain a small set of golden queries, run them continuously in the background, and compare their top results and diagnostic metrics against a known-good baseline. That gives an early warning before users complain.

My first operational change would be to alert on rolling degradation in `top_score` and `score_spread` for those golden queries and for live traffic cohorts. Those two metrics are already close to the relevance problem and are much more actionable than generic latency or error dashboards. Once the degradation is visible, I would inspect whether the issue comes from embeddings, corpus changes, retrieval recall, or reranking logic.

### What I would do with more time

- separate `retrieval_query_text` from the returned refined query
- improve domain-specific query expansions for operational searches
- make the refiner stronger at rejecting broad CRM/business software matches
- add endpoint-level integration tests
- add startup fail-fast checks around vector store readiness
- keep tuning retrieval quality using the README example queries
