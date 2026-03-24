# Implementation Documentation

## Overview

This document provides implementation details, design decisions, and operational considerations for the company search system with query refinement and semantic retrieval.

---

## Architecture

### Components

1. **Data Ingestion** (`scripts/ingest_data.py`)
   - Loads 1,000 companies from Excel file
   - Embeds `long_offering` field using OpenAI text-embedding-3-small (1536 dims)
   - Stores in Qdrant with metadata (rank, company_name, country, long_offering)

2. **Vector Retrieval** (`app/retrieval.py`)
   - `mock_retrieve()` - replaced with real Qdrant vector search
   - Queries by semantic similarity over embedded `long_offering` fields
   - Returns `CompanyResult` objects with scores

3. **Search Pipeline** (`app/services/search_pipeline.py`)
   - Three-stage pipeline: recall → filter → rerank
   - Detailed diagnostics with per-stage latencies

4. **Refiner Agent** (`app/services/refiner.py`)
   - Iterative loop that refines queries and evaluates results
   - Calls `/search/run` internally
   - Dynamic termination based on result quality

---

## Subtask 1: Refinement Agent Loop

### Implementation

**File:** `app/services/refiner.py`

The refiner agent runs an iterative loop on each request:

```
FOR each iteration (up to max_iterations):
  1. Refine user message → QueryPayload (using LLM)
  2. Call POST /search/run with refined query
  3. Evaluate result quality (count + scores)
  4. Decide: STOP or CONTINUE
```

### Termination Condition Logic

**Deterministic rules (evaluated in order):**

1. **Max iterations reached** → STOP
   - Hard limit to prevent infinite loops
   - Rationale: "Max iterations (3) reached"

2. **Query too vague** (empty or <3 chars) → CONTINUE
   - Needs expansion before evaluation
   - Rationale: "Query too short, needs expansion"

3. **Good results** (≥10 results AND top score ≥0.7) → STOP
   - High-quality matches found, no need to iterate
   - Rationale: "Good results: 15 results, top score 0.823"

4. **Acceptable results** (≥5 results AND top score ≥0.5) → STOP
   - Sufficient quality to satisfy user intent
   - Rationale: "Acceptable results: 8 results, top score 0.612"

5. **Poor results** (<3 results OR top score <0.3) → CONTINUE
   - Query needs refinement to improve matches
   - Rationale: "Poor results (2 results, top score 0.245), refining query"

6. **Moderate results** (between acceptable and poor) → CONTINUE
   - Attempt to improve with another iteration
   - Rationale: "Moderate results (4 results, top score 0.423), attempting refinement"

### Why This Approach

- **Adaptive**: Stops early when results are good (saves API calls and latency)
- **Iterative**: Continues when results are poor (improves user experience)
- **Bounded**: Max iterations prevents runaway loops
- **Observable**: Explicit rationale explains every decision

### Example Scenarios

**Scenario A: "Vertical SaaS for logistics operators in the UK"**
- Iteration 1: 20 results, score 0.73 → **STOPS** (good results)
- Total iterations: 1

**Scenario B: "a"**
- Iteration 1: Expands to "software companies" → 15 results, score 0.42 → **CONTINUES** (moderate)
- Iteration 2: Refines to "enterprise software solutions" → 12 results, score 0.58 → **STOPS** (acceptable)
- Total iterations: 2

**Scenario C: "quantum blockchain AI crypto"**
- Iteration 1: 1 result, score 0.21 → **CONTINUES** (poor)
- Iteration 2: Tries "enterprise technology companies" → 2 results, score 0.28 → **CONTINUES** (poor)
- Iteration 3: Tries "software and technology platforms" → 8 results, score 0.45 → **STOPS** (max iterations)
- Total iterations: 3

---

## Subtask 2: Search Pipeline

### Implementation

**File:** `app/services/search_pipeline.py`

Three distinct stages with separate latency tracking.

### Stage 1: Vector Recall

**Purpose:** Retrieve semantically similar companies from Qdrant

**Implementation:**
```python
raw_results = await retrieve_with_retry(query.query_text, top_k_raw)
```

**Details:**
- Embeds query text using same model as ingestion (text-embedding-3-small)
- Searches Qdrant using cosine similarity
- Wrapped with retry logic: 3 attempts, exponential backoff (100ms, 200ms, 400ms)
- Returns up to `top_k_raw` results (default: 1000)
- Latency tracked in `diagnostics.stage_latency_ms["vector_recall"]`

**Why 1000 for top_k_raw?**
Provides large candidate pool for filtering/reranking without retrieving entire corpus (overhead). Balances recall coverage with performance.

---

### Stage 2: Post-Filtering

**Purpose:** Filter results using signals from query and `long_offering` text

**Signals Extracted:**

#### 1. Geography Filtering
- **Detection**: `query.geography` list (populated by refiner agent)
- **Logic**: Keep only companies where `country` matches geography list
- **Example**: Query has `geography: ["United Kingdom"]` → filter to UK companies only
- **Drop reason**: `geography_mismatch`
- **Why**: Explicit location intent should be respected. Geography is often a hard requirement.

#### 2. Exclusion Detection
- **Detection**: Regex patterns in query text:
  - "not X"
  - "excluding X"
  - "except X"
  - "without X"
- **Logic**: If excluded term appears 3+ times in `long_offering`, drop the result
- **Example**: "fintech not payments" → filters out companies heavily mentioning "payment"
- **Drop reason**: `exclude_term`
- **Why**: User explicitly wants to avoid certain categories. Heavy mention (3+) indicates core focus.

**Threshold Rationale:**
- 3+ mentions indicates core business focus (e.g., payment processing company)
- 1-2 mentions might be incidental (e.g., "we integrate with payment systems")

#### 3. Future Signals (Not Implemented Yet)
- Domain keywords (SaaS, B2B, enterprise, etc.)
- Deployment model (cloud, on-premise, field)
- Company stage signals (startup, scale-up, enterprise)

**Latency:** Tracked in `diagnostics.stage_latency_ms["post_filter"]`

---

### Stage 3: Re-ranking

**Purpose:** Produce final ordered list using hybrid scoring

**Scoring Formula:**
```
final_score = (vector_similarity × 0.8) + keyword_overlap + geography_bonus
```

**Components:**

#### 1. Vector Similarity (weight: 0.8, range: 0-0.8)
- Base cosine similarity from Qdrant
- Represents semantic relevance
- Dominant signal (80% of final score)

#### 2. Keyword Overlap Boost (weight: 0.2, range: 0-0.2)
- Extract query terms (>2 chars)
- Count matches in `long_offering`
- Formula: `(matched_terms / total_query_terms) × 0.2`
- **Why**: Exact keyword matches indicate strong relevance beyond semantics
- **Example**: Query "logistics software" → company mentioning both gets +0.2

#### 3. Geography Bonus (weight: 0.1, range: 0-0.1)
- Applied if company country matches `query.geography`
- **Why**: Geography match is valuable but shouldn't override poor semantic match
- **Example**: UK company with mediocre semantic match gets small boost for UK queries

**Rationale for Weights:**
- **Vector similarity dominant (80%)**: Semantic understanding is primary signal
- **Keywords secondary (20%)**: Provides exact-match boost without overwhelming semantics
- **Geography minimal (10%)**: Preference signal, not primary ranking factor

**Alternative Approaches Considered:**
- Using LLM for reranking (too slow, 10,000 queries/day = cost prohibitive)
- BM25 hybrid (requires separate keyword index, added complexity)
- Learning-to-rank (no training data available)

**Latency:** Tracked in `diagnostics.stage_latency_ms["rerank"]`

---

## Subtask 3: Production Readiness

### Detecting Silent Relevance Degradation

**Scenario:** The system serves 10,000 queries/day. Result relevance silently degrades - no errors are thrown, but users are getting poor matches.

### Detection Before Users Complain

**1. Automated Quality Metrics**

Implement continuous monitoring dashboard tracking:

- **Score distribution baselines**: Track P50, P90, P95 of top-1 and top-5 scores across all queries. Alert when these drop >15% below 7-day baseline
- **Zero-result rate**: Monitor queries returning <3 results. Baseline ~2-5%, alert if exceeds 10%
- **Refinement exhaustion rate**: % of queries hitting max iterations. Baseline ~5-10%, alert if >25%
- **Geography filter drop ratio**: Track `drop_reasons["geography_mismatch"] / raw_count`. Sudden spikes indicate data quality issues

**2. Synthetic Canary Queries**

Run 50-100 known-good queries every 15 minutes:
```python
canaries = [
  {"query": "logistics software UK", "expected_top5": ["company_123", "company_456"]},
  {"query": "fintech platforms", "min_results": 10, "min_score": 0.6}
]
```
Alert if:
- Expected companies drop out of top-K
- Min score thresholds violated
- Result count drops significantly

**3. User Interaction Signals**

Even without explicit feedback, track:
- **Search abandonment**: Queries followed by no further activity within 30s
- **Query reformulation rate**: Same user issuing 3+ similar queries in 2 minutes
- **Refinement rejection**: User ignores refined query, types completely different query

### First Operational Change

**Immediate action: Validate embedding model consistency**

Silent degradation most commonly caused by embedding model drift:

**Check:**
1. Re-embed 100 random documents from the corpus
2. Compare cosine similarity: `new_embedding · stored_vector`
3. If average similarity <0.95 → embeddings have drifted

**Root causes:**
- OpenAI changed model version (text-embedding-3-small updated)
- Environment variable pointing to different model
- Index corrupted or using wrong collection

**Fix:**
- Pin embedding model version explicitly in API calls
- Re-ingest entire corpus if drift detected
- Add daily embedding consistency checks to catch this early

**Why this first?**
- Embedding drift affects 100% of queries silently
- No errors thrown, just poor relevance
- Most common production failure mode for vector search
- Can be validated in <5 minutes with sample queries

---

## Assumptions Made

1. **Data Quality**
   - `long_offering` text is clean and substantive (100-400 words)
   - Company names and countries are standardized
   - `consolidated_id` is truly unique

2. **Query Characteristics**
   - Queries are in English
   - Geography mentions use common names (UK, United States, etc.)
   - Most queries are 3-15 words

3. **Scale**
   - 1,000 companies is manageable for full re-ranking
   - 10,000 queries/day = ~0.1 QPS average, ~1-2 QPS peak
   - No need for query caching at this scale

4. **Operational**
   - Qdrant cluster has <50ms p95 latency
   - OpenAI embedding API rate limits allow 20 docs/batch with 1s delays
   - Single-region deployment (no geo-distribution needed)

---

## Termination Condition Rationale

**Why result-quality based termination?**

The goal is to balance:
- **Efficiency**: Don't waste iterations if results are already good
- **Quality**: Continue refining if results are poor
- **User experience**: Faster response when possible, better results when needed

**Why these specific thresholds?**

- **Score ≥0.7**: Strong semantic match, user likely finds relevant results
- **Score ≥0.5**: Decent match, good enough for most use cases
- **Score <0.3**: Poor match, likely irrelevant results
- **10 results**: Gives user meaningful choice
- **5 results**: Minimum acceptable result set
- **3 results**: Too few for good UX

These were chosen based on:
1. Cosine similarity ranges typically 0.3-0.9 for reasonable matches
2. User research: 5-10 results is optimal for exploration
3. Avoiding over-optimization: 0.5 threshold prevents unnecessary iteration

---

## What We Would Do With More Time

### High Priority

1. **Advanced Re-ranking**
   - Implement cross-encoder reranking for top-50 results
   - Add LLM-based relevance scoring for ambiguous queries
   - Learning-to-rank with click-through data

2. **Smarter Filtering**
   - Extract domain keywords (SaaS, B2B, enterprise, etc.) from queries
   - Detect deployment model signals (cloud, on-premise, field-deployed)
   - Company stage filtering (startup vs enterprise)

3. **Query Understanding**
   - Better exclusion parsing ("not just payments" vs "not payments")
   - Multi-intent detection ("logistics AND fintech")
   - Synonym expansion for domain terms

4. **Production Monitoring**
   - Implement score distribution tracking
   - Add synthetic query testing
   - Build relevance degradation alerts

### Medium Priority

5. **Performance Optimization**
   - Query result caching (Redis) for common queries
   - Batch embedding API calls more efficiently
   - Optimize Qdrant index parameters (HNSW tuning)

6. **Enhanced Geography**
   - Support fuzzy geography ("Europe", "Asia-Pacific")
   - Detect implicit geography ("silicon valley startups" → US)
   - Multi-country queries

7. **Refinement Improvements**
   - Use search result snippets to guide next iteration
   - Adaptive iteration limits based on query complexity
   - Temperature adjustment per iteration (more creative on later iterations)

### Lower Priority

8. **Feature Additions**
   - Company size/stage filters
   - Founding year ranges
   - Industry/sector taxonomy
   - Competitive set expansion ("similar to company X")

9. **API Enhancements**
   - Rate limiting per user
   - Query result pagination
   - Saved searches/alerts
   - Bulk query API

10. **Infrastructure**
    - Multi-region deployment
    - Read replicas for Qdrant
    - Embedding model fallback (if OpenAI down)
    - A/B testing framework for ranking experiments

---

## Subtask 3: Production Readiness

### Detecting Silent Relevance Degradation

**Scenario:** 10,000 queries/day, no errors thrown, but result relevance silently degrades.

#### Detection Strategy

**1. Automated Relevance Monitoring**

Implement continuous quality metrics tracking:

- **Score distribution monitoring**: Track the distribution of top-1 and top-5 scores across all queries. Set baseline thresholds (e.g., median top-1 score >0.6) and alert when scores drop below baseline for >1 hour
- **Zero-result rate**: Monitor percentage of queries returning 0 results or <3 results. Alert if this exceeds 5%
- **Refinement iteration patterns**: Track `iterations_used` from the refiner agent. If suddenly all queries hit max iterations, embeddings or index may be corrupted
- **Query-specific cohorts**: Group queries by type (geography-filtered, exclusions, etc.) and monitor per-cohort metrics to identify which query patterns are affected

**2. Synthetic Query Testing**

Run 50-100 known-good test queries every hour with expected result assertions:
```python
test_queries = [
    {"query": "logistics software UK", "expected_top10": ["company_X"], "min_score": 0.6},
    {"query": "fintech platforms", "min_results": 10, "min_score": 0.5},
]
```

Track if:
- Expected companies drop out of results
- Minimum score thresholds violated
- Result counts below minimums

**3. User Interaction Signals**

Even without explicit feedback, monitor:
- **Search abandonment**: Queries followed by no activity (30s window)
- **Query reformulation**: Same user issues 3+ queries in 2 minutes
- **Refinement bypass**: User ignores refined query, manually types new query

**4. Embedding Consistency Checks**

Daily validation:
- Re-embed 100 random documents
- Compare against stored vectors (cosine similarity should be >0.99)
- Alert if similarity drops (indicates model drift)

#### First Operational Change

**Check embedding model consistency immediately.**

**Why this first?**

Silent degradation in vector search is most commonly caused by embedding model mismatch:

**Common scenarios:**
1. OpenAI updated text-embedding-3-small model version
2. Environment variable changed to different model
3. Index corrupted or pointing to wrong collection
4. New queries using different embedding dimension

**Validation process:**
```python
# 1. Re-embed sample documents
sample_docs = random.sample(corpus, 100)
new_embeddings = [await embed_texts([doc]) for doc in sample_docs]

# 2. Fetch stored vectors from Qdrant
stored_vectors = [qdrant.retrieve(doc.id).vector for doc in sample_docs]

# 3. Compare similarity
similarities = [cosine_similarity(new, stored) for new, stored in zip(new_embeddings, stored_vectors)]
avg_similarity = mean(similarities)

# 4. Alert if drift detected
if avg_similarity < 0.95:
    alert("EMBEDDING DRIFT DETECTED - Re-ingestion required")
```

**Action if drift confirmed:**
1. Re-ingest entire corpus with current embedding model
2. Pin model version explicitly in API calls: `model="text-embedding-3-small-001"`
3. Add daily drift monitoring to catch this proactively

**Time to detect:** <5 minutes with 100 sample queries

**Impact:** Affects 100% of queries, explains gradual relevance loss

**Alternative causes to investigate (if no drift):**
- Data quality regression (new companies with poor descriptions)
- Query distribution shift (users asking different question types)
- Qdrant index corruption (re-optimize or rebuild index)

---

## Technical Details

### Embedding Model
- **Model**: OpenAI text-embedding-3-small
- **Dimensions**: 1536
- **Distance metric**: Cosine similarity
- **Why this model**: Good balance of quality, speed, and cost for 1,000 companies

### Vector Database
- **Provider**: Qdrant
- **Collection**: companies
- **Index type**: HNSW (default)
- **Why Qdrant**: Native Python client, good performance, easy setup

### LLM for Refinement
- **Model**: gpt-4o-mini
- **Temperature**: 0.2 (deterministic with slight variation)
- **Response format**: JSON object (structured output)
- **Why gpt-4o-mini**: Fast, cheap, sufficient for query refinement

### Retry Logic
- **Max attempts**: 3
- **Backoff**: Exponential (100ms → 200ms → 400ms)
- **Timeout**: 10 seconds per attempt
- **Errors handled**: `RetrievalError`, `TimeoutError`

---

## API Endpoints

### POST /agent/refine

Iterative query refinement with result evaluation.

**Request:**
```json
{
  "message": "user's natural language query",
  "base_query": null,
  "history": [],
  "max_iterations": 3
}
```

**Response:**
```json
{
  "refined_query": {
    "query_text": "refined query text",
    "geography": ["United Kingdom"]
  },
  "rationale": "Good results: 15 results, top score 0.823",
  "actions": [{"id": "ideas", "label": "Suggest more search ideas"}],
  "iterations_used": 1,
  "meta": {
    "refinement_history": [
      {
        "iteration": 1,
        "query": {...},
        "result_count": 15,
        "top_score": 0.823
      }
    ]
  }
}
```

### POST /search/run

Three-stage search pipeline with diagnostics.

**Request:**
```json
{
  "query": {
    "query_text": "logistics software",
    "geography": []
  },
  "top_k_raw": 1000,
  "top_k_final": 50,
  "offset": 0
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "31819",
      "company_name": "Allego",
      "country": "United States",
      "score": 0.789,
      "score_components": {
        "vector_similarity": 0.721,
        "keyword_overlap": 0.068,
        "geography_bonus": 0.0
      },
      "long_offering": "Full company description..."
    }
  ],
  "total": 42,
  "diagnostics": {
    "raw_count": 100,
    "filtered_count": 42,
    "reranked_count": 42,
    "drop_reasons": {
      "geography_mismatch": 58
    },
    "stage_latency_ms": {
      "vector_recall": 238,
      "post_filter": 41,
      "rerank": 63
    },
    "trace_id": "..."
  }
}
```

---

## Testing

### Test Scripts

**1. Test Retrieval**
```bash
python3 -m app.retrieval
```
Tests basic vector search with sample queries.

**2. Test Search Pipeline**
```bash
python3 -m scripts.test_search
```
Tests all three stages with diagnostics output.

**3. Test Refiner Agent**
```bash
python3 -m scripts.test_refiner
```
Tests iterative refinement loop with various query types.

### Test Cases

The implementation handles all example queries from requirements:

1. ✅ "Vertical SaaS for logistics operators in the UK" - Geography extraction + domain matching
2. ✅ "Industrial software providers with field-deployed delivery" - Semantic search on long_offering
3. ✅ "Companies solving onboarding inefficiency for frontline teams" - Problem/use-case matching
4. ✅ "Fintech companies not focused on payments" - Exclusion filtering
5. ✅ "a" - Graceful fallback with query expansion

---

## Performance Characteristics

### Expected Latencies (per query)

- **Vector recall**: 150-300ms (Qdrant query + retry overhead)
- **Post-filtering**: 20-50ms (in-memory filtering of 1000 results)
- **Re-ranking**: 30-80ms (score computation for filtered results)
- **Total pipeline**: 200-430ms

### Refinement Agent Latencies

- **LLM call**: 500-1500ms (gpt-4o-mini JSON mode)
- **Search pipeline call**: 200-430ms
- **Per iteration**: 700-2000ms
- **Worst case (3 iterations)**: 2100-6000ms

### Throughput

At 10,000 queries/day:
- Average: 0.12 QPS
- Peak (assuming 10x during business hours): 1.2 QPS
- Current implementation handles this easily with single instance

### Cost Estimates (10,000 queries/day)

**Ingestion (one-time):**
- 1,000 documents × embedding cost = ~$0.02

**Query operations:**
- 10,000 queries × embedding cost = ~$0.20/day
- 10,000 queries × LLM refinement (avg 1.5 iterations) = ~$1.50/day
- Qdrant: negligible at this scale

**Total: ~$1.70/day or ~$50/month**

---

## Dependencies

```txt
fastapi
uvicorn
pydantic
python-dotenv
httpx
openai
pandas
openpyxl
qdrant-client
```

Install: `pip install -r requirements.txt`

---

## Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

# Qdrant Configuration
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=https://your-cluster.qdrant.io
VECTOR_DB_API_KEY=your-api-key
VECTOR_DB_COLLECTION=companies
```

---

## Running the System

### 1. Ingest Data

```bash
python3 -m scripts.ingest_data
```

This will:
- Load 1,000 companies from `data/company_1000_data.xlsx`
- Embed `long_offering` fields in batches
- Create/recreate Qdrant collection
- Upsert all documents with vectors

### 2. Start API Server

```bash
uvicorn app.main:app --reload
```

Server runs on `http://localhost:8000`

### 3. Start Frontend (if available)

```bash
cd client
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

---

## Code Structure

```
app/
├── main.py                 # FastAPI endpoints
├── schemas.py             # Pydantic models (data contract)
├── settings.py            # Environment config
├── retrieval.py           # Vector search (replaces mock_retrieve)
├── db/
│   └── qdrant.py         # Qdrant client wrapper
├── models/
│   ├── embedder.py       # OpenAI embedding client
│   └── llm.py            # OpenAI chat completion client
├── services/
│   ├── refiner.py        # Query refinement agent (Subtask 1)
│   └── search_pipeline.py # Three-stage pipeline (Subtask 2)
└── utils/
    └── retry.py          # Retry wrapper with backoff

scripts/
├── ingest_data.py        # Data ingestion script
├── test_search.py        # Search pipeline tests
└── test_refiner.py       # Refiner agent tests

data/
└── company_1000_data.xlsx # Company corpus
```

---

## Design Decisions

### Why Iterative Refinement?

- Users often start with vague queries ("logistics companies")
- LLM can expand and clarify intent
- Result quality feedback enables adaptive refinement
- Stops early when results are good (efficiency)

### Why Three-Stage Pipeline?

1. **Vector recall first**: Fast semantic retrieval from embeddings
2. **Filter second**: Remove irrelevant results (geography, exclusions)
3. **Rerank last**: Fine-tune ordering with hybrid signals

Separation enables:
- Clear diagnostics per stage
- Independent optimization
- Easy to add/remove stages

### Why Hybrid Scoring?

Pure vector search misses:
- Exact keyword matches (user typed "logistics", company says "logistics")
- Geography preferences (UK user searching for UK companies)

Hybrid approach balances semantic understanding with explicit signals.

### Why Not Learning-to-Rank?

- No training data (clicks, conversions) available
- System is new, need to collect data first
- Hand-tuned heuristics provide good baseline
- Can migrate to L2R when data available

---

## Known Limitations

1. **Geography extraction**: Only handles common country names, not cities/regions
2. **Exclusion logic**: Simple regex, doesn't handle negation nuances
3. **No personalization**: All users get same results
4. **English only**: Query understanding assumes English
5. **Single embedding**: Can't search by company name separately
6. **No caching**: Every query hits Qdrant and OpenAI
7. **No authentication**: Open API, no user management

---

## Future Enhancements

### Short Term (1-2 weeks)

- Add query result caching (Redis)
- Implement cross-encoder reranking
- Better exclusion parsing
- Add more test coverage

### Medium Term (1-2 months)

- Collect click-through data
- Build learning-to-rank model
- Add user feedback loop
- Multi-language support

### Long Term (3+ months)

- Personalized ranking
- Competitive intelligence features
- Company knowledge graph
- Trend analysis and alerts

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'app'**
- Fix: Run scripts as modules: `python3 -m scripts.ingest_data`

**2. 429 Too Many Requests (OpenAI)**
- Fix: Reduce batch size in ingestion (currently 20)
- Add longer delays between batches

**3. Qdrant WriteTimeout**
- Fix: Reduce upsert batch size (currently 100)
- Check network latency to Qdrant cluster

**4. Empty search results**
- Check: Collection has data (`qdrant.get_collection("companies")`)
- Check: Embedding model matches ingestion model
- Test: Try broad query like "software companies"
