# Blueknight Solution Notes

## What is implemented

This project now includes:

- `POST /search/run`
  - vector recall over embedded `long_offering`
  - post-filtering for score floor, geography, exclusions, and light domain mismatch
  - deterministic reranking with structured score components
- `POST /agent/refine`
  - multishot refinement loop
  - deterministic controller for action selection and stopping
  - semantic diagnoser with heuristic fallback when no OpenAI key is available
- Qdrant embedded storage
- CSV ingestion and corpus versioning
- retries, timeout handling, concurrency protection, and memoization
- unit tests for parsing, normalization, filtering, reranking, controller behavior, and retrieval wrapper behavior

## Assumptions

- The actual corpus file in this repo is `company_1000_data - Results.csv`, not `data/companies.csv`.
- The relevant fields are:
  - `Consolidated ID`
  - `Company Name`
  - `Country`
  - `Long Offering`
- OpenAI embeddings are used for ingestion and query embedding.
- Qdrant embedded mode is sufficient for this take-home and keeps setup lightweight.

## Termination rationale

The refinement loop is intentionally controller-driven rather than LLM-driven. The LLM or heuristic diagnoser can suggest what is wrong with the current result set, but deterministic control logic owns stopping.

The loop currently stops when one of these conditions is met:

- results are strong and stable enough to stop successfully
- no meaningful improvement is observed for the configured number of iterations
- no usable results remain after refinement attempts
- a regression is detected and the controller reverts to the best known query
- `max_iterations` is reached

This keeps the loop auditable and prevents the system from becoming a prompt-only refinement chain.

## Remaining improvement areas

The main remaining weakness is not architecture but quality tuning. The system runs end to end, but some queries still retrieve semantically adjacent software companies instead of strongly on-target operational vendors.

### Improvement 1: Retrieval quality

The current recall sometimes surfaces broad CRM or business software candidates for operational queries such as warehouse or logistics software.

The next improvement would be to separate:

- `refined_query.query_text` for API output
- `retrieval_query_text` for embedding recall

The retrieval query should expand operational intent more explicitly. For example, a warehouse query should retrieve with terms closer to:

- warehouse management
- WMS
- inventory operations
- fulfillment
- distribution center
- logistics operations

This should improve first-stage recall before reranking even runs.

### Improvement 2: Refiner quality

The multishot loop is structurally correct, but the rewrite quality can still improve.

The next improvement would be to make the diagnoser and controller more aggressive when the top results are:

- geographically correct
- but operationally wrong
- and dominated by generic CRM, sales, or business-management software

In that case the controller should prefer:

- `narrow_domain`
- `rewrite_query_text`
- stronger operational query expansions

rather than simply stopping after low improvement.

## What I would do with more time

- add an internal `retrieval_query_text` path distinct from the returned refined query
- improve synonym and phrase expansion for operational domains
- add stronger domain mismatch filtering for broad business software
- add endpoint-level integration tests
- add startup fail-fast checks for missing vector store and corpus version
- expand README examples with observed real outputs from the implemented system
