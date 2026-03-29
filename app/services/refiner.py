from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.schemas import Action, QueryPayload, RefineRequest, RefineResponse, SearchRequest
from app.utils.json_contract import parse_json_contract

load_dotenv()

_openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling)
# ---------------------------------------------------------------------------

_REFINE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "refine_query",
        "description": (
            "Extract a structured M&A company search query from the user's natural language intent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": (
                        """A semantically rich search phrase for cosine-similarity matching against
                        company long_offering descriptions. Capture the core business problem, 
                        target market, and value proposition being sought — not just category labels. 
                        For degenerate or vague inputs (single char, gibberish), use the safe fallback: 
                        'enterprise B2B software company'."""
                    ),
                },
                "geography": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Countries or regions explicitly or implicitly mentioned. "
                        "Resolve informal references: 'UK' → 'United Kingdom', 'US' → 'United States'. "
                        "Return empty list if no geography signal present."
                    ),
                },
                "domain": {
                    "type": "string",
                    "description": (
                        "Primary industry or business domain (e.g. 'logistics', 'fintech', 'healthcare', "
                        "'HR tech'). One word or short phrase. Empty string if not determinable."
                    ),
                },
                "include_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific signals that should be present in matching company descriptions. "
                        "Capture ALL of: deployment/delivery model signals ('SaaS', 'on-premise', "
                        "'field-deployed', 'cloud'), use-case terms ('onboarding', 'frontline'), "
                        "and any other keyword the user explicitly emphasised. "
                        "Examples: ['field-deployed', 'frontline'] or ['SaaS', 'logistics operators']."
                    ),
                },
                "exclude_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Concepts or terms that must NOT appear in matching companies. "
                        "Use when user says 'not', 'excluding', 'avoid', 'not focused on'. "
                        "Example: 'Fintech not focused on payments' → ['payments']."
                    ),
                },
                "query_label": {
                    "type": "string",
                    "enum": ["valid", "irrelevant"],
                    "description": (
                        "Classify the user's input before processing: "
                        "'valid' — a clear, actionable M&A company search intent; "
                        "'irrelevant' — completely unrelated to finding acquisition targets (e.g. weather, cooking, personal questions). Or not an M&A search request (e.g. asking for advice, explanations, or definitions). "
                        "Only 'valid' proceeds to the search pipeline."
                    ),
                },
                "classification_reason": {
                    "type": "string",
                    "description": (
                        "One sentence explaining why this label was assigned. "
                        "For 'valid', briefly confirm the intent. "
                        "For all other labels, state clearly what is missing or wrong so the user can fix it."
                    ),
                },
            },
            "required": [
                "query_text", "geography", "exclude_terms", "include_keywords",
                "query_label", "classification_reason",
            ],
        },
    },
}

_EVALUATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evaluate_results",
        "description": "Evaluate search result quality against the user's M&A search intent.",
        "parameters": {
            "type": "object",
            "properties": {
                "quality_score": {
                    "type": "number",
                    "description": (
                        "Float 0.0–1.0: how well the top results match the user's full intent. "
                        "1.0 = strong match on all dimensions. 0.0 = completely irrelevant. "
                        "Score >= 0.75 means results are acceptable."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": "Explanation of the quality score.",
                },
                "feedback": {
                    "type": "string",
                    "description": (
                        "Actionable guidance for the next refinement if quality_score < 0.75. "
                        "Name exactly what signals to add, remove, or emphasise. "
                        "Empty string if quality_score >= 0.75."
                    ),
                },
            },
            "required": ["quality_score", "rationale", "feedback"],
        },
    },
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = """\
You are an expert M&A search analyst helping identify acquisition targets.

Users describe the type of company they are looking for in natural language. \
Your job is to FIRST classify the user's input, then extract a structured, semantically \
precise search query that will be used for vector similarity search against a corpus of \
~1,000 company descriptions.

── Step 1: Classify the input ──────────────────────────────────────────────────
Set query_label to one of:
  • valid      — a clear, actionable search query related to companies.
  • irrelevant — completely vague, unclear)

Only classify as "irrelevant" if the query is clearly unrelated to companies
or cannot reasonably map to identifying businesses (e.g., weather, cooking,
definitions, general advice).

Set classification_reason to one sentence explaining the label.
If query_label is NOT "valid", you must still populate all other fields with safe defaults \
( \
  - query_text = "enterprise B2B software company" \
  - geography = "" \
  - include_keywords = [] \
  - exclude_terms = [] \
)

── Step 2: Extract the query (only meaningful when query_label == "valid") ─────

You MUST extract and normalize geography BEFORE constructing query_text.
### 2.1 Geography Extraction & Normalization (MANDATORY)
- Identify any geographic references in the user input.
- Convert ALL informal, abbreviated, or variant forms into STANDARD country names.

Each company description (long_offering) is a 100–400 word paragraph describing:
- What the company does (product or service)
- Who it serves (target market, company size, industry vertical)
- How it delivers value (deployment model, channels, geography)

Guidelines:
- query_text must be rich with domain-specific terminology that would plausibly appear \
  in matching company descriptions — not a direct paraphrase of the user's words.
- Capture the PROBLEM being solved and the TARGET MARKET alongside the product category.
- For exclusion signals ("not focused on X", "excluding Y", "not Y"), populate exclude_terms.
- For deployment signals ("field-deployed", "on-premise", "not cloud-only", "SaaS"), add those
  exact terms to include_keywords — do not use a separate field for them.
- For geography, resolve informal references: "UK" → "United Kingdom", "US" → "United States".
"""

_EVALUATE_SYSTEM = """\
You are evaluating whether M&A company search results adequately match a user's stated intent.

You will receive the user's original natural language query and a sample of the top \
retrieved company descriptions (including their similarity scores).

Assess quality holistically across these dimensions:
- Domain match: do results operate in the industry the user specified?
- Geography: is any geography constraint respected?
- Deployment model: is the delivery model correct (SaaS vs field-deployed vs on-premise)?
- Exclusion compliance: are explicitly excluded concepts absent from top results?
- Problem/use-case alignment: do results reflect the user's problem framing, \
  not just a surface category tag?

Scoring guide:
  1.0 — All dimensions match; strong, relevant, diverse results
  0.8 — Good results with minor gaps on one dimension
  0.6 — Core domain present but key signals missed (geography, deployment, exclusions)
  0.4 — Loosely related results; missing major dimensions
  0.2 — Mostly irrelevant
  0.0 — Completely off or empty

If quality_score < 0.75, populate feedback with specific, actionable guidance on what \
to change in the next query iteration. Otherwise leave feedback empty.
"""

# ---------------------------------------------------------------------------
# QueryRefinerAgent
# ---------------------------------------------------------------------------


class QueryRefinerAgent:
    """
    Agentic loop: iteratively refines a user query and evaluates result quality.

    Each iteration runs two LLM tool calls:
      1. REFINE  — extracts / updates a structured QueryPayload from user intent + prev feedback
      2. EVALUATE — scores the search results and decides whether to iterate

    Termination (explicit and deterministic — first condition that fires wins):
      (a) quality_score >= 0.75  — results are good enough
      (b) should_continue == False returned by the evaluator
      (c) iterations_used == max_iterations — hard cap
    """

    _INVALID_LABELS = {"irrelevant"}

    async def refine(self, request: RefineRequest) -> RefineResponse:
        current_query = request.base_query or default_query_payload()
        prev_feedback: str = ""
        evaluation: dict[str, Any] = {}
        iterations_used = 0

        for i in range(1, request.max_iterations + 1):
            iterations_used = i

            # ── Step 1: Refine (includes classification on iteration 1) ──────
            current_query, query_label, classification_reason = await self._call_refine(
                message=request.message,
                history=request.history,
                prev_feedback=prev_feedback,
                current_query=current_query,
            )
            print(
                f"[1] REFINE iteration={i} | "
                f"query_label={query_label!r} | "
                f"query_text={current_query.query_text!r} | "
                f"geography={current_query.geography} | "
                f"domain={current_query.domain!r} | "
                f"include_keywords={current_query.include_keywords} | "
                f"exclude_terms={current_query.exclude_terms}"
            )

            # ── Early exit for invalid queries (classification only on iter 1)
            if i == 1 and query_label in self._INVALID_LABELS:
                return RefineResponse(
                    refined_query=default_query_payload(),
                    rationale=f"Invalid query ({query_label}): {classification_reason}",
                    actions=[Action(id="refine_query", label="Please provide a specific M&A company search query")],
                    iterations_used=0,
                    meta={
                        "query_label": query_label,
                        "classification_reason": classification_reason,
                        "trace_id": request.trace_id,
                    },
                )

            # ── Step 2: Search (calls POST /search/run internally) ───────────
            results = await self._search(current_query, request.trace_id)

            # ── Step 3: Evaluate ─────────────────────────────────────────────
            evaluation = await self._call_evaluate(
                original_message=request.message,
                results=results,
                iteration=i,
            )

            quality_score: float = float(evaluation.get("quality_score", 0.0))
            print(
                f"[2] EVALUATE iteration={i} | "
                f"quality_score={quality_score} | "
                f"rationale={evaluation.get('rationale', '')!r} | "
                f"feedback={evaluation.get('feedback', '')!r}"
            )

            # Termination (a): quality is good enough; (b): hard cap handled by for-loop
            if quality_score >= 0.75:
                break

            prev_feedback = evaluation.get("feedback", "")

        rationale: str = evaluation.get(
            "rationale",
            f"Completed {iterations_used} iteration(s). Max iterations reached.",
        )

        # ── Final full search with best refined query ────────────────────────
        final_search = await self._final_search(current_query, request.trace_id)

        return RefineResponse(
            refined_query=current_query,
            rationale=rationale,
            actions=default_actions(),
            iterations_used=iterations_used,
            meta={
                "quality_score": evaluation.get("quality_score", 0.0),
                "trace_id": request.trace_id,
                "results": final_search.get("results", []),
                "total": final_search.get("total", 0),
                "diagnostics": final_search.get("diagnostics", {}),
            },
        )

    # ── Refine LLM call ──────────────────────────────────────────────────────

    async def _call_refine(
        self,
        message: str,
        history: list[dict[str, Any]],
        prev_feedback: str,
        current_query: QueryPayload,
    ) -> tuple[QueryPayload, str, str]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": _REFINE_SYSTEM}]

        for turn in history:
            if turn.get("role") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])})

        user_content = f"User intent: {message}"
        if prev_feedback:
            user_content += (
                f"\n\nFeedback from previous iteration "
                f"(use this to improve the query):\n{prev_feedback}"
            )
        messages.append({"role": "user", "content": user_content})

        response = await _openai.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=[_REFINE_TOOL],
            tool_choice={"type": "function", "function": {"name": "refine_query"}},
            temperature=0.2,
        )

        raw_args = response.choices[0].message.tool_calls[0].function.arguments
        parsed = parse_json_contract(raw_args)

        if parsed.get("_parse_error"):
            return current_query, "valid", ""  # keep previous query on parse failure

        query_label: str = parsed.get("query_label", "valid")
        classification_reason: str = parsed.get("classification_reason", "")

        return (
            QueryPayload(
                query_text=(
                    parsed.get("query_text", "")
                    or current_query.query_text
                    or "enterprise B2B software company"
                ),
                geography=parsed.get("geography", current_query.geography),
                domain=parsed.get("domain", current_query.domain),
                include_keywords=parsed.get("include_keywords", current_query.include_keywords),
                exclude_terms=parsed.get("exclude_terms", current_query.exclude_terms),
            ),
            query_label,
            classification_reason,
        )

    # ── Evaluate LLM call ────────────────────────────────────────────────────

    async def _call_evaluate(
        self,
        original_message: str,
        results: list[dict[str, Any]],
        iteration: int,
    ) -> dict[str, Any]:
        if results:
            snippets = []
            for r in results[:5]:
                name = r.get("company_name", "Unknown")
                country = r.get("country", "")
                score = r.get("score", 0.0)
                offering = (r.get("long_offering", ""))[:300]
                snippets.append(
                    f"• {name} ({country}) [similarity={score:.3f}]\n  {offering}..."
                )
            results_text = "\n\n".join(snippets)
        else:
            results_text = "(no results returned — search pipeline may not be implemented yet)"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EVALUATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User intent: {original_message}\n\n"
                    f"Iteration: {iteration}\n\n"
                    f"Top search results:\n{results_text}"
                ),
            },
        ]

        response = await _openai.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=[_EVALUATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "evaluate_results"}},
            temperature=0.0,
        )

        raw_args = response.choices[0].message.tool_calls[0].function.arguments
        parsed = parse_json_contract(raw_args)

        if parsed.get("_parse_error"):
            return {
                "quality_score": 0.0,
                "rationale": "Could not parse evaluator response. Stopping.",
                "feedback": "",
            }

        return parsed

    # ── Internal /search/run call (ASGI transport — no HTTP round-trip) ──────

    async def _search(
        self, query: QueryPayload, trace_id: str
    ) -> list[dict[str, Any]]:
        """Small search (top_k=20) used during the evaluation loop."""
        resp_json = await self._call_search_run(query, trace_id, top_k_raw=20, top_k_final=10)
        return resp_json.get("results", []) if resp_json else []

    async def _final_search(
        self, query: QueryPayload, trace_id: str
    ) -> dict[str, Any]:
        """Full search (top_k_raw=1000, top_k_final=50) after the loop completes."""
        return await self._call_search_run(query, trace_id, top_k_raw=1000, top_k_final=50) or {}

    async def _call_search_run(
        self,
        query: QueryPayload,
        trace_id: str,
        top_k_raw: int,
        top_k_final: int,
    ) -> dict[str, Any] | None:
        from app.main import app as fastapi_app  # lazy — avoids circular import

        search_request = SearchRequest(
            query=query,
            top_k_raw=top_k_raw,
            top_k_final=top_k_final,
            trace_id=trace_id,
        )

        try:
            transport = httpx.ASGITransport(app=fastapi_app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/search/run",
                    json=search_request.model_dump(),
                    timeout=60.0,
                )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_actions() -> list[Action]:
    return [Action(id="ideas", label="Suggest more search ideas")]


def default_query_payload() -> QueryPayload:
    return QueryPayload()


