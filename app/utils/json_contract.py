from __future__ import annotations

import json
import re
from typing import Any


def parse_json_contract(raw: str) -> dict[str, Any]:
    """
    Robustly parse a JSON object from raw LLM output.

    Attempts in order:
      1. Direct json.loads — handles clean JSON strings (e.g. tool call arguments)
      2. Strip markdown fences (```json ... ``` or ``` ... ```) and retry
      3. Regex-extract the first {...} block and retry
      4. Return a deterministic error shape so callers never receive an exception
    """
    if not raw or not raw.strip():
        return {"_parse_error": True, "raw": ""}

    # ── Attempt 1: direct parse ──────────────────────────────────────────────
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: strip markdown fences ────────────────────────────────────
    stripped = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # ── Attempt 3: extract first {...} block ─────────────────────────────────
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # ── Attempt 4: deterministic error shape ─────────────────────────────────
    return {"_parse_error": True, "raw": raw[:300]}


