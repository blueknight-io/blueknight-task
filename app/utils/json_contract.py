from __future__ import annotations

import json
import re
from typing import Any


def parse_json_contract(raw: str) -> dict[str, Any]:
    candidates = [raw.strip()]

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    if fenced != raw.strip():
        candidates.append(fenced.strip())

    match = re.search(r"\{.*\}", raw.strip(), flags=re.DOTALL)
    if match:
        candidates.append(match.group(0).strip())

    last_error = "No JSON object found"
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = f"Expected top-level object, got {type(parsed).__name__}"

    return {"_parse_error": last_error, "_raw": raw}
