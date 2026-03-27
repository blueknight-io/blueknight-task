from __future__ import annotations

from app.utils.json_contract import parse_json_contract


def test_parse_json_contract_accepts_direct_json() -> None:
    parsed = parse_json_contract('{"answer": 1}')
    assert parsed == {"answer": 1}


def test_parse_json_contract_accepts_fenced_json() -> None:
    parsed = parse_json_contract('```json\n{"status":"ok"}\n```')
    assert parsed == {"status": "ok"}


def test_parse_json_contract_extracts_embedded_json() -> None:
    parsed = parse_json_contract('Result follows:\n{"query":"abc","score":0.7}\nThanks')
    assert parsed == {"query": "abc", "score": 0.7}


def test_parse_json_contract_returns_error_shape_on_invalid_json() -> None:
    parsed = parse_json_contract("not valid json")
    assert "_parse_error" in parsed
    assert parsed["_raw"] == "not valid json"


def test_parse_json_contract_rejects_top_level_arrays() -> None:
    parsed = parse_json_contract('["a", "b"]')
    assert parsed["_parse_error"].startswith("Expected top-level object")
