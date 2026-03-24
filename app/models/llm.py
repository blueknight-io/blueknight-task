"""OpenAI chat completions via HTTP (httpx)."""

from __future__ import annotations

from typing import Any

import httpx

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    response_format: dict[str, Any] | None = None,
    api_key: str,
    timeout_s: float = 120.0,
) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        body["response_format"] = response_format
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(OPENAI_CHAT_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]
