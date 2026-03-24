import type { RefineRequest, RefineResponse, SearchRequest, SearchResponse } from "./types";

function apiBase(): string {
  const base = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "";
  return base || "http://127.0.0.1:8000";
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${apiBase()}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 500)}`);
  }
  return res.json() as Promise<T>;
}

export async function refineQuery(payload: RefineRequest): Promise<RefineResponse> {
  return postJson<RefineResponse>("/agent/refine", payload);
}

export async function runSearch(payload: SearchRequest): Promise<SearchResponse> {
  return postJson<SearchResponse>("/search/run", payload);
}
