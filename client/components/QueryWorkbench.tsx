"use client";

import { useState } from "react";
import { refineQuery, runSearch } from "@/services/api";
import type { QueryPayload, RefineResponse, SearchResponse } from "@/services/types";

export function QueryWorkbench() {
  const [message, setMessage] = useState(
    "Vertical SaaS for logistics operators in the UK",
  );
  const [busy, setBusy] = useState<"idle" | "refine" | "search">("idle");
  const [error, setError] = useState<string | null>(null);
  const [refined, setRefined] = useState<RefineResponse | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);

  const structured: QueryPayload = refined?.refined_query ?? {
    query_text: message,
    geography: [],
  };

  async function onRefine() {
    setError(null);
    setBusy("refine");
    try {
      const res = await refineQuery({
        message,
        history: [],
        max_iterations: 3,
      });
      setRefined(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refine failed");
    } finally {
      setBusy("idle");
    }
  }

  async function onSearch() {
    setError(null);
    setBusy("search");
    try {
      const res = await runSearch({
        query: structured,
        top_k_raw: 100,
        top_k_final: 10,
      });
      setSearch(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setBusy("idle");
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 py-8 sm:px-6">
      <section className="flex flex-col gap-3">
        <label
          htmlFor="intent"
          className="text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Describe what you are looking for
        </label>
        <textarea
          id="intent"
          rows={4}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-sm outline-none ring-zinc-400 focus:ring-2 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
          placeholder="Natural language query…"
        />
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onRefine}
            disabled={busy !== "idle" || !message.trim()}
            className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            {busy === "refine" ? "Refining…" : "Refine query"}
          </button>
          <button
            type="button"
            onClick={onSearch}
            disabled={busy !== "idle"}
            className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium text-zinc-900 transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800"
          >
            {busy === "search" ? "Searching…" : "Run search"}
          </button>
        </div>
      </section>

      {error && (
        <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
          {error}
        </p>
      )}

      {refined && (
        <section className="rounded-xl border border-zinc-200 bg-zinc-50/80 p-4 dark:border-zinc-800 dark:bg-zinc-900/40">
          <h2 className="mb-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Refined query
          </h2>
          <p className="text-sm text-zinc-700 dark:text-zinc-300">
            <span className="font-medium">Text:</span>{" "}
            {refined.refined_query.query_text || "—"}
          </p>
          {refined.refined_query.geography.length > 0 && (
            <p className="mt-1 text-sm text-zinc-700 dark:text-zinc-300">
              <span className="font-medium">Geography:</span>{" "}
              {refined.refined_query.geography.join(", ")}
            </p>
          )}
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-500">
            {refined.rationale}
          </p>
          
          {refined.actions && refined.actions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {refined.actions.map((action) => (
                <button
                  key={action.id}
                  type="button"
                  className="rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
          
          {refined.meta?.refinement_history && refined.meta.refinement_history.length > 1 && (
            <div className="mt-4 border-t border-zinc-200 pt-3 dark:border-zinc-700">
              <h3 className="mb-2 text-xs font-medium text-zinc-600 dark:text-zinc-400">
                Refinement History ({refined.iterations_used} iterations)
              </h3>
              <div className="flex flex-col gap-2">
                {refined.meta.refinement_history.map((h) => (
                  <div
                    key={h.iteration}
                    className="rounded-md bg-white/60 px-3 py-2 dark:bg-zinc-800/40"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                        Iteration {h.iteration}
                      </span>
                      <span className="text-xs tabular-nums text-zinc-500">
                        {h.result_count} results, score {h.top_score.toFixed(3)}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
                      {h.query.query_text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {search && search.results.length > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Top results ({search.total})
          </h2>
          <ul className="flex flex-col gap-3">
            {search.results.slice(0, 5).map((r) => (
              <li
                key={r.id}
                className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-medium text-zinc-900 dark:text-zinc-100">
                    {r.company_name || r.id}
                  </span>
                  <span className="text-xs tabular-nums text-zinc-500">
                    score {r.score.toFixed(3)}
                  </span>
                </div>
                <p className="text-xs text-zinc-500">{r.country}</p>
                {r.long_offering && (
                  <p className="mt-2 line-clamp-3 text-sm text-zinc-600 dark:text-zinc-400">
                    {r.long_offering}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
