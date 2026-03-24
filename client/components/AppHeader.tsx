export function AppHeader() {
  return (
    <header className="border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <div className="mx-auto flex max-w-3xl flex-col gap-1 px-4 py-6 sm:px-6">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Company search
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Refine intent and run retrieval against the FastAPI backend.
        </p>
      </div>
    </header>
  );
}
