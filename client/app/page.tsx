import { AppHeader } from "@/components/AppHeader";
import { QueryWorkbench } from "@/components/QueryWorkbench";

export default function Home() {
  return (
    <div className="min-h-screen bg-zinc-100 dark:bg-zinc-950">
      <AppHeader />
      <QueryWorkbench />
    </div>
  );
}
