/**
 * Instant, client-side "project status" reply for the Assistant tab. Renders
 * from the already-fetched tasks table (react-query cache shared with the
 * Tasks tab) — no LLM call, no network round trip of its own beyond an
 * optional progress-URL lookup done by the caller.
 */
import { Link } from "react-router-dom";
import { useI18n } from "@/lib/i18n";
import type { ClientTaskBlock } from "@/lib/types";

function parseFloatSafe(v: string | undefined): number {
  if (!v) return NaN;
  const n = parseFloat(v.replace(/,/g, "").trim());
  return Number.isFinite(n) ? n : NaN;
}

function parseDateSafe(v: string | undefined): Date | null {
  if (!v || !v.trim()) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatM2(n: number): string {
  const rounded = Math.round(n);
  const withSpaces = rounded.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  return `${withSpaces} m²`;
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}

export function ProjectCard({
  block,
  statusColors,
  progressUrl,
}: {
  block: ClientTaskBlock;
  statusColors: Record<string, string>;
  progressUrl?: string;
}) {
  const { t } = useI18n();

  const statusCounts = new Map<string, number>();
  for (const row of block.rows) {
    const status = row["Task Status"];
    if (!status || !status.trim()) continue;
    statusCounts.set(status, (statusCounts.get(status) ?? 0) + 1);
  }

  const totalArea = block.rows.reduce((sum, row) => {
    const n = parseFloatSafe(row["Cladding Area (m2)"]);
    return Number.isFinite(n) ? sum + n : sum;
  }, 0);

  const dispatchedArea = block.rows.reduce((sum, row) => {
    if (!row["Date of Dispatch"] || !row["Date of Dispatch"].trim()) return sum;
    const n = parseFloatSafe(row["Cladding Area (m2)"]);
    return Number.isFinite(n) ? sum + n : sum;
  }, 0);

  const now = Date.now();
  const nextDeadline = block.rows
    .map((row) => parseDateSafe(row["Deadline"]))
    .filter((d): d is Date => d !== null && d.getTime() >= now)
    .sort((a, b) => a.getTime() - b.getTime())[0];

  const lastDispatch = block.rows
    .map((row) => parseDateSafe(row["Date of Dispatch"]))
    .filter((d): d is Date => d !== null)
    .sort((a, b) => b.getTime() - a.getTime())[0];

  return (
    <div className="mt-1 max-w-sm rounded-lg border border-border bg-card overflow-hidden text-foreground" dir="ltr">
      <div className="px-3 py-2 border-b border-border font-semibold text-sm">{block.project_name}</div>

      {statusCounts.size > 0 && (
        <div className="flex flex-wrap gap-1 px-3 py-2 border-b border-border">
          {[...statusCounts.entries()].map(([status, count]) => {
            const hex = statusColors[status];
            return (
              <span
                key={status}
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-white whitespace-nowrap"
                style={{ backgroundColor: hex ? `#${hex}` : "#6b7280" }}
              >
                {status} &middot; {count}
              </span>
            );
          })}
        </div>
      )}

      <div className="grid grid-cols-2 gap-px bg-border">
        <Cell label={t("card.total")} value={formatM2(totalArea)} />
        <Cell label={t("card.dispatched")} value={formatM2(dispatchedArea)} />
        <Cell label={t("card.nextDeadline")} value={nextDeadline ? nextDeadline.toLocaleDateString() : "—"} />
        <Cell label={t("card.lastDispatch")} value={lastDispatch ? lastDispatch.toLocaleDateString() : "—"} />
      </div>

      <div className="flex items-center gap-3 px-3 py-2 border-t border-border text-xs">
        <Link to="/client/tasks" className="text-primary hover:underline">
          {t("card.openTasks")}
        </Link>
        {progressUrl && (
          <a href={progressUrl} target="_blank" rel="noreferrer" className="text-primary hover:underline">
            {t("card.progressPage")}
          </a>
        )}
      </div>
    </div>
  );
}
