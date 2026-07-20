/**
 * Cabinet landing tab: the tasks table (RAV BARIACH columns) with Excel/Monday-style
 * filtering — a global search box that matches any cell, plus per-column header
 * filter popovers with a checkbox list of distinct values (multi-select per column,
 * AND across columns). Data refetches every 60s; Monday webhooks keep the server
 * cache fresh, so no manual refresh button is needed.
 *
 * On top of the table sits a mini-dashboard (completion ring, cladding m², a
 * 6-week dispatch strip, and a "waiting on you" counter) plus quick-filter chips.
 * Both are derived from the currently scoped rows (selected project, or all) and
 * combine (AND) with the existing search/column filters.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Checkbox } from "@/components/ui/checkbox";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import type { ClientTaskBlock } from "@/lib/types";
import { AlertTriangle, Download, Filter, Loader2 } from "lucide-react";

type ColumnFilters = Record<string, Set<string>>;
type ChipKey = "all" | "inProgress" | "waitingOnYou" | "dispatched7d";

// Monday-hosted attachments carry a presigned S3 url that expires in ~1h, so
// the snapshot (cached for hours) never bakes one in — it stores the asset
// id instead ("invoice.pdf (monday-asset:555)") and the browser resolves a
// fresh url from the server at click time. External links (Dropbox etc.,
// "drawing.pdf (https://dropbox.com/...)") don't expire and open directly.
type LinkEntry = { label: string } & ({ url: string } | { assetId: string });

// Possibly several entries, one per line, when a task has multiple
// attachments. Returns null when the cell has no link at all, so callers
// fall back to plain text (e.g. "Task", "Customer").
function parseLinkEntries(raw: string): LinkEntry[] | null {
  if (!raw || !/(https?:\/\/|monday-asset:)/.test(raw)) return null;
  const entries: LinkEntry[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const named = trimmed.match(/^(.*)\s\((https?:\/\/[^\s)]+|monday-asset:[^\s)]+)\)$/);
    if (named) {
      const label = named[1].trim() || "File";
      const target = named[2];
      entries.push(
        target.startsWith("monday-asset:")
          ? { label, assetId: target.slice("monday-asset:".length) }
          : { label, url: target },
      );
      continue;
    }
    const bare = trimmed.match(/^(https?:\/\/\S+)$/);
    if (bare) {
      const last = decodeURIComponent(bare[1].split("/").filter(Boolean).pop() || "");
      entries.push({ label: last.split("?")[0] || "Open", url: bare[1] });
    }
  }
  return entries.length ? entries : null;
}

// One clickable row: external links open directly; Monday-hosted assets
// resolve a fresh download url on click (never reuse a stored one — it may
// have expired) and open it once ready.
function FileLinkRow({ entry, dense }: { entry: LinkEntry; dense?: boolean }) {
  const { token } = useAuth();
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  const rowClass = dense
    ? "flex items-center gap-2 text-sm px-1 py-1.5 rounded hover:bg-muted w-full text-start"
    : "inline-flex items-center gap-1 min-w-0 overflow-hidden text-primary hover:underline";
  const labelClass = dense ? "flex-1 truncate" : "min-w-0 overflow-hidden text-ellipsis whitespace-nowrap";

  if ("url" in entry) {
    return (
      <a href={entry.url} download target="_blank" rel="noreferrer" title={entry.url} className={rowClass}>
        {busy ? <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" /> : <Download className="h-3.5 w-3.5 shrink-0" />}
        <span className={labelClass}>{entry.label}</span>
      </a>
    );
  }

  const openFresh = async () => {
    if (busy || !token) return;
    setBusy(true);
    setFailed(false);
    try {
      const url = await api.clientPortal.fileUrl(token, entry.assetId);
      window.open(url, "_blank", "noopener,noreferrer");
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <button type="button" onClick={openFresh} disabled={busy} className={`${rowClass} disabled:opacity-60`}>
      {busy ? (
        <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
      ) : failed ? (
        <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-destructive" />
      ) : (
        <Download className="h-3.5 w-3.5 shrink-0" />
      )}
      <span className={labelClass}>{entry.label}</span>
    </button>
  );
}

// One-line, truncated by design — the raw URL/asset id never sits in visible
// text ("under the hood" per spec). A single attachment renders as a direct
// clickable row; multiple attachments collapse into a "N files" trigger that
// opens a popover listing every one, each independently clickable.
function LinkCell({ entries }: { entries: LinkEntry[] }) {
  const { t } = useI18n();
  if (entries.length === 1) {
    return (
      <span className="inline-flex min-w-0 max-w-full overflow-hidden">
        <FileLinkRow entry={entries[0]} />
      </span>
    );
  }
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1.5 text-primary hover:underline shrink-0"
        >
          <Download className="h-3.5 w-3.5 shrink-0" />
          {entries.length} {t("tasks.files")}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-1.5 text-foreground">
        {entries.map((entry, i) => (
          <FileLinkRow key={i} entry={entry} dense />
        ))}
      </PopoverContent>
    </Popover>
  );
}

function statusChip(value: string, colors: Record<string, string>) {
  const hex = colors[value];
  if (!hex) return <span>{value}</span>;
  return (
    <span
      className="inline-block rounded px-2 py-0.5 text-xs font-medium text-white whitespace-nowrap"
      style={{ backgroundColor: `#${hex}` }}
    >
      {value}
    </span>
  );
}

function rowMatches(row: Record<string, string>, search: string, filters: ColumnFilters): boolean {
  if (search) {
    const q = search.toLowerCase();
    if (!Object.values(row).some((v) => String(v ?? "").toLowerCase().includes(q))) return false;
  }
  for (const [column, values] of Object.entries(filters)) {
    if (!values.size) continue;
    if (!values.has(String(row[column] ?? ""))) return false;
  }
  return true;
}

function chipMatches(row: Record<string, string>, chip: ChipKey): boolean {
  switch (chip) {
    case "inProgress":
      return row["Task Status"] === "In Progress";
    case "waitingOnYou":
      return row["Task Status"] === "Waiting on Client";
    case "dispatched7d":
      return isWithinLastDays(row["Date of Dispatch"], 7);
    default:
      return true;
  }
}

function parseArea(v: string | undefined): number | null {
  if (!v) return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

function parseDateSafe(v: string | undefined): Date | null {
  if (!v) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

function isWithinLastDays(v: string | undefined, days: number): boolean {
  const d = parseDateSafe(v);
  if (!d) return false;
  const diffMs = Date.now() - d.getTime();
  return diffMs >= 0 && diffMs <= days * 86_400_000;
}

// ISO-8601 week key ("2026-W29") so the 6-week dispatch strip groups by
// calendar week rather than by rolling 7-day windows.
function isoWeekKey(d: Date): string {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = (date.getUTCDay() + 6) % 7; // Monday = 0
  date.setUTCDate(date.getUTCDate() - dayNum + 3);
  const firstThursday = new Date(Date.UTC(date.getUTCFullYear(), 0, 4));
  const firstDayNum = (firstThursday.getUTCDay() + 6) % 7;
  firstThursday.setUTCDate(firstThursday.getUTCDate() - firstDayNum + 3);
  const week = 1 + Math.round((date.getTime() - firstThursday.getTime()) / (7 * 86_400_000));
  return `${date.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

function lastIsoWeekKeys(count: number): string[] {
  const keys: string[] = [];
  for (let i = count - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i * 7);
    keys.push(isoWeekKey(d));
  }
  return keys;
}

function formatThousands(n: number): string {
  return Math.round(n).toLocaleString("en-US").replace(/,/g, " ");
}

type DashboardStats = {
  total: number;
  doneCount: number;
  completionPct: number;
  areaPct: number;
  totalArea: number;
  dispatchedArea: number;
  dispatchedShare: number;
  weeklyDispatchCounts: number[];
  thisWeekCount: number;
  waitingOnClientCount: number;
};

function computeDashboardStats(rows: Record<string, string>[]): DashboardStats {
  const weekKeys = lastIsoWeekKeys(6);
  const weeklyDispatchCounts = new Array(6).fill(0) as number[];
  let doneCount = 0;
  let waitingOnClientCount = 0;
  let totalArea = 0;
  let doneArea = 0;
  let dispatchedArea = 0;

  for (const row of rows) {
    const status = row["Task Status"] ?? "";
    if (status === "Done") doneCount++;
    if (status === "Waiting on Client") waitingOnClientCount++;

    const area = parseArea(row["Cladding Area (m2)"]);
    if (area != null) {
      totalArea += area;
      if (status === "Done") doneArea += area;
    }

    const dispatchDate = row["Date of Dispatch"];
    if (dispatchDate) {
      if (area != null) dispatchedArea += area;
      const d = parseDateSafe(dispatchDate);
      if (d) {
        const idx = weekKeys.indexOf(isoWeekKey(d));
        if (idx >= 0) weeklyDispatchCounts[idx] += 1;
      }
    }
  }

  const total = rows.length;
  return {
    total,
    doneCount,
    completionPct: total > 0 ? Math.round((doneCount / total) * 100) : 0,
    areaPct: totalArea > 0 ? Math.round((doneArea / totalArea) * 100) : 0,
    totalArea,
    dispatchedArea,
    dispatchedShare: totalArea > 0 ? dispatchedArea / totalArea : 0,
    weeklyDispatchCounts,
    thisWeekCount: weeklyDispatchCounts[weeklyDispatchCounts.length - 1] ?? 0,
    waitingOnClientCount,
  };
}

function DashboardStrip({
  rows,
  updatedAt,
  lang,
  onWaitingClick,
}: {
  rows: Record<string, string>[];
  updatedAt: string | null;
  lang: string;
  onWaitingClick: () => void;
}) {
  const { t } = useI18n();
  const stats = useMemo(() => computeDashboardStats(rows), [rows]);

  const RADIUS = 26;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  const ringOffset = CIRCUMFERENCE * (1 - stats.completionPct / 100);
  const maxWeekly = Math.max(1, ...stats.weeklyDispatchCounts);

  return (
    <div className="space-y-1.5">
      {updatedAt && (
        <p className="text-xs text-muted-foreground text-end">
          {t("dash.updated")}: {new Date(updatedAt).toLocaleTimeString(lang === "he" ? "he-IL" : "en-GB")}
        </p>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* a. Completion */}
        <div className="rounded-lg border border-border bg-card p-3 flex items-center gap-3">
          <svg width="56" height="56" viewBox="0 0 64 64" className="shrink-0">
            <circle cx="32" cy="32" r={RADIUS} fill="none" stroke="hsl(var(--muted))" strokeWidth="6" />
            <circle
              cx="32"
              cy="32"
              r={RADIUS}
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={ringOffset}
              transform="rotate(-90 32 32)"
            />
            <text x="32" y="37" textAnchor="middle" fontSize="15" fontWeight="600" className="fill-foreground">
              {stats.completionPct}%
            </text>
          </svg>
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground truncate">{t("dash.completion")}</p>
            <p className="text-sm font-semibold truncate">
              {stats.doneCount} / {stats.total} {t("dash.tasksDone")}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              {stats.areaPct}% {t("dash.byArea")}
            </p>
          </div>
        </div>

        {/* b. Cladding */}
        <div className="rounded-lg border border-border bg-card p-3 flex flex-col justify-between gap-2">
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground truncate">{t("dash.cladding")}</p>
            <p className="text-sm font-semibold truncate">{formatThousands(stats.totalArea)} m²</p>
            <p className="text-xs text-muted-foreground truncate">
              {formatThousands(stats.dispatchedArea)} m² {t("dash.dispatched")}
            </p>
          </div>
          <div className="h-1.5 w-full rounded-full overflow-hidden bg-muted">
            <div
              className="h-full rounded-full"
              style={{ width: `${Math.min(100, stats.dispatchedShare * 100)}%`, background: "#007EB5" }}
            />
          </div>
        </div>

        {/* c. Dispatches, 6 weeks */}
        <div className="rounded-lg border border-border bg-card p-3">
          <p className="text-xs font-medium text-muted-foreground truncate mb-2">{t("dash.dispatches")}</p>
          <div className="flex items-end gap-1 h-8">
            {stats.weeklyDispatchCounts.map((count, i) => {
              const isCurrentWeek = i === stats.weeklyDispatchCounts.length - 1;
              const heightPx = Math.max(3, Math.round((count / maxWeekly) * 32));
              return (
                <div
                  key={i}
                  className="w-2 rounded-sm"
                  style={{ height: `${heightPx}px`, backgroundColor: "#007EB5", opacity: isCurrentWeek ? 1 : 0.3 }}
                  title={String(count)}
                />
              );
            })}
          </div>
          <p className="text-xs text-muted-foreground mt-1.5">
            {stats.thisWeekCount} {t("dash.thisWeek")}
          </p>
        </div>

        {/* d. Waiting on you */}
        <button
          type="button"
          onClick={onWaitingClick}
          className="rounded-lg border border-border bg-card p-3 text-start hover:border-primary/50 transition-colors"
        >
          <p className="text-2xl font-bold" style={{ color: "#175A63" }}>
            {stats.waitingOnClientCount}
          </p>
          <p className="text-xs font-medium text-muted-foreground">{t("dash.waitingOnYou")}</p>
        </button>
      </div>
    </div>
  );
}

function QuickChips({
  active,
  counts,
  onChange,
}: {
  active: ChipKey;
  counts: Record<ChipKey, number>;
  onChange: (chip: ChipKey) => void;
}) {
  const { t } = useI18n();
  const items: { key: ChipKey; label: string }[] = [
    { key: "all", label: t("chips.all") },
    { key: "inProgress", label: t("chips.inProgress") },
    { key: "waitingOnYou", label: t("chips.waitingOnYou") },
    { key: "dispatched7d", label: t("chips.dispatched7d") },
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {items.map((item) => {
        const isActive = active === item.key;
        return (
          <button
            key={item.key}
            type="button"
            onClick={() => onChange(item.key)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              isActive
                ? "bg-primary text-primary-foreground"
                : "border border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            {item.label} <span className={isActive ? "opacity-80" : "opacity-70"}>({counts[item.key]})</span>
          </button>
        );
      })}
    </div>
  );
}

function ColumnFilterPopover({
  column,
  blocks,
  filters,
  setFilters,
}: {
  column: string;
  blocks: ClientTaskBlock[];
  filters: ColumnFilters;
  setFilters: (f: ColumnFilters) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const allValues = useMemo(() => {
    const values = new Set<string>();
    blocks.forEach((b) => b.rows.forEach((r) => values.add(String(r[column] ?? "").trim())));
    return [...values].sort((a, b) => a.localeCompare(b));
  }, [blocks, column]);

  const shownValues = query
    ? allValues.filter((v) => (v || "—").toLowerCase().includes(query.toLowerCase()))
    : allValues;

  const selected = filters[column] ?? new Set<string>();
  const isActive = selected.size > 0;

  const updateColumn = (next: Set<string>) => {
    const nextFilters = { ...filters };
    if (next.size) nextFilters[column] = next;
    else delete nextFilters[column];
    setFilters(nextFilters);
  };

  const toggleValue = (value: string, checked: boolean) => {
    const next = new Set(selected);
    if (checked) next.add(value);
    else next.delete(value);
    updateColumn(next);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          className={`shrink-0 ${isActive ? "text-primary" : "text-muted-foreground hover:text-foreground"}`}
          aria-label={column}
        >
          <Filter className="h-3 w-3" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-56 p-2 text-foreground">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t("tasks.filterSearch")}
          className="h-7 text-xs mb-2"
        />
        <div className="flex items-center justify-between mb-1.5 text-xs">
          <button
            className="text-primary hover:underline"
            onClick={() => updateColumn(new Set(allValues))}
          >
            {t("tasks.selectAll")}
          </button>
          <button
            className="text-muted-foreground hover:text-destructive"
            onClick={() => updateColumn(new Set())}
          >
            {t("tasks.clearColumn")}
          </button>
        </div>
        <div className="max-h-56 overflow-y-auto space-y-1">
          {shownValues.map((v) => (
            <label key={v} className="flex items-center gap-2 text-xs cursor-pointer py-0.5">
              <Checkbox
                checked={selected.has(v)}
                onCheckedChange={(checked) => toggleValue(v, checked === true)}
              />
              <span className="truncate">{v || "—"}</span>
            </label>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

function TaskTable({
  rows,
  headers,
  colors,
  blocks,
  filters,
  setFilters,
}: {
  rows: Record<string, string>[];
  headers: string[];
  colors: Record<string, string>;
  blocks: ClientTaskBlock[];
  filters: ColumnFilters;
  setFilters: (f: ColumnFilters) => void;
}) {
  const { t } = useI18n();
  if (!rows.length) return <p className="text-sm text-muted-foreground py-4">{t("tasks.empty")}</p>;
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="min-w-full text-sm client-task-table" dir="ltr">
        <thead className="bg-muted/60">
          <tr>
            {headers.map((h) => (
              <th key={h} className="px-3 py-2 text-left font-semibold whitespace-nowrap text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  {h}
                  <ColumnFilterPopover column={h} blocks={blocks} filters={filters} setFilters={setFilters} />
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-border hover:bg-muted/30">
              {headers.map((h, hi) => {
                const value = String(row[h] ?? "");
                const isTitle = hi === 0;
                const isEmpty = !isTitle && value.trim() === "";
                const links = h === "Task Status" ? null : parseLinkEntries(value);
                const cellClass = [
                  "px-3 py-2 max-w-[28rem] overflow-hidden text-ellipsis",
                  links ? "" : "whitespace-nowrap",
                  isTitle ? "cell-title" : "",
                  isEmpty ? "cell-empty" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <td key={h} data-label={h} className={cellClass}>
                    {h === "Task Status" ? statusChip(value, colors) : links ? <LinkCell entries={links} /> : value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ClientTasksPage() {
  const { token } = useAuth();
  const { t, lang } = useI18n();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState<ColumnFilters>({});
  const [activeChip, setActiveChip] = useState<ChipKey>("all");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["client-tasks"],
    queryFn: () => api.clientPortal.tasksTable(token!),
    enabled: Boolean(token),
    refetchInterval: 60_000,
  });

  const hasFilters = Object.keys(filters).length > 0;
  const hasChip = activeChip !== "all";
  const blocks = data?.projects ?? [];
  // Filter popovers must offer values only from the selected project's rows,
  // so the column-value universe is scoped BEFORE search/filters are applied.
  const scopedBlocks = useMemo(
    () => (selectedId == null ? blocks : blocks.filter((b) => b.project_id === selectedId)),
    [blocks, selectedId],
  );
  // Dashboard + chip counts are derived from the scoped rows only — they must
  // NOT react to search/column filters or the active chip itself.
  const scopedRows = useMemo(() => scopedBlocks.flatMap((b) => b.rows), [scopedBlocks]);
  const latestFetchedAt = useMemo(() => {
    let latest: string | null = null;
    for (const b of scopedBlocks) {
      if (b.fetched_at && (!latest || b.fetched_at > latest)) latest = b.fetched_at;
    }
    return latest;
  }, [scopedBlocks]);
  const chipCounts = useMemo<Record<ChipKey, number>>(
    () => ({
      all: scopedRows.length,
      inProgress: scopedRows.filter((r) => r["Task Status"] === "In Progress").length,
      waitingOnYou: scopedRows.filter((r) => r["Task Status"] === "Waiting on Client").length,
      dispatched7d: scopedRows.filter((r) => isWithinLastDays(r["Date of Dispatch"], 7)).length,
    }),
    [scopedRows],
  );

  const visible = useMemo(() => {
    if (!search && !hasFilters && !hasChip) return scopedBlocks;
    return scopedBlocks
      .map((b) => ({ ...b, rows: b.rows.filter((r) => rowMatches(r, search, filters) && chipMatches(r, activeChip)) }))
      .filter((b) => b.rows.length > 0 || (!search && !hasFilters && !hasChip));
  }, [scopedBlocks, search, filters, hasFilters, activeChip, hasChip]);

  const nothingFound = Boolean((search || hasFilters || hasChip) && blocks.length && !visible.length);

  return (
    <ClientShell>
      <div className="p-4 sm:p-6 space-y-4 overflow-y-auto">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-semibold">{t("tasks.title")}</h1>
          {blocks.length > 1 && (
            <select
              className="h-9 rounded-md border border-border bg-card px-3 text-sm"
              value={selectedId ?? ""}
              onChange={(e) => {
                setSelectedId(e.target.value ? Number(e.target.value) : null);
                setFilters({}); // stale filters from another project would hide everything
              }}
            >
              <option value="">{t("common.allProjects")}</option>
              {blocks.map((b) => (
                <option key={b.project_id} value={b.project_id}>
                  {b.project_name}
                </option>
              ))}
            </select>
          )}
        </div>

        {data && blocks.length > 0 && (
          <DashboardStrip
            rows={scopedRows}
            updatedAt={latestFetchedAt}
            lang={lang}
            onWaitingClick={() => setActiveChip("waitingOnYou")}
          />
        )}

        {data && blocks.length > 0 && (
          <QuickChips active={activeChip} counts={chipCounts} onChange={setActiveChip} />
        )}

        {data && blocks.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t("tasks.search")}
              className="w-64 max-w-full"
            />
            {(hasFilters || search || hasChip) && (
              <button
                className="text-xs text-primary hover:underline"
                onClick={() => {
                  setFilters({});
                  setSearch("");
                  setActiveChip("all");
                }}
              >
                {t("tasks.clearFilters")}
              </button>
            )}
          </div>
        )}

        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-40 w-full" />
          </div>
        )}
        {isError && <p className="text-sm text-destructive">{t("common.error")}</p>}
        {!isLoading && !isError && !blocks.length && (
          <p className="text-sm text-muted-foreground">{t("tasks.empty")}</p>
        )}
        {nothingFound && <p className="text-sm text-muted-foreground">{t("tasks.noMatches")}</p>}

        {visible.map((block) => (
          <section key={block.project_id} className="space-y-2">
            <div className="flex items-baseline justify-between gap-3">
              <h2 className="text-base font-semibold">{block.project_name}</h2>
              {block.fetched_at && (
                <span className="text-xs text-muted-foreground">
                  {t("tasks.updated")}: {new Date(block.fetched_at).toLocaleTimeString(lang === "he" ? "he-IL" : "en-GB")}
                </span>
              )}
            </div>
            {block.error ? (
              <p className="text-sm text-destructive py-4">{t("tasks.loadError")}</p>
            ) : (
              <TaskTable
                rows={block.rows}
                headers={data!.headers}
                colors={data!.status_colors}
                blocks={scopedBlocks}
                filters={filters}
                setFilters={setFilters}
              />
            )}
          </section>
        ))}
      </div>
    </ClientShell>
  );
}
