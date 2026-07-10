/**
 * Cabinet landing tab: the tasks table (RAV BARIACH columns) with Excel/Monday-style
 * filtering — a global search box that matches any cell, plus per-column header
 * filter popovers with a checkbox list of distinct values (multi-select per column,
 * AND across columns). Data refetches every 60s; Monday webhooks keep the server
 * cache fresh, so no manual refresh button is needed.
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
import { Filter } from "lucide-react";

type ColumnFilters = Record<string, Set<string>>;

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
      <table className="min-w-full text-sm" dir="ltr">
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
              {headers.map((h) => (
                <td key={h} className="px-3 py-2 whitespace-nowrap max-w-[28rem] overflow-hidden text-ellipsis">
                  {h === "Task Status" ? statusChip(String(row[h] ?? ""), colors) : String(row[h] ?? "")}
                </td>
              ))}
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

  const { data, isLoading, isError } = useQuery({
    queryKey: ["client-tasks"],
    queryFn: () => api.clientPortal.tasksTable(token!),
    enabled: Boolean(token),
    refetchInterval: 60_000,
  });

  const hasFilters = Object.keys(filters).length > 0;
  const blocks = data?.projects ?? [];
  const visible = useMemo(() => {
    const byProject = selectedId == null ? blocks : blocks.filter((b) => b.project_id === selectedId);
    if (!search && !hasFilters) return byProject;
    return byProject
      .map((b) => ({ ...b, rows: b.rows.filter((r) => rowMatches(r, search, filters)) }))
      .filter((b) => b.rows.length > 0 || (!search && !hasFilters));
  }, [blocks, selectedId, search, filters, hasFilters]);

  const nothingFound = Boolean((search || hasFilters) && blocks.length && !visible.length);

  return (
    <ClientShell>
      <div className="p-4 sm:p-6 space-y-4 overflow-y-auto">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-semibold">{t("tasks.title")}</h1>
          {blocks.length > 1 && (
            <select
              className="h-9 rounded-md border border-border bg-card px-3 text-sm"
              value={selectedId ?? ""}
              onChange={(e) => setSelectedId(e.target.value ? Number(e.target.value) : null)}
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
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t("tasks.search")}
              className="w-64 max-w-full"
            />
            {(hasFilters || search) && (
              <button
                className="text-xs text-primary hover:underline"
                onClick={() => {
                  setFilters({});
                  setSearch("");
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
                blocks={blocks}
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
