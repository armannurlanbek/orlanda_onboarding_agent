/**
 * Cabinet landing tab: the tasks table (RAV BARIACH columns) with Monday-style
 * filtering — a global search box that matches any cell plus stackable
 * column/value filters (AND). Data refetches every 60s; Monday webhooks keep
 * the server cache fresh, so no manual refresh button is needed.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import type { ClientTaskBlock } from "@/lib/types";
import { Plus, X } from "lucide-react";

type ColumnFilter = { column: string; value: string };

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

function rowMatches(row: Record<string, string>, search: string, filters: ColumnFilter[]): boolean {
  if (search) {
    const q = search.toLowerCase();
    if (!Object.values(row).some((v) => String(v ?? "").toLowerCase().includes(q))) return false;
  }
  return filters.every((f) => String(row[f.column] ?? "") === f.value);
}

function FilterBar({
  headers,
  blocks,
  search,
  setSearch,
  filters,
  setFilters,
}: {
  headers: string[];
  blocks: ClientTaskBlock[];
  search: string;
  setSearch: (v: string) => void;
  filters: ColumnFilter[];
  setFilters: (f: ColumnFilter[]) => void;
}) {
  const { t } = useI18n();
  const [draftColumn, setDraftColumn] = useState("");

  const valuesFor = (column: string) => {
    const values = new Set<string>();
    blocks.forEach((b) => b.rows.forEach((r) => {
      const v = String(r[column] ?? "").trim();
      if (v) values.add(v);
    }));
    return [...values].sort();
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder={t("tasks.search")}
        className="w-64 max-w-full"
      />

      {filters.map((f, i) => (
        <span
          key={`${f.column}-${f.value}-${i}`}
          className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 text-foreground px-3 py-1 text-xs"
        >
          <b>{f.column}</b>: {f.value}
          <button
            onClick={() => setFilters(filters.filter((_, j) => j !== i))}
            className="text-muted-foreground hover:text-destructive"
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}

      {draftColumn ? (
        <span className="inline-flex items-center gap-1.5">
          <select
            className="h-8 rounded-md border border-border bg-card px-2 text-xs"
            value={draftColumn}
            onChange={(e) => setDraftColumn(e.target.value)}
          >
            {headers.map((h) => (
              <option key={h} value={h}>{h}</option>
            ))}
          </select>
          <select
            className="h-8 rounded-md border border-border bg-card px-2 text-xs max-w-56"
            defaultValue=""
            onChange={(e) => {
              if (e.target.value) {
                setFilters([...filters, { column: draftColumn, value: e.target.value }]);
                setDraftColumn("");
              }
            }}
          >
            <option value="" disabled>{t("tasks.pickValue")}</option>
            {valuesFor(draftColumn).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <button onClick={() => setDraftColumn("")} className="text-muted-foreground hover:text-foreground">
            <X className="h-3.5 w-3.5" />
          </button>
        </span>
      ) : (
        <Button variant="outline" size="sm" className="gap-1.5 h-8" onClick={() => setDraftColumn(headers[0])}>
          <Plus className="h-3.5 w-3.5" /> {t("tasks.addFilter")}
        </Button>
      )}

      {(filters.length > 0 || search) && (
        <button
          className="text-xs text-primary hover:underline"
          onClick={() => {
            setFilters([]);
            setSearch("");
          }}
        >
          {t("tasks.clearFilters")}
        </button>
      )}
    </div>
  );
}

function TaskTable({
  rows,
  headers,
  colors,
}: {
  rows: Record<string, string>[];
  headers: string[];
  colors: Record<string, string>;
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
                {h}
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
  const [filters, setFilters] = useState<ColumnFilter[]>([]);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["client-tasks"],
    queryFn: () => api.clientPortal.tasksTable(token!),
    enabled: Boolean(token),
    refetchInterval: 60_000,
  });

  const blocks = data?.projects ?? [];
  const visible = useMemo(() => {
    const byProject = selectedId == null ? blocks : blocks.filter((b) => b.project_id === selectedId);
    if (!search && !filters.length) return byProject;
    return byProject
      .map((b) => ({ ...b, rows: b.rows.filter((r) => rowMatches(r, search, filters)) }))
      .filter((b) => b.rows.length > 0 || (!search && !filters.length));
  }, [blocks, selectedId, search, filters]);

  const nothingFound = Boolean((search || filters.length) && blocks.length && !visible.length);

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
          <FilterBar
            headers={data.headers}
            blocks={blocks}
            search={search}
            setSearch={setSearch}
            filters={filters}
            setFilters={setFilters}
          />
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
              <TaskTable rows={block.rows} headers={data!.headers} colors={data!.status_colors} />
            )}
          </section>
        ))}
      </div>
    </ClientShell>
  );
}
