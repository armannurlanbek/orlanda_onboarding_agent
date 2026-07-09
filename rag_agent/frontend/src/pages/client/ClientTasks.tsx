/**
 * Cabinet landing tab: the tasks table (same columns as the RAV BARIACH sheet).
 * One block per project with a project selector when the client has several.
 * Data is cached server-side and invalidated by Monday webhooks; we also
 * refetch every 60s so the client never needs a refresh button.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Skeleton } from "@/components/ui/skeleton";
import type { ClientTaskBlock } from "@/lib/types";

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

function TaskTable({ block, headers, colors }: { block: ClientTaskBlock; headers: string[]; colors: Record<string, string> }) {
  const { t } = useI18n();
  if (block.error) return <p className="text-sm text-destructive py-4">{t("tasks.loadError")}</p>;
  if (!block.rows.length) return <p className="text-sm text-muted-foreground py-4">{t("tasks.empty")}</p>;
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
          {block.rows.map((row, i) => (
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

  const { data, isLoading, isError } = useQuery({
    queryKey: ["client-tasks"],
    queryFn: () => api.clientPortal.tasksTable(token!),
    enabled: Boolean(token),
    refetchInterval: 60_000,
  });

  const blocks = data?.projects ?? [];
  const visible = useMemo(
    () => (selectedId == null ? blocks : blocks.filter((b) => b.project_id === selectedId)),
    [blocks, selectedId],
  );

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
            <TaskTable block={block} headers={data!.headers} colors={data!.status_colors} />
          </section>
        ))}
      </div>
    </ClientShell>
  );
}
