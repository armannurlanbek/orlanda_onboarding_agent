import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/apiClient";
import type { MondayKbStatus } from "@/lib/types";
import { AlertTriangle, ExternalLink, File, FileText, RefreshCw } from "lucide-react";
import { toast } from "sonner";

const POLL_MS = 2500;

function fmtDateTime(value: string | null): string {
  if (!value) return "";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" });
}

/** monday.com "Knowledge base" block for the admin Knowledge modal: sync status, synced files, skipped files. */
export function MondayKbSection({ token, open, refreshKey }: { token: string | null; open: boolean; refreshKey: number }) {
  const [status, setStatus] = useState<MondayKbStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const wasRunning = useRef(false);

  const load = useCallback(async () => {
    if (!token) return;
    try {
      setStatus(await api.knowledge.mondayStatus(token));
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Не удалось получить статус monday");
    }
  }, [token]);

  useEffect(() => {
    if (open) load();
  }, [open, refreshKey, load]);

  // Poll while a sync is running; announce the result once it finishes.
  useEffect(() => {
    if (!open || !status?.running) {
      if (wasRunning.current && status && !status.running) {
        if (status.lastOk) toast.success(`monday: синхронизировано, документов в базе — ${status.files.length}`);
        else if (status.lastError) toast.error(`monday: ${status.lastError}`);
      }
      wasRunning.current = Boolean(status?.running);
      return;
    }
    wasRunning.current = true;
    const timer = setInterval(load, POLL_MS);
    return () => clearInterval(timer);
  }, [open, status, load]);

  const sync = async () => {
    if (!token) return;
    setStarting(true);
    try {
      await api.knowledge.mondaySync(token);
      await load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось запустить синхронизацию");
    } finally {
      setStarting(false);
    }
  };

  const running = Boolean(status?.running) || starting;
  const rootName = status?.roots.map((r) => r.name).filter(Boolean).join(", ") || status?.names.join(", ") || "Knowledge base";

  return (
    <div className="mt-6">
      <div className="px-2 py-2 flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em] truncate">
          monday · {rootName}
        </div>
        <Button size="sm" variant="ghost" onClick={sync} disabled={!status?.enabled || running}>
          <RefreshCw className={`h-3.5 w-3.5 ${running ? "animate-spin" : ""}`} />
          {running ? "Синхронизация…" : "Синхронизировать"}
        </Button>
      </div>

      {status === null && !loadError ? (
        <div className="space-y-2 px-2"><Skeleton className="h-10 w-full" /></div>
      ) : (
        <div className="px-3 space-y-2">
          <StatusLine status={status} loadError={loadError} />
          {status && status.files.length > 0 && (
            <ul className="space-y-1 -mx-1">
              {status.files.map((f) => {
                const Icon = f.kind === "doc" ? FileText : File;
                const where = [f.board, f.item].filter(Boolean).join(" › ") || (f.kind === "doc" ? "monday Doc" : "");
                const body = (
                  <>
                    <Icon className="h-4 w-4 text-primary shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-foreground truncate">{f.name}</div>
                      {where && <div className="text-xs text-muted-foreground truncate">{where}</div>}
                    </div>
                    {f.url && <ExternalLink className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 shrink-0" />}
                  </>
                );
                return (
                  <li key={f.docId}>
                    {f.url ? (
                      <a href={f.url} target="_blank" rel="noreferrer" className="group rounded-lg px-2 py-1.5 flex items-center gap-3 hover:bg-muted/60" title="Открыть в monday">
                        {body}
                      </a>
                    ) : (
                      <div className="rounded-lg px-2 py-1.5 flex items-center gap-3">{body}</div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          {status && status.skipped.length > 0 && (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer select-none py-1">Не проиндексировано: {status.skipped.length}</summary>
              <ul className="mt-1 space-y-1 pl-3">
                {status.skipped.map((s, i) => (
                  <li key={`${s.name}-${i}`}>
                    <span className="text-foreground">{s.name}</span>
                    {s.where ? ` (${s.where})` : ""} — {s.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function StatusLine({ status, loadError }: { status: MondayKbStatus | null; loadError: string | null }) {
  if (loadError || !status) {
    return <div className="text-xs text-destructive">{loadError}</div>;
  }
  if (!status.enabled) {
    return <div className="text-xs text-muted-foreground">Синхронизация с monday отключена на сервере.</div>;
  }
  if (status.running) {
    const p = status.progress;
    return (
      <div className="text-xs text-muted-foreground">
        Синхронизация…{p && p.total > 0 ? ` ${p.done} из ${p.total}${p.current ? ` · ${p.current}` : ""}` : " ищем документы в monday"}
      </div>
    );
  }
  if (status.lastError) {
    return (
      <div className="text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded-md px-2 py-1.5 flex gap-1.5">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
        <span>{status.lastError}</span>
      </div>
    );
  }
  if (!status.configured) {
    return (
      <div className="text-xs text-muted-foreground">
        Нужен доступ к monday: задайте RAG_MONDAY_KB_API_TOKEN на сервере или подключите monday под учётной записью администратора.
      </div>
    );
  }
  const auto = status.intervalMinutes > 0 ? `автообновление каждые ${status.intervalMinutes} мин` : "автообновление выключено";
  return (
    <div className="text-xs text-muted-foreground">
      {status.lastFinishedAt ? `Обновлено ${fmtDateTime(status.lastFinishedAt)}` : "Первая синхронизация скоро начнётся"}
      {` · документов: ${status.files.length} · ${auto}`}
      {!status.complete && <span className="text-warning"> · часть источников не прочитана, см. ниже</span>}
    </div>
  );
}
