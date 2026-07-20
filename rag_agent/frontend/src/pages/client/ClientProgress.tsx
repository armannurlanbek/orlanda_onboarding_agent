/**
 * Progress tab: embeds the existing progress-tracking page (/p/{token}) in an
 * iframe with a project selector. The progress page has its own EN/HE toggle,
 * so we pass the current cabinet language via its ?lang= query param.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Skeleton } from "@/components/ui/skeleton";
import { ExternalLink } from "lucide-react";

export default function ClientProgressPage() {
  const { token } = useAuth();
  const { t, lang } = useI18n();
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const { data: projects = [], isLoading, isError } = useQuery({
    queryKey: ["client-progress"],
    queryFn: () => api.clientPortal.progress(token!),
    enabled: Boolean(token),
  });

  const current = useMemo(() => {
    if (!projects.length) return null;
    if (selectedId != null) return projects.find((p) => p.id === selectedId) ?? projects[0];
    return projects.find((p) => p.url) ?? projects[0];
  }, [projects, selectedId]);

  const frameUrl = current?.url ? `${current.url}?lang=${lang}` : null;

  return (
    <ClientShell>
      <div className="flex flex-col h-full min-h-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 sm:px-6 py-2 min-h-[3.5rem] shrink-0">
          <h1 className="text-base font-semibold">{t("progress.title")}</h1>
          <div className="flex items-center gap-2">
            {projects.length > 1 && (
              <select
                className="h-9 rounded-md border border-border bg-card px-3 text-sm"
                value={current?.id ?? ""}
                onChange={(e) => setSelectedId(Number(e.target.value))}
              >
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            )}
            {frameUrl && (
              <a
                href={frameUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
              >
                <ExternalLink className="h-4 w-4" /> {t("progress.openTab")}
              </a>
            )}
          </div>
        </div>

        <div className="flex-1 min-h-0">
          {isLoading && (
            <div className="p-6 space-y-3">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-64 w-full" />
            </div>
          )}
          {isError && <p className="p-6 text-sm text-destructive">{t("common.error")}</p>}
          {!isLoading && !isError && !frameUrl && (
            <p className="p-6 text-sm text-muted-foreground">{t("progress.empty")}</p>
          )}
          {frameUrl && (
            <iframe
              key={frameUrl}
              src={frameUrl}
              title={current?.name ?? "Progress"}
              className="w-full h-full border-0"
            />
          )}
        </div>
      </div>
    </ClientShell>
  );
}
