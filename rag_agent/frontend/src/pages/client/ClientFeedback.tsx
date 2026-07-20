/**
 * Feedback tab: per-client feedback-form links pulled from the Monday clients
 * board (EN/HE columns) via orlanda-api. The current cabinet language picks
 * which form to highlight; e-mail contact stays as a fallback.
 */
import { useQuery } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { ExternalLink, Mail } from "lucide-react";

const FEEDBACK_EMAIL = (import.meta.env.VITE_FEEDBACK_EMAIL as string | undefined)?.trim() || "admin@orlanda.info";

export default function ClientFeedbackPage() {
  const { token } = useAuth();
  const { t, lang } = useI18n();

  const { data: links = [], isLoading } = useQuery({
    queryKey: ["client-feedback"],
    queryFn: () => api.clientPortal.feedback(token!),
    enabled: Boolean(token),
  });

  return (
    <ClientShell>
      <div className="p-4 sm:p-6 max-w-xl space-y-4">
        <h1 className="text-xl font-semibold">{t("feedback.title")}</h1>
        <p className="text-sm text-muted-foreground">{t("feedback.text")}</p>

        {isLoading && <Skeleton className="h-16 w-full" />}

        <div className="space-y-3">
          {links.map((link) => (
            <div key={link.client} className="flex flex-wrap items-center gap-2">
              {links.length > 1 && link.client && (
                <span className="text-sm font-medium w-full">{link.client}</span>
              )}
              {link.url_en && (
                <Button asChild variant={lang === "he" ? "outline" : "default"} className="gap-2">
                  <a href={link.url_en} target="_blank" rel="noreferrer">
                    <ExternalLink className="h-4 w-4" /> {t("feedback.formEn")}
                  </a>
                </Button>
              )}
              {link.url_he && (
                <Button asChild variant={lang === "he" ? "default" : "outline"} className="gap-2">
                  <a href={link.url_he} target="_blank" rel="noreferrer">
                    <ExternalLink className="h-4 w-4" /> {t("feedback.formHe")}
                  </a>
                </Button>
              )}
            </div>
          ))}
        </div>

        <div>
          <Button asChild variant="outline" className="gap-2">
            <a href={`mailto:${FEEDBACK_EMAIL}`}>
              <Mail className="h-4 w-4" /> {t("feedback.emailUs")}
            </a>
          </Button>
        </div>
      </div>
    </ClientShell>
  );
}
