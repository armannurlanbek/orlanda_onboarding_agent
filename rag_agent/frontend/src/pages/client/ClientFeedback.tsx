/**
 * Feedback tab: contact links. The external form URL is optional and comes
 * from the frontend build env (VITE_FEEDBACK_URL); e-mail is always shown.
 */
import { ClientShell } from "@/components/ClientShell";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n";
import { Mail, MessageSquareHeart } from "lucide-react";

const FEEDBACK_URL = (import.meta.env.VITE_FEEDBACK_URL as string | undefined)?.trim() || "";
const FEEDBACK_EMAIL = (import.meta.env.VITE_FEEDBACK_EMAIL as string | undefined)?.trim() || "admin@orlanda.info";

export default function ClientFeedbackPage() {
  const { t } = useI18n();

  return (
    <ClientShell>
      <div className="p-6 max-w-xl space-y-4">
        <h1 className="text-xl font-semibold">{t("feedback.title")}</h1>
        <p className="text-sm text-muted-foreground">{t("feedback.text")}</p>
        <div className="flex flex-wrap gap-3">
          <Button asChild className="gap-2">
            <a href={`mailto:${FEEDBACK_EMAIL}`}>
              <Mail className="h-4 w-4" /> {t("feedback.emailUs")}
            </a>
          </Button>
          {FEEDBACK_URL && (
            <Button asChild variant="outline" className="gap-2">
              <a href={FEEDBACK_URL} target="_blank" rel="noreferrer">
                <MessageSquareHeart className="h-4 w-4" /> {t("feedback.title")}
              </a>
            </Button>
          )}
        </div>
      </div>
    </ClientShell>
  );
}
