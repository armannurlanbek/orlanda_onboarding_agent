import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { ArrowLeft, CheckCircle2, Loader2, Plug } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/apiClient";
import type { MondayStatus } from "@/lib/types";

const ERROR_REASONS: Record<string, string> = {
  not_configured: "Интеграция не настроена на сервере.",
  invalid_state: "Сессия подключения истекла. Попробуйте ещё раз.",
  missing_code: "monday.com не вернул код авторизации.",
  token_exchange_failed: "Не удалось обменять код на токен.",
  no_token: "monday.com не вернул токен доступа.",
  store_failed: "Не удалось сохранить подключение.",
  access_denied: "Доступ к monday.com не предоставлен.",
};

export default function SettingsPage() {
  const { token } = useAuth();
  const [params, setParams] = useSearchParams();
  const [status, setStatus] = useState<MondayStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const handledParam = useRef(false);

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      setStatus(await api.integrations.monday.status(token));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось загрузить статус интеграции");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Handle the post-OAuth redirect (?monday=connected|error&reason=...), then clear the params.
  useEffect(() => {
    const result = params.get("monday");
    if (!result || handledParam.current) return;
    handledParam.current = true;
    if (result === "connected") {
      toast.success("monday.com подключён");
    } else if (result === "error") {
      const reason = params.get("reason") || "";
      toast.error(ERROR_REASONS[reason] || "Не удалось подключить monday.com");
    }
    params.delete("monday");
    params.delete("reason");
    setParams(params, { replace: true });
  }, [params, setParams]);

  const connect = async () => {
    if (!token) return;
    setBusy(true);
    try {
      const url = await api.integrations.monday.authorizeUrl(token);
      if (!url) throw new Error("Пустой адрес авторизации");
      window.location.href = url; // leave the SPA for monday consent
    } catch (e) {
      setBusy(false);
      toast.error(e instanceof Error ? e.message : "Не удалось начать подключение");
    }
  };

  const disconnect = async () => {
    if (!token) return;
    if (!window.confirm("Отключить monday.com? Ассистент потеряет доступ к вашим доскам.")) return;
    setBusy(true);
    try {
      await api.integrations.monday.disconnect(token);
      toast.success("monday.com отключён");
      await refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось отключить");
    } finally {
      setBusy(false);
    }
  };

  const scopes = (status?.scope || "").split(/\s+/).filter(Boolean);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-card/85 backdrop-blur-md">
        <div className="px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <Link to="/chat" aria-label="На главную" className="shrink-0"><Logo /></Link>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/chat"><ArrowLeft className="h-4 w-4" /> В чат</Link>
          </Button>
        </div>
      </header>

      <main className="flex-1 px-4 sm:px-6 py-8">
        <div className="mx-auto max-w-2xl space-y-6">
          <div>
            <h1 className="font-display text-2xl font-semibold tracking-tight">Интеграции</h1>
            <p className="text-sm text-muted-foreground mt-1">Подключите внешние сервисы к ассистенту.</p>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <CardTitle className="flex items-center gap-2">
                    <Plug className="h-5 w-5" /> monday.com
                  </CardTitle>
                  <CardDescription>
                    Спрашивайте ассистента о ваших досках и задачах. Доступ ограничен вашими
                    правами в monday — вы видите только то, что доступно вам.
                  </CardDescription>
                </div>
                {!loading && status?.connected ? (
                  <Badge className="gap-1 shrink-0" variant="secondary">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Подключено
                  </Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {loading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> Загрузка…
                </div>
              ) : !status?.enabled ? (
                <p className="text-sm text-muted-foreground">
                  Интеграция monday.com не настроена администратором.
                </p>
              ) : status.connected ? (
                <>
                  <div className="space-y-1 text-sm">
                    {status.mondayUserName ? (
                      <div>
                        Аккаунт: <span className="font-medium">{status.mondayUserName}</span>
                      </div>
                    ) : null}
                    {status.connectedAt ? (
                      <div className="text-muted-foreground">
                        Подключено: {new Date(status.connectedAt).toLocaleString("ru-RU")}
                      </div>
                    ) : null}
                  </div>
                  {scopes.length ? (
                    <div className="flex flex-wrap gap-1.5">
                      {scopes.map((s) => (
                        <Badge key={s} variant="outline" className="font-normal">{s}</Badge>
                      ))}
                    </div>
                  ) : null}
                  <Separator />
                  <Button variant="outline" onClick={disconnect} disabled={busy}>
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Отключить
                  </Button>
                </>
              ) : (
                <Button onClick={connect} disabled={busy}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plug className="h-4 w-4" />}
                  Подключить monday.com
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
