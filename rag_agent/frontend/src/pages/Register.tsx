/**
 * Invite-based client registration: /register?invite=TOKEN.
 * Shows the invite preview (company + projects), asks for e-mail + password,
 * and lands the new client in the cabinet. Bilingual EN/HE like the cabinet.
 */
import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { I18nProvider, useI18n } from "@/lib/i18n";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

function RegisterForm() {
  const [params] = useSearchParams();
  const inviteToken = params.get("invite") ?? "";
  const { registerClient } = useAuth();
  const { lang, setLang, t } = useI18n();
  const nav = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [repeat, setRepeat] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const { data: preview, isLoading } = useQuery({
    queryKey: ["invite", inviteToken],
    queryFn: () => api.clientPortal.invitePreview(inviteToken),
    enabled: Boolean(inviteToken),
  });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password !== repeat) {
      setError(t("register.mismatch"));
      return;
    }
    setBusy(true);
    try {
      await registerClient(inviteToken, email.trim(), password);
      nav("/client/tasks", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.error"));
    } finally {
      setBusy(false);
    }
  };

  const invalid = !inviteToken || (preview && !preview.valid);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <Logo />
            <div className="flex rounded-md border border-border overflow-hidden text-xs font-semibold">
              <button
                type="button"
                onClick={() => setLang("en")}
                className={`px-2.5 py-1 ${lang === "en" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
              >
                EN
              </button>
              <button
                type="button"
                onClick={() => setLang("he")}
                className={`px-2.5 py-1 ${lang === "he" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
              >
                עב
              </button>
            </div>
          </div>
          <div>
            <h1 className="text-lg font-semibold">{t("register.title")}</h1>
            <p className="text-sm text-muted-foreground">{t("register.subtitle")}</p>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {isLoading && <p className="text-sm text-muted-foreground">{t("register.checking")}</p>}

          {!isLoading && invalid && (
            <p className="text-sm text-destructive">{preview?.error || t("register.invalidInvite")}</p>
          )}

          {!isLoading && preview?.valid && (
            <>
              {(preview.company_name || (preview.project_names?.length ?? 0) > 0) && (
                <div className="rounded-md bg-muted/60 p-3 text-sm space-y-1">
                  {preview.company_name && <div className="font-medium">{preview.company_name}</div>}
                  {(preview.project_names?.length ?? 0) > 0 && (
                    <div className="text-muted-foreground">
                      {t("register.projects")}: {preview.project_names!.join(", ")}
                    </div>
                  )}
                </div>
              )}

              <form onSubmit={submit} className="space-y-3">
                <div className="space-y-1.5">
                  <Label htmlFor="email">{t("register.email")}</Label>
                  <Input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} dir="ltr" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="password">{t("register.password")}</Label>
                  <Input id="password" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} dir="ltr" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="repeat">{t("register.repeat")}</Label>
                  <Input id="repeat" type="password" required value={repeat} onChange={(e) => setRepeat(e.target.value)} dir="ltr" />
                </div>
                {error && <p className="text-sm text-destructive">{error}</p>}
                <Button type="submit" className="w-full" disabled={busy}>
                  {busy ? t("common.loading") : t("register.submit")}
                </Button>
              </form>
            </>
          )}

          <p className="text-sm text-muted-foreground text-center">
            {t("register.haveAccount")}{" "}
            <Link to="/auth" className="text-primary hover:underline">
              {t("register.signIn")}
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default function RegisterPage() {
  return (
    <I18nProvider>
      <RegisterForm />
    </I18nProvider>
  );
}
