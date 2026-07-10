/**
 * Settings tab: account (e-mail) + password management for the client cabinet.
 * The e-mail change also renames the client's identity in orlanda-api (ACL +
 * assistant history) via POST /client/portal/email — see rag_agent/api.py.
 */
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { ClientShell } from "@/components/ClientShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";

function PasswordInput({
  id,
  value,
  onChange,
  minLength,
  autoComplete = "new-password",
}: {
  id: string;
  value: string;
  onChange: (v: string) => void;
  minLength?: number;
  autoComplete?: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="relative" dir="ltr">
      <Input
        id={id}
        type={visible ? "text" : "password"}
        required
        minLength={minLength}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="pr-10"
        autoComplete={autoComplete}
      />
      <button
        type="button"
        tabIndex={-1}
        onClick={() => setVisible((v) => !v)}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
        aria-label={visible ? "Hide password" : "Show password"}
      >
        {visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
      </button>
    </div>
  );
}

function AccountCard() {
  const { user, changeEmail } = useAuth();
  const { t } = useI18n();

  const [newEmail, setNewEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess(false);
    setBusy(true);
    try {
      await changeEmail(newEmail.trim(), password);
      setNewEmail("");
      setPassword("");
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.error"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("settings.account")}</CardTitle>
        <CardDescription>{user?.username}</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-3 max-w-sm">
          <div className="space-y-1.5">
            <Label htmlFor="new-email">{t("settings.newEmail")}</Label>
            <Input
              id="new-email"
              type="email"
              required
              value={newEmail}
              onChange={(e) => setNewEmail(e.target.value)}
              dir="ltr"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="account-password">{t("settings.currentPassword")}</Label>
            <PasswordInput id="account-password" value={password} onChange={setPassword} autoComplete="current-password" />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {success && <p className="text-sm text-green-600 dark:text-green-500">{t("settings.saved")}</p>}
          <Button type="submit" disabled={busy}>
            {busy ? t("common.loading") : t("settings.changeEmail")}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function PasswordCard() {
  const { changePassword } = useAuth();
  const { t } = useI18n();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess(false);
    setBusy(true);
    try {
      await changePassword({ currentPassword, newPassword, repeatPassword });
      setCurrentPassword("");
      setNewPassword("");
      setRepeatPassword("");
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("common.error"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("settings.password")}</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-3 max-w-sm">
          <div className="space-y-1.5">
            <Label htmlFor="current-password">{t("settings.currentPassword")}</Label>
            <PasswordInput
              id="current-password"
              value={currentPassword}
              onChange={setCurrentPassword}
              autoComplete="current-password"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="new-password">{t("settings.newPassword")}</Label>
            <PasswordInput id="new-password" value={newPassword} onChange={setNewPassword} minLength={8} />
            <p className="text-xs text-muted-foreground">{t("register.passwordHint")}</p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="repeat-password">{t("settings.repeatPassword")}</Label>
            <PasswordInput id="repeat-password" value={repeatPassword} onChange={setRepeatPassword} />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {success && <p className="text-sm text-green-600 dark:text-green-500">{t("settings.saved")}</p>}
          <Button type="submit" disabled={busy}>
            {busy ? t("common.loading") : t("settings.changePassword")}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

export default function ClientSettingsPage() {
  const { t } = useI18n();

  return (
    <ClientShell>
      <div className="p-6 max-w-xl space-y-6">
        <h1 className="text-xl font-semibold">{t("settings.title")}</h1>
        <AccountCard />
        <PasswordCard />
      </div>
    </ClientShell>
  );
}
