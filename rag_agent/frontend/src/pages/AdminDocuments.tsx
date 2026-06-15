import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { KnowledgeModal } from "@/components/KnowledgeModal";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import type { DocMeta } from "@/lib/types";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";

const SAVE_DELAY_MS = 650;

export default function AdminDocumentsPage() {
  const { token, changePassword } = useAuth();
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);
  const [docs, setDocs] = useState<DocMeta[] | null>(null);
  const [savingIds, setSavingIds] = useState<Record<string, boolean>>({});
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  useEffect(() => {
    if (!token) return;
    api.admin
      .listDocMeta(token)
      .then(setDocs)
      .catch((e) => toast.error(e instanceof Error ? e.message : "Ошибка загрузки метаданных"));
  }, [token]);

  useEffect(() => {
    return () => {
      for (const timer of timersRef.current.values()) clearTimeout(timer);
      timersRef.current.clear();
    };
  }, []);

  const scheduleSave = (id: string, expiresAt: string) => {
    const prev = timersRef.current.get(id);
    if (prev) clearTimeout(prev);
    const timer = setTimeout(async () => {
      if (!token) return;
      try {
        setSavingIds((s) => ({ ...s, [id]: true }));
        await api.admin.updateDocMetaExpiry(token, id, expiresAt);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Не удалось сохранить срок действия");
      } finally {
        setSavingIds((s) => ({ ...s, [id]: false }));
      }
    }, SAVE_DELAY_MS);
    timersRef.current.set(id, timer);
  };

  const rows = useMemo(() => docs ?? Array.from({ length: 8 }).map((_, idx) => ({ id: `sk-${idx}` } as DocMeta)), [docs]);

  const submitPasswordChange = async () => {
    setSettingsError(null);
    if (!newPassword || !repeatPassword) {
      setSettingsError("Заполните новый пароль и подтверждение");
      return;
    }
    if (newPassword !== repeatPassword) {
      setSettingsError("Новый пароль и подтверждение не совпадают");
      return;
    }
    if (!currentPassword.trim()) {
      setSettingsError("Введите текущий пароль");
      return;
    }
    setChangingPassword(true);
    try {
      await changePassword({ currentPassword, newPassword, repeatPassword });
      setCurrentPassword("");
      setNewPassword("");
      setRepeatPassword("");
      setSettingsOpen(false);
      setSettingsError(null);
      toast.success("Пароль успешно обновлён");
    } catch (e) {
      setSettingsError(e instanceof Error ? e.message : "Не удалось сменить пароль");
    } finally {
      setChangingPassword(false);
    }
  };

  return (
    <AppShell onOpenKnowledge={() => setKnowledgeOpen(true)} onOpenSettings={() => setSettingsOpen(true)}>
      <div className="px-4 sm:px-8 py-6 max-w-7xl mx-auto w-full space-y-6">
        <div className="flex items-center gap-3">
          <Link to="/chat" className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" /> К чату
          </Link>
        </div>

        <div className="space-y-2">
          <h1 className="font-display text-3xl font-semibold tracking-tight text-foreground">Метаданные документов</h1>
          <div className="flex items-center gap-2 text-sm">
            <Link to="/admin/logs" className="text-muted-foreground hover:text-foreground">Журнал диалогов</Link>
            <span className="text-muted-foreground">·</span>
            <Link to="/admin/documents" className="font-medium text-foreground">Метаданные документов</Link>
          </div>
        </div>

        <section className="surface-card rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-border">
            <p className="text-xs text-muted-foreground">Изменение даты истечения сохраняется автоматически</p>
          </div>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Документ</TableHead>
                  <TableHead>Владелец</TableHead>
                  <TableHead className="w-[160px]">Проверен</TableHead>
                  <TableHead className="w-[180px]">Истекает</TableHead>
                  <TableHead className="w-[140px]">Статус</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((doc, idx) => {
                  if (docs === null) {
                    return (
                      <TableRow key={`sk-${idx}`}>
                        <TableCell colSpan={5}>
                          <Skeleton className="h-8 w-full" />
                        </TableCell>
                      </TableRow>
                    );
                  }
                  return (
                    <TableRow key={doc.id}>
                      <TableCell className="font-medium text-foreground">{doc.name}</TableCell>
                      <TableCell>{doc.owner}</TableCell>
                      <TableCell className="text-muted-foreground">{formatDateRu(doc.reviewedAt)}</TableCell>
                      <TableCell>
                        <Input
                          type="date"
                          value={doc.expiresAt}
                          onChange={(e) => {
                            const expiresAt = e.target.value;
                            setDocs((current) => current?.map((entry) => entry.id === doc.id ? { ...entry, expiresAt, status: getDocStatus(expiresAt) } : entry) ?? null);
                            scheduleSave(doc.id, expiresAt);
                          }}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        {savingIds[doc.id] ? (
                          <Badge variant="outline" className="border-primary/30 text-primary bg-primary/5">Сохраняется…</Badge>
                        ) : (
                          <DocStatus status={doc.status} />
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </section>
      </div>
      <KnowledgeModal open={knowledgeOpen} onOpenChange={setKnowledgeOpen} />

      <Dialog
        open={settingsOpen}
        onOpenChange={(open) => {
          setSettingsOpen(open);
          if (!open) setSettingsError(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Настройки аккаунта</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">Смените пароль в любое время.</p>
            <div className="space-y-1">
              <Label htmlFor="admin-docs-current-password">Текущий пароль</Label>
              <Input id="admin-docs-current-password" type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} autoComplete="current-password" />
            </div>
            <div className="space-y-1">
              <Label htmlFor="admin-docs-new-password">Новый пароль</Label>
              <Input id="admin-docs-new-password" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} autoComplete="new-password" />
            </div>
            <div className="space-y-1">
              <Label htmlFor="admin-docs-repeat-password">Повторите новый пароль</Label>
              <Input id="admin-docs-repeat-password" type="password" value={repeatPassword} onChange={(e) => setRepeatPassword(e.target.value)} autoComplete="new-password" />
            </div>
            {settingsError && (
              <div className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
                {settingsError}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setSettingsOpen(false)} disabled={changingPassword}>Отмена</Button>
            <Button onClick={submitPasswordChange} disabled={changingPassword}>
              {changingPassword ? "Сохранение..." : "Сменить пароль"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

function formatDateRu(value: string): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("ru-RU");
}

function getDocStatus(expiresAt: string): DocMeta["status"] {
  if (!expiresAt) return "actual";
  const expiresTs = Date.parse(expiresAt);
  if (!Number.isFinite(expiresTs)) return "actual";
  const days = (expiresTs - Date.now()) / (24 * 60 * 60 * 1000);
  if (days < 0) return "expired";
  if (days <= 30) return "review_soon";
  return "actual";
}

function DocStatus({ status }: { status: DocMeta["status"] }) {
  const map = {
    actual: { label: "Актуален", cls: "border-success/30 text-success bg-success/5" },
    review_soon: { label: "Скоро ревизия", cls: "border-warning/30 text-warning bg-warning/5" },
    expired: { label: "Просрочен", cls: "border-destructive/30 text-destructive bg-destructive/5" },
  } as const;
  return <Badge variant="outline" className={map[status].cls}>{map[status].label}</Badge>;
}
