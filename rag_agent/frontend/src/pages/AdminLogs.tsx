import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { KnowledgeModal } from "@/components/KnowledgeModal";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import type { AdminLog } from "@/lib/types";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { toast } from "sonner";

const PAGE_SIZE = 8;
const SAVE_DELAY_MS = 650;

type Draft = {
  score: string;
  correctAnswer: string;
  status: AdminLog["status"];
};

export default function AdminLogsPage() {
  const { token, changePassword } = useAuth();
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);
  const [logs, setLogs] = useState<AdminLog[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [savingIds, setSavingIds] = useState<Record<string, boolean>>({});
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const load = () => {
    if (!token) return;
    api.admin
      .listLogs(token, page, PAGE_SIZE)
      .then((result) => {
        setLogs(result.items);
        setTotal(result.total);
      })
      .catch((e) => toast.error(e instanceof Error ? e.message : "Ошибка загрузки журнала"));
  };

  useEffect(() => {
    load();
  }, [page, token]);

  useEffect(() => {
    if (!logs) return;
    const next: Record<string, Draft> = {};
    for (const log of logs) {
      next[log.id] = {
        score: log.reviewScore == null ? "" : String(log.reviewScore),
        correctAnswer: log.correctAnswer ?? "",
        status: log.status,
      };
    }
    setDrafts(next);
  }, [logs]);

  useEffect(() => {
    return () => {
      for (const timer of timersRef.current.values()) clearTimeout(timer);
      timersRef.current.clear();
    };
  }, []);

  const scheduleSave = (id: string, candidate: Draft) => {
    const prev = timersRef.current.get(id);
    if (prev) clearTimeout(prev);
    const timer = setTimeout(async () => {
      if (!token) return;
      const score = Number(candidate.score);
      if (!Number.isFinite(score) || score < 1 || score > 10) return;
      try {
        setSavingIds((s) => ({ ...s, [id]: true }));
        await api.admin.reviewLog(token, id, score, candidate.correctAnswer.trim() || undefined);
        setDrafts((current) => ({
          ...current,
          [id]: {
            ...(current[id] ?? candidate),
            score: String(Math.round(score)),
            status: "ok",
          },
        }));
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Не удалось сохранить оценку");
      } finally {
        setSavingIds((s) => ({ ...s, [id]: false }));
      }
    }, SAVE_DELAY_MS);
    timersRef.current.set(id, timer);
  };

  const rows = useMemo(() => logs ?? Array.from({ length: PAGE_SIZE }).map((_, idx) => ({ id: `sk-${idx}` } as AdminLog)), [logs]);

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
          <h1 className="font-display text-3xl font-semibold text-foreground">Журнал диалогов</h1>
          <div className="flex items-center gap-2 text-sm">
            <Link to="/admin/logs" className="font-medium text-foreground">Журнал диалогов</Link>
            <span className="text-muted-foreground">·</span>
            <Link to="/admin/documents" className="text-muted-foreground hover:text-foreground">Метаданные документов</Link>
          </div>
        </div>

        <section className="surface-card rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-border flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Оценка 1-10 и правильный ответ сохраняются автоматически</p>
            <div className="text-xs text-muted-foreground">Всего: {total}</div>
          </div>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[140px]">Время</TableHead>
                  <TableHead>Пользователь</TableHead>
                  <TableHead>Сообщение</TableHead>
                  <TableHead>Модель</TableHead>
                  <TableHead className="w-[110px]">Оценка</TableHead>
                  <TableHead className="min-w-[260px]">Правильный ответ</TableHead>
                  <TableHead className="w-[140px]">Статус</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((log, idx) => {
                  if (logs === null) {
                    return (
                      <TableRow key={`sk-${idx}`}>
                        <TableCell colSpan={7}>
                          <Skeleton className="h-8 w-full" />
                        </TableCell>
                      </TableRow>
                    );
                  }

                  const draft = drafts[log.id] ?? { score: "", correctAnswer: "", status: log.status };
                  const scoreValue = draft.score;
                  const scoreNum = Number(scoreValue);
                  const invalidScore = scoreValue !== "" && (!Number.isFinite(scoreNum) || scoreNum < 1 || scoreNum > 10);

                  return (
                    <TableRow key={log.id}>
                      <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                        {new Date(log.createdAt).toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" })}
                      </TableCell>
                      <TableCell className="text-sm">{log.username}</TableCell>
                      <TableCell className="text-sm max-w-[320px] truncate">{log.message}</TableCell>
                      <TableCell className="text-xs font-mono text-muted-foreground">{log.model}</TableCell>
                      <TableCell>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          step={1}
                          value={scoreValue}
                          onChange={(e) => {
                            const next = { ...draft, score: e.target.value };
                            setDrafts((s) => ({ ...s, [log.id]: next }));
                            scheduleSave(log.id, next);
                          }}
                          className={`h-9 ${invalidScore ? "border-destructive focus-visible:ring-destructive" : ""}`}
                        />
                      </TableCell>
                      <TableCell>
                        <Textarea
                          value={draft.correctAnswer}
                          onChange={(e) => {
                            const next = { ...draft, correctAnswer: e.target.value };
                            setDrafts((s) => ({ ...s, [log.id]: next }));
                            scheduleSave(log.id, next);
                          }}
                          className="min-h-[72px]"
                          placeholder="Введите правильный ответ"
                        />
                      </TableCell>
                      <TableCell>
                        {savingIds[log.id] ? (
                          <Badge variant="outline" className="border-primary/30 text-primary bg-primary/5">Сохраняется…</Badge>
                        ) : invalidScore ? (
                          <Badge variant="outline" className="border-destructive/30 text-destructive bg-destructive/5">Оценка 1-10</Badge>
                        ) : (
                          <StatusPill status={draft.status} />
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          <div className="flex items-center justify-between px-5 py-3 border-t border-border">
            <div className="text-xs text-muted-foreground">Стр. {page} из {totalPages}</div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
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
              <Label htmlFor="admin-current-password">Текущий пароль</Label>
              <Input id="admin-current-password" type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} autoComplete="current-password" />
            </div>
            <div className="space-y-1">
              <Label htmlFor="admin-new-password">Новый пароль</Label>
              <Input id="admin-new-password" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} autoComplete="new-password" />
            </div>
            <div className="space-y-1">
              <Label htmlFor="admin-repeat-password">Повторите новый пароль</Label>
              <Input id="admin-repeat-password" type="password" value={repeatPassword} onChange={(e) => setRepeatPassword(e.target.value)} autoComplete="new-password" />
            </div>
            {settingsError && (
              <div className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
                {settingsError}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setSettingsOpen(false)} disabled={changingPassword}>Отмена</Button>
            <Button onClick={submitPasswordChange} className="btn-gradient" disabled={changingPassword}>
              {changingPassword ? "Сохранение..." : "Сменить пароль"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

function StatusPill({ status }: { status: AdminLog["status"] }) {
  const map = {
    ok: { label: "Оценен", cls: "border-success/30 text-success bg-success/5" },
    review: { label: "Без оценки", cls: "border-warning/30 text-warning bg-warning/5" },
    flagged: { label: "Внимание", cls: "border-destructive/30 text-destructive bg-destructive/5" },
  } as const;
  return <Badge variant="outline" className={map[status].cls}>{map[status].label}</Badge>;
}
