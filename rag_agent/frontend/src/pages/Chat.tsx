import { useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ChatMessage, TypingIndicator } from "@/components/ChatMessage";
import { KnowledgeModal } from "@/components/KnowledgeModal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import type { Conversation, Message } from "@/lib/types";
import { MessageSquarePlus, Menu, Send, Sparkles, Trash2 } from "lucide-react";
import { toast } from "sonner";

export default function ChatPage() {
  const { token, user, changePassword } = useAuth();
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[] | null>(null);
  const [sending, setSending] = useState(false);
  const [input, setInput] = useState("");
  const [newOpen, setNewOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [mobileSheet, setMobileSheet] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const loadConversations = async (selectFirst = false) => {
    if (!token) return;
    const list = await api.chat.listConversations(token);
    setConversations(list);
    if (selectFirst && list[0] && !activeId) setActiveId(list[0].id);
  };

  useEffect(() => {
    loadConversations(true).catch(() => {
      setConversations([]);
    });
  }, [token]);

  useEffect(() => {
    if (user?.mustChangePassword) {
      setSettingsOpen(true);
      setSettingsError("Требуется сменить временный пароль перед началом работы.");
    }
  }, [user?.mustChangePassword]);

  useEffect(() => {
    if (!token || !activeId) { setMessages([]); return; }
    setMessages(null);
    api.chat.getHistory(token, activeId).then(setMessages).catch((e) => {
      toast.error(e instanceof Error ? e.message : "Ошибка загрузки истории");
      setMessages([]);
    });
  }, [token, activeId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const createConv = async () => {
    if (!user || !token) return;
    if (user.mustChangePassword) { toast.error("Сначала смените временный пароль"); return; }
    if (!newTitle.trim()) { toast.error("Введите название диалога"); return; }
    const c = await api.chat.createConversation(token, newTitle.trim());
    setNewOpen(false); setNewTitle("");
    setActiveId(c.id);
    await loadConversations();
    setMobileSheet(false);
  };

  const deleteConv = async (id: string) => {
    if (!token || !user) return;
    if (user.mustChangePassword) { toast.error("Сначала смените временный пароль"); return; }
    if (!confirm("Удалить диалог?")) return;
    await api.chat.deleteConversation(token, id);
    if (activeId === id) setActiveId(null);
    toast.success("Диалог удалён");
    await loadConversations(true);
  };

  const streamControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      streamControllerRef.current?.abort();
      streamControllerRef.current = null;
    };
  }, []);

  const send = (text?: string) => {
    if (!token || !user) return;
    if (user.mustChangePassword) {
      toast.error("Сначала смените временный пароль");
      return;
    }
    const msg = (text ?? input).trim();
    if (!msg || !activeId || sending) return;
    setInput("");

    const userMsgId = `tmp-u-${Date.now()}`;
    const assistantMsgId = `tmp-a-${Date.now()}`;
    let bubbleStarted = false;

    setMessages((m) => [
      ...(m ?? []),
      { id: userMsgId, role: "user", content: msg, createdAt: new Date().toISOString() },
    ]);
    setSending(true);

    const ensureAssistantBubble = (initialContent = "") => {
      if (bubbleStarted) return;
      bubbleStarted = true;
      setMessages((m) => [
        ...(m ?? []),
        {
          id: assistantMsgId,
          role: "assistant",
          content: initialContent,
          createdAt: new Date().toISOString(),
          toolEvents: [],
        },
      ]);
    };

    streamControllerRef.current = api.chat.sendMessageStream(token, activeId, msg, {
      onDelta: (chunk) => {
        if (!bubbleStarted) {
          ensureAssistantBubble(chunk);
        } else {
          setMessages((m) =>
            m?.map((x) => (x.id === assistantMsgId ? { ...x, content: x.content + chunk } : x)) ?? null,
          );
        }
      },
      onToolStart: (name) => {
        ensureAssistantBubble();
        setMessages((m) =>
          m?.map((x) => {
            if (x.id !== assistantMsgId) return x;
            const events = [
              ...(x.toolEvents ?? []),
              {
                id: `t-${Date.now()}-${name}`,
                name,
                status: "success" as const,
                detail: "выполняется…",
              },
            ];
            return { ...x, toolEvents: events };
          }) ?? null,
        );
      },
      onToolEnd: (name, status) => {
        setMessages((m) =>
          m?.map((x) => {
            if (x.id !== assistantMsgId) return x;
            const events = (x.toolEvents ?? []).slice();
            for (let i = events.length - 1; i >= 0; i--) {
              if (events[i].name === name && events[i].detail === "выполняется…") {
                events[i] = { ...events[i], status, detail: status === "error" ? "ошибка" : "готово" };
                break;
              }
            }
            return { ...x, toolEvents: events };
          }) ?? null,
        );
      },
      onDone: ({ response, sources, toolEvents }) => {
        ensureAssistantBubble(response);
        setMessages((m) =>
          m?.map((x) =>
            x.id === assistantMsgId
              ? {
                  ...x,
                  content: response || x.content,
                  sources,
                  toolEvents: toolEvents.length ? toolEvents : x.toolEvents,
                }
              : x,
          ) ?? null,
        );
        setSending(false);
        streamControllerRef.current = null;
        loadConversations().catch(() => undefined);
      },
      onError: (errMsg) => {
        setMessages((m) =>
          m?.filter((x) => x.id !== assistantMsgId || (x.content?.trim().length ?? 0) > 0) ?? null,
        );
        toast.error(errMsg || "Ошибка отправки");
        setSending(false);
        streamControllerRef.current = null;
      },
    });
  };

  const ConvList = (
    <div className="flex flex-col h-full">
      <div className="p-3">
        <Button onClick={() => setNewOpen(true)} className="w-full justify-start" disabled={Boolean(user?.mustChangePassword)}>
          <MessageSquarePlus className="h-4 w-4" /> Новый диалог
        </Button>
      </div>
      <div className="flex-1 overflow-auto px-2 pb-3">
        <div className="px-2 py-1.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-[0.08em]">Диалоги</div>
        {conversations === null ? (
          <div className="space-y-2 px-2">{[...Array(3)].map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
        ) : conversations.length === 0 ? (
          <div className="px-4 py-4 text-sm text-muted-foreground">Нет диалогов</div>
        ) : (
          <ul className="space-y-0.5">
            {conversations.map((c) => (
              <li key={c.id}>
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => { setActiveId(c.id); setMobileSheet(false); }}
                  onKeyDown={(e) => { if (e.key === "Enter") { setActiveId(c.id); setMobileSheet(false); } }}
                  className={`group rounded-lg px-3 py-2 flex items-center gap-2 cursor-pointer transition-colors ${activeId === c.id ? "bg-accent text-accent-foreground" : "hover:bg-muted/60"}`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-sm truncate">{c.title}</div>
                    <div className="text-xs text-muted-foreground">{new Date(c.updatedAt).toLocaleDateString("ru-RU")}</div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); deleteConv(c.id); }}
                    aria-label="Удалить диалог"
                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive p-1"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );

  const submitPasswordChange = async () => {
    if (!user) return;
    setSettingsError(null);
    if (!newPassword || !repeatPassword) {
      setSettingsError("Заполните новый пароль и подтверждение");
      return;
    }
    if (newPassword !== repeatPassword) {
      setSettingsError("Новый пароль и подтверждение не совпадают");
      return;
    }
    if (!user.mustChangePassword && !currentPassword.trim()) {
      setSettingsError("Введите текущий пароль");
      return;
    }
    setChangingPassword(true);
    try {
      await changePassword({
        currentPassword: user.mustChangePassword ? "" : currentPassword,
        newPassword,
        repeatPassword,
      });
      setCurrentPassword("");
      setNewPassword("");
      setRepeatPassword("");
      setSettingsOpen(false);
      setSettingsError(null);
      toast.success("Пароль успешно обновлён");
      if (token && activeId) {
        await api.chat.getHistory(token, activeId).then((history) => setMessages(history)).catch(() => undefined);
      }
    } catch (e) {
      setSettingsError(e instanceof Error ? e.message : "Не удалось сменить пароль");
    } finally {
      setChangingPassword(false);
    }
  };

  return (
    <AppShell
      lockLayout
      onOpenKnowledge={() => {
        if (user?.mustChangePassword) {
          toast.error("Сначала смените временный пароль");
          setSettingsOpen(true);
          return;
        }
        setKnowledgeOpen(true);
      }}
      onOpenSettings={() => setSettingsOpen(true)}
    >
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[280px_1fr] overflow-hidden">
        <aside className="hidden lg:block min-h-0 border-r border-border bg-sidebar overflow-hidden">{ConvList}</aside>

        <section className="flex min-h-0 flex-col overflow-hidden bg-background">
          <div className="lg:hidden border-b border-border bg-card px-4 py-2 flex items-center justify-between">
            <Sheet open={mobileSheet} onOpenChange={setMobileSheet}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="sm"><Menu className="h-4 w-4" /> Диалоги</Button>
              </SheetTrigger>
              <SheetContent side="left" className="p-0 w-[300px]">
                <SheetHeader className="p-4 border-b border-border"><SheetTitle>Диалоги</SheetTitle></SheetHeader>
                {ConvList}
              </SheetContent>
            </Sheet>
            <div className="text-sm text-muted-foreground truncate">
              {conversations?.find((c) => c.id === activeId)?.title ?? "Выберите диалог"}
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 min-h-0 overflow-auto px-4 sm:px-8 py-6">
            <div className="max-w-3xl mx-auto space-y-4">
              {!activeId ? (
                <EmptyState />
              ) : messages === null ? (
                <div className="space-y-3">{[...Array(3)].map((_, i) => <Skeleton key={i} className="h-16 w-3/4" />)}</div>
              ) : messages.length === 0 ? (
                <EmptyState />
              ) : (
                messages.map((m) => <ChatMessage key={m.id} message={m} />)
              )}
              {sending && <TypingIndicator />}
            </div>
          </div>

          <div className="border-t border-border bg-card px-4 sm:px-6 py-3">
            <div className="max-w-3xl mx-auto space-y-2">
              <div className="flex items-end gap-2">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  placeholder={user?.mustChangePassword ? "Сначала смените временный пароль" : activeId ? "Спросите ассистента…" : "Создайте диалог, чтобы начать"}
                  disabled={!activeId || sending || Boolean(user?.mustChangePassword)}
                  className="min-h-[52px] max-h-[160px] resize-none"
                />
                <Button onClick={() => send()} disabled={!activeId || sending || !input.trim() || Boolean(user?.mustChangePassword)} className="h-[52px] px-5">
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs text-muted-foreground">Enter — отправить · Shift+Enter — перенос</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      <KnowledgeModal open={knowledgeOpen} onOpenChange={setKnowledgeOpen} />

      <Dialog
        open={settingsOpen}
        onOpenChange={(open) => {
          if (user?.mustChangePassword && !open) return;
          setSettingsOpen(open);
          if (!open) setSettingsError(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{user?.mustChangePassword ? "Смена временного пароля" : "Настройки аккаунта"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              {user?.mustChangePassword
                ? "Это обязательный шаг при первом входе. Задайте новый пароль и повторите его."
                : "Смените пароль в любое время."}
            </p>
            {!user?.mustChangePassword && (
              <div className="space-y-1">
                <Label htmlFor="current-password">Текущий пароль</Label>
                <Input
                  id="current-password"
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  autoComplete="current-password"
                />
              </div>
            )}
            <div className="space-y-1">
              <Label htmlFor="new-password">Новый пароль</Label>
              <Input
                id="new-password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                autoComplete="new-password"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="repeat-password">Повторите новый пароль</Label>
              <Input
                id="repeat-password"
                type="password"
                value={repeatPassword}
                onChange={(e) => setRepeatPassword(e.target.value)}
                autoComplete="new-password"
              />
            </div>
            {settingsError && (
              <div className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md px-3 py-2">
                {settingsError}
              </div>
            )}
          </div>
          <DialogFooter>
            {!user?.mustChangePassword && (
              <Button variant="ghost" onClick={() => setSettingsOpen(false)} disabled={changingPassword}>
                Отмена
              </Button>
            )}
            <Button onClick={submitPasswordChange} disabled={changingPassword}>
              {changingPassword ? "Сохранение..." : "Сменить пароль"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={newOpen} onOpenChange={setNewOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader><DialogTitle>Новый диалог</DialogTitle></DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="ctitle">Название</Label>
            <Input
              id="ctitle"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void createConv(); }}
              placeholder="Например: Отпуск 2025"
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setNewOpen(false)}>Отмена</Button>
            <Button onClick={() => void createConv()}>Создать</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center text-center pt-16 pb-6">
      <div className="h-14 w-14 rounded-xl bg-accent ring-1 ring-border flex items-center justify-center mb-5">
        <Sparkles className="h-7 w-7 text-primary" />
      </div>
      <h2 className="font-display text-2xl font-semibold tracking-tight text-foreground mb-2">Orlanda Engineering HR Agent</h2>
      <p className="text-sm leading-relaxed text-muted-foreground max-w-md mb-6">
        Это HR-агент Orlanda Engineering. Здесь можно задавать вопросы по бизнес-процессам и инженерным задачам.
      </p>
    </div>
  );
}
