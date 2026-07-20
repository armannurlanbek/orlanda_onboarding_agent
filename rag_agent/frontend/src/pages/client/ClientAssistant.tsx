/**
 * Assistant tab with multiple conversations (e.g. one per project). The chat
 * list lives server-side (Redis via orlanda-api); selecting a chat loads its
 * history, "+" starts a fresh one, deleting removes it permanently.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { MarkdownLite } from "@/components/MarkdownLite";
import { ProjectCard } from "@/components/ProjectCard";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import type { ClientChatReply, ClientTaskBlock } from "@/lib/types";
import { Menu, MessageSquare, Plus, Send, Trash2 } from "lucide-react";

type CardPayload = {
  block: ClientTaskBlock;
  statusColors: Record<string, string>;
  progressUrl?: string;
  extraCount: number;
};

type Turn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  table?: ClientChatReply["table"];
  card?: CardPayload;
};

function ReplyTable({ table }: { table: NonNullable<ClientChatReply["table"]> }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border mt-2">
      <table className="min-w-full text-xs" dir="ltr">
        {table.title && <caption className="px-2 py-1 text-left font-semibold">{table.title}</caption>}
        <thead className="bg-muted/60">
          <tr>
            {table.columns.map((c) => (
              <th key={c} className="px-2 py-1 text-left font-semibold whitespace-nowrap">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-t border-border">
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 whitespace-nowrap">{String(cell ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type Chat = { id: string; title: string; updated_at: string };

function ChatList({
  chats,
  loading,
  activeChat,
  onSelect,
  onDelete,
  t,
}: {
  chats: Chat[];
  loading: boolean;
  activeChat: string;
  onSelect: (chatId: string) => void;
  onDelete: (chatId: string) => void;
  t: (key: string) => string;
}) {
  return (
    <div className="flex-1 overflow-y-auto p-2 space-y-1">
      {chats.map((chat) => (
        <div
          key={chat.id}
          className={`group flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm cursor-pointer ${
            chat.id === activeChat ? "bg-primary/10 text-foreground" : "text-muted-foreground hover:bg-muted"
          }`}
          onClick={() => onSelect(chat.id)}
        >
          <MessageSquare className="h-3.5 w-3.5 shrink-0" />
          <span className="flex-1 truncate" title={chat.title}>{chat.title}</span>
          <button
            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(chat.id);
            }}
            aria-label={t("assistant.deleteChat")}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}
      {!loading && !chats.length && <p className="text-xs text-muted-foreground px-2 py-1">{t("assistant.noChats")}</p>}
    </div>
  );
}

const newChatId = () => `c${Date.now().toString(36)}`;

export default function ClientAssistantPage() {
  const { token } = useAuth();
  const { t, dir } = useI18n();
  const qc = useQueryClient();

  const [activeChat, setActiveChat] = useState<string>(newChatId());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [chatsSheetOpen, setChatsSheetOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const chatsQ = useQuery({
    queryKey: ["client-chats"],
    queryFn: () => api.clientPortal.chats(token!),
    enabled: Boolean(token),
  });

  const scrollDown = () => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

  const openChat = async (chatId: string) => {
    setActiveChat(chatId);
    setTurns([]);
    setChatsSheetOpen(false);
    if (!token) return;
    try {
      const messages = await api.clientPortal.chatHistory(token, chatId);
      setTurns(
        messages.map((m, i) => ({
          id: `h-${chatId}-${i}`,
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content,
        })),
      );
    } catch {
      // history is best-effort; the chat still works
    }
    scrollDown();
  };

  const startNewChat = () => {
    setActiveChat(newChatId());
    setTurns([]);
    setInput("");
    setChatsSheetOpen(false);
  };

  const deleteChat = async (chatId: string) => {
    if (!token) return;
    try {
      await api.clientPortal.deleteChat(token, chatId);
    } catch {
      // ignore
    }
    qc.invalidateQueries({ queryKey: ["client-chats"] });
    if (chatId === activeChat) startNewChat();
  };

  const sendMessage = async (message: string) => {
    if (!message || busy || !token) return;
    const isFirstMessage = turns.length === 0;
    setTurns((prev) => [...prev, { id: `u-${Date.now()}`, role: "user", content: message }]);
    setBusy(true);
    scrollDown();
    try {
      const reply = await api.clientPortal.chat(token, message, activeChat);
      setTurns((prev) => [
        ...prev,
        { id: `a-${Date.now()}`, role: "assistant", content: reply.answer, table: reply.table ?? undefined },
      ]);
      if (isFirstMessage) qc.invalidateQueries({ queryKey: ["client-chats"] });
    } catch (e) {
      setTurns((prev) => [
        ...prev,
        { id: `e-${Date.now()}`, role: "assistant", content: e instanceof Error ? e.message : t("common.error") },
      ]);
    } finally {
      setBusy(false);
      scrollDown();
    }
  };

  const send = () => {
    const message = input.trim();
    if (!message) return;
    setInput("");
    sendMessage(message);
  };

  // "Project status" chip: answered entirely from the already-cached tasks
  // table (shared with the Tasks tab query key) — no LLM round trip.
  const showStatusCard = async () => {
    if (busy || !token) return;
    setTurns((prev) => [...prev, { id: `u-${Date.now()}`, role: "user", content: t("suggest.status") }]);
    scrollDown();
    try {
      const data = await qc.fetchQuery({
        queryKey: ["client-tasks"],
        queryFn: () => api.clientPortal.tasksTable(token),
        staleTime: 60_000,
      });
      const [firstBlock, ...rest] = data.projects;
      if (!firstBlock) {
        setTurns((prev) => [...prev, { id: `a-${Date.now()}`, role: "assistant", content: t("tasks.empty") }]);
        return;
      }
      let progressUrl: string | undefined;
      try {
        const projects = await qc.fetchQuery({
          queryKey: ["client-progress"],
          queryFn: () => api.clientPortal.progress(token),
          staleTime: 5 * 60_000,
        });
        progressUrl = projects.find((p) => p.name === firstBlock.project_name)?.url ?? undefined;
      } catch {
        // progress link is best-effort; the card still renders without it
      }
      setTurns((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: "",
          card: { block: firstBlock, statusColors: data.status_colors, progressUrl, extraCount: rest.length },
        },
      ]);
    } catch (e) {
      setTurns((prev) => [
        ...prev,
        { id: `e-${Date.now()}`, role: "assistant", content: e instanceof Error ? e.message : t("common.error") },
      ]);
    } finally {
      scrollDown();
    }
  };

  useEffect(() => {
    // Land in the latest existing conversation on first open.
    const chats = chatsQ.data;
    if (chats && chats.length && turns.length === 0 && !chats.some((c) => c.id === activeChat)) {
      openChat(chats[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatsQ.data]);

  const chats = chatsQ.data ?? [];

  return (
    <ClientShell>
      <div className="flex h-full min-h-0">
        {/* Desktop chat list */}
        <aside className="hidden md:flex w-56 shrink-0 border-e border-border flex-col bg-muted/20">
          <div className="p-2 border-b border-border">
            <Button variant="outline" size="sm" className="w-full gap-2" onClick={startNewChat}>
              <Plus className="h-4 w-4" /> {t("assistant.newChat")}
            </Button>
          </div>
          <ChatList
            chats={chats}
            loading={chatsQ.isLoading}
            activeChat={activeChat}
            onSelect={openChat}
            onDelete={deleteChat}
            t={t}
          />
        </aside>

        <div className="flex-1 min-w-0 flex flex-col">
          {/* Mobile header: chats sheet trigger + new chat, replaces the desktop aside below md */}
          <div className="md:hidden flex items-center gap-2 border-b border-border px-3 h-14 shrink-0">
            <Sheet open={chatsSheetOpen} onOpenChange={setChatsSheetOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" aria-label={t("assistant.newChat")}>
                  <Menu className="h-4 w-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side={dir === "rtl" ? "right" : "left"} className="w-72 p-0 flex flex-col">
                <SheetHeader className="p-3 border-b border-border text-start">
                  <SheetTitle>{t("assistant.title")}</SheetTitle>
                </SheetHeader>
                <ChatList
                  chats={chats}
                  loading={chatsQ.isLoading}
                  activeChat={activeChat}
                  onSelect={openChat}
                  onDelete={deleteChat}
                  t={t}
                />
              </SheetContent>
            </Sheet>
            <h1 className="flex-1 text-base font-semibold truncate">{t("assistant.title")}</h1>
            <Button variant="outline" size="icon" onClick={startNewChat} aria-label={t("assistant.newChat")}>
              <Plus className="h-4 w-4" />
            </Button>
          </div>

          <div className="hidden md:flex items-center border-b border-border px-4 sm:px-6 h-14 shrink-0">
            <h1 className="text-base font-semibold">{t("assistant.title")}</h1>
          </div>

          <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-4">
            {!turns.length && <p className="text-sm text-muted-foreground max-w-lg">{t("assistant.hello")}</p>}
            {turns.map((turn) => (
              <div key={turn.id} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-lg px-3.5 py-2.5 text-sm ${
                    turn.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                  }`}
                >
                  {turn.role === "assistant" && turn.content && <MarkdownLite text={turn.content} />}
                  {turn.role === "user" && turn.content}
                  {turn.table && <ReplyTable table={turn.table} />}
                  {turn.card && (
                    <>
                      <ProjectCard
                        block={turn.card.block}
                        statusColors={turn.card.statusColors}
                        progressUrl={turn.card.progressUrl}
                      />
                      {turn.card.extraCount > 0 && (
                        <Link to="/client/tasks" className="mt-1.5 block text-xs text-primary hover:underline">
                          …{t("card.openTasks")} ({turn.card.extraCount}+)
                        </Link>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
            {busy && <p className="text-sm text-muted-foreground animate-pulse">{t("assistant.thinking")}</p>}
            <div ref={bottomRef} />
          </div>

          <div className="border-t border-border p-3 sm:p-4 shrink-0 space-y-2">
            {!turns.length && (
              <div className="flex flex-wrap gap-2 max-w-3xl mx-auto">
                <button
                  className="rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={() => sendMessage("What's new?")}
                  disabled={busy}
                >
                  {t("suggest.whatsNew")}
                </button>
                <button
                  className="rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={showStatusCard}
                  disabled={busy}
                >
                  {t("suggest.status")}
                </button>
                <button
                  className="rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={() => sendMessage("Latest drawings")}
                  disabled={busy}
                >
                  {t("suggest.drawings")}
                </button>
              </div>
            )}
            <div className="flex items-end gap-2 max-w-3xl mx-auto">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder={t("assistant.placeholder")}
                rows={2}
                className="resize-none"
              />
              <Button onClick={send} disabled={busy || !input.trim()} className="gap-2 shrink-0">
                <Send className="h-4 w-4" /> {t("assistant.send")}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </ClientShell>
  );
}
