/**
 * Assistant tab with multiple conversations (e.g. one per project). The chat
 * list lives server-side (Redis via orlanda-api); selecting a chat loads its
 * history, "+" starts a fresh one, deleting removes it permanently.
 */
import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ClientShell } from "@/components/ClientShell";
import { MarkdownLite } from "@/components/MarkdownLite";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import type { ClientChatReply } from "@/lib/types";
import { MessageSquare, Plus, Send, Trash2 } from "lucide-react";

type Turn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  table?: ClientChatReply["table"];
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

const newChatId = () => `c${Date.now().toString(36)}`;

export default function ClientAssistantPage() {
  const { token } = useAuth();
  const { t } = useI18n();
  const qc = useQueryClient();

  const [activeChat, setActiveChat] = useState<string>(newChatId());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
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

  const send = async () => {
    const message = input.trim();
    if (!message || busy || !token) return;
    const isFirstMessage = turns.length === 0;
    setInput("");
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

  useEffect(() => {
    // Land in the latest existing conversation on first open.
    const chats = chatsQ.data;
    if (chats && chats.length && turns.length === 0 && !chats.some((c) => c.id === activeChat)) {
      openChat(chats[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatsQ.data]);

  return (
    <ClientShell>
      <div className="flex h-screen min-h-0">
        <aside className="w-56 shrink-0 border-e border-border flex flex-col bg-muted/20">
          <div className="p-2 border-b border-border">
            <Button variant="outline" size="sm" className="w-full gap-2" onClick={startNewChat}>
              <Plus className="h-4 w-4" /> {t("assistant.newChat")}
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {(chatsQ.data ?? []).map((chat) => (
              <div
                key={chat.id}
                className={`group flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm cursor-pointer ${
                  chat.id === activeChat ? "bg-primary/10 text-foreground" : "text-muted-foreground hover:bg-muted"
                }`}
                onClick={() => openChat(chat.id)}
              >
                <MessageSquare className="h-3.5 w-3.5 shrink-0" />
                <span className="flex-1 truncate" title={chat.title}>{chat.title}</span>
                <button
                  className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteChat(chat.id);
                  }}
                  aria-label={t("assistant.deleteChat")}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            {!chatsQ.isLoading && !(chatsQ.data ?? []).length && (
              <p className="text-xs text-muted-foreground px-2 py-1">{t("assistant.noChats")}</p>
            )}
          </div>
        </aside>

        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex items-center border-b border-border px-4 sm:px-6 h-14 shrink-0">
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
                  {turn.role === "assistant" ? <MarkdownLite text={turn.content} /> : turn.content}
                  {turn.table && <ReplyTable table={turn.table} />}
                </div>
              </div>
            ))}
            {busy && <p className="text-sm text-muted-foreground animate-pulse">{t("assistant.thinking")}</p>}
            <div ref={bottomRef} />
          </div>

          <div className="border-t border-border p-3 sm:p-4 shrink-0">
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
