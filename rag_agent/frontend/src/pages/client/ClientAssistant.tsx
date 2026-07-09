/**
 * Assistant tab: chat with the OrlandaBot brain (proxied through the platform
 * to orlanda-api). History lives server-side in Redis per client; the page
 * keeps only the visible transcript of the current browser session.
 */
import { useRef, useState } from "react";
import { ClientShell } from "@/components/ClientShell";
import { MarkdownLite } from "@/components/MarkdownLite";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import type { ClientChatReply } from "@/lib/types";
import { RotateCcw, Send } from "lucide-react";

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

export default function ClientAssistantPage() {
  const { token } = useAuth();
  const { t } = useI18n();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollDown = () => setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

  const send = async () => {
    const message = input.trim();
    if (!message || busy || !token) return;
    setInput("");
    setTurns((prev) => [...prev, { id: `u-${Date.now()}`, role: "user", content: message }]);
    setBusy(true);
    scrollDown();
    try {
      const reply = await api.clientPortal.chat(token, message);
      setTurns((prev) => [
        ...prev,
        { id: `a-${Date.now()}`, role: "assistant", content: reply.answer, table: reply.table ?? undefined },
      ]);
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

  const reset = async () => {
    if (!token || busy) return;
    try {
      await api.clientPortal.chatReset(token);
    } catch {
      // history reset is best-effort
    }
    setTurns([]);
  };

  return (
    <ClientShell>
      <div className="flex flex-col h-screen">
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 sm:px-6 h-14 shrink-0">
          <h1 className="text-base font-semibold">{t("assistant.title")}</h1>
          <Button variant="ghost" size="sm" onClick={reset} className="gap-2">
            <RotateCcw className="h-4 w-4" /> {t("assistant.newChat")}
          </Button>
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
    </ClientShell>
  );
}
