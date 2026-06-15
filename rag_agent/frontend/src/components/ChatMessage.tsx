import { useState } from "react";
import type { Message } from "@/lib/types";
import { MarkdownLite } from "./MarkdownLite";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { CheckCircle2, ChevronDown, FileText, Sparkles, XCircle } from "lucide-react";

export function ChatMessage({ message }: { message: Message }) {
  const [openTools, setOpenTools] = useState(false);
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[80%] rounded-lg rounded-tr-sm bg-primary text-primary-foreground px-4 py-2.5 shadow-soft">
          <div className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="h-8 w-8 rounded-md bg-accent flex items-center justify-center shrink-0 ring-1 ring-border">
        <Sparkles className="h-4 w-4 text-primary" />
      </div>
      <div className="flex-1 max-w-[80%] space-y-2 min-w-0">
        <div className="rounded-lg rounded-tl-sm bg-card border border-border px-4 py-3 shadow-soft overflow-hidden">
          <MarkdownLite text={message.content} />
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-muted-foreground">Источники:</span>
            {message.sources.map((s, i) => (
              <span key={i} className="inline-flex items-center gap-1 rounded-md bg-secondary text-secondary-foreground px-2 py-0.5">
                <FileText className="h-3 w-3" />
                {s.file}{s.page ? ` · стр. ${s.page}` : ""}
              </span>
            ))}
          </div>
        )}

        {message.toolEvents && message.toolEvents.length > 0 && (
          <Collapsible open={openTools} onOpenChange={setOpenTools}>
            <CollapsibleTrigger className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
              <ChevronDown className={`h-3 w-3 transition-transform ${openTools ? "rotate-180" : ""}`} />
              Agent activity ({message.toolEvents.length})
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-1.5 space-y-1">
              {message.toolEvents.map((e) => (
                <div key={e.id} className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/50 rounded-md px-2 py-1">
                  {e.status === "success" ? <CheckCircle2 className="h-3.5 w-3.5 text-success" /> : <XCircle className="h-3.5 w-3.5 text-destructive" />}
                  <span className="font-mono text-foreground">{e.name}</span>
                  <span className="truncate">— {e.detail}</span>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="h-8 w-8 rounded-md bg-accent flex items-center justify-center shrink-0 ring-1 ring-border">
        <Sparkles className="h-4 w-4 text-primary" />
      </div>
      <div className="rounded-lg rounded-tl-sm bg-card border border-border px-4 py-3 shadow-soft">
        <div className="flex items-center gap-1.5">
          <span className="typing-dot" /><span className="typing-dot" /><span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
