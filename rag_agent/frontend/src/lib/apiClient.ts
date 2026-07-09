import type {
  AdminLog,
  ClientChatReply,
  ClientInvite,
  ClientProgressProject,
  ClientTasksTable,
  Conversation,
  DocMeta,
  MemorySettings,
  Message,
  MondayStatus,
  OrlandaProject,
  PdfFile,
  TextBlock,
  ToolEvent,
  User,
  UserMemory,
} from "@/lib/types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "";
const TOKEN_KEY = "orlanda.token";

type RequestOptions = {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  token?: string | null;
  body?: unknown;
  isMultipart?: boolean;
};

function makeUrl(path: string): string {
  if (!API_BASE_URL) return path;
  return `${API_BASE_URL}${path}`;
}

async function parseError(res: Response): Promise<string> {
  const fallback = `${res.status} ${res.statusText}`.trim();
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail)) {
      const msg = detail
        .map((x) => (typeof x === "string" ? x : x?.msg || x?.message || JSON.stringify(x)))
        .join(" ");
      if (msg) return msg;
    }
  } catch {
    // ignore parse error
  }
  return fallback || "Ошибка запроса";
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const token = options.token ?? localStorage.getItem(TOKEN_KEY);
  const headers: Record<string, string> = {};
  if (!options.isMultipart) {
    headers["Content-Type"] = "application/json";
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(makeUrl(path), {
    method: options.method ?? "GET",
    headers,
    body: options.body == null ? undefined : options.isMultipart ? (options.body as BodyInit) : JSON.stringify(options.body),
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

async function requestBlob(path: string, token?: string | null): Promise<Blob> {
  const authToken = token ?? localStorage.getItem(TOKEN_KEY);
  const res = await fetch(makeUrl(path), {
    headers: authToken ? { Authorization: `Bearer ${authToken}` } : undefined,
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return res.blob();
}

function normalizeRole(role: unknown): "user" | "assistant" {
  const value = String(role || "").toLowerCase();
  if (value === "assistant" || value === "ai") return "assistant";
  return "user";
}

function normalizeEvent(event: any, idx: number): ToolEvent {
  const statusRaw = String(event?.status || "").toLowerCase();
  return {
    id: String(event?.id || event?.ts || `${Date.now()}-${idx}`),
    name: String(event?.tool_name || event?.name || event?.source || "event"),
    status: statusRaw === "error" ? "error" : "success",
    detail: String(event?.message || event?.detail || ""),
  };
}

function toIso(value: unknown): string {
  if (typeof value === "string" && value.trim()) return value;
  return new Date().toISOString();
}

function mapFileToPdf(f: any): PdfFile {
  return {
    path: String(f?.path || ""),
    name: String(f?.name || f?.path || "document.pdf"),
    sizeBytes: Number(f?.size || 0),
    pages: Math.max(1, Number(f?.pages || 1)),
    uploadedAt: toIso(f?.uploaded_at),
    ragText: "",
    ragOverride: false,
  };
}

function deriveDocStatus(expired: boolean, expiresAt: string): "actual" | "review_soon" | "expired" {
  if (expired) return "expired";
  if (!expiresAt) return "actual";
  const expiresTs = Date.parse(expiresAt);
  if (!Number.isFinite(expiresTs)) return "actual";
  const days = (expiresTs - Date.now()) / (24 * 60 * 60 * 1000);
  if (days <= 30) return "review_soon";
  return "actual";
}

function toDateInput(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "";
  const raw = value.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

function mapMemory(m: any): UserMemory {
  const category = String(m?.category || "fact");
  return {
    id: String(m?.id || ""),
    content: String(m?.content || ""),
    category: (["fact", "preference", "task_recipe"].includes(category) ? category : "fact") as UserMemory["category"],
    source: String(m?.source || "agent"),
    createdAt: toIso(m?.created_at),
    updatedAt: toIso(m?.updated_at),
  };
}

export const api = {
  auth: {
    async login(username: string, password: string): Promise<{ token: string; user: User }> {
      const data = await request<{ token: string; username: string }>("/auth/login", {
        method: "POST",
        body: { username, password },
      });
      const me = await this.me(data.token);
      return { token: data.token, user: me };
    },
    async register(username: string, password: string): Promise<{ token: string; user: User }> {
      const data = await request<{ token: string; username: string }>("/auth/register", {
        method: "POST",
        body: { username, password },
      });
      const me = await this.me(data.token);
      return { token: data.token, user: me };
    },
    async logout(token?: string | null): Promise<void> {
      await request<{ ok: boolean }>("/auth/logout", { method: "POST", token: token ?? null });
    },
    async me(token?: string | null): Promise<User> {
      const data = await request<{ username: string; role: "user" | "admin" | "client"; must_change_password?: boolean }>("/auth/me", { token: token ?? null });
      return {
        username: data.username,
        role: data.role,
        displayName: data.username,
        mustChangePassword: Boolean(data.must_change_password),
      };
    },
    async changePassword(
      token: string,
      payload: { currentPassword?: string; newPassword: string; repeatPassword: string },
    ): Promise<{ token: string; user: User }> {
      const data = await request<{ token: string; username: string; role: "user" | "admin" | "client"; must_change_password?: boolean }>(
        "/auth/password/change",
        {
          method: "POST",
          token,
          body: {
            current_password: payload.currentPassword ?? "",
            new_password: payload.newPassword,
            repeat_password: payload.repeatPassword,
          },
        },
      );
      return {
        token: data.token,
        user: {
          username: data.username,
          role: data.role,
          displayName: data.username,
          mustChangePassword: Boolean(data.must_change_password),
        },
      };
    },
  },

  clientPortal: {
    async invitePreview(token: string): Promise<{ valid: boolean; error?: string; company_name?: string; project_names?: string[] }> {
      return request(`/invites/${encodeURIComponent(token)}`);
    },
    async registerClient(inviteToken: string, email: string, password: string): Promise<{ token: string; user: User }> {
      const data = await request<{ token: string; username: string; role: string }>("/auth/register-client", {
        method: "POST",
        body: { invite_token: inviteToken, email, password },
      });
      return {
        token: data.token,
        user: { username: data.username, role: "client", displayName: data.username, mustChangePassword: false },
      };
    },
    async tasksTable(token: string): Promise<ClientTasksTable> {
      return request("/client/portal/tasks-table", { token });
    },
    async chat(token: string, message: string): Promise<ClientChatReply> {
      return request("/client/portal/chat", { method: "POST", token, body: { message } });
    },
    async chatReset(token: string): Promise<void> {
      await request("/client/portal/chat/reset", { method: "POST", token });
    },
    async progress(token: string): Promise<ClientProgressProject[]> {
      const data = await request<{ projects: ClientProgressProject[] }>("/client/portal/progress", { token });
      return data.projects || [];
    },
  },

  adminInvites: {
    async orlandaProjects(token: string): Promise<OrlandaProject[]> {
      const data = await request<{ projects: OrlandaProject[] }>("/admin/orlanda/projects", { token });
      return data.projects || [];
    },
    async list(token: string): Promise<ClientInvite[]> {
      const data = await request<{ invites: ClientInvite[] }>("/admin/invites", { token });
      return data.invites || [];
    },
    async create(
      token: string,
      payload: { project_ids: number[]; project_names: string[]; company_name?: string; max_uses?: number; expires_days?: number },
    ): Promise<ClientInvite> {
      const data = await request<{ invite: ClientInvite }>("/admin/invites", { method: "POST", token, body: payload });
      return data.invite;
    },
    async remove(token: string, inviteToken: string): Promise<void> {
      await request(`/admin/invites/${encodeURIComponent(inviteToken)}`, { method: "DELETE", token });
    },
  },

  chat: {
    async listConversations(token: string): Promise<Conversation[]> {
      const data = await request<{ conversations: Array<{ id: string; title?: string; last_activity_ts?: string }> }>(
        "/chat/conversations",
        { token },
      );
      const conversations = (data.conversations || [])
        .map((entry) => {
          const id = String(entry?.id || "").trim();
          if (!id) return null;
          return {
            id,
            title: String(entry?.title || id),
            updatedAt: toIso(entry?.last_activity_ts),
          } satisfies Conversation;
        })
        .filter((x): x is Conversation => Boolean(x));
      if (!conversations.length) {
        return [{ id: "default", title: "Основной диалог", updatedAt: new Date().toISOString() }];
      }
      return conversations.sort((a, b) => toIso(b.updatedAt).localeCompare(toIso(a.updatedAt)));
    },
    async createConversation(token: string, title: string): Promise<Conversation> {
      const localId = `c_${Date.now().toString(36)}`;
      const data = await request<{
        conversation: { id: string; title: string; last_activity_ts?: string };
      }>("/chat/conversations", {
        method: "POST",
        token,
        body: { id: localId, title: title.trim() || "Новый диалог" },
      });
      return {
        id: String(data?.conversation?.id || localId),
        title: String(data?.conversation?.title || title || "Новый диалог"),
        updatedAt: toIso(data?.conversation?.last_activity_ts),
      };
    },
    async deleteConversation(token: string, conversationId: string): Promise<void> {
      await request(`/chat/conversation?conversation_id=${encodeURIComponent(conversationId)}`, {
        method: "DELETE",
        token,
      });
    },
    async getHistory(token: string, conversationId: string): Promise<Message[]> {
      const data = await request<{ messages: Array<{ role: string; content: string }> }>(
        `/chat/history?conversation_id=${encodeURIComponent(conversationId)}`,
        { token },
      );
      // The backend can persist the same assistant answer more than once: its
      // best-effort "ensure assistant turn" guard appends a duplicate whenever its
      // exact-string match misses (structured-JSON vs streamed text, or whitespace
      // differences), and conversations that doubled before the backend fix shipped
      // still hold those duplicate rows. Collapse a repeated assistant turn within
      // the SAME exchange — i.e. an assistant message whose whitespace-normalized
      // text matches the previous assistant turn with no user message since. We
      // reset on every user turn, so legitimate distinct Q&A pairs are never merged.
      // Stable ids (no Date.now) keep React keys consistent across refetches.
      const norm = (s: string) => s.trim().replace(/\s+/g, " ");
      const result: Message[] = [];
      let lastAssistantNorm: string | null = null;
      (data.messages || []).forEach((m, index) => {
        const role = normalizeRole(m.role);
        const content = String(m.content || "");
        const n = norm(content);
        if (role === "assistant") {
          if (n && n === lastAssistantNorm) return; // duplicate assistant turn for this exchange
          lastAssistantNorm = n;
        } else {
          lastAssistantNorm = null; // a user turn starts a new exchange
        }
        result.push({
          id: `h-${conversationId}-${index}`,
          role,
          content,
          createdAt: new Date().toISOString(),
        });
      });
      return result;
    },
    async sendMessage(
      token: string,
      conversationId: string,
      message: string,
    ): Promise<Message> {
      const data = await request<{
        response: string;
        sources: Array<{ file: string; page?: number | string }>;
        tool_events: any[];
      }>(`/chat?conversation_id=${encodeURIComponent(conversationId)}`, {
        method: "POST",
        token,
        body: { message },
      });
      return {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: String(data.response || ""),
        createdAt: new Date().toISOString(),
        sources: Array.isArray(data.sources)
          ? data.sources.map((s) => ({
              file: String(s.file || ""),
              page: s.page == null ? undefined : Number.isFinite(Number(s.page)) ? Number(s.page) + 1 : undefined,
            }))
          : [],
        toolEvents: Array.isArray(data.tool_events) ? data.tool_events.map(normalizeEvent) : [],
      };
    },
    /**
     * Streaming chat. Calls `/chat/stream` and invokes callbacks as SSE events arrive.
     * onDelta — every text chunk; onToolStart/onToolEnd — tool activity; onDone — final
     * payload with sources + tool_events; onError — fatal error string.
     * Returns an AbortController so the caller can cancel.
     */
    sendMessageStream(
      token: string,
      conversationId: string,
      message: string,
      callbacks: {
        onDelta?: (text: string) => void;
        onToolStart?: (name: string) => void;
        onToolEnd?: (name: string, status: "success" | "error") => void;
        onDone?: (payload: {
          response: string;
          sources: Array<{ file: string; page?: number }>;
          toolEvents: ToolEvent[];
        }) => void;
        onError?: (msg: string) => void;
      },
    ): AbortController {
      const controller = new AbortController();
      (async () => {
        try {
          const res = await fetch(makeUrl(`/chat/stream?conversation_id=${encodeURIComponent(conversationId)}`), {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ message }),
            signal: controller.signal,
          });
          if (!res.ok || !res.body) {
            const errText = await parseError(res);
            callbacks.onError?.(errText);
            return;
          }
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // SSE messages are separated by blank lines (\n\n)
            let sepIdx;
            while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
              const rawEvent = buffer.slice(0, sepIdx);
              buffer = buffer.slice(sepIdx + 2);
              const dataLine = rawEvent
                .split("\n")
                .find((l) => l.startsWith("data:"));
              if (!dataLine) continue;
              const payloadStr = dataLine.slice(5).trim();
              if (!payloadStr) continue;
              try {
                const payload = JSON.parse(payloadStr);
                switch (payload.type) {
                  case "delta":
                    if (typeof payload.text === "string") callbacks.onDelta?.(payload.text);
                    break;
                  case "tool_start":
                    callbacks.onToolStart?.(String(payload.name || ""));
                    break;
                  case "tool_end":
                    callbacks.onToolEnd?.(
                      String(payload.name || ""),
                      payload.status === "error" ? "error" : "success",
                    );
                    break;
                  case "done":
                    callbacks.onDone?.({
                      response: String(payload.response || ""),
                      sources: Array.isArray(payload.sources)
                        ? payload.sources.map((s: any) => ({
                            file: String(s.file || ""),
                            page:
                              s.page == null
                                ? undefined
                                : Number.isFinite(Number(s.page))
                                ? Number(s.page) + 1
                                : undefined,
                          }))
                        : [],
                      toolEvents: Array.isArray(payload.tool_events)
                        ? payload.tool_events.map(normalizeEvent)
                        : [],
                    });
                    break;
                  case "error":
                    callbacks.onError?.(String(payload.message || "Ошибка стриминга"));
                    break;
                }
              } catch {
                // ignore malformed event
              }
            }
          }
        } catch (e) {
          if ((e as DOMException)?.name === "AbortError") return;
          callbacks.onError?.(e instanceof Error ? e.message : "Ошибка соединения");
        }
      })();
      return controller;
    },
    touchConversation(_conversationId: string): void {
      // No-op: conversation ordering is now backend-derived via /chat/conversations.
    },
  },

  knowledge: {
    async listFiles(token: string): Promise<PdfFile[]> {
      const data = await request<{ files: any[] }>("/knowledge/files", { token });
      return (data.files || []).map(mapFileToPdf);
    },
    async getPreviewBlob(token: string, path: string): Promise<Blob> {
      return requestBlob(`/knowledge/files/preview?path=${encodeURIComponent(path)}`, token);
    },
    async getFileText(token: string, path: string): Promise<{ text: string; source: "override" | "extracted"; path: string }> {
      return request(`/knowledge/files/text?path=${encodeURIComponent(path)}`, { token });
    },
    async updateRagText(token: string, path: string, text: string): Promise<{ path: string; source: "override" | "extracted" }> {
      return request("/knowledge/files/text", {
        method: "PUT",
        token,
        body: { path, text },
      });
    },
    async uploadFile(token: string, file: File): Promise<void> {
      const formData = new FormData();
      formData.append("file", file);
      await request("/knowledge/upload", {
        method: "POST",
        token,
        body: formData,
        isMultipart: true,
      });
    },
    async deleteFile(token: string, path: string): Promise<void> {
      await request(`/knowledge/files?path=${encodeURIComponent(path)}`, { method: "DELETE", token });
    },
    async listBlocks(token: string): Promise<TextBlock[]> {
      const data = await request<{ items: any[] }>("/knowledge/items", { token });
      return (data.items || []).map((it) => ({
        id: String(it.id),
        name: String(it.name || "Без названия"),
        content: String(it.content || ""),
        updatedAt: toIso(it.last_updated_at || it.created_at),
      }));
    },
    async getBlock(token: string, id: string): Promise<TextBlock> {
      const it = await request<any>(`/knowledge/items/${encodeURIComponent(id)}`, { token });
      return {
        id: String(it.id),
        name: String(it.name || "Без названия"),
        content: String(it.content || ""),
        updatedAt: toIso(it.last_updated_at || it.created_at),
      };
    },
    async upsertBlock(token: string, block: { id?: string; name: string; content: string }): Promise<void> {
      if (block.id) {
        await request(`/knowledge/items/${encodeURIComponent(block.id)}`, {
          method: "PATCH",
          token,
          body: { name: block.name, content: block.content },
        });
        return;
      }
      await request("/knowledge/items", {
        method: "POST",
        token,
        body: { name: block.name, content: block.content },
      });
    },
    async deleteBlock(token: string, id: string): Promise<void> {
      await request(`/knowledge/items/${encodeURIComponent(id)}`, { method: "DELETE", token });
    },
    async reindex(token: string): Promise<void> {
      await request("/knowledge/reindex", { method: "POST", token });
    },
  },

  admin: {
    async listLogs(token: string, page: number, pageSize: number): Promise<{ items: AdminLog[]; total: number }> {
      const offset = Math.max(0, (page - 1) * pageSize);
      const data = await request<{ entries: any[]; total: number }>(
        `/admin/logs?limit=${encodeURIComponent(String(pageSize))}&offset=${encodeURIComponent(String(offset))}`,
        { token },
      );
      return {
        total: Number(data.total || 0),
        items: (data.entries || []).map((entry) => {
          const score = entry?.score == null ? null : Number(entry.score);
          const status: AdminLog["status"] =
            entry?.error ? "flagged" : score == null ? "review" : "ok";
          return {
            id: String(entry?.id || ""),
            createdAt: toIso(entry?.timestamp),
            username: String(entry?.username || ""),
            message: String(entry?.question || ""),
            answer: String(entry?.answer || ""),
            model: String(entry?.model || "n/a"),
            reviewScore: Number.isFinite(score as number) ? score : null,
            status,
            correctAnswer: String(entry?.correct_answer || ""),
          };
        }),
      };
    },
    async reviewLog(token: string, id: string, score: number, correctAnswer?: string): Promise<void> {
      await request(`/admin/logs/${encodeURIComponent(id)}/review`, {
        method: "PATCH",
        token,
        body: { score: Math.max(1, Math.min(10, Math.round(score))), correct_answer: correctAnswer ?? "" },
      });
    },
    async listDocMeta(token: string): Promise<DocMeta[]> {
      const data = await request<{ pdfs: any[]; items: any[] }>("/admin/documents/metadata", { token });
      const pdfs = (data.pdfs || []).map((it, idx) => ({
        id: `pdf-${idx}-${it.path || it.name || "unknown"}`,
        name: String(it.name || it.path || "PDF"),
        owner: String(it.responsible || "—"),
        reviewedAt: toDateInput(it.last_updated_at),
        expiresAt: toDateInput(it.expires_at),
        status: deriveDocStatus(Boolean(it.expired), String(it.expires_at || "")),
      })) satisfies DocMeta[];
      const items = (data.items || []).map((it, idx) => ({
        id: `item-${idx}-${it.id || it.name || "unknown"}`,
        name: String(it.name || "Текстовый блок"),
        owner: String(it.responsible || "—"),
        reviewedAt: toDateInput(it.last_updated_at),
        expiresAt: toDateInput(it.expires_at),
        status: deriveDocStatus(Boolean(it.expired), String(it.expires_at || "")),
      })) satisfies DocMeta[];
      return [...pdfs, ...items];
    },
    async updateDocMetaExpiry(token: string, id: string, expiresAt: string): Promise<void> {
      await request(`/admin/documents/metadata/${encodeURIComponent(id)}`, {
        method: "PATCH",
        token,
        body: { expires_at: expiresAt || null },
      });
    },
  },

  integrations: {
    monday: {
      async status(token: string): Promise<MondayStatus> {
        const data = await request<{
          connected: boolean;
          enabled?: boolean;
          scope?: string;
          account_id?: string | null;
          monday_user_name?: string | null;
          connected_at?: string | null;
        }>("/integrations/monday/status", { token });
        return {
          connected: Boolean(data.connected),
          enabled: Boolean(data.enabled),
          scope: data.scope ?? "",
          accountId: data.account_id ?? null,
          mondayUserName: data.monday_user_name ?? null,
          connectedAt: data.connected_at ?? null,
        };
      },
      async authorizeUrl(token: string): Promise<string> {
        const data = await request<{ authorize_url: string }>("/integrations/monday/authorize", { token });
        return String(data.authorize_url || "");
      },
      async disconnect(token: string): Promise<boolean> {
        const data = await request<{ ok: boolean; disconnected: boolean }>("/integrations/monday", {
          method: "DELETE",
          token,
        });
        return Boolean(data.disconnected);
      },
    },
  },

  memory: {
    async list(token: string): Promise<UserMemory[]> {
      const data = await request<{ memories: any[] }>("/memories", { token });
      return (data.memories || []).map(mapMemory);
    },
    async add(token: string, content: string, category: string): Promise<UserMemory> {
      const data = await request<{ memory: any }>("/memories", {
        method: "POST",
        token,
        body: { content, category },
      });
      return mapMemory(data.memory);
    },
    async update(token: string, id: string, content: string): Promise<UserMemory> {
      const data = await request<{ memory: any }>(`/memories/${encodeURIComponent(id)}`, {
        method: "PATCH",
        token,
        body: { content },
      });
      return mapMemory(data.memory);
    },
    async remove(token: string, id: string): Promise<void> {
      await request(`/memories/${encodeURIComponent(id)}`, { method: "DELETE", token });
    },
    async clearAll(token: string): Promise<number> {
      const data = await request<{ deleted: number }>("/memories", { method: "DELETE", token });
      return Number(data.deleted || 0);
    },
    async getSettings(token: string): Promise<MemorySettings> {
      const data = await request<{ enabled: boolean; globally_enabled: boolean }>("/me/memory-settings", { token });
      return { enabled: Boolean(data.enabled), globallyEnabled: Boolean(data.globally_enabled) };
    },
    async setSettings(token: string, enabled: boolean): Promise<MemorySettings> {
      const data = await request<{ enabled: boolean; globally_enabled: boolean }>("/me/memory-settings", {
        method: "PUT",
        token,
        body: { enabled },
      });
      return { enabled: Boolean(data.enabled), globallyEnabled: Boolean(data.globally_enabled) };
    },
  },

  adminMemory: {
    async list(token: string, username: string): Promise<{ enabled: boolean; memories: UserMemory[] }> {
      const data = await request<{ enabled: boolean; memories: any[] }>(
        `/admin/users/${encodeURIComponent(username)}/memories`,
        { token },
      );
      return { enabled: Boolean(data.enabled), memories: (data.memories || []).map(mapMemory) };
    },
    async add(token: string, username: string, content: string, category: string): Promise<UserMemory> {
      const data = await request<{ memory: any }>(`/admin/users/${encodeURIComponent(username)}/memories`, {
        method: "POST",
        token,
        body: { content, category },
      });
      return mapMemory(data.memory);
    },
    async update(token: string, username: string, id: string, content: string): Promise<UserMemory> {
      const data = await request<{ memory: any }>(
        `/admin/users/${encodeURIComponent(username)}/memories/${encodeURIComponent(id)}`,
        { method: "PATCH", token, body: { content } },
      );
      return mapMemory(data.memory);
    },
    async remove(token: string, username: string, id: string): Promise<void> {
      await request(
        `/admin/users/${encodeURIComponent(username)}/memories/${encodeURIComponent(id)}`,
        { method: "DELETE", token },
      );
    },
  },
};
