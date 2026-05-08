// Mock API client mirroring the documented REST contract.
// Swap this module to use real fetch() when wiring to FastAPI.

import {
  SEED_USERS, SEED_CONVERSATIONS, SEED_PDFS, SEED_TEXT_BLOCKS, SEED_ADMIN_LOGS, SEED_DOC_META,
  type Conversation, type Message, type PdfFile, type TextBlock, type AdminLog, type DocMeta, type User,
} from "./data";

const delay = (ms = 350) => new Promise((r) => setTimeout(r, ms));
const uid = () => Math.random().toString(36).slice(2, 10);

// In-memory mutable stores (persist within session)
let conversations: Conversation[] = JSON.parse(JSON.stringify(SEED_CONVERSATIONS));
let pdfs: PdfFile[] = JSON.parse(JSON.stringify(SEED_PDFS));
let blocks: TextBlock[] = JSON.parse(JSON.stringify(SEED_TEXT_BLOCKS));
let adminLogs: AdminLog[] = JSON.parse(JSON.stringify(SEED_ADMIN_LOGS));
const docMeta: DocMeta[] = JSON.parse(JSON.stringify(SEED_DOC_META));

export const api = {
  // ───────── auth ─────────
  async login(username: string, password: string): Promise<{ token: string; user: User }> {
    await delay();
    const seed = SEED_USERS[username];
    if (!seed || seed.password !== password) throw new Error("Неверный логин или пароль");
    return { token: `mock-${username}-${Date.now()}`, user: seed.user };
  },
  async register(username: string, password: string): Promise<{ token: string; user: User }> {
    await delay();
    if (!username || password.length < 4) throw new Error("Пароль должен содержать минимум 4 символа");
    if (SEED_USERS[username]) throw new Error("Пользователь уже существует");
    const user: User = { username, role: "user", displayName: username, mustChangePassword: false };
    return { token: `mock-${username}-${Date.now()}`, user };
  },
  async logout() { await delay(120); },
  async me(token: string): Promise<User> {
    await delay(120);
    const name = token.split("-")[1];
    return SEED_USERS[name]?.user ?? { username: name, role: "user", displayName: name, mustChangePassword: false };
  },

  // ───────── chat ─────────
  async listConversations(): Promise<Conversation[]> {
    await delay(180);
    return [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  },
  async createConversation(title: string): Promise<Conversation> {
    await delay(150);
    const c: Conversation = { id: uid(), title, updatedAt: new Date().toISOString(), messages: [] };
    conversations = [c, ...conversations];
    return c;
  },
  async deleteConversation(id: string) {
    await delay(150);
    conversations = conversations.filter((c) => c.id !== id);
  },
  async getHistory(conversationId: string): Promise<Message[]> {
    await delay(200);
    return conversations.find((c) => c.id === conversationId)?.messages ?? [];
  },
  async sendMessage(conversationId: string, message: string): Promise<Message> {
    await delay(900);
    const conv = conversations.find((c) => c.id === conversationId);
    if (!conv) throw new Error("Диалог не найден");
    const userMsg: Message = { id: uid(), role: "user", content: message, createdAt: new Date().toISOString() };
    const assistant: Message = {
      id: uid(),
      role: "assistant",
      content: `По вашему вопросу «${message}» в базе знаний найдены релевантные положения. **Краткий ответ**: см. источники ниже для деталей.`,
      sources: [{ file: "Регламент_закупок_2024.pdf", page: 7 }],
      toolEvents: [
        { id: uid(), name: "knowledge.search", status: "success", detail: "Найдено 3 фрагмента" },
      ],
      createdAt: new Date().toISOString(),
    };
    conv.messages = [...conv.messages, userMsg, assistant];
    conv.updatedAt = new Date().toISOString();
    return assistant;
  },
  // ───────── knowledge ─────────
  async listFiles(): Promise<PdfFile[]> { await delay(180); return [...pdfs]; },
  async uploadFile(file: File): Promise<PdfFile> {
    await delay(700);
    const created: PdfFile = {
      path: `/docs/${uid()}_${file.name}`,
      name: file.name,
      sizeBytes: file.size,
      pages: Math.max(1, Math.round(file.size / 50_000)),
      uploadedAt: new Date().toISOString(),
      ragText: "Извлечённый текст появится после индексации…",
      ragOverride: false,
    };
    pdfs = [created, ...pdfs];
    return created;
  },
  async deleteFile(path: string) { await delay(150); pdfs = pdfs.filter((p) => p.path !== path); },
  async updateRagText(path: string, text: string) {
    await delay(200);
    const f = pdfs.find((p) => p.path === path);
    if (f) { f.ragText = text; f.ragOverride = true; }
  },
  async listBlocks(): Promise<TextBlock[]> { await delay(150); return [...blocks]; },
  async upsertBlock(b: Omit<TextBlock, "updatedAt"> & { id?: string }): Promise<TextBlock> {
    await delay(200);
    const now = new Date().toISOString();
    if (b.id && blocks.find((x) => x.id === b.id)) {
      blocks = blocks.map((x) => (x.id === b.id ? { ...x, ...b, updatedAt: now } as TextBlock : x));
      return blocks.find((x) => x.id === b.id)!;
    }
    const created: TextBlock = { id: uid(), name: b.name, content: b.content, updatedAt: now };
    blocks = [created, ...blocks];
    return created;
  },
  async deleteBlock(id: string) { await delay(150); blocks = blocks.filter((b) => b.id !== id); },
  async reindex() { await delay(900); return { ok: true }; },

  // ───────── admin ─────────
  async listLogs(page: number, pageSize: number): Promise<{ items: AdminLog[]; total: number }> {
    await delay(220);
    const start = (page - 1) * pageSize;
    return { items: adminLogs.slice(start, start + pageSize), total: adminLogs.length };
  },
  async reviewLog(id: string, score: number, correctAnswer?: string) {
    await delay(180);
    const normalized = Math.max(1, Math.min(10, Math.round(score)));
    adminLogs = adminLogs.map((l) => l.id === id ? { ...l, reviewScore: normalized, correctAnswer: correctAnswer ?? l.correctAnswer, status: "ok" } : l);
  },
  async listDocMeta(): Promise<DocMeta[]> { await delay(150); return [...docMeta]; },
  async updateDocMetaExpiry(id: string, expiresAt: string) {
    await delay(180);
    const nextStatus: DocMeta["status"] = !expiresAt
      ? "actual"
      : Date.parse(expiresAt) < Date.now()
        ? "expired"
        : (Date.parse(expiresAt) - Date.now()) / (24 * 60 * 60 * 1000) <= 30
          ? "review_soon"
          : "actual";
    for (const doc of docMeta) {
      if (doc.id === id) {
        doc.expiresAt = expiresAt;
        doc.status = nextStatus;
      }
    }
  },
};
