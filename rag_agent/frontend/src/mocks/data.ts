// Static seed data for the Orlanda prototype. Replace with real API later.

export type User = { username: string; role: "user" | "admin"; displayName: string; mustChangePassword: boolean };

export type ToolEvent = { id: string; name: string; status: "success" | "error"; detail: string };
export type Source = { file: string; page?: number };
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  toolEvents?: ToolEvent[];
  createdAt: string;
};
export type Conversation = { id: string; title: string; updatedAt: string; messages: Message[] };

export type PdfFile = { path: string; name: string; sizeBytes: number; pages: number; uploadedAt: string; ragText: string; ragOverride: boolean };
export type TextBlock = { id: string; name: string; content: string; updatedAt: string };

export type AdminLog = {
  id: string;
  createdAt: string;
  username: string;
  message: string;
  model: string;
  reviewScore: number | null;
  status: "ok" | "review" | "flagged";
  correctAnswer?: string;
};

export type DocMeta = {
  id: string;
  name: string;
  owner: string;
  reviewedAt: string;
  expiresAt: string;
  status: "actual" | "review_soon" | "expired";
};

export const SEED_USERS: Record<string, { password: string; user: User }> = {
  user: { password: "user", user: { username: "user", role: "user", displayName: "Иван Петров", mustChangePassword: false } },
  admin: { password: "admin", user: { username: "admin", role: "admin", displayName: "Админ Орланда", mustChangePassword: false } },
};

export const SEED_CONVERSATIONS: Conversation[] = [
  {
    id: "c1",
    title: "Регламент закупок 2024",
    updatedAt: "2025-04-18T09:14:00Z",
    messages: [
      { id: "m1", role: "user", content: "Какой лимит закупки без согласования с финдиректором?", createdAt: "2025-04-18T09:10:00Z" },
      {
        id: "m2",
        role: "assistant",
        content: "Согласно **Регламенту закупок 2024** (раздел 3.2), руководитель подразделения может согласовывать закупки до **150 000 ₽** без участия финансового директора. Свыше этой суммы требуется виза финдира и второй подписант.",
        sources: [{ file: "Регламент_закупок_2024.pdf", page: 7 }, { file: "Регламент_закупок_2024.pdf", page: 12 }],
        toolEvents: [
          { id: "t1", name: "knowledge.search", status: "success", detail: "Найдено 4 фрагмента" },
          { id: "t2", name: "knowledge.rerank", status: "success", detail: "top-2 отобрано" },
        ],
        createdAt: "2025-04-18T09:14:00Z",
      },
    ],
  },
  {
    id: "c2",
    title: "Командировки: суточные",
    updatedAt: "2025-04-17T15:42:00Z",
    messages: [
      { id: "m3", role: "user", content: "Размер суточных по России и СНГ?", createdAt: "2025-04-17T15:40:00Z" },
      {
        id: "m4",
        role: "assistant",
        content: "Суточные: **Россия — 2 500 ₽/день**, **СНГ — 3 500 ₽/день**, дальнее зарубежье — по таблице приложения.",
        sources: [{ file: "Положение_о_командировках.pdf", page: 3 }],
        toolEvents: [{ id: "t3", name: "knowledge.search", status: "success", detail: "Найдено 2 фрагмента" }],
        createdAt: "2025-04-17T15:42:00Z",
      },
    ],
  },
  {
    id: "c3",
    title: "Отпуск без сохранения",
    updatedAt: "2025-04-15T11:00:00Z",
    messages: [],
  },
];

export const SEED_PDFS: PdfFile[] = [
  { path: "/docs/regl_zakupok_2024.pdf", name: "Регламент_закупок_2024.pdf", sizeBytes: 1_240_000, pages: 24, uploadedAt: "2025-02-10T10:00:00Z", ragText: "Регламент закупок Orlanda Engineering, редакция 2024 года. Раздел 3.2 устанавливает лимиты согласования…", ragOverride: false },
  { path: "/docs/komandirovki.pdf", name: "Положение_о_командировках.pdf", sizeBytes: 612_400, pages: 11, uploadedAt: "2025-01-22T08:30:00Z", ragText: "Положение о служебных командировках сотрудников. Суточные по России — 2 500 ₽…", ragOverride: true },
  { path: "/docs/ohrana_truda.pdf", name: "Инструкция_по_охране_труда.pdf", sizeBytes: 980_000, pages: 18, uploadedAt: "2025-03-04T12:15:00Z", ragText: "Инструкция по охране труда для офисных сотрудников…", ragOverride: false },
  { path: "/docs/it_security.pdf", name: "Политика_ИБ_v3.pdf", sizeBytes: 2_140_000, pages: 31, uploadedAt: "2025-03-28T17:45:00Z", ragText: "Политика информационной безопасности, версия 3…", ragOverride: false },
];

export const SEED_TEXT_BLOCKS: TextBlock[] = [
  { id: "b1", name: "График работы офиса", content: "Пн–Пт 09:00–18:00, перерыв 13:00–14:00. Удалённый день — пятница (по согласованию с руководителем).", updatedAt: "2025-04-01T10:00:00Z" },
  { id: "b2", name: "Контакты HR", content: "HR-партнёр: hr@orlanda.example, доб. 204. Расчёт зарплаты: payroll@orlanda.example.", updatedAt: "2025-04-05T14:00:00Z" },
  { id: "b3", name: "Корпоративные скидки", content: "Партнёрские программы для сотрудников: фитнес, страхование, обучение. Подробности в портале льгот.", updatedAt: "2025-03-20T09:00:00Z" },
];

export const SEED_ADMIN_LOGS: AdminLog[] = Array.from({ length: 28 }).map((_, i) => ({
  id: `log_${i + 1}`,
  createdAt: new Date(Date.now() - i * 3.6e6 * 7).toISOString(),
  username: ["user", "i.petrov", "a.smirnova", "k.ivanov", "m.belova"][i % 5],
  message: [
    "Какой лимит закупки без согласования?",
    "Размер суточных по России?",
    "Как оформить удалённый день?",
    "Сроки согласования отпуска",
    "Кому писать по доступам в 1С?",
    "Какая политика по паролям?",
  ][i % 6],
  model: i % 3 === 0 ? "gpt-4o-mini" : "gpt-4.1",
  reviewScore: ([null, 5, 4, 3, null, 5] as const)[i % 6],
  status: (["ok", "ok", "review", "ok", "flagged", "ok"] as const)[i % 6],
  correctAnswer: i % 4 === 0 ? "Сверьтесь с регламентом, раздел 3.2." : undefined,
}));

export const SEED_DOC_META: DocMeta[] = [
  { id: "d1", name: "Регламент закупок 2024", owner: "Финансовый отдел", reviewedAt: "2025-01-15", expiresAt: "2026-01-15", status: "actual" },
  { id: "d2", name: "Положение о командировках", owner: "HR", reviewedAt: "2024-09-01", expiresAt: "2025-09-01", status: "review_soon" },
  { id: "d3", name: "Инструкция по охране труда", owner: "ОТиТБ", reviewedAt: "2024-03-10", expiresAt: "2025-03-10", status: "expired" },
  { id: "d4", name: "Политика ИБ v3", owner: "ИТ-безопасность", reviewedAt: "2025-03-28", expiresAt: "2026-03-28", status: "actual" },
  { id: "d5", name: "Кодекс этики", owner: "Юридический отдел", reviewedAt: "2024-11-20", expiresAt: "2025-11-20", status: "actual" },
  { id: "d6", name: "Положение о премировании", owner: "HR", reviewedAt: "2024-06-01", expiresAt: "2025-06-01", status: "review_soon" },
];
