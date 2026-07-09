export type UserRole = "user" | "admin" | "client";

export type User = {
  username: string;
  role: UserRole;
  displayName: string;
  mustChangePassword: boolean;
};

export type Source = {
  file: string;
  page?: number;
};

export type ToolEvent = {
  id: string;
  name: string;
  status: "success" | "error";
  detail: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  sources?: Source[];
  toolEvents?: ToolEvent[];
};

export type Conversation = {
  id: string;
  title: string;
  updatedAt: string;
};

export type PdfFile = {
  path: string;
  name: string;
  sizeBytes: number;
  pages: number;
  uploadedAt: string;
  ragText: string;
  ragOverride: boolean;
};

export type TextBlock = {
  id: string;
  name: string;
  content: string;
  updatedAt: string;
};

export type AdminLog = {
  id: string;
  createdAt: string;
  username: string;
  message: string;
  answer: string;
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

export type MondayStatus = {
  connected: boolean;
  // false when the integration is not configured on the server (no client id/secret).
  enabled: boolean;
  scope?: string;
  accountId?: string | null;
  mondayUserName?: string | null;
  connectedAt?: string | null;
};

export type MemoryCategory = "fact" | "preference" | "task_recipe";

export type UserMemory = {
  // Short stable handle used to reference the memory in API calls.
  id: string;
  content: string;
  category: MemoryCategory;
  // Who created it: agent | user | admin.
  source: string;
  createdAt: string;
  updatedAt: string;
};

// ── Client portal ────────────────────────────────────────────────────────────

export type ClientTaskBlock = {
  project_id: number;
  project_name: string;
  rows: Record<string, string>[];
  fetched_at: string;
  error?: boolean;
};

export type ClientTasksTable = {
  headers: string[];
  status_colors: Record<string, string>;
  projects: ClientTaskBlock[];
};

export type ClientChatReply = {
  answer: string;
  table: { title?: string; columns: string[]; rows: string[][] } | null;
};

export type ClientProgressProject = {
  id: number;
  name: string;
  url: string | null;
};

export type ClientInvite = {
  token: string;
  url?: string;
  company_name: string;
  project_ids: number[];
  project_names: string[];
  max_uses: number;
  used_count: number;
  expires_at: string | null;
  created_by: string;
  created_at: string | null;
};

export type OrlandaProject = {
  id: number;
  name: string;
  monday_item_id: string;
};

export type MemorySettings = {
  // Per-user toggle.
  enabled: boolean;
  // false when the feature is disabled platform-wide (global kill-switch).
  globallyEnabled: boolean;
};
