export type UserRole = "user" | "admin";

export type User = {
  username: string;
  role: UserRole;
  displayName: string;
  mustChangePassword: boolean;
};

export type MondayConnectionStatus = {
  enabled: boolean;
  connected: boolean;
  mondayUserId?: string | null;
  mondayAccountId?: string | null;
  scope?: string | null;
  expiresAt?: string | null;
  revoked?: boolean;
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
