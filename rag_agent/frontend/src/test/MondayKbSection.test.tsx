import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { MondayKbStatus } from "@/lib/types";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const statusMock = vi.fn();
const syncMock = vi.fn();

vi.mock("@/lib/apiClient", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/apiClient")>();
  return {
    ...actual,
    api: {
      knowledge: {
        mondayStatus: (...a: unknown[]) => statusMock(...a),
        mondaySync: (...a: unknown[]) => syncMock(...a),
      },
    },
  };
});

import { MondayKbSection } from "@/components/MondayKbSection";
import { mapMondayKbStatus } from "@/lib/apiClient";

const base: MondayKbStatus = mapMondayKbStatus({
  enabled: true,
  configured: true,
  token_source: "RAG_MONDAY_KB_API_TOKEN",
  running: false,
  last_ok: true,
  last_finished_at: "2026-11-05T10:00:00+00:00",
  roots: [{ kind: "folder", id: "500", name: "Knowledge base" }],
  stats: { complete: true },
  interval_minutes: 30,
  names: ["Knowledge base"],
  files: [
    { doc_id: "monday_asset:1", type: "monday_file", name: "onboarding.pdf", board: "Policies", item: "Отпуска", url: "https://m/1", chunks: 4 },
    { doc_id: "monday_doc:9", type: "monday_doc", name: "Handbook", url: "", chunks: 2 },
  ],
  skipped: [{ name: "photo.png", where: "Policies › Отпуска", reason: "изображение" }],
});

describe("MondayKbSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    statusMock.mockResolvedValue(base);
    syncMock.mockResolvedValue(true);
  });

  it("lists synced monday files with a link to the item and the skipped files", async () => {
    render(<MondayKbSection token="t" open refreshKey={0} />);
    expect(await screen.findByText("onboarding.pdf")).toBeInTheDocument();
    expect(screen.getByText("Policies › Отпуска").closest("a")).toHaveAttribute("href", "https://m/1");
    expect(screen.getByText("Handbook")).toBeInTheDocument();
    expect(screen.getByText(/документов: 2/)).toBeInTheDocument();
    expect(screen.getByText(/Не проиндексировано: 1/)).toBeInTheDocument();
    expect(screen.getByText(/monday · Knowledge base/)).toBeInTheDocument();
  });

  it("starts a sync and refreshes the status", async () => {
    render(<MondayKbSection token="t" open refreshKey={0} />);
    await screen.findByText("onboarding.pdf");
    statusMock.mockResolvedValueOnce({ ...base, running: true, progress: { done: 1, total: 3, current: "a.pdf" } });
    fireEvent.click(screen.getByRole("button", { name: /Синхронизировать/ }));
    await waitFor(() => expect(syncMock).toHaveBeenCalledWith("t"));
    expect(await screen.findByText(/1 из 3 · a\.pdf/)).toBeInTheDocument();
  });

  it("shows the sync error", async () => {
    statusMock.mockResolvedValue({ ...base, lastOk: false, lastError: "Нет токена monday", files: [] });
    render(<MondayKbSection token="t" open refreshKey={0} />);
    expect(await screen.findByText("Нет токена monday")).toBeInTheDocument();
  });
});
