import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({ token: "test-token" }),
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const listMock = vi.fn();
const addMock = vi.fn();
const removeMock = vi.fn();
const getSettingsMock = vi.fn();
const setSettingsMock = vi.fn();

vi.mock("@/lib/apiClient", () => ({
  api: {
    memory: {
      list: (...a: unknown[]) => listMock(...a),
      add: (...a: unknown[]) => addMock(...a),
      update: vi.fn(),
      remove: (...a: unknown[]) => removeMock(...a),
      clearAll: vi.fn(),
      getSettings: (...a: unknown[]) => getSettingsMock(...a),
      setSettings: (...a: unknown[]) => setSettingsMock(...a),
    },
  },
}));

import { MemorySettingsCard } from "@/components/MemorySettingsCard";

describe("MemorySettingsCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getSettingsMock.mockResolvedValue({ enabled: true, globallyEnabled: true });
    listMock.mockResolvedValue([
      {
        id: "abc12345",
        content: "Отвечать кратко по-русски",
        category: "preference",
        source: "agent",
        createdAt: "",
        updatedAt: "",
      },
    ]);
    addMock.mockResolvedValue({
      id: "new00001",
      content: "Я работаю в отделе продаж",
      category: "fact",
      source: "user",
      createdAt: "",
      updatedAt: "",
    });
    removeMock.mockResolvedValue(undefined);
    setSettingsMock.mockResolvedValue({ enabled: false, globallyEnabled: true });
  });

  it("renders existing memories with their category", async () => {
    render(<MemorySettingsCard />);
    expect(await screen.findByText("Отвечать кратко по-русски")).toBeInTheDocument();
    // "Предпочтение" also appears as a category <option>; assert the badge (non-option) too.
    const labels = screen.getAllByText("Предпочтение");
    expect(labels.some((el) => el.tagName !== "OPTION")).toBe(true);
  });

  it("adds a memory via the API", async () => {
    render(<MemorySettingsCard />);
    await screen.findByText("Отвечать кратко по-русски");
    fireEvent.change(screen.getByPlaceholderText("Добавить запись в память…"), {
      target: { value: "Я работаю в отделе продаж" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Добавить/ }));
    await waitFor(() =>
      expect(addMock).toHaveBeenCalledWith("test-token", "Я работаю в отделе продаж", "fact"),
    );
  });

  it("deletes a memory via the API", async () => {
    render(<MemorySettingsCard />);
    await screen.findByText("Отвечать кратко по-русски");
    fireEvent.click(screen.getByRole("button", { name: "Удалить" }));
    await waitFor(() => expect(removeMock).toHaveBeenCalledWith("test-token", "abc12345"));
  });
});
