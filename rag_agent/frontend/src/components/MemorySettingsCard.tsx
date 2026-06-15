import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Brain, Check, Loader2, Pencil, Plus, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/apiClient";
import type { MemoryCategory, MemorySettings, UserMemory } from "@/lib/types";

const CATEGORY_LABELS: Record<string, string> = {
  fact: "Факт",
  preference: "Предпочтение",
  task_recipe: "Рецепт задачи",
};
const CATEGORIES: MemoryCategory[] = ["fact", "preference", "task_recipe"];

export function MemorySettingsCard() {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState<MemorySettings | null>(null);
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [newContent, setNewContent] = useState("");
  const [newCategory, setNewCategory] = useState<MemoryCategory>("fact");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [s, list] = await Promise.all([api.memory.getSettings(token), api.memory.list(token)]);
      setSettings(s);
      setMemories(list);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось загрузить память");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggle = async (enabled: boolean) => {
    if (!token) return;
    setBusy(true);
    try {
      setSettings(await api.memory.setSettings(token, enabled));
      toast.success(enabled ? "Память включена" : "Память отключена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось изменить настройку");
    } finally {
      setBusy(false);
    }
  };

  const add = async () => {
    if (!token || !newContent.trim()) return;
    setBusy(true);
    try {
      await api.memory.add(token, newContent.trim(), newCategory);
      setNewContent("");
      setNewCategory("fact");
      await refresh();
      toast.success("Запись добавлена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось добавить запись");
    } finally {
      setBusy(false);
    }
  };

  const saveEdit = async (id: string) => {
    if (!token || !editText.trim()) return;
    setBusy(true);
    try {
      await api.memory.update(token, id, editText.trim());
      setEditingId(null);
      setEditText("");
      await refresh();
      toast.success("Запись обновлена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось обновить запись");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    if (!token) return;
    setBusy(true);
    try {
      await api.memory.remove(token, id);
      setMemories((prev) => prev.filter((m) => m.id !== id));
      toast.success("Запись удалена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось удалить запись");
    } finally {
      setBusy(false);
    }
  };

  const clearAll = async () => {
    if (!token || !memories.length) return;
    if (!window.confirm("Удалить все записи памяти? Это нельзя отменить.")) return;
    setBusy(true);
    try {
      const n = await api.memory.clearAll(token);
      setMemories([]);
      toast.success(`Удалено записей: ${n}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось очистить память");
    } finally {
      setBusy(false);
    }
  };

  const enabled = settings?.enabled ?? false;
  const globallyDisabled = settings ? !settings.globallyEnabled : false;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" /> Память
            </CardTitle>
            <CardDescription>
              Ассистент запоминает важные факты, предпочтения и проверенные решения, чтобы
              использовать их в будущих беседах. Вы можете просматривать и редактировать записи.
            </CardDescription>
          </div>
          {!loading && !globallyDisabled ? (
            <Switch checked={enabled} onCheckedChange={toggle} disabled={busy} aria-label="Долгосрочная память" />
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Загрузка…
          </div>
        ) : globallyDisabled ? (
          <p className="text-sm text-muted-foreground">Память отключена администратором.</p>
        ) : !enabled ? (
          <p className="text-sm text-muted-foreground">
            Память выключена. Включите её, чтобы ассистент запоминал важное между беседами.
          </p>
        ) : (
          <>
            {/* Add new memory */}
            <div className="space-y-2">
              <Textarea
                value={newContent}
                onChange={(e) => setNewContent(e.target.value)}
                placeholder="Добавить запись в память…"
                rows={2}
                maxLength={4000}
              />
              <div className="flex flex-wrap items-center gap-2">
                <select
                  value={newCategory}
                  onChange={(e) => setNewCategory(e.target.value as MemoryCategory)}
                  className="h-9 rounded-md border border-input bg-background px-2 text-sm"
                >
                  {CATEGORIES.map((c) => (
                    <option key={c} value={c}>{CATEGORY_LABELS[c]}</option>
                  ))}
                </select>
                <Button size="sm" onClick={add} disabled={busy || !newContent.trim()}>
                  <Plus className="h-4 w-4" /> Добавить
                </Button>
              </div>
            </div>

            <Separator />

            {/* Memory list */}
            {memories.length === 0 ? (
              <p className="text-sm text-muted-foreground">Пока ничего не сохранено.</p>
            ) : (
              <ul className="space-y-2">
                {memories.map((m) => (
                  <li key={m.id} className="rounded-md border border-border p-3">
                    {editingId === m.id ? (
                      <div className="space-y-2">
                        <Textarea
                          value={editText}
                          onChange={(e) => setEditText(e.target.value)}
                          rows={2}
                          maxLength={4000}
                        />
                        <div className="flex gap-2">
                          <Button size="sm" onClick={() => saveEdit(m.id)} disabled={busy || !editText.trim()}>
                            <Check className="h-4 w-4" /> Сохранить
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => setEditingId(null)} disabled={busy}>
                            <X className="h-4 w-4" /> Отмена
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1.5 min-w-0">
                          <Badge variant="outline" className="font-normal">
                            {CATEGORY_LABELS[m.category] || m.category}
                          </Badge>
                          <p className="text-sm break-words">{m.content}</p>
                        </div>
                        <div className="flex shrink-0 gap-1">
                          <Button
                            size="icon"
                            variant="ghost"
                            aria-label="Редактировать"
                            onClick={() => {
                              setEditingId(m.id);
                              setEditText(m.content);
                            }}
                            disabled={busy}
                          >
                            <Pencil className="h-4 w-4" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            aria-label="Удалить"
                            onClick={() => remove(m.id)}
                            disabled={busy}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}

            {memories.length > 0 ? (
              <>
                <Separator />
                <Button variant="outline" size="sm" onClick={clearAll} disabled={busy}>
                  <Trash2 className="h-4 w-4" /> Очистить всё
                </Button>
              </>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}
