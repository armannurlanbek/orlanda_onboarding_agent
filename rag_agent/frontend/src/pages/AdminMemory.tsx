import { useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { ArrowLeft, Check, Loader2, Pencil, Plus, Search, Trash2, X } from "lucide-react";
import { Logo } from "@/components/Logo";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/apiClient";
import type { MemoryCategory, UserMemory } from "@/lib/types";

const CATEGORY_LABELS: Record<string, string> = {
  fact: "Факт",
  preference: "Предпочтение",
  task_recipe: "Рецепт задачи",
};
const CATEGORIES: MemoryCategory[] = ["fact", "preference", "task_recipe"];

export default function AdminMemoryPage() {
  const { token } = useAuth();
  const [username, setUsername] = useState("");
  const [loaded, setLoaded] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [memories, setMemories] = useState<UserMemory[]>([]);
  const [newContent, setNewContent] = useState("");
  const [newCategory, setNewCategory] = useState<MemoryCategory>("fact");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const load = async (name: string) => {
    if (!token || !name.trim()) return;
    setLoading(true);
    try {
      const res = await api.adminMemory.list(token, name.trim());
      setEnabled(res.enabled);
      setMemories(res.memories);
      setLoaded(name.trim());
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось загрузить память пользователя");
      setLoaded(null);
      setMemories([]);
      setEnabled(null);
    } finally {
      setLoading(false);
    }
  };

  const add = async () => {
    if (!token || !loaded || !newContent.trim()) return;
    setBusy(true);
    try {
      await api.adminMemory.add(token, loaded, newContent.trim(), newCategory);
      setNewContent("");
      setNewCategory("fact");
      await load(loaded);
      toast.success("Запись добавлена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось добавить запись");
    } finally {
      setBusy(false);
    }
  };

  const saveEdit = async (id: string) => {
    if (!token || !loaded || !editText.trim()) return;
    setBusy(true);
    try {
      await api.adminMemory.update(token, loaded, id, editText.trim());
      setEditingId(null);
      setEditText("");
      await load(loaded);
      toast.success("Запись обновлена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось обновить запись");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    if (!token || !loaded) return;
    setBusy(true);
    try {
      await api.adminMemory.remove(token, loaded, id);
      setMemories((prev) => prev.filter((m) => m.id !== id));
      toast.success("Запись удалена");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Не удалось удалить запись");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-card/85 backdrop-blur-md">
        <div className="px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <Link to="/chat" aria-label="На главную" className="shrink-0"><Logo /></Link>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/chat"><ArrowLeft className="h-4 w-4" /> В чат</Link>
          </Button>
        </div>
      </header>

      <main className="flex-1 px-4 sm:px-6 py-8">
        <div className="mx-auto max-w-2xl space-y-6">
          <div>
            <h1 className="font-display text-2xl font-semibold tracking-tight">Память пользователей</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Просмотр и управление долгосрочной памятью любого пользователя. Все изменения
              записываются в журнал аудита.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Найти пользователя</CardTitle>
            </CardHeader>
            <CardContent>
              <form
                className="flex flex-wrap items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void load(username);
                }}
              >
                <Input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="имя пользователя или email"
                  className="max-w-xs"
                />
                <Button type="submit" size="sm" disabled={loading || !username.trim()}>
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  Загрузить
                </Button>
              </form>
            </CardContent>
          </Card>

          {loaded ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <span>@{loaded}</span>
                  <Badge variant={enabled ? "secondary" : "outline"} className="font-normal">
                    {enabled ? "память включена" : "память выключена"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Textarea
                    value={newContent}
                    onChange={(e) => setNewContent(e.target.value)}
                    placeholder="Добавить запись в память пользователя…"
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

                {memories.length === 0 ? (
                  <p className="text-sm text-muted-foreground">У пользователя нет сохранённых записей.</p>
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
              </CardContent>
            </Card>
          ) : null}
        </div>
      </main>
    </div>
  );
}
