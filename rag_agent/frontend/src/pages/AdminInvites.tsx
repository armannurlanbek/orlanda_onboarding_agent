/**
 * Admin page: client invite links + access management for existing client
 * accounts. Project pickers filter as you type (the catalog is 300+ projects,
 * live-synced from Monday by orlanda-api). Admin UI stays Russian.
 */
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppShell } from "@/components/AppShell";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";
import type { CustomerDirectoryEntry, OrlandaProject } from "@/lib/types";
import { Copy, KeyRound, Pencil, Trash2, X } from "lucide-react";

function CustomerSelect({
  customers,
  value,
  onChange,
}: {
  customers: CustomerDirectoryEntry[];
  value: string;
  onChange: (customer: string) => void;
}) {
  return (
    <select
      className="h-9 rounded-md border border-border bg-card px-3 text-sm w-full"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">— все проекты —</option>
      {customers.map((c) => (
        <option key={c.customer} value={c.customer}>
          {c.customer} ({c.projects.length})
        </option>
      ))}
    </select>
  );
}

function restrictProjects(
  projects: OrlandaProject[],
  customers: CustomerDirectoryEntry[],
  customer: string,
): OrlandaProject[] {
  if (!customer) return projects;
  const allowed = new Set(customers.find((c) => c.customer === customer)?.projects.map((p) => p.id) ?? []);
  return projects.filter((p) => allowed.has(p.id));
}

function ProjectPicker({
  projects,
  selected,
  onToggle,
}: {
  projects: OrlandaProject[];
  selected: Set<number>;
  onToggle: (id: number) => void;
}) {
  const [search, setSearch] = useState("");
  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return projects;
    return projects.filter((p) => p.name.toLowerCase().includes(q));
  }, [projects, search]);

  return (
    <div className="space-y-2">
      <Input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Поиск проекта по названию…"
      />
      {selected.size > 0 && (
        <div className="text-xs text-muted-foreground">
          Выбрано: {selected.size}{" "}
          <button className="text-primary hover:underline" onClick={() => [...selected].forEach(onToggle)}>
            сбросить
          </button>
        </div>
      )}
      <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3 max-h-64 overflow-y-auto rounded-md border border-border p-3">
        {visible.map((p) => (
          <label key={p.id} className="flex items-center gap-2 text-sm cursor-pointer">
            <Checkbox checked={selected.has(p.id)} onCheckedChange={() => onToggle(p.id)} />
            <span className="truncate" title={p.name}>{p.name}</span>
          </label>
        ))}
        {!visible.length && <p className="text-sm text-muted-foreground col-span-full">Ничего не найдено</p>}
      </div>
    </div>
  );
}

function ClientAccessEditor({
  username,
  projects,
  customers,
  onClose,
}: {
  username: string;
  projects: OrlandaProject[];
  customers: CustomerDirectoryEntry[];
  onClose: () => void;
}) {
  const { token } = useAuth();
  const { toast } = useToast();
  const [selected, setSelected] = useState<Set<number> | null>(null);
  const [customer, setCustomer] = useState("");

  const accessQ = useQuery({
    queryKey: ["client-access", username],
    queryFn: () => api.adminInvites.clientAccess(token!, username),
    enabled: Boolean(token),
  });

  const current = selected ?? new Set((accessQ.data ?? []).map((p) => p.id));

  const toggle = (id: number) =>
    setSelected(() => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const saveM = useMutation({
    mutationFn: () => api.adminInvites.setClientAccess(token!, username, [...current]),
    onSuccess: () => {
      toast({ title: "Доступы обновлены", description: username });
      onClose();
    },
    onError: (e) =>
      toast({ title: "Ошибка", description: e instanceof Error ? e.message : "Не удалось сохранить", variant: "destructive" }),
  });

  return (
    <div className="rounded-md border border-border p-3 mt-2 space-y-3 bg-muted/30">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Доступы: {username}</span>
        <Button variant="ghost" size="sm" onClick={onClose}><X className="h-4 w-4" /></Button>
      </div>
      {accessQ.isLoading ? (
        <Skeleton className="h-24 w-full" />
      ) : (
        <>
          <CustomerSelect customers={customers} value={customer} onChange={setCustomer} />
          <ProjectPicker
            projects={restrictProjects(projects, customers, customer)}
            selected={current}
            onToggle={toggle}
          />
        </>
      )}
      <Button size="sm" onClick={() => saveM.mutate()} disabled={saveM.isPending || accessQ.isLoading}>
        {saveM.isPending ? "Сохранение…" : "Сохранить доступы"}
      </Button>
    </div>
  );
}

export default function AdminInvitesPage() {
  const { token } = useAuth();
  const { toast } = useToast();
  const qc = useQueryClient();

  const [company, setCompany] = useState("");
  const [maxUses, setMaxUses] = useState(1);
  const [expiresDays, setExpiresDays] = useState(14);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [editingClient, setEditingClient] = useState<string | null>(null);
  const [inviteCustomer, setInviteCustomer] = useState("");

  const projectsQ = useQuery({
    queryKey: ["orlanda-projects"],
    queryFn: () => api.adminInvites.orlandaProjects(token!),
    enabled: Boolean(token),
    staleTime: 60_000,
  });
  const customersQ = useQuery({
    queryKey: ["orlanda-customers"],
    queryFn: () => api.adminInvites.customers(token!),
    enabled: Boolean(token),
    staleTime: 60_000,
  });
  const invitesQ = useQuery({
    queryKey: ["client-invites"],
    queryFn: () => api.adminInvites.list(token!),
    enabled: Boolean(token),
  });
  const clientsQ = useQuery({
    queryKey: ["client-accounts"],
    queryFn: () => api.adminInvites.listClients(token!),
    enabled: Boolean(token),
  });

  const createM = useMutation({
    mutationFn: () => {
      const projects = (projectsQ.data ?? []).filter((p) => selected.has(p.id));
      return api.adminInvites.create(token!, {
        project_ids: projects.map((p) => p.id),
        project_names: projects.map((p) => p.name),
        company_name: company.trim(),
        max_uses: maxUses,
        expires_days: expiresDays,
      });
    },
    onSuccess: (invite) => {
      qc.invalidateQueries({ queryKey: ["client-invites"] });
      setSelected(new Set());
      setCompany("");
      if (invite.url) {
        navigator.clipboard?.writeText(invite.url).catch(() => undefined);
        toast({ title: "Приглашение создано", description: "Ссылка скопирована в буфер обмена" });
      }
    },
    onError: (e) => toast({ title: "Ошибка", description: e instanceof Error ? e.message : "Не удалось создать приглашение", variant: "destructive" }),
  });

  const deleteM = useMutation({
    mutationFn: (inviteToken: string) => api.adminInvites.remove(token!, inviteToken),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["client-invites"] }),
  });

  const resetPwdM = useMutation({
    mutationFn: (username: string) => api.adminInvites.resetClientPassword(token!, username),
    onSuccess: (newPassword, username) => {
      navigator.clipboard?.writeText(newPassword).catch(() => undefined);
      toast({
        title: `Новый пароль для ${username}`,
        description: `${newPassword} — скопирован в буфер, передайте клиенту`,
        duration: 30000,
      });
    },
    onError: (e) =>
      toast({ title: "Ошибка", description: e instanceof Error ? e.message : "Не удалось сбросить пароль", variant: "destructive" }),
  });

  const toggle = (id: number) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const copyUrl = (url?: string) => {
    if (!url) return;
    navigator.clipboard?.writeText(url).catch(() => undefined);
    toast({ title: "Ссылка скопирована" });
  };

  return (
    <AppShell onOpenKnowledge={() => undefined} onOpenSettings={() => undefined}>
      <div className="p-4 sm:p-6 space-y-6 max-w-5xl w-full mx-auto overflow-y-auto">
        <h1 className="text-xl font-semibold">Приглашения клиентов</h1>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Новое приглашение</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1.5 sm:col-span-1">
                <Label htmlFor="company">Компания клиента</Label>
                <Input id="company" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Например: RAV BARIACH" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="uses">Макс. регистраций</Label>
                <Input id="uses" type="number" min={1} max={100} value={maxUses} onChange={(e) => setMaxUses(Number(e.target.value) || 1)} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="days">Срок действия (дней)</Label>
                <Input id="days" type="number" min={1} max={365} value={expiresDays} onChange={(e) => setExpiresDays(Number(e.target.value) || 14)} />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Клиент (по задачам Monday) — сузит список проектов</Label>
              <CustomerSelect
                customers={customersQ.data ?? []}
                value={inviteCustomer}
                onChange={(customer) => {
                  setInviteCustomer(customer);
                  setSelected(new Set()); // старый выбор мог быть вне нового списка
                  if (customer && (!company.trim() || company === inviteCustomer)) setCompany(customer);
                }}
              />
            </div>

            <div className="space-y-2">
              <Label>Проекты (доступ клиента)</Label>
              {projectsQ.isLoading && <Skeleton className="h-24 w-full" />}
              {projectsQ.isError && <p className="text-sm text-destructive">Не удалось загрузить проекты из OrlandaBot</p>}
              {projectsQ.data && (
                <ProjectPicker
                  projects={restrictProjects(projectsQ.data, customersQ.data ?? [], inviteCustomer)}
                  selected={selected}
                  onToggle={toggle}
                />
              )}
            </div>

            <Button onClick={() => createM.mutate()} disabled={!selected.size || createM.isPending}>
              {createM.isPending ? "Создание…" : "Создать ссылку-приглашение"}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Клиентские аккаунты</CardTitle>
          </CardHeader>
          <CardContent>
            {clientsQ.isLoading && <Skeleton className="h-16 w-full" />}
            {!clientsQ.isLoading && !(clientsQ.data ?? []).length && (
              <p className="text-sm text-muted-foreground">Зарегистрированных клиентов пока нет.</p>
            )}
            <div className="space-y-2">
              {(clientsQ.data ?? []).map((c) => (
                <div key={c.username}>
                  <div className="flex flex-wrap items-center gap-3 rounded-md border border-border p-3 text-sm">
                    <div className="flex-1 min-w-48">
                      <div className="font-medium">{c.username}</div>
                      <div className="text-xs text-muted-foreground">
                        {c.created_at && `зарегистрирован ${new Date(c.created_at).toLocaleDateString("ru-RU")}`}
                        {!c.is_active && " · отключён"}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1.5"
                      onClick={() => setEditingClient(editingClient === c.username ? null : c.username)}
                    >
                      <Pencil className="h-3.5 w-3.5" /> Доступы
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1.5"
                      disabled={resetPwdM.isPending}
                      onClick={() => {
                        if (window.confirm(`Сбросить пароль для ${c.username}? Текущие сессии клиента будут разлогинены.`)) {
                          resetPwdM.mutate(c.username);
                        }
                      }}
                    >
                      <KeyRound className="h-3.5 w-3.5" /> Сбросить пароль
                    </Button>
                  </div>
                  {editingClient === c.username && projectsQ.data && (
                    <ClientAccessEditor
                      username={c.username}
                      projects={projectsQ.data}
                      customers={customersQ.data ?? []}
                      onClose={() => setEditingClient(null)}
                    />
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Активные приглашения</CardTitle>
          </CardHeader>
          <CardContent>
            {invitesQ.isLoading && <Skeleton className="h-24 w-full" />}
            {!invitesQ.isLoading && !(invitesQ.data ?? []).length && (
              <p className="text-sm text-muted-foreground">Приглашений пока нет.</p>
            )}
            <div className="space-y-2">
              {(invitesQ.data ?? []).map((inv) => (
                <div key={inv.token} className="flex flex-wrap items-center gap-3 rounded-md border border-border p-3 text-sm">
                  <div className="flex-1 min-w-48">
                    <div className="font-medium">{inv.company_name || "Без названия"}</div>
                    <div className="text-xs text-muted-foreground">
                      {inv.project_names.join(", ") || `${inv.project_ids.length} проект(ов)`}
                      {" · "}использовано {inv.used_count}/{inv.max_uses}
                      {inv.expires_at && ` · до ${new Date(inv.expires_at).toLocaleDateString("ru-RU")}`}
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="gap-1.5" onClick={() => copyUrl(inv.url)}>
                    <Copy className="h-3.5 w-3.5" /> Копировать ссылку
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive"
                    onClick={() => deleteM.mutate(inv.token)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
