/**
 * Admin page: client invite links. Pick OrlandaBot projects, generate a link
 * with the access baked in, copy it and send to the client. Admin UI stays
 * Russian like the rest of the admin pages.
 */
import { useState } from "react";
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
import { Copy, Trash2 } from "lucide-react";

export default function AdminInvitesPage() {
  const { token } = useAuth();
  const { toast } = useToast();
  const qc = useQueryClient();

  const [company, setCompany] = useState("");
  const [maxUses, setMaxUses] = useState(1);
  const [expiresDays, setExpiresDays] = useState(14);
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const projectsQ = useQuery({
    queryKey: ["orlanda-projects"],
    queryFn: () => api.adminInvites.orlandaProjects(token!),
    enabled: Boolean(token),
  });
  const invitesQ = useQuery({
    queryKey: ["client-invites"],
    queryFn: () => api.adminInvites.list(token!),
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
              <Label>Проекты (доступ клиента)</Label>
              {projectsQ.isLoading && <Skeleton className="h-24 w-full" />}
              {projectsQ.isError && <p className="text-sm text-destructive">Не удалось загрузить проекты из OrlandaBot</p>}
              <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3 max-h-64 overflow-y-auto rounded-md border border-border p-3">
                {(projectsQ.data ?? []).map((p) => (
                  <label key={p.id} className="flex items-center gap-2 text-sm cursor-pointer">
                    <Checkbox checked={selected.has(p.id)} onCheckedChange={() => toggle(p.id)} />
                    <span className="truncate">{p.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <Button onClick={() => createM.mutate()} disabled={!selected.size || createM.isPending}>
              {createM.isPending ? "Создание…" : "Создать ссылку-приглашение"}
            </Button>
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
