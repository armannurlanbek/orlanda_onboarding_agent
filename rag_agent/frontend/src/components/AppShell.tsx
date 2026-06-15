import { Link, useNavigate } from "react-router-dom";
import { Logo } from "./Logo";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { useAuth } from "@/lib/auth";
import { BookOpen, ChevronDown, LogOut, Plug, Settings, ShieldCheck } from "lucide-react";

type Props = {
  onOpenKnowledge: () => void;
  onOpenSettings: () => void;
  children: React.ReactNode;
  lockLayout?: boolean;
};

export function AppShell({ onOpenKnowledge, onOpenSettings, children, lockLayout = false }: Props) {
  const { user, logout } = useAuth();
  const nav = useNavigate();

  return (
    <div className={`min-h-screen flex flex-col bg-background ${lockLayout ? "h-screen overflow-hidden" : ""}`}>
      <header className="sticky top-0 z-40 border-b border-border bg-card/85 backdrop-blur-md">
        <div className="px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <Link to="/chat" aria-label="На главную" className="shrink-0"><Logo /></Link>

          <div className="flex items-center gap-1.5">
            <Button variant="ghost" size="sm" onClick={onOpenKnowledge} className="hidden sm:inline-flex">
              <BookOpen className="h-4 w-4" /> Документы
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="gap-2">
                  <div className="h-7 w-7 rounded-md bg-primary text-primary-foreground flex items-center justify-center text-xs font-semibold">
                    {user?.displayName?.[0] ?? "U"}
                  </div>
                  <span className="hidden sm:inline text-sm font-medium">{user?.displayName}</span>
                  <ChevronDown className="h-3.5 w-3.5 opacity-60" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>
                  <div className="font-medium">{user?.displayName}</div>
                  <div className="text-xs text-muted-foreground font-normal">@{user?.username} · {user?.role === "admin" ? "Администратор" : "Сотрудник"}</div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onOpenKnowledge}><BookOpen className="h-4 w-4" /> Документы</DropdownMenuItem>
                <DropdownMenuItem onClick={onOpenSettings}><Settings className="h-4 w-4" /> Настройки</DropdownMenuItem>
                <DropdownMenuItem onClick={() => nav("/settings")}><Plug className="h-4 w-4" /> Интеграции</DropdownMenuItem>
                {user?.role === "admin" && (
                  <>
                    <DropdownMenuItem onClick={() => nav("/admin/logs")}><ShieldCheck className="h-4 w-4" /> Журнал диалогов</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => nav("/admin/documents")}><ShieldCheck className="h-4 w-4" /> Метаданные документов</DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={async () => { await logout(); nav("/auth"); }} className="text-destructive focus:text-destructive">
                  <LogOut className="h-4 w-4" /> Выйти
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <main className={`flex-1 min-h-0 flex flex-col ${lockLayout ? "overflow-hidden" : ""}`}>{children}</main>
    </div>
  );
}
