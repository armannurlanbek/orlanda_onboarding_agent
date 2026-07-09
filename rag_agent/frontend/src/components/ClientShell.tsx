/**
 * Client cabinet layout: left sidebar (mirrors to the right in Hebrew/RTL via
 * the dir attribute the I18nProvider stamps on <html>) with tab navigation,
 * language toggle and logout. All labels come from the i18n dictionary.
 */
import { NavLink, useNavigate } from "react-router-dom";
import { Logo } from "./Logo";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Bot, ListTodo, LogOut, MessageSquareHeart, TrendingUp } from "lucide-react";

const tabs = [
  { to: "/client/tasks", key: "nav.tasks", icon: ListTodo },
  { to: "/client/assistant", key: "nav.assistant", icon: Bot },
  { to: "/client/progress", key: "nav.progress", icon: TrendingUp },
  { to: "/client/feedback", key: "nav.feedback", icon: MessageSquareHeart },
];

export function ClientShell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const { lang, setLang, t } = useI18n();
  const nav = useNavigate();

  return (
    <div className="min-h-screen flex bg-background">
      <aside className="w-56 shrink-0 border-e border-border bg-card flex flex-col">
        <div className="h-16 flex items-center px-4 border-b border-border">
          <Logo />
        </div>

        <nav className="flex-1 p-2 space-y-1">
          {tabs.map(({ to, key, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {t(key)}
            </NavLink>
          ))}
        </nav>

        <div className="p-3 border-t border-border space-y-3">
          <div className="flex rounded-md border border-border overflow-hidden text-xs font-semibold">
            <button
              onClick={() => setLang("en")}
              className={`flex-1 py-1.5 ${lang === "en" ? "bg-primary text-primary-foreground" : "bg-transparent text-muted-foreground hover:bg-muted"}`}
            >
              EN
            </button>
            <button
              onClick={() => setLang("he")}
              className={`flex-1 py-1.5 ${lang === "he" ? "bg-primary text-primary-foreground" : "bg-transparent text-muted-foreground hover:bg-muted"}`}
            >
              עב
            </button>
          </div>

          <div className="text-xs text-muted-foreground truncate" title={user?.username}>
            {user?.username}
          </div>
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={async () => {
              await logout();
              nav("/auth");
            }}
          >
            <LogOut className="h-4 w-4" /> {t("common.logout")}
          </Button>
        </div>
      </aside>

      <main className="flex-1 min-w-0 flex flex-col">{children}</main>
    </div>
  );
}
