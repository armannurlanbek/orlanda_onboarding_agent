/**
 * Client cabinet layout: left sidebar (mirrors to the right in Hebrew/RTL via
 * the dir attribute the I18nProvider stamps on <html>) with tab navigation,
 * language toggle and logout. All labels come from the i18n dictionary.
 */
import { NavLink, useNavigate } from "react-router-dom";
import { ClientLogo } from "./ClientLogo";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Bot, ListTodo, LogOut, MessageSquareHeart, Settings, TrendingUp } from "lucide-react";

const tabs = [
  { to: "/client/tasks", key: "nav.tasks", icon: ListTodo },
  { to: "/client/assistant", key: "nav.assistant", icon: Bot },
  { to: "/client/progress", key: "nav.progress", icon: TrendingUp },
  { to: "/client/feedback", key: "nav.feedback", icon: MessageSquareHeart },
  { to: "/client/settings", key: "nav.settings", icon: Settings },
];

export function ClientShell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const { lang, setLang, t } = useI18n();
  const nav = useNavigate();

  const doLogout = async () => {
    await logout();
    nav("/auth");
  };

  return (
    <div className="client-theme min-h-screen flex flex-col md:flex-row bg-background">
      {/* Mobile top bar (below md): logo + compact lang toggle + logout icon */}
      <header className="md:hidden flex items-center justify-between h-14 px-3 border-b border-border bg-card gap-2">
        <ClientLogo className="[&_img]:h-7 [&>div]:text-[8px]" />
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex rounded-md border border-border overflow-hidden text-[10px] font-semibold">
            <button
              onClick={() => setLang("en")}
              className={`px-2 py-1 ${lang === "en" ? "bg-primary text-primary-foreground" : "bg-transparent text-muted-foreground hover:bg-muted"}`}
            >
              EN
            </button>
            <button
              onClick={() => setLang("he")}
              className={`px-2 py-1 ${lang === "he" ? "bg-primary text-primary-foreground" : "bg-transparent text-muted-foreground hover:bg-muted"}`}
            >
              עב
            </button>
          </div>
          <button
            type="button"
            onClick={doLogout}
            aria-label={t("common.logout")}
            className="p-2 rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </header>

      <aside className="hidden md:flex w-56 shrink-0 border-e border-border bg-card flex-col">
        <div className="h-16 flex items-center px-4 border-b border-border">
          <ClientLogo />
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
            onClick={doLogout}
          >
            <LogOut className="h-4 w-4" /> {t("common.logout")}
          </Button>
        </div>
      </aside>

      <main className="flex-1 min-w-0 flex flex-col pb-16 md:pb-0">{children}</main>

      {/* Mobile bottom navigation (below md) */}
      <nav
        className="md:hidden fixed bottom-0 inset-x-0 z-40 border-t border-border bg-card grid grid-cols-5 pb-[env(safe-area-inset-bottom)]"
      >
        {tabs.map(({ to, key, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex flex-col items-center justify-center gap-0.5 py-1.5 mx-1 my-1 rounded-lg text-[10px] font-medium transition-colors ${
                isActive ? "text-primary bg-primary/10" : "text-muted-foreground"
              }`
            }
          >
            <Icon className="h-5 w-5 shrink-0" />
            <span className="truncate max-w-full">{t(key)}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
