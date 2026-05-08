import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/apiClient";
import type { User } from "@/lib/types";

type AuthState = {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (u: string, p: string) => Promise<void>;
  register: (u: string, p: string) => Promise<void>;
  changePassword: (payload: { currentPassword?: string; newPassword: string; repeatPassword: string }) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);
const TOKEN_KEY = "orlanda.token";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (!stored) { setLoading(false); return; }
    api.auth.me(stored).then((u) => { setUser(u); setToken(stored); }).catch(() => localStorage.removeItem(TOKEN_KEY)).finally(() => setLoading(false));
  }, []);

  const persist = (t: string, u: User) => {
    localStorage.setItem(TOKEN_KEY, t);
    setToken(t); setUser(u);
  };

  return (
    <AuthContext.Provider
      value={{
        user, token, loading,
        login: async (u, p) => { const r = await api.auth.login(u, p); persist(r.token, r.user); },
        register: async (u, p) => { const r = await api.auth.register(u, p); persist(r.token, r.user); },
        changePassword: async (payload) => {
          if (!token) throw new Error("Сессия отсутствует");
          const r = await api.auth.changePassword(token, payload);
          persist(r.token, r.user);
        },
        logout: async () => { await api.auth.logout(token); localStorage.removeItem(TOKEN_KEY); setUser(null); setToken(null); },
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
