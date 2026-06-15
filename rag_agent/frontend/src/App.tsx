import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/lib/auth";
import { RequireAuth } from "@/components/RequireAuth";
import AuthPage from "./pages/Auth";
import ChatPage from "./pages/Chat";
import AdminPage from "./pages/Admin";
import AdminLogsPage from "./pages/AdminLogs";
import AdminDocumentsPage from "./pages/AdminDocuments";
import AdminMemoryPage from "./pages/AdminMemory";
import SettingsPage from "./pages/Settings";
import ComponentsPage from "./pages/Components";
import NotFound from "./pages/NotFound.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider delayDuration={200}>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/auth" element={<AuthPage />} />
            <Route path="/chat" element={<RequireAuth><ChatPage /></RequireAuth>} />
            <Route path="/settings" element={<RequireAuth><SettingsPage /></RequireAuth>} />
            <Route path="/admin" element={<RequireAuth adminOnly><AdminPage /></RequireAuth>} />
            <Route path="/admin/logs" element={<RequireAuth adminOnly><AdminLogsPage /></RequireAuth>} />
            <Route path="/admin/documents" element={<RequireAuth adminOnly><AdminDocumentsPage /></RequireAuth>} />
            <Route path="/admin/memory" element={<RequireAuth adminOnly><AdminMemoryPage /></RequireAuth>} />
            <Route path="/components" element={<ComponentsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
