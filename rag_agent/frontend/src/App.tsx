import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/lib/auth";
import { I18nProvider } from "@/lib/i18n";
import { RequireAuth } from "@/components/RequireAuth";
import AuthPage from "./pages/Auth";
import ChatPage from "./pages/Chat";
import AdminPage from "./pages/Admin";
import AdminLogsPage from "./pages/AdminLogs";
import AdminDocumentsPage from "./pages/AdminDocuments";
import AdminMemoryPage from "./pages/AdminMemory";
import AdminInvitesPage from "./pages/AdminInvites";
import SettingsPage from "./pages/Settings";
import ComponentsPage from "./pages/Components";
import RegisterPage from "./pages/Register";
import ClientTasksPage from "./pages/client/ClientTasks";
import ClientAssistantPage from "./pages/client/ClientAssistant";
import ClientProgressPage from "./pages/client/ClientProgress";
import ClientFeedbackPage from "./pages/client/ClientFeedback";
import NotFound from "./pages/NotFound.tsx";

const queryClient = new QueryClient();

// Client cabinet routes: auth-gated to clients (admins allowed for testing)
// and wrapped in the EN/HE i18n provider that also flips the layout to RTL.
const ClientRoute = ({ children }: { children: React.ReactNode }) => (
  <RequireAuth clientOnly>
    <I18nProvider>{children}</I18nProvider>
  </RequireAuth>
);

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
            <Route path="/register" element={<RegisterPage />} />
            <Route path="/chat" element={<RequireAuth><ChatPage /></RequireAuth>} />
            <Route path="/settings" element={<RequireAuth><SettingsPage /></RequireAuth>} />
            <Route path="/admin" element={<RequireAuth adminOnly><AdminPage /></RequireAuth>} />
            <Route path="/admin/logs" element={<RequireAuth adminOnly><AdminLogsPage /></RequireAuth>} />
            <Route path="/admin/documents" element={<RequireAuth adminOnly><AdminDocumentsPage /></RequireAuth>} />
            <Route path="/admin/memory" element={<RequireAuth adminOnly><AdminMemoryPage /></RequireAuth>} />
            <Route path="/admin/invites" element={<RequireAuth adminOnly><AdminInvitesPage /></RequireAuth>} />
            <Route path="/client" element={<Navigate to="/client/tasks" replace />} />
            <Route path="/client/tasks" element={<ClientRoute><ClientTasksPage /></ClientRoute>} />
            <Route path="/client/assistant" element={<ClientRoute><ClientAssistantPage /></ClientRoute>} />
            <Route path="/client/progress" element={<ClientRoute><ClientProgressPage /></ClientRoute>} />
            <Route path="/client/feedback" element={<ClientRoute><ClientFeedbackPage /></ClientRoute>} />
            <Route path="/components" element={<ComponentsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
