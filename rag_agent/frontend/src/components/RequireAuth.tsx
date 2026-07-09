import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/lib/auth";
import { Skeleton } from "@/components/ui/skeleton";

export function RequireAuth({
  children,
  adminOnly = false,
  clientOnly = false,
}: {
  children: React.ReactNode;
  adminOnly?: boolean;
  clientOnly?: boolean;
}) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="space-y-3 w-64">
          <Skeleton className="h-6 w-40 mx-auto" />
          <Skeleton className="h-4 w-56" />
          <Skeleton className="h-4 w-48" />
        </div>
      </div>
    );
  }
  if (!user) return <Navigate to="/auth" state={{ from: location }} replace />;
  if (adminOnly && user.role !== "admin") return <Navigate to="/chat" replace />;
  // Clients live in their own cabinet; employee pages bounce them there.
  // Admins may open the cabinet too (for testing what clients see).
  if (clientOnly && user.role !== "client" && user.role !== "admin") return <Navigate to="/chat" replace />;
  if (!clientOnly && user.role === "client") return <Navigate to="/client/tasks" replace />;
  return <>{children}</>;
}
