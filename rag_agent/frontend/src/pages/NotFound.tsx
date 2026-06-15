import { useLocation } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-muted/40 p-4">
      <div className="surface-card rounded-lg px-10 py-12 text-center max-w-sm w-full">
        <div className="font-display text-6xl font-semibold tracking-tight text-primary">404</div>
        <p className="mt-3 mb-6 text-base text-muted-foreground">Oops! Page not found</p>
        <a href="/" className="inline-flex items-center text-sm font-medium text-primary underline underline-offset-4 hover:text-primary/80">
          Return to Home
        </a>
      </div>
    </div>
  );
};

export default NotFound;
