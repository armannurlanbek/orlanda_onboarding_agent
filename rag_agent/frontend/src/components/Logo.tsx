export function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="h-8 w-8 rounded-xl bg-gradient-primary shadow-glow flex items-center justify-center">
        <span className="font-display font-bold text-primary-foreground text-base leading-none">O</span>
      </div>
      <div className="leading-tight">
        <div className="font-display font-semibold text-foreground text-base">Orlanda</div>
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Knowledge AI</div>
      </div>
    </div>
  );
}
