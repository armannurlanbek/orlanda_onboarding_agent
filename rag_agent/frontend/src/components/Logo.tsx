export function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div className="h-8 w-8 rounded-md bg-primary flex items-center justify-center shadow-soft">
        <span className="font-display font-semibold text-primary-foreground text-[15px] leading-none">O</span>
      </div>
      <div className="leading-tight">
        <div className="font-display font-semibold text-foreground text-[15px] tracking-tight">Orlanda</div>
        <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">Knowledge AI</div>
      </div>
    </div>
  );
}
