import logoUrl from "@/assets/orlanda-logo.png";
import { useI18n } from "@/lib/i18n";

export function ClientLogo({ className = "" }: { className?: string }) {
  const { t } = useI18n();
  return (
    <div className={`flex flex-col items-start gap-1 ${className}`}>
      <img src={logoUrl} alt="Orlanda" className="h-9 w-auto object-contain" />
      <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
        {t("brand.cabinet")}
      </div>
    </div>
  );
}
