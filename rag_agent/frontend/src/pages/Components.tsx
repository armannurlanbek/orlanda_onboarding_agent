import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ArrowLeft, Sparkles } from "lucide-react";
import { Logo } from "@/components/Logo";

export default function ComponentsPage() {
  return (
    <div className="min-h-screen bg-gradient-surface">
      <header className="border-b border-border bg-card/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <Logo />
          <Link to="/chat" className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" /> К приложению
          </Link>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-10 space-y-12">
        <div>
          <h1 className="font-display text-4xl font-semibold text-foreground">Дизайн-система</h1>
          <p className="text-muted-foreground mt-1">Токены и компоненты Orlanda Knowledge AI · для инженерной передачи</p>
        </div>

        <Section title="Цветовые токены">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { n: "background", c: "bg-background border" },
              { n: "card", c: "bg-card border" },
              { n: "primary", c: "bg-primary" },
              { n: "primary-glow", c: "bg-primary-glow" },
              { n: "secondary", c: "bg-secondary" },
              { n: "accent", c: "bg-accent" },
              { n: "muted", c: "bg-muted" },
              { n: "border", c: "bg-border" },
              { n: "destructive", c: "bg-destructive" },
              { n: "success", c: "bg-success" },
              { n: "warning", c: "bg-warning" },
              { n: "gradient-primary", c: "bg-gradient-primary" },
            ].map((t) => (
              <div key={t.n} className="rounded-xl overflow-hidden border border-border">
                <div className={`h-16 ${t.c}`} />
                <div className="px-3 py-2 text-xs font-mono text-foreground bg-card">{t.n}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Типографика">
          <div className="space-y-2">
            <div className="font-display text-4xl font-semibold">Display 4xl · Source Serif</div>
            <div className="font-display text-2xl font-semibold">Display 2xl</div>
            <div className="text-base text-foreground">Body Inter — основной текст интерфейса</div>
            <div className="text-sm text-muted-foreground">Caption · вспомогательный muted-foreground</div>
            <code className="text-xs font-mono bg-muted rounded px-2 py-1">monospace · код и техн. метки</code>
          </div>
        </Section>

        <Section title="Кнопки">
          <div className="flex flex-wrap gap-3">
            <Button className="btn-gradient">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">Destructive</Button>
            <Button disabled>Disabled</Button>
          </div>
        </Section>

        <Section title="Поля и чипы">
          <div className="grid sm:grid-cols-2 gap-4 max-w-xl">
            <Input placeholder="Обычное поле" />
            <Input placeholder="Disabled" disabled />
          </div>
          <div className="flex flex-wrap gap-2 mt-4">
            <Badge>Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="outline" className="border-success/30 text-success bg-success/5">Connected</Badge>
            <Badge variant="outline" className="border-warning/30 text-warning bg-warning/5">Review</Badge>
            <Badge variant="outline" className="border-destructive/30 text-destructive bg-destructive/5">Flagged</Badge>
          </div>
        </Section>

        <Section title="Скелетоны и пустые состояния">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Skeleton className="h-5 w-48" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
            </div>
            <div className="rounded-xl border border-dashed border-border p-6 text-center">
              <Sparkles className="h-6 w-6 mx-auto text-primary mb-2" />
              <div className="font-medium">Пусто</div>
              <div className="text-sm text-muted-foreground">Создайте первый элемент</div>
            </div>
          </div>
        </Section>

        <Section title="Табы">
          <Tabs defaultValue="a" className="max-w-md">
            <TabsList className="grid grid-cols-2"><TabsTrigger value="a">Вход</TabsTrigger><TabsTrigger value="b">Регистрация</TabsTrigger></TabsList>
            <TabsContent value="a" className="text-sm text-muted-foreground pt-3">Содержимое таба «Вход»</TabsContent>
            <TabsContent value="b" className="text-sm text-muted-foreground pt-3">Содержимое таба «Регистрация»</TabsContent>
          </Tabs>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="font-display text-2xl font-semibold text-foreground">{title}</h2>
      <div className="surface-card rounded-xl p-6">{children}</div>
    </section>
  );
}
