import { useEffect, useRef, useState } from "react";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";
import type { PdfFile, TextBlock } from "@/lib/types";
import { File, FileText, Plus, RefreshCw, Trash2, Upload, X } from "lucide-react";
import { toast } from "sonner";

const fmtKb = (b: number) => b < 1024 * 1024 ? `${(b / 1024).toFixed(0)} КБ` : `${(b / 1024 / 1024).toFixed(1)} МБ`;

export function KnowledgeModal({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const { token } = useAuth();
  const [pdfs, setPdfs] = useState<PdfFile[] | null>(null);
  const [blocks, setBlocks] = useState<TextBlock[] | null>(null);
  const [selected, setSelected] = useState<PdfFile | null>(null);
  const [ragText, setRagText] = useState("");
  const [ragSource, setRagSource] = useState<"override" | "extracted" | "">("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [editingBlock, setEditingBlock] = useState<TextBlock | null>(null);
  const [newBlockOpen, setNewBlockOpen] = useState(false);
  const [blockName, setBlockName] = useState("");
  const [blockContent, setBlockContent] = useState("");
  const [reindexing, setReindexing] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const reload = async () => {
    if (!token) return;
    const [p, b] = await Promise.all([api.knowledge.listFiles(token), api.knowledge.listBlocks(token)]);
    setPdfs(p); setBlocks(b);
    if (p.length && !selected) {
      setSelected(p[0]);
    }
    if (!p.length) {
      setSelected(null);
      setRagText("");
      setRagSource("");
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
    }
  };

  useEffect(() => { if (open) reload(); }, [open, token]);

  useEffect(() => {
    if (!token || !selected?.path) return;
    let active = true;
    api.knowledge.getFileText(token, selected.path)
      .then((data) => {
        if (!active) return;
        setRagText(data.text || "");
        setRagSource(data.source || "");
      })
      .catch((e) => toast.error(e instanceof Error ? e.message : "Ошибка загрузки текста"));
    api.knowledge.getPreviewBlob(token, selected.path)
      .then((blob) => {
        if (!active) return;
        const nextUrl = URL.createObjectURL(blob);
        setPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return nextUrl;
        });
      })
      .catch(() => {
        setPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return null;
        });
      });
    return () => { active = false; };
  }, [token, selected?.path]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onSelectPdf = (p: PdfFile) => { setSelected(p); };

  const onUpload = async (files: FileList | null) => {
    if (!token) return;
    if (!files?.length) return;
    for (const f of Array.from(files)) {
      if (!f.name.toLowerCase().endsWith(".pdf")) { toast.error(`${f.name}: только PDF`); continue; }
      await api.knowledge.uploadFile(token, f);
      toast.success(`Загружено: ${f.name}`);
    }
    reload();
  };

  const onDeletePdf = async (p: PdfFile) => {
    if (!token) return;
    if (!confirm(`Удалить «${p.name}»?`)) return;
    await api.knowledge.deleteFile(token, p.path);
    toast.success("Файл удалён");
    if (selected?.path === p.path) { setSelected(null); setRagText(""); }
    reload();
  };

  const onSaveRag = async () => {
    if (!token) return;
    if (!selected) return;
    const saved = await api.knowledge.updateRagText(token, selected.path, ragText);
    setRagSource(saved.source);
    toast.success("Текст RAG сохранён");
    reload();
  };

  const openNewBlock = () => { setEditingBlock(null); setBlockName(""); setBlockContent(""); setNewBlockOpen(true); };
  const openEditBlock = (b: TextBlock) => { setEditingBlock(b); setBlockName(b.name); setBlockContent(b.content); setNewBlockOpen(true); };
  const saveBlock = async () => {
    if (!token) return;
    if (!blockName.trim()) { toast.error("Введите название"); return; }
    await api.knowledge.upsertBlock(token, { id: editingBlock?.id, name: blockName.trim(), content: blockContent });
    toast.success(editingBlock ? "Блок обновлён" : "Блок создан");
    setNewBlockOpen(false); reload();
  };
  const deleteBlock = async (b: TextBlock) => {
    if (!token) return;
    if (!confirm(`Удалить блок «${b.name}»?`)) return;
    await api.knowledge.deleteBlock(token, b.id); toast.success("Блок удалён"); reload();
  };

  const reindex = async () => {
    if (!token) return;
    setReindexing(true);
    try { await api.knowledge.reindex(token); toast.success("Индекс обновлён"); }
    finally { setReindexing(false); }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-6xl w-[95vw] h-[90vh] p-0 gap-0 flex flex-col">
          <DialogHeader className="px-6 py-4 border-b border-border">
            <DialogTitle className="font-display text-xl">Документы в базе знаний</DialogTitle>
          </DialogHeader>

          <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_1.2fr] overflow-hidden">
            {/* LEFT */}
            <div className="border-r border-border flex flex-col overflow-hidden">
              <div className="p-4 border-b border-border">
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); onUpload(e.dataTransfer.files); }}
                  className={`rounded-xl border-2 border-dashed p-5 text-center cursor-pointer transition-colors ${dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/50"}`}
                  onClick={() => fileRef.current?.click()}
                >
                  <Upload className="h-6 w-6 mx-auto text-muted-foreground mb-2" />
                  <div className="text-sm font-medium text-foreground">Перетащите PDF сюда</div>
                  <div className="text-xs text-muted-foreground">или нажмите, чтобы выбрать файл</div>
                  <input ref={fileRef} type="file" accept="application/pdf" multiple className="hidden" onChange={(e) => onUpload(e.target.files)} />
                </div>
              </div>

              <div className="flex-1 overflow-auto px-2 pb-4">
                <div className="px-2 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">PDF документы</div>
                {pdfs === null ? (
                  <div className="space-y-2 px-2">{[...Array(3)].map((_, i) => <Skeleton key={i} className="h-14 w-full" />)}</div>
                ) : pdfs.length === 0 ? (
                  <div className="px-4 py-6 text-sm text-muted-foreground">Пока нет документов.</div>
                ) : (
                  <ul className="space-y-1">
                    {pdfs.map((p) => (
                      <li key={p.path}>
                        <button
                          onClick={() => onSelectPdf(p)}
                          className={`w-full text-left rounded-lg px-3 py-2 flex items-center gap-3 group transition-colors ${selected?.path === p.path ? "bg-accent" : "hover:bg-muted/60"}`}
                        >
                          <File className="h-4 w-4 text-primary shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm text-foreground truncate">{p.name}</div>
                            <div className="text-xs text-muted-foreground">{fmtKb(p.sizeBytes)}</div>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); onDeletePdf(p); }}
                            aria-label="Удалить"
                            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive p-1"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}

                <div className="mt-6 px-2 py-2 flex items-center justify-between">
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Текстовые блоки</div>
                  <Button size="sm" variant="ghost" onClick={openNewBlock}><Plus className="h-3.5 w-3.5" /> Новый</Button>
                </div>
                {blocks === null ? (
                  <div className="space-y-2 px-2">{[...Array(2)].map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
                ) : (
                  <ul className="space-y-1">
                    {blocks.map((b) => (
                      <li key={b.id} className="rounded-lg px-3 py-2 hover:bg-muted/60 group">
                        <div className="flex items-start gap-2">
                          <FileText className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-foreground truncate">{b.name}</div>
                            <div className="text-xs text-muted-foreground line-clamp-2">{b.content}</div>
                          </div>
                          <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1">
                            <button onClick={() => openEditBlock(b)} className="text-xs text-primary hover:underline">Изм.</button>
                            <button onClick={() => deleteBlock(b)} aria-label="Удалить" className="text-muted-foreground hover:text-destructive p-1"><Trash2 className="h-3.5 w-3.5" /></button>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* RIGHT */}
            <div className="flex flex-col overflow-hidden">
              <div className="flex-1 overflow-auto p-6 space-y-4">
                <div className="rounded-xl border border-border bg-muted/40 aspect-[4/3] flex items-center justify-center text-muted-foreground">
                  {selected && previewUrl ? (
                    <iframe
                      src={previewUrl}
                      title={`Предпросмотр ${selected.name}`}
                      className="h-full w-full rounded-xl border-0"
                    />
                  ) : selected ? (
                    <div className="text-center">
                      <File className="h-10 w-10 mx-auto mb-2 text-primary/60" />
                      <div className="text-sm font-medium text-foreground">{selected.name}</div>
                      <div className="text-xs">Предпросмотр недоступен</div>
                    </div>
                  ) : (
                    <div className="text-sm">Выберите документ слева</div>
                  )}
                </div>

                {selected && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="rag">Текст для поиска (RAG)</Label>
                      <span className={`text-xs ${ragSource === "override" ? "text-primary" : "text-muted-foreground"}`}>
                        {ragSource === "override" ? "сохранённое переопределение" : "извлечено автоматически"}
                      </span>
                    </div>
                    <Textarea id="rag" value={ragText} onChange={(e) => setRagText(e.target.value)} className="min-h-[160px] font-mono text-xs" />
                    <div className="flex justify-end">
                      <Button size="sm" onClick={onSaveRag} className="btn-gradient">Сохранить</Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <DialogFooter className="px-6 py-3 border-t border-border bg-muted/30 sm:justify-between">
            <Button variant="outline" onClick={reindex} disabled={reindexing}>
              <RefreshCw className={`h-4 w-4 ${reindexing ? "animate-spin" : ""}`} /> Обновить индекс
            </Button>
            <Button variant="ghost" onClick={() => onOpenChange(false)}><X className="h-4 w-4" /> Закрыть</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Block create/edit */}
      <Dialog open={newBlockOpen} onOpenChange={setNewBlockOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{editingBlock ? "Изменить блок" : "Новый текстовый блок"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="bname">Название</Label>
              <Input id="bname" value={blockName} onChange={(e) => setBlockName(e.target.value)} placeholder="Например: График работы" />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="bcontent">Содержимое</Label>
              <Textarea id="bcontent" value={blockContent} onChange={(e) => setBlockContent(e.target.value)} className="min-h-[140px]" />
            </div>
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setNewBlockOpen(false)}>Отмена</Button>
            <Button onClick={saveBlock} className="btn-gradient">Сохранить</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
