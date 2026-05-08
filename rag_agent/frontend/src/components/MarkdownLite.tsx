import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";

export function MarkdownLite({ text }: { text: string }) {
  return (
    <div className="markdown-body text-sm leading-relaxed text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          table({ children }) {
            return (
              <div className="my-2 overflow-x-auto rounded-md border border-border">
                <table className="min-w-full border-collapse text-xs">{children}</table>
              </div>
            );
          },
          thead({ children }) {
            return <thead className="bg-muted">{children}</thead>;
          },
          th({ children }) {
            return (
              <th className="border-b border-border px-2.5 py-1.5 text-left font-semibold text-foreground">
                {children}
              </th>
            );
          },
          td({ children }) {
            return <td className="border-b border-border px-2.5 py-1.5 align-top">{children}</td>;
          },
          tr({ children }) {
            return <tr className="even:bg-muted/30">{children}</tr>;
          },
          // react-markdown v9: `code` no longer receives an `inline` prop.
          // Block-level code has a `language-*` className; inline code does not.
          code({ className, children }) {
            const isBlock = /language-/.test(String(className ?? ""));
            if (!isBlock) {
              return (
                <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em] text-foreground">
                  {children}
                </code>
              );
            }
            return <code className={className}>{children}</code>;
          },
          pre({ children }) {
            return (
              <pre className="my-2 overflow-x-auto rounded-md bg-muted p-3 text-xs">{children}</pre>
            );
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer noopener"
                className="text-primary underline underline-offset-2 hover:text-primary/80"
              >
                {children}
              </a>
            );
          },
          ul({ children }) {
            return <ul className="my-1.5 list-disc space-y-0.5 pl-5">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="my-1.5 list-decimal space-y-0.5 pl-5">{children}</ol>;
          },
          li({ children }) {
            return <li className="leading-snug">{children}</li>;
          },
          h1({ children }) {
            return <h1 className="mb-1 mt-3 text-lg font-semibold">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="mb-1 mt-2.5 text-base font-semibold">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="mb-1 mt-2 text-sm font-semibold">{children}</h3>;
          },
          h4({ children }) {
            return <h4 className="mb-0.5 mt-2 text-sm font-semibold">{children}</h4>;
          },
          p({ children }) {
            return <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>;
          },
          strong({ children }) {
            return <strong className="font-semibold text-foreground">{children}</strong>;
          },
          em({ children }) {
            return <em className="italic">{children}</em>;
          },
          blockquote({ children }) {
            return (
              <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground">
                {children}
              </blockquote>
            );
          },
          hr() {
            return <hr className="my-3 border-border" />;
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
