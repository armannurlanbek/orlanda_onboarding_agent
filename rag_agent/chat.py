"""
Chat with the RAG agent. Index is loaded from disk (no re-indexing each run).
Shows which documents were used for each answer.
"""
from rag_agent.agent import agent, Context
from rag_agent.rag_tool import get_last_sources

CONFIG = {"configurable": {"thread_id": "chat_1"}}


def _format_sources(sources: list[dict]) -> str:
    """Deduplicate and format (file, page) for display."""
    seen = set()
    parts = []
    for s in sources:
        key = (s.get("file", "?"), s.get("page", "?"))
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{key[0]} (стр. {key[1]})")
    return ", ".join(parts) if parts else "—"


def main():
    messages = []
    print("Чат с ассистентом (база знаний RAG). Для выхода введите exit или quit.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "выход"):
            print("До свидания.")
            break

        messages.append({"role": "user", "content": user_input})

        response = agent.invoke(
            {"messages": messages},
            config=CONFIG,
            context=Context(user_id="1"),
        )

        content = response["structured_response"].response_content
        messages.append({"role": "assistant", "content": content})

        sources = get_last_sources()
        if sources:
            print(f"\n📎 Источники: {_format_sources(sources)}\n")
        else:
            print("\n📎 Источники: не использовались\n")
        print(f"Ассистент: {content}\n")


if __name__ == "__main__":
    main()
