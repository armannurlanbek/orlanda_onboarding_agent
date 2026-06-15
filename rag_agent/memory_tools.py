"""Long-term memory tools bound to a specific user.

The agent calls these mid-conversation to persist knowledge across separate chats (see
rag_agent.user_memory). Tools are **sync** on purpose: DB access is sync, so they run on BOTH
the sync /chat path and the async /chat/stream path (unlike the async-only monday tools).

Errors never propagate into the agent loop — they are returned as a short string the model can
react to (mirrors the "monday tool errors are non-fatal" decision in monday_tools.py).
"""
from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, StructuredTool

from rag_agent import user_memory
from rag_agent.config import RAG_MAX_USER_MEMORIES, memory_enabled
from rag_agent.user_memory import VALID_CATEGORIES

log = logging.getLogger(__name__)


def get_memory_tools_for_user(user_id, thread_id: str | None = None) -> list[BaseTool]:
    """Return save/update/delete_memory tools bound to ``user_id``.

    Empty list when memory is globally disabled or the user has turned memory off — so the
    agent simply has no memory tools in that case.
    """
    if not memory_enabled() or not user_memory.get_memory_enabled(user_id):
        return []

    def _save_memory(content: str, category: str = "fact") -> str:
        try:
            res = user_memory.add_memory(
                user_id, content, category, source="agent", thread_id=thread_id
            )
        except Exception as exc:  # noqa: BLE001 - surface to the model, don't kill the chat
            log.warning("save_memory failed: %s", exc)
            return "Не удалось сохранить память (внутренняя ошибка). Продолжай без сохранения."
        status = res.get("status")
        if status == "added":
            mem = res["memory"]
            return (
                f"Сохранено в долгосрочную память [mem_{mem['id']}] ({mem['category']}). "
                "Кратко сообщи пользователю, что ты это запомнил."
            )
        if status == "duplicate":
            return "Похожая запись уже есть в памяти — ничего не добавлено."
        if status == "full":
            return (
                f"Память заполнена (лимит {RAG_MAX_USER_MEMORIES}). Попроси пользователя "
                "удалить ненужные записи в Настройках."
            )
        return "Память не сохранена: пустой или некорректный текст."

    def _update_memory(memory_id: str, new_content: str) -> str:
        try:
            mem = user_memory.update_memory(user_id, memory_id, new_content)
        except Exception as exc:  # noqa: BLE001
            log.warning("update_memory failed: %s", exc)
            return "Не удалось обновить память (внутренняя ошибка)."
        if mem is None:
            return f"Память [mem_{memory_id}] не найдена."
        return f"Память [mem_{mem['id']}] обновлена. Кратко сообщи об этом пользователю."

    def _delete_memory(memory_id: str) -> str:
        try:
            ok = user_memory.delete_memory(user_id, memory_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("delete_memory failed: %s", exc)
            return "Не удалось удалить память (внутренняя ошибка)."
        if not ok:
            return f"Память [mem_{memory_id}] не найдена."
        return "Память удалена. Кратко сообщи об этом пользователю."

    cats = ", ".join(VALID_CATEGORIES)
    save_tool = StructuredTool.from_function(
        func=_save_memory,
        name="save_memory",
        description=(
            "Сохрани долгосрочный факт о пользователе, его предпочтение, или ПРОВЕРЕННЫЙ рецепт "
            "задачи (например, какие именно параметры/фильтры дали нужный результат), чтобы "
            "использовать это в будущих беседах. Сохраняй устойчивые сведения и подтверждённые "
            f"решения, а не разовые мелочи. category — одно из: {cats}. НЕ сохраняй чувствительные "
            "данные (HR, зарплата, здоровье), секреты/токены/пароли."
        ),
    )
    update_tool = StructuredTool.from_function(
        func=_update_memory,
        name="update_memory",
        description=(
            "Обнови содержимое ранее сохранённой памяти. memory_id — идентификатор из метки "
            "[mem_xxxx] в блоке памяти (например, 3f9a1b2c)."
        ),
    )
    delete_tool = StructuredTool.from_function(
        func=_delete_memory,
        name="delete_memory",
        description=(
            "Удали ранее сохранённую память, которая стала неверной или ненужной. memory_id — "
            "идентификатор из метки [mem_xxxx] в блоке памяти."
        ),
    )
    return [save_tool, update_tool, delete_tool]
