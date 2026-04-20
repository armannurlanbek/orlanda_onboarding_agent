# RAG Agent

Агент на LangChain с RAG по PDF из папки `knowledge_base`. Отвечает на русском, только по контексту из документов.

## Запуск

### Чат в терминале

```bash
python -m rag_agent.chat
```

### Индексация PDF

Положите PDF в папку **knowledge_base** в корне проекта, затем:

```bash
python -m rag_agent.indexing
```

Перезапускайте индексацию после добавления или изменения PDF.

### Локальный веб-интерфейс (FastAPI)

Запустите сервер и откройте в браузере **http://127.0.0.1:8000**. Сначала войдите или зарегистрируйтесь — история чата привязана к вашему аккаунту (thread_id = логин). Когда вернётесь и снова войдёте, увидите прежний диалог (если задан CHECKPOINT_DB).

```bash
# История в памяти (после перезапуска сервера — новая история)
python -m rag_agent.api

# История сохраняется между перезапусками — при следующем входе в аккаунт чат восстановится
set CHECKPOINT_DB=./data/checkpoints.db
mkdir data
python -m rag_agent.api
```

В браузере: **Вход** / **Регистрация** (логин и пароль), затем чат, кнопка «Выйти».

- **GET /** — страница с формой входа и чатом.
- **POST /auth/register**, **POST /auth/login** — регистрация и вход (тело: `{"username", "password"}`; ответ: `{"token", "username"}`).
- **GET /auth/me** — текущий пользователь (заголовок `Authorization: Bearer <token>`).
- **POST /chat** — отправка сообщения (заголовок `Authorization: Bearer <token>`; тело: `{"message": "…"}`). thread_id берётся из токена (ваш логин).
- **GET /health** — проверка работы.

Подробности и чеклист для продакшена: [PRODUCTION.md](PRODUCTION.md).

## Переменные окружения

- **OPENAI_API_KEY** — обязателен для RAG-эмбеддингов (индексация и поиск).
- **CHECKPOINT_DB** — путь к файлу SQLite для истории (например `./data/checkpoints.db`); если не задан, история только в памяти.
- **RAG_AGENT_MODEL**, **RAG_AGENT_MAX_TOKENS**, **RAG_AGENT_API_PORT** — опционально (см. `config.py`).
  - Рекомендуемое значение: `RAG_AGENT_MODEL=openai:gpt-4o-mini`
- Для контроля 429 можно уменьшить контекст:
  - `RAG_RETRIEVE_TOP_K` (по умолчанию `4`)
  - `RAG_MAX_CHARS_PER_CHUNK` (по умолчанию `1200`)
  - `RAG_MAX_TOTAL_CONTEXT_CHARS` (по умолчанию `6000`)
- Для истории чата:
  - `RAG_MAX_HISTORY_MESSAGES` — порог количества user/assistant сообщений в одном диалоге
  - `RAG_HISTORY_KEEP_LAST_MESSAGES` — сколько последних сообщений оставить после сжатия истории
  - `RAG_HISTORY_SUMMARY_MAX_TOKEN_LIMIT` — токен-бюджет суммаризации старых сообщений

### Переключение модели без перезапуска (admin)

- **GET `/admin/model`** — вернуть текущую модель.
- **PUT `/admin/model`** — поменять модель в рантайме. Только OpenAI-модели. Тело: `{"model":"openai:gpt-4o-mini"}`.
- **GET `/admin/history/threads`** — диагностический срез по историям чатов (какие диалоги близки к порогу сжатия/превышают его).
