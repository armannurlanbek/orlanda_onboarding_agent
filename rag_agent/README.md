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
- **ANTHROPIC_API_KEY** — обязателен, если `RAG_AGENT_MODEL` начинается с `anthropic:`.
- **CHECKPOINT_DB** — путь к файлу SQLite для истории (например `./data/checkpoints.db`); если не задан, история только в памяти.
- **RAG_AGENT_MODEL**, **RAG_AGENT_MAX_TOKENS**, **RAG_AGENT_API_PORT** — опционально (см. `config.py`).
  - Пример для Sonnet: `RAG_AGENT_MODEL=anthropic:claude-sonnet-4-6`
- Для контроля 429 можно уменьшить контекст:
  - `RAG_RETRIEVE_TOP_K` (по умолчанию `4`)
  - `RAG_MAX_CHARS_PER_CHUNK` (по умолчанию `1200`)
  - `RAG_MAX_TOTAL_CONTEXT_CHARS` (по умолчанию `6000`)

### Переключение модели без перезапуска (admin)

- **GET `/admin/model`** — вернуть текущую модель.
- **PUT `/admin/model`** — поменять модель в рантайме. Тело: `{"model":"anthropic:claude-sonnet-4-6"}`.

## Monday.com Connector

Поддерживается опциональный per-user коннектор Monday:
- Пользователь может **подключить** или **не подключать** Monday.
- Подключение выполняется через OAuth в браузере.
- После подключения в чате становятся доступны Monday tools (прямой вызов Monday API).
- В ответе отображается блок **Agent activity** с событиями вызовов Monday tools.

### Что настроить

Добавьте в `.env`:

```bash
MONDAY_CLIENT_ID=...
MONDAY_CLIENT_SECRET=...
MONDAY_OAUTH_REDIRECT_URI=http://127.0.0.1:8000/integrations/monday/callback
MONDAY_OAUTH_SCOPES=me:read boards:read boards:write
MONDAY_ENCRYPTION_KEY=...
```

Важно: `MONDAY_OAUTH_REDIRECT_URI` должен совпадать с redirect URL в приложении Monday.

### Проверка локально

1. Запустите API: `python -m rag_agent.api`
2. Войдите в веб-интерфейс.
3. Нажмите **Connect Monday**.
4. Пройдите OAuth consent в Monday.
5. Вернитесь в чат и задайте Monday-запрос (например про boards/items).
6. Убедитесь, что:
   - статус стал `Monday: connected`
   - в сообщении ассистента отображается `Agent activity`
7. Нажмите **Disconnect Monday** и проверьте, что Monday tools больше не используются.
