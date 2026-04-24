# RAG Agent

Агент на LangChain с RAG по PDF из папки `knowledge_base`. Отвечает на русском и использует контекст из индекса.

## Быстрый запуск

### 1) Индексация знаний

Положите PDF в `knowledge_base` в корне проекта и выполните:

```bash
python -m rag_agent.indexing
```

После добавления/изменения документов запускайте индексацию снова.

### 2) Запуск API (веб-интерфейс)

```bash
python -m rag_agent.api
```

Откройте `http://127.0.0.1:8000`.

## Архитектура базы данных

Аутентификация и сессии работают только через PostgreSQL.

- Таблица `users`:
  - `id` (UUID, PK)
  - `username` (unique)
  - `password_hash` (Argon2; legacy SHA-256 из импорта обновляется при успешном логине)
  - `role` (`admin` / `user`)
  - `is_active`
  - `must_change_password` (обязательная смена пароля при первом входе)
  - `password_changed_at`, `temp_password_issued_at`
  - `created_at`, `updated_at`
- Таблица `auth_sessions`:
  - `id` (UUID, PK)
  - `token_hash` (SHA-256 от `SECRET_KEY:token`; сам токен в БД не хранится)
  - `user_id` (FK -> `users.id`, `ON DELETE CASCADE`)
  - `expires_at`, `created_at`

Миграции:
- `001_users`
- `002_auth_sessions`
- `003_widen_username`
- `004_user_password_lifecycle`

## Настройка PostgreSQL

### Обязательно

1. Создайте БД (например `rag_agent`).
2. В корне проекта создайте `.env` и укажите:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/rag_agent
```

3. Примените миграции:

```bash
python -m alembic upgrade head
```

Если `DATABASE_URL` не задан, API не стартует.

## Правила логинов и регистрации

Допускаются только:
- логины из `RAG_AGENT_ADMIN_USERNAMES` (короткий формат без `@`),
- или email вида `local@orlanda.info` (домен задается `RAG_ALLOWED_EMAIL_DOMAIN`).

Пароль при регистрации:
- минимум `RAG_MIN_PASSWORD_LENGTH` (по умолчанию 12),
- максимум `RAG_MAX_PASSWORD_LENGTH` (по умолчанию 128),
- минимум 1 буква и 1 цифра.

Роли:
- если username есть в `RAG_AGENT_ADMIN_USERNAMES` -> `admin`,
- иначе `user`.

## Как работать с пользователями

Есть два варианта:
- через API (рекомендуется для первичного заведения сотрудников),
- через SQL (для ручного администрирования).

### 1) Создать сотрудника с временным паролем (admin API)

`POST /admin/users/provision` (только admin)

Тело:

```json
{
  "username": "employee@orlanda.info",
  "role": "user"
}
```

Ответ:

```json
{
  "ok": true,
  "user": {
    "username": "employee@orlanda.info",
    "role": "user",
    "must_change_password": true,
    "temporary_password": "..."
  }
}
```

`temporary_password` показывается один раз. Передайте его сотруднику безопасным каналом.

Подключиться к БД в psql:

```sql
\conninfo
\c rag_agent
```

Посмотреть пользователей:

```sql
SELECT id, username, role, is_active, created_at
FROM users
ORDER BY created_at DESC;
```

Сменить роль:

```sql
UPDATE users SET role = 'admin' WHERE username = 'user@orlanda.info';
UPDATE users SET role = 'user'  WHERE username = 'user@orlanda.info';
```

Отключить пользователя (без удаления):

```sql
UPDATE users SET is_active = false WHERE username = 'user@orlanda.info';
```

Удалить пользователя:

```sql
DELETE FROM users WHERE username = 'user@orlanda.info';
```

При удалении пользователя его сессии удаляются автоматически (`ON DELETE CASCADE`).

## Миграция со старого `users.json`

Runtime-код больше не использует `users.json`, но есть one-time импорт:

```bash
python -m rag_agent.import_json_users
```

Рекомендуемый процесс decommission:
1. Импортировать пользователей в PostgreSQL.
2. Проверить логин для нужных аккаунтов.
3. Сделать бэкап БД.
4. Удалить или архивировать `rag_agent/data/users.json`.

## Сессии и logout

- При логине создается bearer token.
- Сервер хранит только `token_hash` в `auth_sessions`.
- `POST /auth/logout` инвалидирует текущий токен.
- Истекшие сессии чистятся при проверке токена.
- Если `must_change_password=true`, рабочие маршруты блокируются до смены пароля.

## Полезные эндпоинты auth

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/password/change`
- `POST /auth/logout`
- `GET /auth/me`
- `GET /admin/retrieval/debug?q=...&limit=12` — отладка retrieval-пайплайна (только admin).

`/auth/login` и `/auth/me` возвращают `must_change_password`, поэтому UI может показать принудительное окно смены пароля.

## Переменные окружения (важные)

- `DATABASE_URL` — обязательно.
- `RAG_AGENT_SECRET_KEY` — обязательно в production.
- `RAG_AGENT_ADMIN_USERNAMES` — список админ-логинов.
- `RAG_ALLOWED_EMAIL_DOMAIN` — допустимый email-домен (по умолчанию `orlanda.info`).
- `RAG_MIN_PASSWORD_LENGTH`, `RAG_MAX_PASSWORD_LENGTH`
- `RAG_SESSION_EXPIRY_DAYS`
- `OPENAI_API_KEY` — обязателен для RAG-эмбеддингов.
- `CHECKPOINT_BACKEND` — backend checkpointer (`postgres` по умолчанию, также `sqlite`/`memory`).
- `CHECKPOINT_POSTGRES_URL` — optional DSN для checkpointer в PostgreSQL (если не задан, берется `DATABASE_URL`).
- `CHECKPOINT_DB` — путь к SQLite-файлу, используется только при `CHECKPOINT_BACKEND=sqlite`.
- Retrieval quality tuning (Phase A):
  - `RAG_RETRIEVE_TOP_K` — сколько чанков попадет в итоговый контекст.
  - `RAG_RETRIEVE_FETCH_K` — размер пула кандидатов до финального отбора.
  - `RAG_ENABLE_MMR` — включить diversity-отбор (max marginal relevance).
  - `RAG_MMR_LAMBDA` — баланс релевантность/разнообразие (0.0..1.0).
  - `RAG_QUERY_REWRITE_MAX` — максимум query-variants для workflow-вопросов.
  - `RAG_NEIGHBOR_PAGE_WINDOW` — сколько соседних страниц добавлять вокруг найденной.
  - `RAG_NEIGHBOR_MAX_CHUNKS` — лимит соседних чанков в ответе.
  - `RAG_RETRIEVAL_LOG_TOP` — сколько top-кандидатов логировать в diagnostics.
  - `RAG_ENABLE_HYBRID_RETRIEVAL` — включить гибридный поиск (dense + BM25).
  - `RAG_BM25_TOP_K` — сколько BM25-кандидатов брать на каждый query-variant.
  - `RAG_RERANK_CANDIDATES_K` — размер пула для финального rerank.
  - `RAG_RRF_K` — константа reciprocal rank fusion (обычно 50-100).
  - `RAG_ENABLE_CROSS_ENCODER_RERANK` — включить cross-encoder reranker (опционально).
  - `RAG_CROSS_ENCODER_MODEL` — модель cross-encoder (по умолчанию `cross-encoder/ms-marco-MiniLM-L-6-v2`).

## Бэкапы

Код не делает автоматические бэкапы — это операционная задача.

Минимум для production:
- регулярный бэкап PostgreSQL (`pg_dump` или managed DB backups),
- возможность point-in-time recovery (если поддерживается платформой),
- безопасное хранение секретов (`DATABASE_URL`, `RAG_AGENT_SECRET_KEY`, `MONDAY_ENCRYPTION_KEY`) вне git,
- периодическая проверка восстановления из бэкапа.

Пример manual backup:

```bash
pg_dump "postgresql://USER:PASSWORD@HOST:5432/rag_agent" > rag_agent_backup.sql
```

---

Подробный production-чеклист: [PRODUCTION.md](PRODUCTION.md).

## Retrieval Evaluation (Phase C)

Для замера качества retrieval на наборе эталонных вопросов:

1) Создайте `rag_agent/data/retrieval_eval.jsonl`:

```json
{"query":"what is next step after estimation","must_include_any":["shop drawing"]}
{"query":"who approves shop drawing","must_include_all":["approve","manager"]}
```

2) Запустите оценку:

```bash
python -m rag_agent.eval_retrieval --dataset rag_agent/data/retrieval_eval.jsonl --k 8 --show-failures
```

Скрипт печатает `Hit@K`, `MRR@K` и список проблемных запросов.

## PostgreSQL + pgvector Indexing

Индекс RAG мигрирован с FAISS на PostgreSQL (`pgvector`) с инкрементальной индексацией:
- добавление/изменение PDF -> переиндексируется только этот PDF,
- добавление/изменение text item -> переиндексируется только этот item,
- удаление PDF/item -> удаляются только его чанки.

Перед миграцией убедитесь, что расширение `vector` доступно в PostgreSQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Если команда возвращает ошибку `extension "vector" is not available`, расширение не установлено на сервере БД (нужно установить на стороне хоста PostgreSQL).

После установки расширения:

```bash
python -m alembic upgrade head
python -m rag_agent.backfill_pgvector
python -m rag_agent.backfill_state_postgres
python -m rag_agent.migrate_checkpoints_to_postgres
```
