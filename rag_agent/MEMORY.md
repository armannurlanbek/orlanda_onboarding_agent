# How memory works in this agent

In this project there are **two different “memories”**. They do different jobs.

---

## 1. Knowledge base (RAG) — “what the agent knows”

- **What it is:** The indexed PDFs in the **FAISS vector store** (built by `python -m rag_agent.indexing`).
- **Where it lives:** On disk in `rag_agent/index_store/`. Loaded once when the first search runs.
- **Role:** When the user asks a question, the agent calls the **retrieve_context** tool, which searches this index and returns relevant text chunks. The model then answers **from that context only**. So the agent “remembers” company info only through this search, not by storing facts in conversation.
- **Long-term?** Yes. The index stays until you re-run indexing. It does **not** depend on the user or the conversation.

---

## 2. Conversation memory — “what was said in this chat”

- **What it is:** The **list of messages** in one conversation (user + assistant turns). The model needs this so it can say things like “As I said above…” or “You asked about X earlier.”
- **Where it lives:** In the **checkpointer**.
  - **InMemorySaver** (default): in RAM. One “conversation” per **thread_id**. Lost when you stop the app.
  - **SqliteSaver** (when `CHECKPOINT_DB` is set): in a SQLite file (e.g. `./data/checkpoints.db`). One conversation per **thread_id**, and it **survives restarts**.
- **Role:** On each request the agent:
  1. Loads the **state for that thread_id** (all previous messages, if any).
  2. Appends your **new** message.
  3. Runs the graph (calls tools, gets model reply).
  4. Saves the updated state (including the new assistant reply) back for that thread_id.

So “long-term” conversation memory = **SqliteSaver + same thread_id over time**. “Short-term” = **InMemorySaver** (same thread_id only until the process exits).

---

## Summary

| What              | Where it lives              | Long-term? | Controlled by        |
|-------------------|-----------------------------|------------|-----------------------|
| Company knowledge | FAISS index (index_store)   | Yes        | Re-run indexing       |
| Chat history      | Checkpointer (RAM or SQLite)| SQLite: yes; RAM: no | `CHECKPOINT_DB`, `thread_id` |

- **Same thread_id** (e.g. one user, or one Slack channel) = one continuous conversation; the model sees all previous messages in that thread.
- **New thread_id** = new conversation; no previous messages (but the **same** RAG index is used for search).

No other “long-term memory” is used in this agent: no extra database of facts, no user profiles—only the RAG index and the checkpointer state per thread.
