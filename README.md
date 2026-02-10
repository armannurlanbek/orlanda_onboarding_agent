# Orlanda Engineering Onboarding AI Agent

An AI agent that chats with new employees and answers questions about **Orlanda Engineering** company business processes using a knowledge base built from company documents.

## Features

- **Knowledge vector database**: Company files (PDF, TXT, MD, DOCX) are embedded and stored in ChromaDB.
- **RAG-based answers**: The agent retrieves relevant chunks and generates answers from them.
- **Chat interface**: Simple CLI to ask questions and get onboarding help.

## Production Steps (Micro Step by Micro Step)

| Step | Description |
|------|-------------|
| 1 | Project structure, `requirements.txt`, README |
| 2 | Config and environment (paths, model settings) |
| 3 | Document loader for company files |
| 4 | Text chunking for embedding |
| 5 | Embeddings + ChromaDB vector store |
| 6 | Build/index knowledge base from company folder |
| 7 | RAG retrieval (query → relevant chunks) |
| 8 | LLM integration for answer generation |
| 9 | Chat agent conversation loop |
| 10 | CLI interface and run instructions |

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add company documents** into `company_knowledge/` (PDF, TXT, MD, DOCX).

4. **Configure** (optional): Copy `.env.example` to `.env` and set your LLM API URL and key if needed.

5. **Build the knowledge base** (first time):
   ```bash
   python -m src.build_knowledge_base
   ```

6. **Run the chat agent**:
   ```bash
   python -m src.chat
   ```

## Project Structure

```
onboarding_bot_from_scratch/
├── company_knowledge/     # Put Orlanda Engineering docs here
├── data/                  # ChromaDB and artifacts (auto-created)
├── src/
│   ├── config.py         # Settings
│   ├── load_documents.py # Load PDF, TXT, MD, DOCX
│   ├── chunking.py       # Split text for embedding
│   ├── embeddings.py     # Embeddings + vector store
│   ├── rag.py            # Retrieve + generate
│   ├── llm.py            # LLM client
│   ├── agent.py          # Chat agent
│   ├── build_knowledge_base.py  # Index company_knowledge/
│   └── chat.py           # CLI entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

## License

Internal use — Orlanda Engineering.
