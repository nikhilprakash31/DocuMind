# DocuMind 📚

**DocuMind** is a NotebookLM-inspired local Document Intelligence and Grounded Conversational RAG (Retrieval-Augmented Generation) assistant. It allows you to chat with single or multi-document PDF libraries with transparent source citations, verifiable evidence cards, and embedded PDF inspection—operating entirely on your local machine with private LLMs and embeddings via Ollama.

---

## Key Features

- **Local & Private Ingestion**:
  - Validates and parses PDF documents using `pypdf` and `langchain-community`.
  - SHA-256 content hashing for deterministic document IDs and instant duplicate detection.
  - Recursive chunking with metadata preservation (document ID, filename, page numbers).
  - Persistent document registry backed by SQLite (`registry.db`).

- **Vector Storage & Semantic Retrieval**:
  - Persistent ChromaDB vector store configured with cosine similarity.
  - Nomic embedding adapter (`nomic-embed-text`, 768 dimensions) enforcing task-specific asymmetric prefixes (`search_document: ` / `search_query: `) for optimal retrieval accuracy.
  - Document-level and multi-document filtering for scoped question answering.

- **Grounded Generation & Citation Verification**:
  - Local LLM generation via Ollama (`llama3.2:3b`) with structured context grounding.
  - Strict citation mapping `[Page X, filename.pdf]` cross-referenced against retrieved chunks.
  - Visual evidence cards displaying similarity scores, verified badge indicators, and exact snippet quotes.

- **Conversational Memory & Reformulation**:
  - Contextual query reformulation rewriting follow-up questions into standalone queries based on dialogue history.
  - Multi-session isolation allowing concurrent independent research threads.

- **NotebookLM-Inspired UI**:
  - Interactive Streamlit frontend featuring collapsible sidebars, multi-notebook navigation, and document upload management.
  - Synchronized PDF viewer rendering exact cited pages directly within the workspace.
  - Persistent Light and Dark themes saved atomically to local preferences.

---

## Architecture Overview

```
                               ┌──────────────────────────────────────────────┐
                               │                 Streamlit UI                 │
                               │           (src/frontend/app.py)              │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │               DocuMindService                │
                               │        (src/service/app_service.py)          │
                               └───────┬──────────────────────────────┬───────┘
                                       │                              │
                   ┌───────────────────┴──────────┐       ┌───────────┴───────────────────┐
                   ▼                              ▼       ▼                               ▼
       ┌──────────────────────┐        ┌───────────────┐ ┌───────────────────┐ ┌───────────────────┐
       │  Ingestion Pipeline  │        │ Conversational│ │   Vector Store    │ │ Document Registry │
       │ (loader, splitter,   │        │   RAG Chain   │ │ (ChromaDB + Nomic │ │     (SQLite)      │
       │     validator)       │        │ (reformulator,│ │    embeddings)    │ └───────────────────┘
       └──────────────────────┘        │  generation)  │ └───────────────────┘
                                       └───────────────┘
```

---

## Prerequisites

1. **Python 3.11+** installed.
2. **`uv` package manager**: Recommended for fast, reproducible dependency management.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama**: Download and install from [ollama.com](https://ollama.com).
   Ensure Ollama is running (`http://localhost:11434`):
   ```bash
   ollama serve
   ```
4. **Required Ollama Models**:
   ```bash
   # Generation model
   ollama pull llama3.2:3b

   # Embedding model
   ollama pull nomic-embed-text
   ```

---

## Installation & Setup

1. **Clone or enter the project directory**:
   ```bash
   cd DocuMind
   ```

2. **Sync locked dependencies**:
   ```bash
   uv sync --locked
   ```

---

## Running the Application

Launch the Streamlit web workspace:
```bash
uv run streamlit run src/frontend/app.py
```

Once running, access the application in your browser at `http://localhost:8501`.

### Using the App:
1. **Upload Documents**: Use the left sidebar to drag and drop or upload PDF files.
2. **Select Scope**: Ask questions across all documents or restrict retrieval to specific files.
3. **Inspect Citations**: Click on citation pills or evidence cards to preview the exact source page in the embedded PDF viewer.
4. **Manage Sessions**: Create multiple conversation threads or switch between light and dark themes in the sidebar.

---

## Running Verification & Tests

To execute the test suite (157 unit and integration tests):
```bash
uv run pytest -q
```

To run code quality and linting checks:
```bash
uv run ruff check .
```

---

## Repository Structure

```
DocuMind/
├── pyproject.toml              # Project dependencies and tool configurations
├── uv.lock                     # Deterministic dependency lockfile
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules for runtime data and caches
├── src/
│   ├── citations/              # Evidence mapping and citation locating
│   ├── conversation/           # Session management, history, and reformulation
│   ├── core/                   # Shared data models and custom exceptions
│   ├── frontend/               # Streamlit UI shell, styles, and theme persistence
│   ├── generation/             # RAG generation chains and prompt templates
│   ├── ingestion/              # PDF loading, page extraction, splitting, validation
│   ├── retrieval/              # Vector retriever and search abstraction
│   ├── service/                # Unified application service facade
│   ├── storage/                # SQLite registry, ChromaDB vector store, embeddings
│   └── utils/                  # Cryptographic hashing and deterministic ID utils
└── tests/
    ├── conftest.py             # Shared pytest fixtures
    └── unit/                   # Comprehensive unit and integration test suites
```
