# Multi-Hop Agentic Codebase RAG Analyzer

An AI-powered developer tool designed to accelerate codebase onboarding, architecture exploration, and feature auditing. Unlike standard naive semantic search tools that lose structural context, this platform utilizes **Abstract Syntax Tree (AST) parsing** for intelligent chunking and an **Agentic Loop (Multi-Hop Routing)** to traverse complex cross-file dependencies autonomously.

---

## 🚀 Core Features & Architecture

*   **AST-Aware Structural Chunking:** Uses `Tree-sitter` to parse codebases into native syntax trees. Instead of fracturing files using arbitrary character limits, code is chunked naturally into complete logical units (functions, methods, declarations, and structs).
*   **Privacy-First Embedding Pipeline:** Generates mathematical vectors locally using `all-MiniLM-L6-v2` through LangChain. Entire source codes remain 100% on-premises during the database ingestion and retrieval loops.
*   **Multi-Hop Tool-Calling Agent:** Built with the latest LangChain unified agent framework. When encountering nested dependencies (e.g., `Function A` in `worker.go` calling `Function B` in `utils.go`), the agent pauses execution, triggers a recursive `search_codebase` tool invocation, and resolves the context graph before synthesizing the final answer.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | LangChain / Python | Multi-hop agent loop & system abstraction |
| **Package Management**| `uv` | Hyper-fast dependency and venv synchronization |
| **AST Parser** | Tree-sitter (Go / Rust) | Structural syntax code splitting |
| **Embeddings** | `all-MiniLM-L6-v2` | High-performance, offline local text-to-vector |
| **Vector Database** | ChromaDB | Local persistent embedded vector engine |
| **Inference Engine** | Gemini 2.5 Flash | Real-time agentic tool invocation & synthesis |

---

## 📦 Installation & Setup

Ensure you have Python 3.10+ and `uv` installed.

### 1. Initialize the Environment
Clone this repository to your machine, navigate to the folder, and sync dependencies:
```bash
uv sync