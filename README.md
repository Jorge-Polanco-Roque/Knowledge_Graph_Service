<p align="center">
  <img src="screenshots/19_autofit_dark.png" alt="NeuroGraph — Brain Plasticity Knowledge Graph" width="100%">
</p>

<h1 align="center">NeuroGraph</h1>

<p align="center">
  <strong>A brain-inspired knowledge graph that learns, strengthens, and prunes connections — just like biological neural networks.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=white" alt="React 18">
  <img src="https://img.shields.io/badge/TypeScript-strict-3178c6?logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2-ff6f00" alt="LangGraph">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Overview

NeuroGraph is an AI-powered knowledge management system that simulates **brain plasticity**. As you converse with the agent, it dynamically builds a knowledge graph — forming new connections, strengthening frequently used pathways through **Hebbian learning**, and pruning unused ones — mirroring how biological neural networks operate.

The application features a split-view interface: a **3D interactive brain visualization** on the left and a **chat interface** on the right, showing in real-time how concepts connect and knowledge evolves.

<p align="center">
  <img src="screenshots/18_light_theme.png" alt="NeuroGraph Light Theme" width="100%">
</p>

## Key Features

- **Adaptive Knowledge Graph** — Concepts and relationships evolve through conversation, growing stronger with use and fading when neglected
- **Brain Plasticity Model** — Implements Hebbian learning, Long-Term Potentiation (LTP), Long-Term Depression (LTD), synaptic pruning, and spreading activation
- **Dual Memory System** — Short-Term Memory (STM) for active conversation context and Long-Term Memory (LTM) for persistent knowledge, with consolidation between them
- **3D Neural Visualization** — Interactive, rotatable 3D graph with cluster detection, color-coded communities, and real-time animation of graph changes
- **Node Search** — Autocomplete search bar to find and fly to any node in the graph, with cluster color and importance indicators
- **Markdown Chat** — AI responses rendered with full Markdown support (bold, italic, lists, code blocks, headings)
- **RAG (Retrieval-Augmented Generation)** — 4,844 document chunks + 60 auto-generated summaries stored in ChromaDB with OpenAI embeddings. Dual retrieval: semantic + graph-guided search. Default: chunks-only mode (top 7) for maximum accuracy. Configurable via `CONTEXT_MODE` env var. Source collision prevention via folder/filename keys
- **Graph ↔ Document Cross-Linking** — Every node tracks which documents it was extracted from (`source_files` metadata). Hover a node to see its source documents. Enables bidirectional navigation between structured knowledge and raw evidence
- **Contextual Spreading Activation** — Query-weighted neural propagation. Node embeddings cached in `.npz`; cosine similarity between query and neighbors modulates which pathways activate, focusing context on what matters for each question
- **Semantic Deduplication** — Embedding-based cosine similarity detects and merges quasi-duplicate nodes (e.g., "AI Fairness" + "Fairness in AI"), cleaning the graph and consolidating knowledge
- **Cluster Summaries** — LLM-generated 2-3 sentence descriptions per cluster, using top nodes + relationships + document excerpts
- **Response Visualization** — Activated nodes glow in the 3D graph when the agent responds; camera flies to the activation region; fades after 8 seconds
- **Grounding & Anti-Hallucination** — Post-processing grounding check verifies specific claims (article numbers, statistics, citations) against the context. Self-hallucination loop prevention: conversation files excluded from RAG to prevent the agent from citing its own previous hallucinations as evidence. Documents are the primary source of truth over graph inferences
- **Streaming Responses** — Token-level WebSocket streaming. Pre-generation pipeline runs first (~1.5s), then LLM tokens stream in real-time. Time-to-first-token drops from 5-10s to ~1.5s
- **Security** — Prompt injection defense (immutable identity, injection pattern detection), concept sanitization (blocklist for sensitive terms), user ID validation (regex, prevents path traversal)
- **Query Routing** — Adaptive pipeline: GPT-4o-mini classifies queries as greeting/simple/complex, skipping unnecessary pipeline stages. 10/10 accuracy, greetings respond in ~2s vs 6-8s for full queries
- **Language-Aware** — Detects query language (Spanish, English, Chinese, Arabic, French, Portuguese, German) and responds in the same language
- **Document Comparison** — Detects "article_1 vs article_3" references and searches within each document separately for targeted comparison
- **Knowledge Gap Detection** — Identifies concepts without document evidence. Reports per-node and per-cluster coverage gaps
- **Node Confidence Visualization** — Low-confidence nodes (few sources, low activation) render semi-transparent in the 3D graph
- **User Feedback** — Thumbs up/down per response, stored as JSONL with analytics summary endpoint
- **Graph Timeline** — Timestamped snapshots on ingestion events for tracking knowledge evolution
- **Intelligent Agent** — Critical thinking system prompt that challenges false premises, grounds responses in graph knowledge via fuzzy concept matching, and validates relationship types
- **AI-Powered Concept Extraction** — GPT-4o extracts entities, relationships, and semantic meaning from natural conversation
- **File Ingestion** — Ingest documents (.docx, .pdf, .xlsx, .pptx, .csv, .json, .txt, .md), code files, and images (via GPT-4o vision) with auto-generated document summaries
- **Real-Time Updates** — WebSocket-driven graph diffs push changes instantly to the frontend
- **Light & Dark Themes** — Full theme support with neural-inspired color palettes
- **Performance Optimizations** — Stress-tested: context capped (93% token reduction), hub edge limiting (30/node), smart embedding cache, ghost node pruning. Query latency: 11.3s → 5.4s. Cost: ~$0.019/query
- **Per-User Graphs** — Each user gets an isolated, persistent knowledge graph with feedback history and timeline

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐  │
│  │   3D Brain Graph         │  │   Chat Interface            │  │
│  │   react-force-graph-3d   │  │   WebSocket + REST          │  │
│  │   Three.js Sprites       │  │   Markdown rendering        │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Metrics Bar · Cluster Legend · Search · Theme Toggle   │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket + REST
┌────────────────────────────▼────────────────────────────────────┐
│                       Backend (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   LangGraph Agent                         │   │
│  │   Extract → Activate → Build Context → Generate → Update  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Knowledge Graph  │  │  Plasticity   │  │  Memory System   │   │
│  │ (NetworkX)       │  │  Engine       │  │  (STM + LTM)     │   │
│  └─────────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Brain Plasticity Model

NeuroGraph implements six biologically-inspired plasticity mechanisms:

| Mechanism | What it does | Visual effect |
|---|---|---|
| **Hebbian Learning** | Co-mentioned concepts strengthen their connection | Edge glows and thickens |
| **LTP** (Long-Term Potentiation) | Repeated activation increases node importance | Node grows larger |
| **LTD** (Long-Term Depression) | Unused connections decay over time | Edge fades and thins |
| **Neurogenesis** | New concepts create new nodes | Node appears with expansion |
| **Synaptic Pruning** | Edges below weight threshold are removed | Edge dissolves |
| **Spreading Activation** | Activating a node partially activates neighbors | Light propagates through edges |

### Dual Memory System

| | Short-Term Memory (STM) | Long-Term Memory (LTM) |
|---|---|---|
| **Analogy** | Hippocampus | Neocortex |
| **Capacity** | ~7 concepts (Miller's Law) | Unlimited |
| **Plasticity** | High — rapid decay | Low — slow decay |
| **Persistence** | Current session | Across sessions (JSON) |
| **Consolidation** | Concepts exceeding activation threshold promote to LTM | — |

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.11+ | Runtime |
| FastAPI | REST API + WebSocket server |
| LangGraph | Agent orchestration (stateful workflows) |
| LangChain + OpenAI | LLM integration (GPT-4o) |
| NetworkX | In-memory knowledge graph |
| Pydantic v2 | Data validation |
| ChromaDB | Vector store for RAG document chunks |

### Frontend
| Technology | Purpose |
|---|---|
| React 18 + TypeScript | UI framework |
| Vite | Build tool |
| react-force-graph-3d | 3D graph visualization |
| Three.js | Sprite-based node rendering |
| Tailwind CSS v4 | Styling |
| Framer Motion | Animations |
| Zustand | State management |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Jorge-Polanco-Roque/Graph_Knowledge_Base.git
cd Graph_Knowledge_Base
```

**2. Set up the backend**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**3. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

**4. Set up the frontend**

```bash
cd ../frontend
npm install
```

### Running

Start both services in separate terminals:

```bash
# Terminal 1 — Backend (port 8899)
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8899

# Terminal 2 — Frontend (port 3477)
cd frontend
npm run dev
```

Open **http://localhost:3477** in your browser.

## Usage

### Chat

Start typing in the chat panel. The AI agent will:
1. Extract concepts and relationships from your message
2. Activate relevant existing knowledge via spreading activation
3. Generate a context-aware response
4. Update the knowledge graph in real-time

Watch the 3D graph grow and evolve as you converse.

### File Ingestion

Click **Ingest Files** to upload documents into the knowledge graph. Supported formats:

| Format | Notes |
|---|---|
| `.docx`, `.pdf`, `.txt`, `.md` | Text documents |
| `.xlsx`, `.csv` | Tabular data |
| `.pptx` | Presentations |
| `.json` | Structured data |
| `.py`, `.js`, `.ts`, etc. | Code files |
| `.png`, `.jpg` | Images (analyzed via GPT-4o vision) |

Files are organized in the `inputs/` folder with typed subfolders (`documents`, `people`, `projects`, `knowledge`, `code`, `media`, `conversations`, `data`). A `.taxonomy.json` file defines extraction hints per folder type.

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a message, receive response + graph diff |
| `GET` | `/api/graph/{user_id}` | Get full graph state |
| `GET` | `/api/graph/{user_id}/stats` | Get graph metrics |
| `DELETE` | `/api/graph/{user_id}` | Reset user's graph |

### WebSocket

Connect to `ws://localhost:8899/ws/{user_id}` for real-time bidirectional communication.

**Client → Server:**
```json
{ "type": "message", "content": "Tell me about neural networks" }
```

**Server → Client:**
```json
{
  "type": "graph_diff",
  "added_nodes": [],
  "added_edges": [],
  "updated_nodes": [],
  "removed_edges": []
}
```

## Project Structure

```
graph_knowledge_base/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app entry point
│   │   ├── config.py             # Settings (pydantic-settings)
│   │   ├── models/               # Pydantic data models
│   │   ├── graph_kb/             # Knowledge graph + plasticity engine
│   │   ├── agent/                # LangGraph agent definition
│   │   └── api/                  # REST routes + WebSocket handlers
│   ├── data/users/               # Per-user persistent graph storage
│   └── pyproject.toml
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── BrainGraph/       # 3D force graph visualization
│       │   ├── Chat/             # Chat panel + message bubbles
│       │   ├── Layout/           # Split view layout
│       │   └── Metrics/          # Bottom stats bar
│       ├── hooks/                # WebSocket + Zustand store
│       ├── types/                # TypeScript interfaces
│       └── utils/                # Graph data transformations
│
├── inputs/                       # File ingestion folder
└── screenshots/                  # App screenshots
```

## License

MIT

---

<p align="center">
  Built with neuroscience-inspired algorithms and modern web technologies.
</p>
