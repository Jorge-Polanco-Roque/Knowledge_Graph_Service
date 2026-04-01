<p align="center">
  <img src="screenshots/19_autofit_dark.png" alt="NeuroGraph" width="100%">
</p>

<h1 align="center">NeuroGraph</h1>

<p align="center">
  <strong>A brain-inspired knowledge graph that learns, strengthens, and prunes connections in real time — just like biological neural networks.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2-ff6f00" alt="LangGraph">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## What is NeuroGraph?

NeuroGraph is an AI agent with a **living memory**. Instead of treating each conversation as stateless, it builds and maintains a knowledge graph that evolves over time — forming new concepts, strengthening connections through repeated use, and pruning neglected pathways.

Think of it as a brain you can talk to. Every conversation makes it smarter.

```
You: "What is fairness in hiring?"
                │
                ▼
┌─────────── NeuroGraph Brain ───────────┐
│                                         │
│  1. Extracts concepts: fairness, hiring │
│  2. Searches documents (RAG)            │
│  3. Activates related knowledge         │
│  4. Generates grounded response         │
│  5. Strengthens used connections         │
│  6. Creates new nodes if needed         │
│                                         │
│  Graph before: 300 nodes, 2600 edges    │
│  Graph after:  302 nodes, 2608 edges    │
└─────────────────────────────────────────┘
                │
                ▼
Agent: "Fairness in hiring refers to..." (grounded in documents)
```

---

## How It Works

### The Brain: 6 Plasticity Mechanisms

NeuroGraph implements biologically-inspired learning at the graph level:

| Mechanism | What happens | Brain analogy |
|---|---|---|
| **Hebbian Learning** | Concepts mentioned together get stronger connections | "Neurons that fire together, wire together" |
| **LTP** (Long-Term Potentiation) | Repeated activation increases a concept's importance | Memory consolidation through repetition |
| **LTD** (Long-Term Depression) | Unused connections decay over time | Forgetting unused information |
| **Neurogenesis** | New concepts create new nodes | Learning new things |
| **Synaptic Pruning** | Weak connections are removed | Brain optimization |
| **Spreading Activation** | Activating one concept partially activates neighbors | Associative thinking |

### Dual Memory System

| | Short-Term Memory (STM) | Long-Term Memory (LTM) |
|---|---|---|
| **Analogy** | Hippocampus | Neocortex |
| **Capacity** | ~7 concepts (Miller's Law) | Unlimited |
| **Decay** | Fast (seconds) | Slow (days) |
| **Persistence** | Current session only | Saved to disk (survives restarts) |
| **Promotion** | High-activation STM concepts consolidate into LTM | — |

### The Agent Pipeline

Every message goes through a 6-step pipeline orchestrated by LangGraph:

```
User message
     │
     ▼
┌─────────────────┐
│ 1. Route Query   │  GPT-4o-mini classifies: greeting | simple | complex
└────────┬────────┘  Greetings skip the pipeline entirely (~1s vs ~5s)
         │
         ▼
┌─────────────────┐
│ 2. Extract       │  GPT-4o identifies concepts, entities, relationships
│    Concepts      │  "fairness in hiring" → [fairness, hiring, discrimination]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Activate      │  Spreading activation propagates through the graph
│    Memory        │  Query-weighted: relevant pathways get stronger signal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Retrieve      │  ChromaDB semantic search + graph-guided retrieval
│    Documents     │  Returns top 7 document chunks as evidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Generate      │  GPT-4o generates response grounded in documents
│    Response      │  Tokens stream via WebSocket in real time
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Update        │  Apply Hebbian learning, LTP, LTD, pruning
│    Graph         │  Consolidate STM → LTM if thresholds met
└─────────────────┘
```

### RAG (Retrieval-Augmented Generation)

Documents are ingested, chunked, and embedded in ChromaDB. At query time, two retrieval strategies work together:

1. **Semantic search** — finds chunks textually similar to the query
2. **Graph-guided search** — finds chunks from documents that produced the activated concepts

This means the graph enhances retrieval: if you ask about "fairness" and the graph knows that concept came from `article_3.docx`, the system searches *within* that specific document too.

### Anti-Hallucination

- **Document-first**: responses are grounded in actual document excerpts, not graph label inferences
- **Grounding check**: post-processing scans for specific claims (article numbers, statistics) and verifies against context
- **Self-reference loop prevention**: conversation files are excluded from RAG ingestion
- **Context poisoning defense**: the agent is instructed to distrust user-embedded "facts" and verify against documents
- **User corrections**: thumbs-down feedback with comments becomes highest-priority context in future queries

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Any Frontend / Client                        │
│              (React, mobile app, another backend)                │
└────────────────────────┬─────────────────────────────────────────┘
                         │  REST (/api/*) + WebSocket (/ws/*)
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    NeuroGraph Brain (FastAPI)                     │
│                                                                   │
│  ┌── Auth ──────────────────────────────────────────────────┐    │
│  │  API key validation · Per-request OpenAI key · Rate limit │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌── Agent Pipeline (LangGraph) ────────────────────────────┐    │
│  │  Route → Extract → Activate → Retrieve → Generate → Update│   │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─── Knowledge Graph ──┐  ┌─── Document Store ──┐              │
│  │  NetworkX (in-memory) │  │  ChromaDB (vectors) │              │
│  │  Plasticity engine    │  │  Semantic search     │              │
│  │  Spreading activation │  │  4800+ chunks        │              │
│  └───────────────────────┘  └─────────────────────┘              │
│                                                                   │
│  Storage: /data/users/{user_id}/                                 │
│    ├── graph.json           (knowledge graph)                    │
│    ├── chroma/              (vector embeddings)                  │
│    ├── node_embeddings.npz  (spreading activation cache)         │
│    └── feedback.jsonl       (user corrections)                   │
└──────────────────────────────────────────────────────────────────┘
```

The brain is **headless** — it exposes REST + WebSocket endpoints and any client can connect to it. The included React frontend and test UI (`UI/index.html`) are just two examples.

### Two Keys You Need to Understand

**1. `NEUROGRAPH_API_KEY` — Who can access the brain**

This is the **service key**. It protects the API from unauthorized access. Set it as an environment variable when starting the server. Every request must include it:

```
Authorization: Bearer nk_your_secret_key
```

If not set, authentication is disabled (local development mode). You can set multiple keys separated by commas for key rotation: `NEUROGRAPH_API_KEY=nk_key1,nk_key2`.

**2. `user_id` — Whose brain is it**

Every endpoint and WebSocket connection requires a `user_id`. Each user gets a **completely isolated brain**:

```
/data/users/
├── alice/              ← Alice's brain (her graph, her documents, her feedback)
│   ├── graph.json
│   ├── chroma/
│   └── feedback.jsonl
├── bob/                ← Bob's brain (completely separate)
│   ├── graph.json
│   ├── chroma/
│   └── feedback.jsonl
└── team-project/       ← A shared brain for a team
    ├── graph.json
    └── chroma/
```

Two users asking the same question will get different answers based on what each brain has learned. A new `user_id` starts with an empty brain — no knowledge until you ingest documents or start chatting.

The `user_id` is chosen by the client (alphanumeric, hyphens, underscores, max 64 chars). There is no user registration — any valid ID creates a brain on first use.

**3. `X-OpenAI-Key` — Who pays for the AI (optional)**

By default, the server uses its own `OPENAI_API_KEY`. But clients can provide their own key per request:

```
X-OpenAI-Key: sk-client-own-key
```

This enables a "bring your own key" model where each client pays OpenAI directly.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

### Option A: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/Jorge-Polanco-Roque/Graph_Knowledge_Base.git
cd Graph_Knowledge_Base

# 2. Set up the backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Configure your API key
cp .env.example .env
```

Edit `backend/.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
```

```bash
# 4. Start the brain
uvicorn app.main:app --reload --port 8899
```

The brain is now running at `http://localhost:8899`.

### Option B: Docker

```bash
# 1. Clone the repository
git clone https://github.com/Jorge-Polanco-Roque/Graph_Knowledge_Base.git
cd Graph_Knowledge_Base

# 2. Create .env in the project root
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
echo 'NEUROGRAPH_API_KEY=nk_choose_a_secret_key' >> .env

# 3. Start with Docker Compose
docker compose up -d
```

The brain is now running at `http://localhost:8899` with authentication enabled.

### Verify It Works

```bash
# Health check (no auth required)
curl http://localhost:8899/health
# → {"status":"healthy","version":"1.0.0"}

# Chat (auth required if NEUROGRAPH_API_KEY is set)
curl -X POST http://localhost:8899/api/chat \
  -H "Authorization: Bearer nk_choose_a_secret_key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "my-user", "message": "Hello, what can you do?"}'
```

### Test Values for Quick Testing

Use these values to get started immediately:

| Field | Test Value | Notes |
|---|---|---|
| **Host** | `localhost:8899` | Default backend port |
| **API Key** | `nk_test_key_12345` | Set via `NEUROGRAPH_API_KEY=nk_test_key_12345` when starting the server |
| **User ID** | `default-user` | Pre-existing brain with ingested data (if available) |
| **User ID** | `test-user` | Fresh empty brain for testing from scratch |
| **OpenAI Key** | *(your own `sk-...`)* | Required in `backend/.env` or via `OPENAI_API_KEY` env var |

**Example — start the server with test auth and try it:**

```bash
# Start with test API key
cd backend
source .venv/bin/activate
NEUROGRAPH_API_KEY=nk_test_key_12345 uvicorn app.main:app --port 8899

# In another terminal — test REST
curl -X POST http://localhost:8899/api/chat \
  -H "Authorization: Bearer nk_test_key_12345" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user", "message": "What is artificial intelligence?"}'

# Test WebSocket with wscat
npx wscat -c "ws://localhost:8899/ws/test-user?token=nk_test_key_12345"
# Then type: {"type":"user_message","data":{"content":"Hello"}}
```

### Connect a UI

Open `UI/index.html` in your browser for a minimal chat interface:

1. Set **Host** to `localhost:8899`
2. Set **API Key** to `nk_test_key_12345`
3. Set **User ID** to `default-user` (has data) or `test-user` (empty)
4. Click **Connect** and start chatting

Or set up the full React frontend:

```bash
cd frontend
npm install
npm run dev    # Opens at http://localhost:3477
```

<p align="center">
  <img src="screenshots/18_light_theme.png" alt="NeuroGraph Light Theme" width="100%">
</p>

---

## Ingesting Documents

The brain starts empty. Feed it documents to build knowledge:

### Step 1: Add files to `backend/inputs/`

```
backend/inputs/
├── documents/          # .docx, .pdf, .txt, .md
├── data/               # .csv, .json, .xlsx
├── code/               # .py, .js, .ts
├── media/              # .png, .jpg (analyzed via GPT-4o vision)
├── knowledge/          # Reference materials
├── people/             # People profiles
├── projects/           # Project documentation
└── conversations/      # Chat logs (auto-saved)
```

### Step 2: Trigger ingestion

```bash
curl -X POST http://localhost:8899/api/ingest/my-user \
  -H "Authorization: Bearer nk_your_key"
```

### Step 3: Post-ingestion cleanup (recommended)

```bash
# Merge duplicate concepts
curl -X POST http://localhost:8899/api/graph/my-user/deduplicate \
  -H "Authorization: Bearer nk_your_key"

# Remove disconnected low-importance nodes
curl -X POST http://localhost:8899/api/graph/my-user/prune-ghosts \
  -H "Authorization: Bearer nk_your_key"

# Generate cluster descriptions
curl -X POST http://localhost:8899/api/graph/my-user/summarize-clusters \
  -H "Authorization: Bearer nk_your_key"
```

Supported formats: `.docx`, `.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.xlsx`, `.pptx`, `.py`, `.js`, `.ts`, `.png`, `.jpg`

---

## API Reference

### Authentication

When `NEUROGRAPH_API_KEY` is set, all endpoints (except `/health` and `/docs`) require authentication:

```
Authorization: Bearer nk_your_key
# or
X-API-Key: nk_your_key
```

Each client can also provide their own OpenAI key per request:

```
X-OpenAI-Key: sk-client-own-key
```

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/docs` | Interactive API documentation (no auth) |
| `POST` | `/api/chat` | Send message, receive response + graph diff |
| `GET` | `/api/graph/{user_id}` | Full graph snapshot with clusters |
| `GET` | `/api/graph/{user_id}/stats` | Graph metrics |
| `GET` | `/api/graph/{user_id}/gaps` | Knowledge gap analysis |
| `GET` | `/api/graph/{user_id}/timeline` | Graph evolution snapshots |
| `POST` | `/api/graph/{user_id}/deduplicate` | Merge duplicate nodes |
| `POST` | `/api/graph/{user_id}/prune-ghosts` | Remove disconnected nodes |
| `POST` | `/api/graph/{user_id}/summarize-clusters` | Generate cluster descriptions |
| `DELETE` | `/api/graph/{user_id}` | Full reset (graph + vectors + cache) |
| `POST` | `/api/feedback/{user_id}` | Submit feedback (thumbs up/down + comment) |
| `GET` | `/api/feedback/{user_id}/summary` | Feedback analytics |
| `GET` | `/api/ingest/files` | List available input files |
| `POST` | `/api/ingest/{user_id}` | Ingest all files |
| `POST` | `/api/ingest/{user_id}/{filename}` | Ingest single file |

### WebSocket

Connect to `ws://localhost:8899/ws/{user_id}?token=nk_your_key` for real-time streaming.

**Send a message:**

```json
{
  "type": "user_message",
  "data": { "content": "What is fairness in hiring?" }
}
```

**Receive (streaming):**

```json
{"type": "response_chunk", "data": {"content": "Fair"}}
{"type": "response_chunk", "data": {"content": "ness "}}
{"type": "response_chunk", "data": {"content": "in hiring..."}}
{"type": "response_end",   "data": {"content": "...", "extracted_concepts": [...], "graph_diff": {...}}}
{"type": "graph_diff",     "data": {"node_diffs": [...], "edge_diffs": [...], "activated_nodes": [...]}}
{"type": "spreading_activation", "data": {"waves": [["node1", "node2"], ["node3"]]}}
```

### Connecting from Any Language

```javascript
// JavaScript / TypeScript
const ws = new WebSocket('ws://localhost:8899/ws/my-user?token=nk_key');
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'response_chunk') process.stdout.write(msg.data.content);
};
ws.onopen = () => ws.send(JSON.stringify({
  type: 'user_message', data: { content: 'Hello' }
}));
```

```python
# Python
import httpx
r = httpx.post("http://localhost:8899/api/chat",
    headers={"Authorization": "Bearer nk_key"},
    json={"user_id": "my-user", "message": "Hello"})
print(r.json()["response"])
```

```bash
# curl
curl -X POST http://localhost:8899/api/chat \
  -H "Authorization: Bearer nk_key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "my-user", "message": "Hello"}'
```

---

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model for responses |
| `NEUROGRAPH_API_KEY` | *(empty = auth disabled)* | Service authentication key |
| `DATA_DIR` | `./data` | Data persistence directory |
| `CORS_ORIGINS` | `["http://localhost:3477"]` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CONTEXT_MODE` | `chunks_only` | `chunks_only` or `graph_and_chunks` |
| `MAX_CLUSTERS` | `10` | Fixed cluster count for visualization |
| `RAG_TOP_K` | `7` | Number of document chunks retrieved per query |
| `RAG_CHUNK_SIZE` | `800` | Document chunk size (characters) |

---

## Project Structure

```
neurograph/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + middleware stack
│   │   ├── config.py                # Pydantic settings
│   │   ├── api/
│   │   │   ├── routes.py            # REST endpoints
│   │   │   ├── websocket.py         # WebSocket streaming handler
│   │   │   ├── auth.py              # API key auth + security middleware
│   │   │   └── context.py           # Per-request OpenAI key (contextvar)
│   │   ├── agent/
│   │   │   ├── graph_agent.py       # LangGraph state machine
│   │   │   ├── nodes.py             # Pipeline nodes + system prompt
│   │   │   └── state.py             # Agent state definition
│   │   ├── graph_kb/
│   │   │   ├── knowledge_graph.py   # NetworkX graph wrapper
│   │   │   ├── plasticity.py        # Hebbian, LTP, LTD, pruning
│   │   │   ├── spreading.py         # Spreading activation algorithm
│   │   │   ├── memory.py            # STM/LTM manager + consolidation
│   │   │   ├── document_store.py    # ChromaDB vector store
│   │   │   ├── embedding_cache.py   # Node embedding cache (.npz)
│   │   │   ├── deduplication.py     # Semantic node merging
│   │   │   ├── cluster_summarizer.py # LLM cluster descriptions
│   │   │   └── timeline.py          # Graph snapshots
│   │   └── ingestion/
│   │       ├── ingest.py            # File ingestion pipeline
│   │       └── file_reader.py       # Multi-format file parser
│   ├── data/users/                  # Per-user persistent storage
│   ├── inputs/                      # Documents to ingest
│   ├── tests/
│   │   ├── security/               # Pentesting suites (auth, path traversal, injection)
│   │   └── integration/            # E2E pipeline tests
│   ├── Dockerfile                   # Multi-stage production image
│   └── pyproject.toml
│
├── frontend/                        # React + TypeScript (3D graph + chat)
├── UI/                              # Minimal test UI (single HTML file)
├── docker-compose.yml               # One-command deployment
└── CLAUDE.md                        # Detailed technical documentation
```

---

## Security

The brain includes production-grade security hardening:

- **API key authentication** with constant-time comparison (timing-attack resistant)
- **Per-request OpenAI key** support (BYO-key model)
- **Path traversal prevention** on file ingestion endpoints
- **Input validation** — message length (10K), body size (1MB), history depth (50)
- **WebSocket rate limiting** — 1 message/second, max 3 connections per user
- **Error sanitization** — no stack traces or internal paths in responses
- **Security headers** — X-Content-Type-Options, X-Frame-Options, CSP
- **Prompt injection defense** — immutable identity block, injection pattern detection
- **Concept sanitization** — blocklist for sensitive terms (passwords, keys, credentials)
- **Non-root Docker user** — container runs as `neurograph` (UID 1000)
- **54 automated security + E2E tests** passing

---

## Performance

Stress-tested with 300+ nodes and 2600+ edges:

| Metric | Value |
|---|---|
| Simple query (greeting) | ~1-2s |
| Complex query (full pipeline) | ~5-6s |
| Context tokens per query | ~1,200 (93% reduction from 18K) |
| Cost per query | ~$0.019 |
| Time to first token (WebSocket) | ~1.5s |
| Max edges traversed per node | 30 (hub limiting) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.11+ |
| API Framework | FastAPI (REST + WebSocket) |
| Agent Orchestration | LangGraph (stateful graph workflows) |
| LLM | OpenAI GPT-4o (via LangChain) |
| Knowledge Graph | NetworkX (in-memory, JSON persistence) |
| Vector Store | ChromaDB (embedded, SQLite-backed) |
| Embeddings | OpenAI text-embedding-3-small |
| Data Validation | Pydantic v2 |
| Frontend | React 18 + TypeScript + Vite |
| 3D Visualization | react-force-graph-3d + Three.js |
| Styling | Tailwind CSS v4 |
| State Management | Zustand |
| Containerization | Docker (multi-stage build) |

---

## License

MIT

---

<p align="center">
  Built with neuroscience-inspired algorithms and modern web technologies.
</p>
