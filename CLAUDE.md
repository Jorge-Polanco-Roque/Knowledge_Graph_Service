# NeuroGraph — Brain Plasticity Knowledge Graph

## Project Overview

NeuroGraph simulates brain plasticity using a graph knowledge base. An AI agent (powered by OpenAI) uses a dynamic knowledge graph as its memory and context. The graph evolves in real-time as the user converses with the agent — forming new connections, strengthening existing ones, and pruning unused pathways, just like a biological brain.

A React frontend provides a split-view: a 3D brain-like graph visualization (left) and a chat interface (right), showing in real-time how concepts connect and knowledge grows.

## Architecture

```
frontend/ (React + TypeScript)
├── 3D Graph Visualization (react-force-graph-3d + Three.js)
├── Chat Interface
├── Metrics Dashboard
└── WebSocket client for real-time updates

backend/ (Python + FastAPI)
├── LangGraph Agent (orchestration)
├── Knowledge Graph (NetworkX)
├── Plasticity Engine (Hebbian, LTP, LTD, pruning)
├── Memory System (STM + LTM)
├── WebSocket server
└── OpenAI LLM integration
```

## Tech Stack

### Backend
- **Python 3.11+**
- **FastAPI** — REST API + WebSocket server
- **LangGraph** — agent orchestration (stateful graph-based workflows)
- **LangChain + langchain-openai** — LLM integration with OpenAI
- **NetworkX** — in-memory knowledge graph
- **Pydantic v2** — data models and validation
- **uvicorn** — ASGI server

### Frontend
- **React 18** + **TypeScript**
- **Vite** — build tool
- **react-force-graph-3d** — 3D rotatable graph (Three.js sprites with Canvas2D pill textures)
- **react-force-graph-2d** — available as fallback
- **three** — sprite materials for billboard node labels
- **Tailwind CSS v4** — styling
- **Framer Motion** — UI animations
- **Zustand** — lightweight state management

### Visualization Decisions (validated through iteration)
- **Node rendering**: Canvas2D pill textures cached as Three.js Sprites (billboard = always face camera, `depthWrite: false`)
- **Node sizing**: `scale = 2 + importance * 1`, aspect ratio from text width (reduced from 5+imp*2 to prevent overlap)
- **Initial positions**: Fibonacci sphere layout — deterministic, evenly distributed in 3D, radius scales with node count (`R = max(20, N*0.08)`, ultra compact)
- **Force timing**: Forces configured via `onEngineTick` on first tick (not `useEffect`) to guarantee they apply before any layout computation. `d3ReheatSimulation()` called once after configuration.
- **Custom forces**: Cluster force (strength=0.3) groups same-cluster nodes together. Gravity force (strength=0.15) pulls all nodes toward origin to keep clusters close. No custom forces that collapse the graph.
- **Cluster detection**: NetworkX Louvain community detection (resolution=0.5, min cluster size=3)
- **Cluster colors**: 16-color palette, nodes colored by cluster not category
- **Layout forces**: charge=-15, link distance=8, center force (default), `warmupTicks=0` — ultra compact layout
- **Graph freezes** after 300 cooldown ticks (static after layout settles)
- **Auto-fit**: zoomToFit on engine stop + delayed fallbacks at 0.5s, 2s, 5s, 8s (padding=20)
- **Collapsed graph fix**: Root cause was 152+ disconnected components all converging to origin with weak charge. Fibonacci sphere initial positions + force timing via onEngineTick + smaller sprites solved it.
- **Node search**: Search bar (top-left) with autocomplete suggestions, filters by label, shows cluster color and importance. Clicking a suggestion flies the camera to that node.
- **Chat markdown**: Messages rendered with `react-markdown` — supports bold, italic, lists, code blocks, headings.
- **Theme**: Light/Dark toggle, affects background, link opacity, borders
- **Controls**: Theme toggle, Fit to view, Reset camera, Zoom in/out, Node search
- **Ports**: Backend=8899, Frontend=3477

### File Ingestion
- **inputs/ folder** with typed subfolders: documents, people, projects, knowledge, code, media, conversations, data
- **.taxonomy.json** defines extraction hints per folder type
- Custom folders (e.g. AI_Ethics) auto-detected with generic extraction
- Conversations auto-saved on WebSocket disconnect
- Supports: .docx, .pdf, .xlsx, .pptx, .csv, .json, .txt, .md, code files, images (GPT-4o vision)

## Brain Plasticity Model

### Dual Memory System

**Short-Term Memory (STM)** — like the hippocampus:
- Active concepts from the current conversation
- High plasticity, rapid decay (seconds to minutes)
- Limited capacity (~7 concepts, like Miller's Law)
- Nodes have high activation values that decay quickly
- Used to build immediate conversational context

**Long-Term Memory (LTM)** — like the neocortex:
- Persistent graph stored per user (survives across sessions)
- Low plasticity, slow decay (days to weeks)
- Unlimited capacity
- Consolidated from STM through repeated activation
- Persisted to disk as JSON per user (`data/users/{user_id}/graph.json`)

**Consolidation Process** (STM -> LTM):
- When a concept in STM exceeds an activation threshold through repeated use, it consolidates into LTM
- Connections that appear across multiple conversation turns get promoted
- On session end, remaining high-activation STM nodes consolidate to LTM
- Mimics memory consolidation during sleep

### Plasticity Mechanisms

| Mechanism | Implementation | Visual Effect |
|---|---|---|
| **Hebbian Learning** | Co-mentioned concepts get their edge weight increased | Edge glows and thickens |
| **LTP (Long-term Potentiation)** | Repeated activation increases node importance | Node grows and pulses brighter |
| **LTD (Long-term Depression)** | Unused connections decay over time | Edge fades and thins |
| **Neurogenesis** | New concepts create new nodes | Node "births" with expansion animation |
| **Synaptic Pruning** | Edges below weight threshold are removed | Edge dissolves with particle effect |
| **Spreading Activation** | Activating a node partially activates neighbors | Wave of light propagates through edges |

### Plasticity Parameters (tunable)
```python
HEBBIAN_LEARNING_RATE = 0.1      # Edge weight increase per co-activation
LTP_RATE = 0.05                   # Node importance increase per activation
LTD_DECAY_RATE = 0.01            # Per-turn decay for inactive connections
PRUNING_THRESHOLD = 0.05          # Minimum edge weight before pruning
STM_CAPACITY = 7                  # Max active STM concepts
STM_DECAY_RATE = 0.3             # Rapid decay for STM nodes
CONSOLIDATION_THRESHOLD = 0.7    # STM activation needed to consolidate to LTM
SPREADING_FACTOR = 0.5           # Activation spread to neighbors (multiplier)
```

## Agent Flow (LangGraph)

```
User Message
     │
     ▼
┌─────────────┐
│  Extract     │ ── NLP/LLM extracts concepts, entities, relationships
│  Concepts    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Activate    │ ── Load relevant subgraph from LTM, activate in STM
│  Memory      │ ── Spreading activation to related concepts
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Build       │ ── Construct enriched context from active graph
│  Context     │ ── Include concept definitions, relationships, history
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Generate    │ ── LLM generates response using graph context
│  Response    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Update      │ ── Apply plasticity: Hebbian, LTP, LTD, pruning
│  Graph       │ ── Consolidate STM → LTM if thresholds met
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Broadcast   │ ── Send graph diff via WebSocket to frontend
│  Changes     │ ── Include animation hints (new, strengthened, pruned)
└─────────────┘
```

## Data Models

### Node (Concept)
```python
{
    "id": str,                    # Unique identifier
    "label": str,                 # Display name
    "category": str,              # person | place | concept | action | object | event
    "activation": float,          # Current activation level (0.0 - 1.0)
    "importance": float,          # LTP-accumulated importance (0.0 - 1.0)
    "memory_type": str,           # "stm" | "ltm"
    "first_seen": datetime,       # When the concept was first introduced
    "last_activated": datetime,   # Last time the concept was activated
    "activation_count": int,      # Total number of activations
    "metadata": dict              # Additional context, definitions
}
```

### Edge (Connection)
```python
{
    "source": str,                # Source node ID
    "target": str,                # Target node ID
    "weight": float,              # Connection strength (0.0 - 1.0)
    "relationship": str,          # Type: "related_to" | "is_a" | "part_of" | "causes" | etc.
    "co_activation_count": int,   # Times both concepts activated together
    "last_co_activated": datetime
}
```

## Visualization Design

### Theme: Dark Neural Space
- Background: deep black/dark blue gradient (`#0a0a1a` to `#0d1117`)
- Primary accent: electric blue (`#00d4ff`)
- Secondary: neural purple (`#8b5cf6`)
- Warm accents: synaptic orange (`#f97316`)

### Node Colors by Category
- **Persons**: `#3b82f6` (blue)
- **Places**: `#10b981` (green)
- **Concepts**: `#8b5cf6` (purple)
- **Actions**: `#f97316` (orange)
- **Objects**: `#ec4899` (pink)
- **Events**: `#f59e0b` (amber)

### Visual Effects
- **Active nodes**: bloom/glow effect, pulsing size animation
- **STM nodes**: brighter, more saturated, visible pulse
- **LTM nodes**: steady glow, size reflects importance
- **Strong edges**: thick, bright, particle flow
- **Weak edges**: thin, transparent, dashed
- **New node birth**: expand from point with radial burst
- **Pruned edge**: dissolve into particles
- **Spreading activation**: wave of light traveling along edges
- **Consolidation**: STM node transitions color/style to LTM

### Layout
```
┌────────────────────────────────────────────────────────┐
│ NeuroGraph                          [user] [settings]  │
├───────────────────────────────┬────────────────────────┤
│                               │  Conversation          │
│                               │                        │
│     3D Brain Graph            │  ┌──────────────────┐  │
│     (react-force-graph-3d)    │  │ Agent message...  │  │
│                               │  └──────────────────┘  │
│     - rotate, zoom, pan       │  ┌──────────────────┐  │
│     - click node for details  │  │ User message...   │  │
│     - real-time updates       │  └──────────────────┘  │
│                               │                        │
│                               │                        │
├───────────────────────────────┤  ┌──────────────────┐  │
│ Metrics: 42 nodes | 87 edges  │  │ Type message...   │  │
│ STM: 5/7 | Top: "AI" "Graph" │  │            [Send]  │  │
└───────────────────────────────┴────────────────────────┘
```

## File Structure

```
graph_knowledge_base/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
│
├── backend/
│   ├── pyproject.toml
│   ├── .env.example              # OPENAI_API_KEY=sk-...
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app, CORS, lifespan
│   │   ├── config.py             # Settings via pydantic-settings
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py          # Node, Edge, GraphDiff pydantic models
│   │   │   └── messages.py       # ChatMessage, WSMessage models
│   │   │
│   │   ├── graph_kb/
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_graph.py  # NetworkX-backed KG (CRUD operations)
│   │   │   ├── plasticity.py       # PlasticityEngine: Hebbian, LTP, LTD, pruning
│   │   │   ├── spreading.py        # Spreading activation algorithm
│   │   │   └── memory.py           # STM/LTM manager, consolidation logic
│   │   │
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── graph_agent.py    # LangGraph StateGraph definition
│   │   │   ├── nodes.py          # LangGraph node functions (extract, activate, build, generate, update)
│   │   │   └── state.py          # TypedDict state for the agent
│   │   │
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── routes.py         # REST: /chat, /graph/{user_id}, /users
│   │       └── websocket.py      # WS: /ws/{user_id} — real-time graph diffs
│   │
│   ├── data/
│   │   └── users/                # Per-user persistent graph storage
│   │       └── {user_id}/
│   │           └── graph.json
│   │
│   └── tests/
│       ├── test_plasticity.py
│       ├── test_knowledge_graph.py
│       └── test_agent.py
│
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── index.html
    │
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css               # Tailwind + custom neural theme
        │
        ├── components/
        │   ├── Layout/
        │   │   └── SplitView.tsx    # Resizable split panel
        │   ├── BrainGraph/
        │   │   ├── BrainGraph.tsx   # 3D force graph wrapper
        │   │   ├── NodeObject.tsx   # Custom 3D node with glow
        │   │   ├── EdgeObject.tsx   # Custom edge with particles
        │   │   └── effects.ts       # Bloom, pulse, spread animations
        │   ├── Chat/
        │   │   ├── ChatPanel.tsx    # Chat container
        │   │   ├── MessageBubble.tsx
        │   │   └── InputBar.tsx
        │   └── Metrics/
        │       └── MetricsBar.tsx   # Bottom stats bar
        │
        ├── hooks/
        │   ├── useWebSocket.ts      # WS connection + reconnection
        │   └── useGraphStore.ts     # Zustand store for graph state
        │
        ├── types/
        │   ├── graph.ts             # Node, Edge, GraphDiff types
        │   └── messages.ts          # Chat message types
        │
        └── utils/
            └── graphTransforms.ts   # Map backend data to 3D visual props
```

## Commands

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Add OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev  # Vite dev server on port 5173
```

### Tests
```bash
cd backend && pytest
cd frontend && npm test
```

## API Endpoints

### REST
- `POST /api/chat` — Send message, receive response + graph diff
- `GET /api/graph/{user_id}` — Full graph state for a user
- `GET /api/graph/{user_id}/stats` — Graph metrics
- `DELETE /api/graph/{user_id}` — Reset user's graph (fresh brain)

### WebSocket
- `ws://localhost:8000/ws/{user_id}` — Real-time bidirectional:
  - Client → Server: `{ "type": "message", "content": "..." }`
  - Server → Client: `{ "type": "graph_diff", "added_nodes": [...], "added_edges": [...], "updated_nodes": [...], "updated_edges": [...], "removed_edges": [...] }`
  - Server → Client: `{ "type": "agent_response", "content": "...", "activated_concepts": [...] }`
  - Server → Client: `{ "type": "spreading_activation", "waves": [...] }`

## Key Design Decisions

1. **NetworkX over Neo4j**: Keeps the project self-contained with no external DB dependency. Each user's graph is small enough to fit in memory. Persisted as JSON.
2. **OpenAI as LLM provider**: Using `gpt-4o` for concept extraction and response generation via langchain-openai.
3. **LangGraph for agent orchestration**: Provides clear state machine for the extract → activate → build → generate → update pipeline.
4. **WebSocket for real-time**: Graph diffs are pushed immediately, enabling smooth animations. REST is used only for initial state loading.
5. **Dual memory (STM/LTM)**: Adds biological fidelity and practical value — STM keeps conversations coherent, LTM builds lasting knowledge.
6. **Per-user graphs**: Each user has an isolated knowledge graph stored in `data/users/{user_id}/graph.json`.
7. **3D as default visualization**: More immersive and brain-like. A 2D toggle can be added later for accessibility.

## Coding Conventions

- Backend: Python type hints everywhere, async/await for I/O
- Frontend: TypeScript strict mode, functional components, hooks only
- All API communication via typed Pydantic models (backend) and TypeScript interfaces (frontend)
- Graph operations are atomic — plasticity updates happen in a single transaction
- WebSocket messages are JSON with a `type` discriminator field
