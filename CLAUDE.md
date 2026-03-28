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
- **Cluster detection**: NetworkX Louvain community detection (resolution=1.5, min cluster size=3). Resolution increased from 0.5 to 1.5 to produce more granular clusters after fuzzy matching + RAG ingestion dramatically increased edge count (~3500 edges).
- **Cluster colors**: 16-color palette, nodes colored by cluster not category
- **Layout forces**: charge=-15, link distance=8, center force (default), `warmupTicks=0` — ultra compact layout
- **Graph freezes** after 300 cooldown ticks (static after layout settles)
- **Auto-fit**: zoomToFit on engine stop + delayed fallbacks at 0.5s, 2s, 5s, 8s (padding=20)
- **Collapsed graph fix**: Root cause was 152+ disconnected components all converging to origin with weak charge. Fibonacci sphere initial positions + force timing via onEngineTick + smaller sprites solved it.
- **Node search**: Search bar (top-left) with autocomplete suggestions, filters by label, shows cluster color and importance. Clicking a suggestion flies the camera to that node.
- **Chat markdown**: Messages rendered with `react-markdown` — supports bold, italic, lists, code blocks, headings.
- **Static concept tags**: Each message stores its activated concepts at response time (`ChatMessage.concepts`), not from global state. Tags are immutable per message.
- **Theme**: Light/Dark toggle, affects background, link opacity, borders
- **Controls**: Theme toggle, Fit to view, Reset camera, Zoom in/out, Node search
- **Ports**: Backend=8899, Frontend=3477

### Agent Intelligence (validated through iteration)
- **System prompt**: Instructs the LLM to think critically, challenge false premises, distinguish known from unknown, and ground responses in graph context.
- **Fuzzy concept matching**: When the user mentions a concept, `_find_related_nodes()` searches all existing graph nodes by substring and word overlap (not just exact ID match). This ensures the agent uses existing knowledge instead of creating empty stubs.
- **Relationship validation**: LLM-extracted relationship types are validated against the `RelationshipType` enum; invalid values fallback to `related_to` to prevent Pydantic crashes.
- **Spreading activation**: Depth=2, min_activation=0.05, factor=0.5. Context built from activated subgraph includes concepts + relationships for the LLM prompt.

### Cross-Linking: Graph ↔ Documents

#### Problem

Graph nodes and ChromaDB chunks are two independent systems that both originate from the same documents, but have no link between them. When the user asks about "Fairness", the agent does semantic search AND graph activation independently — but doesn't know that the node "Fairness" was extracted from `article_3.docx`, which also has relevant chunks in ChromaDB.

#### Solution: `source_files` as the Bidirectional Link Key

The link between nodes and chunks is the **source filename**:
- Nodes know which files they came from → `node.metadata["source_files"]`
- Chunks know which file they came from → `chunk.metadata["source_file"]`
- To find chunks for a node: get node's `source_files`, query ChromaDB with `where={"source_file": {"$in": [...]}}`

```
article_3.docx
     │
     ├──→ Node "Fairness"  (metadata.source_files = ["article_3.docx"])
     ├──→ Node "Hiring"    (metadata.source_files = ["article_3.docx", "article_5.docx"])
     │
     └──→ Chunk 0 (metadata.source_file = "article_3.docx")
          Chunk 1 (metadata.source_file = "article_3.docx")
          Chunk 2 (metadata.source_file = "article_3.docx")
```

#### Implementation Plan — 4 Steps

**Step 1: Store `source_files` in node metadata during ingestion**

Modify `ingest_file()` in `backend/app/ingestion/ingest.py`. After concepts are activated (`mm.activate_concepts()`), update each activated node's metadata:

```python
for node_id in activated_ids:
    node = mm.graph.get_node(node_id)
    if node:
        sources = list(node.metadata.get("source_files", []))
        if file_data["filename"] not in sources:
            sources.append(file_data["filename"])
            mm.graph.update_node(node_id, metadata={**node.metadata, "source_files": sources})
```

A node extracted from 3 different documents will have `source_files: ["article_1.docx", "article_3.docx", "article_5.docx"]`.

**Step 2: Graph-guided RAG retrieval (dual strategy)**

Modify `retrieve_documents()` in `backend/app/agent/nodes.py`. Instead of only doing semantic search, also retrieve chunks from documents related to activated graph nodes:

```python
def retrieve_documents(state):
    # Strategy 1: Pure semantic search (current)
    semantic_chunks = doc_store.search(query, top_k=3)

    # Strategy 2: Graph-guided — find chunks from source docs of activated nodes
    source_files = set()
    for nid in activated_ids:
        node = mm.graph.get_node(nid)
        if node and node.metadata.get("source_files"):
            source_files.update(node.metadata["source_files"])

    source_chunks = []
    for sf in list(source_files)[:5]:
        source_chunks.extend(doc_store.search(query, top_k=2, where={"source_file": sf}))

    # Merge, deduplicate by chunk ID, return top 5
    return deduplicated(semantic_chunks + source_chunks)[:5]
```

This gives **two retrieval strategies**:
1. **Semantic**: "what chunks are textually similar to the query?"
2. **Graph-guided**: "what chunks come from documents that produced the activated concepts?"

The combination is more targeted — if the user asks about "fairness in hiring" and the graph activates "Fairness" (from article_3.docx), the graph-guided strategy will search WITHIN article_3.docx for relevant passages, even if those passages don't contain the exact words "fairness in hiring".

**Step 3: Show source files in node tooltip (frontend)**

Modify `nodeLabel` callback in `BrainGraph.tsx`. The node's `metadata.source_files` is already sent to the frontend. Add a "Sources" line to the tooltip:

```tsx
const sources = node.metadata?.source_files;
// Add to tooltip HTML:
${sources?.length ? `<div>Sources: ${sources.slice(0,3).join(', ')}</div>` : ''}
```

**Step 4: Add `DocumentStore.search_by_sources()` convenience method**

Add to `document_store.py`:

```python
def search_by_sources(self, query: str, source_files: list[str], top_k: int = 3) -> list[dict]:
    """Search within specific source files only."""
    if not source_files or self._collection.count() == 0:
        return []
    return self.search(query, top_k=top_k, where={"source_file": {"$in": source_files}})
```

#### Files to Modify

| File | Change |
|---|---|
| `backend/app/ingestion/ingest.py` | Store `source_files` in node metadata after activation |
| `backend/app/agent/nodes.py` | Dual retrieval strategy in `retrieve_documents` |
| `backend/app/graph_kb/document_store.py` | Add `search_by_sources()` method |
| `frontend/src/components/BrainGraph/BrainGraph.tsx` | Show source files in node tooltip |

#### What This Enables

- **Smarter retrieval**: Graph knowledge guides document search, not just embeddings
- **Provenance**: Every node knows which documents it came from
- **Transparency**: Users see source files on hover, building trust in the graph
- **Bidirectional navigation**: node → docs (via metadata), docs → nodes (via shared source_file key)

### File Ingestion
- **inputs/ folder** with typed subfolders: documents, people, projects, knowledge, code, media, conversations, data
- **.taxonomy.json** defines extraction hints per folder type
- Custom folders (e.g. AI_Ethics) auto-detected with generic extraction
- Conversations auto-saved on WebSocket disconnect
- Supports: .docx, .pdf, .xlsx, .pptx, .csv, .json, .txt, .md, code files, images (GPT-4o vision)

## RAG Implementation Plan (Document-Grounded Retrieval)

### Problem Statement

Currently, when a user asks "what does the document say about fairness in hiring?", the agent can only answer from concept labels and relationships in the knowledge graph (e.g., "Fairness --[related_to]--> Hiring"). **The original document text is discarded after concept extraction.** The agent cannot cite paragraphs, quote evidence, or reference specific sections from the ingested documents.

### Goal

Add a Retrieval-Augmented Generation (RAG) layer that stores document chunks with embeddings, retrieves relevant passages at query time, and injects them into the LLM context alongside the existing graph context. The agent should be able to cite specific text from documents.

### Architecture Overview

```
INGESTION (enhanced):
File → Read → Chunk → Embed → Store in ChromaDB
                 └──→ LLM Extract Concepts → Knowledge Graph (unchanged)

QUERY (enhanced):
User Message
     │
     ├──→ Embed query → ChromaDB similarity search → Top-K chunks
     │
     ├──→ Extract concepts → Fuzzy match → Spreading activation → Graph context
     │
     └──→ Merge: graph_context + document_chunks → LLM → Response
```

### Tech Stack Additions

| Component | Library | Why |
|---|---|---|
| Vector Store | **ChromaDB** (`chromadb`) | Embedded (no external service), Python-native, persistent, metadata filtering. Mirrors the NetworkX philosophy: self-contained, no infra deps. |
| Embeddings | **OpenAI `text-embedding-3-small`** via `langchain-openai` | Already have the API key and langchain-openai installed. 1536 dimensions, cheap ($0.02/1M tokens). |
| Text Splitting | **`langchain-text-splitters`** (`RecursiveCharacterTextSplitter`) | Smart chunking that respects paragraph/sentence boundaries. Already in the LangChain ecosystem. |

### Implementation Plan — 8 Steps

#### Step 1: Add Dependencies

```toml
# backend/pyproject.toml — add to dependencies
"chromadb>=1.0.0",
"langchain-text-splitters>=0.3.0",
```

No new frontend dependencies required.

#### Step 2: Create Document Store Module

**New file**: `backend/app/graph_kb/document_store.py`

```python
class DocumentStore:
    """ChromaDB-backed document chunk storage with semantic search."""

    def __init__(self, user_id: str, data_dir: Path):
        self._client = chromadb.PersistentClient(path=str(data_dir / "users" / user_id / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # ~150 tokens per chunk
            chunk_overlap=100,   # continuity between chunks
            separators=["\n\n", "\n", ". ", " "],
        )

    def ingest(self, content: str, metadata: dict) -> int:
        """Chunk, embed, and store a document. Returns number of chunks stored."""

    def search(self, query: str, top_k: int = 5, filter: dict = None) -> list[dict]:
        """Semantic search. Returns chunks with content, metadata, and distance."""

    def delete_by_source(self, source_file: str) -> None:
        """Remove all chunks from a specific source file (for re-ingestion)."""
```

**Chunk metadata schema**:
```python
{
    "source_file": "article_1.docx",      # original filename
    "folder": "AI_Ethics/Module_1/Sprint_1",  # folder path
    "folder_type": "documents",            # taxonomy category
    "chunk_index": 3,                      # position in document
    "total_chunks": 12,                    # total chunks in document
    "page": 2,                             # page number (PDF only)
}
```

#### Step 3: Integrate into Ingestion Pipeline

**Modify**: `backend/app/ingestion/ingest.py`

In `ingest_file()`, after reading the file content and before/in parallel with concept extraction:

```python
async def ingest_file(user_id, file_path, folder_context):
    content, image_b64 = read_file(file_path)

    # NEW: Store document chunks (text files only, not images)
    if content and not image_b64:
        doc_store = get_document_store(user_id)
        doc_store.ingest(content, metadata={
            "source_file": file_path.name,
            "folder": str(file_path.parent.relative_to(inputs_dir)),
            "folder_type": folder_context.get("type", "general"),
        })

    # EXISTING: Extract concepts for knowledge graph (unchanged)
    concepts = await extract_concepts_from_content(content, ...)
    mm.activate_concepts(concepts)
```

**Key design decision**: Chunking + embedding happens in parallel with concept extraction. They are independent processes — the graph captures structure (entities, relationships), the vector store captures content (paragraphs, evidence).

#### Step 4: Add RAG Retrieval to Agent Pipeline

**Modify**: `backend/app/agent/nodes.py`

Add a new LangGraph node `retrieve_documents` between `activate_memory` and `build_context`:

```
extract_concepts → activate_memory → retrieve_documents → build_context → generate_response → update_graph
```

```python
def retrieve_documents(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant document chunks via semantic search."""
    user_id = state["user_id"]
    messages = state["messages"]

    # Use the latest user message as the search query
    query = ""
    for msg in reversed(messages):
        if msg.role == "user":
            query = msg.content
            break

    if not query:
        return {"document_chunks": []}

    doc_store = get_document_store(user_id)
    chunks = doc_store.search(query, top_k=5)

    return {"document_chunks": chunks}
```

**Update `AgentState`**:
```python
class AgentState(TypedDict):
    messages: list[ChatMessage]
    user_id: str
    extracted_concepts: list[ExtractedConcept]
    activated_ids: list[str]
    graph_context: str
    document_chunks: list[dict]   # NEW
    response: str
    graph_diff: GraphDiff
    spreading_waves: list[list[str]]
```

#### Step 5: Merge Graph Context + Document Chunks

**Modify**: `build_context` in `backend/app/agent/nodes.py`

```python
def build_context(state: AgentState) -> dict[str, Any]:
    # EXISTING: graph context from spreading activation
    graph_context = mm.get_context_for_prompt(activated_ids)

    # NEW: document chunk context
    chunks = state.get("document_chunks", [])
    if chunks:
        doc_lines = ["\n## Relevant Document Excerpts\n"]
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source_file", "unknown")
            folder = chunk["metadata"].get("folder", "")
            doc_lines.append(f"### [{source}] (from {folder})")
            doc_lines.append(chunk["content"])
            doc_lines.append("")  # blank line separator
        graph_context += "\n".join(doc_lines)

    return {"graph_context": graph_context}
```

The LLM now receives:
```
## Relevant Knowledge Graph Context
### Concepts:
- **Fairness** (importance: 0.80, memory: LTM)
- **Demographic Parity** (importance: 0.65, memory: LTM)
### Relationships:
- Fairness --[related_to, strength: 0.45]--> Hiring

## Relevant Document Excerpts
### [article_3.docx] (from AI_Ethics/Module_1/Sprint_2)
"Fairness in hiring algorithms requires evaluating disparate impact
across protected groups. The 80% rule (four-fifths rule) states that
the selection rate for any group should be at least 80% of the rate
for the group with the highest selection rate..."
```

#### Step 6: Update System Prompt

**Modify**: `SYSTEM_PROMPT` in `backend/app/agent/nodes.py`

Add to the Response Guidelines section:
```
- When document excerpts are provided, **cite them directly**. Use quotes and reference
  the source file name. Prefer evidence from documents over general knowledge.
- Synthesize information across multiple sources when relevant.
```

#### Step 7: Persistence & Data Layout

```
data/users/{user_id}/
├── graph.json          # Knowledge graph (NetworkX, existing)
└── chroma/             # ChromaDB persistent storage (NEW)
    ├── chroma.sqlite3   # metadata + document mapping
    └── *.bin            # HNSW index files
```

**Storage estimates** for ~60 DOCX files:
- Avg 5 pages × 500 words = 2500 words per doc → ~5 chunks per doc
- 60 docs × 5 chunks = ~300 chunks
- Embedding dimension: 1536 (text-embedding-3-small)
- ChromaDB storage: ~5 MB (negligible)
- Embedding API cost: ~300 chunks × 150 tokens = 45K tokens ≈ $0.001

#### Step 8: Graph ↔ Document Cross-Linking (optional enhancement)

Store the source document reference in graph node metadata when a concept is extracted from a specific file:

```python
node.metadata["source_files"] = ["article_3.docx", "article_5.docx"]
```

This enables:
- Click a node in the frontend → see which documents mention it
- "Show me all documents related to this cluster"
- Bidirectional navigation: graph → docs, docs → graph

### Chunking Strategy Details

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 600 chars (~150 tokens) | Small enough for precise retrieval, large enough for coherent context |
| `chunk_overlap` | 100 chars | Prevents splitting mid-sentence, ensures continuity |
| `separators` | `["\n\n", "\n", ". ", " "]` | Respects document structure: paragraphs > lines > sentences > words |
| `top_k` | 5 | Balance between context richness and token budget |
| `embedding_model` | `text-embedding-3-small` | Best cost/performance ratio, 1536 dims, cosine similarity |

### What Changes vs What Stays

| Component | Status |
|---|---|
| Knowledge Graph (NetworkX) | **Unchanged** — still extracts concepts and relationships |
| Plasticity Engine | **Unchanged** — Hebbian, LTP, LTD, pruning still work |
| Spreading Activation | **Unchanged** — still builds graph context |
| Fuzzy Concept Matching | **Unchanged** — still activates related nodes |
| File Reader | **Unchanged** — same parsing for all formats |
| Concept Extraction LLM call | **Unchanged** — same extraction prompt |
| Agent Pipeline | **Extended** — new `retrieve_documents` node added |
| `build_context` | **Extended** — merges graph + document chunks |
| System Prompt | **Extended** — citation guidelines added |
| Frontend | **No changes needed** — markdown rendering already handles quotes/citations |

### Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `backend/pyproject.toml` | Modify | Add chromadb, langchain-text-splitters |
| `backend/app/graph_kb/document_store.py` | **Create** | ChromaDB wrapper: ingest, search, delete |
| `backend/app/ingestion/ingest.py` | Modify | Add chunk+embed step during file ingestion |
| `backend/app/agent/nodes.py` | Modify | Add `retrieve_documents` node, update `build_context` |
| `backend/app/agent/state.py` | Modify | Add `document_chunks` to AgentState |
| `backend/app/agent/graph_agent.py` | Modify | Add `retrieve_documents` to StateGraph |
| `backend/app/config.py` | Modify | Add RAG settings (chunk_size, top_k, embedding_model) |

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| Embedding API cost | text-embedding-3-small is $0.02/1M tokens; 60 docs ≈ $0.001 |
| ChromaDB disk space | ~5 MB for 300 chunks; negligible |
| Slow ingestion (embedding) | Batch embedding calls (ChromaDB handles this); async ingestion |
| Context too large for LLM | Cap document chunks at top_k=5 (~750 tokens); graph context already capped |
| Re-ingestion duplicates | `delete_by_source()` before re-ingesting same file |
| Image files (no text) | Skip chunking for images; only concept extraction (already handled) |

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
