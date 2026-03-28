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
- **System prompt**: Instructs the LLM to think critically, challenge false premises, distinguish known from unknown, and ground responses in graph context. Includes citation guidelines for document excerpts.
- **Fuzzy concept matching**: When the user mentions a concept, `_find_related_nodes()` searches all existing graph nodes by substring and word overlap (not just exact ID match). This ensures the agent uses existing knowledge instead of creating empty stubs.
- **Relationship validation**: LLM-extracted relationship types are validated against the `RelationshipType` enum; invalid values fallback to `related_to` to prevent Pydantic crashes.
- **Contextual spreading activation**: Query-weighted propagation. `NodeEmbeddingCache` pre-computes embeddings for all node labels (persisted as `.npz`). At query time, cosine similarity between query and each neighbor modulates the spreading: `context_factor = 0.3 + 0.7 × similarity`. Relevant pathways get stronger activation, irrelevant ones get dampened. Cache auto-rebuilds when nodes change. Overhead: ~200ms per query (one embedding API call).
- **Semantic deduplication**: `NodeDeduplicator` embeds all node labels, computes cosine similarity matrix, merges nodes above threshold (default 0.85). Union-Find groups duplicates, primary node (highest importance) absorbs metadata/edges/source_files. Endpoint: `POST /api/graph/{user_id}/deduplicate`.
- **Cluster summaries**: LLM-generated 2-3 sentence descriptions per cluster. Uses top 10 nodes + relationships + sample document chunks. Endpoint: `POST /api/graph/{user_id}/summarize-clusters`. Summary shown in cluster legend tooltip.
- **Response visualization**: Activated nodes glow with `createActivatedPillTexture` (brighter fill, white border, outer glow). Camera flies to activation region. Auto-clears after 8 seconds. Uses existing `activatedNodes` from WebSocket `GraphDiff`.

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

## Future Implementation Plans

### Plan A: Semantic Deduplication of Nodes

#### Problem

The graph accumulates quasi-duplicate nodes from different extraction runs: "AI Fairness", "Fairness in AI", "fairness", "Algorithmic Fairness" are 4 nodes that represent the same concept. This fragments knowledge, weakens edge weights (connections are split across duplicates), and bloats the context sent to the LLM.

#### Solution: Embedding-Based Merge

Use the same OpenAI embedding model (`text-embedding-3-small`) to compute cosine similarity between all node labels. Nodes above a similarity threshold are merged into one.

#### Algorithm

```
1. Embed all node labels → N vectors of 1536 dims
2. Compute cosine similarity matrix (N×N)
3. Find pairs where similarity > 0.85
4. Build merge groups (union-find / connected components)
5. For each group:
   a. Pick "primary" node (highest importance)
   b. For each "secondary" node:
      - Merge source_files lists
      - Sum activation_counts
      - Take max(importance)
      - Redirect all edges: any edge pointing to/from secondary → point to primary
      - If redirect creates a duplicate edge, merge weights (keep max)
      - Delete secondary node
6. Recalculate clusters
7. Save graph
```

#### Implementation — 3 Files

**File 1: `backend/app/graph_kb/deduplication.py` (NEW)**

```python
class NodeDeduplicator:
    """Embedding-based node deduplication."""

    def __init__(self, graph: KnowledgeGraph, embedding_model: str = "text-embedding-3-small"):
        self._graph = graph
        self._embeddings = OpenAIEmbeddings(model=embedding_model, api_key=settings.openai_api_key)

    def find_duplicates(self, threshold: float = 0.85) -> list[list[str]]:
        """Return groups of node IDs that are semantically duplicate."""
        nodes = self._graph.all_nodes()
        # Skip folder nodes and very short labels
        candidates = [n for n in nodes if not n.metadata.get("is_folder") and len(n.label) > 2]

        # Batch embed all labels
        labels = [n.label for n in candidates]
        embeddings = self._embeddings.embed_documents(labels)

        # Cosine similarity matrix via numpy
        import numpy as np
        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        similarity = (matrix @ matrix.T) / (norms @ norms.T + 1e-10)

        # Union-Find to group connected duplicates
        parent = list(range(len(candidates)))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b): parent[find(a)] = find(b)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if similarity[i][j] >= threshold:
                    union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(len(candidates)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Return only groups with 2+ members
        return [
            [candidates[i].id for i in group]
            for group in groups.values()
            if len(group) > 1
        ]

    def merge_group(self, node_ids: list[str]) -> str:
        """Merge a group of duplicate nodes into one. Returns the surviving node ID."""
        nodes = [self._graph.get_node(nid) for nid in node_ids]
        nodes = [n for n in nodes if n is not None]
        if len(nodes) < 2:
            return node_ids[0]

        # Primary = highest importance
        primary = max(nodes, key=lambda n: n.importance)
        secondaries = [n for n in nodes if n.id != primary.id]

        for sec in secondaries:
            # Merge metadata
            merged_sources = list(set(
                primary.metadata.get("source_files", []) +
                sec.metadata.get("source_files", [])
            ))
            self._graph.update_node(primary.id,
                activation_count=primary.activation_count + sec.activation_count,
                importance=max(primary.importance, sec.importance),
                metadata={**primary.metadata, "source_files": merged_sources},
            )

            # Redirect edges
            for edge in self._graph.edges_of_node(sec.id):
                other = edge.target if edge.source == sec.id else edge.source
                if other == primary.id:
                    continue  # Skip self-loops
                # Check if primary already has this edge
                existing = self._graph.get_edge(primary.id, other) or self._graph.get_edge(other, primary.id)
                if existing:
                    # Merge weights
                    new_weight = min(1.0, max(existing.weight, edge.weight))
                    self._graph.update_edge(existing.source, existing.target, weight=new_weight)
                else:
                    # Redirect edge
                    new_source = primary.id if edge.source == sec.id else edge.source
                    new_target = primary.id if edge.target == sec.id else edge.target
                    self._graph.add_edge(EdgeData(
                        source=new_source, target=new_target,
                        weight=edge.weight, relationship=edge.relationship,
                        co_activation_count=edge.co_activation_count,
                    ))

            # Delete secondary
            self._graph.remove_node(sec.id)

        return primary.id

    def deduplicate(self, threshold: float = 0.85) -> dict:
        """Full deduplication pipeline. Returns stats."""
        groups = self.find_duplicates(threshold)
        merged_count = 0
        for group in groups:
            self.merge_group(group)
            merged_count += len(group) - 1
        return {"groups_found": len(groups), "nodes_merged": merged_count}
```

**File 2: `backend/app/api/routes.py` — Add endpoint**

```python
@router.post("/api/graph/{user_id}/deduplicate")
async def deduplicate_graph(user_id: str, threshold: float = 0.85):
    mm = MemoryManager(user_id)
    dedup = NodeDeduplicator(mm.graph)
    stats = dedup.deduplicate(threshold)
    mm.save()
    return stats
```

**File 3: `backend/app/graph_kb/knowledge_graph.py` — Add `update_edge` and `edges_of_node` if missing**

Need to verify these methods exist; `edges_of_node` already exists, may need `update_edge`.

#### Cost & Performance

| Metric | Value |
|---|---|
| Embedding cost | 500 nodes × ~5 tokens/label = 2500 tokens ≈ $0.00005 |
| Similarity matrix | 500×500 = 250K comparisons, ~10ms with numpy |
| Merge operations | Depends on duplicates found; typically 50-150 merges |
| Total time | ~3-5 seconds (dominated by embedding API call) |

#### When to Run

- **After ingestion**: Button in frontend "Deduplicate Graph"
- **Not during ingestion**: Would slow down file processing
- **Manual trigger**: POST `/api/graph/{user_id}/deduplicate?threshold=0.85`

---

### Plan B: Cluster Summaries

#### Problem

Clusters contain 20-300 nodes but have no human-readable summary. The cluster label is just the most important node's label (e.g., "AI Systems"), which doesn't describe the cluster's content. The agent has no high-level understanding of what each cluster represents.

#### Solution: LLM-Generated Cluster Summaries

After Louvain clustering, send each cluster's top nodes + relationships + sample document chunks to the LLM for a 2-3 sentence summary.

#### Architecture

```
Louvain Detection → For each cluster:
    Top 10 nodes (by importance)
    + Inter-cluster relationships
    + Sample chunks from nodes' source_files
    → LLM: "Summarize this knowledge cluster in 2-3 sentences"
    → Store summary in ClusterInfo
    → Send to frontend + use in agent context
```

#### Implementation — 5 Files

**Step 1: Extend ClusterInfo model**

`backend/app/models/graph.py` — Add `summary` field:

```python
class ClusterData(BaseModel):
    id: int
    label: str
    node_ids: list[str]
    size: int
    summary: str = ""  # NEW: LLM-generated cluster summary
```

`frontend/src/types/graph.ts` — Add to ClusterInfo:

```typescript
export interface ClusterInfo {
  id: number;
  label: string;
  node_ids: string[];
  size: number;
  summary?: string;  // NEW
}
```

**Step 2: Generate summaries after clustering**

`backend/app/graph_kb/knowledge_graph.py` — New method:

```python
CLUSTER_SUMMARY_PROMPT = """\
You are summarizing a thematic cluster from a knowledge graph.

Cluster label: {label}
Top concepts (by importance):
{concepts}

Key relationships:
{relationships}

{document_context}

Write a 2-3 sentence summary of what this cluster of knowledge represents.
Focus on the main theme, key insights, and how the concepts connect.
Be specific — mention actual concept names.
"""

async def generate_cluster_summaries(self, doc_store=None) -> None:
    """Generate LLM summaries for each cluster."""
    clusters = self.detect_clusters()
    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.2)

    for cluster in clusters:
        if cluster["size"] < 3:
            continue

        # Get top 10 nodes
        cluster_nodes = [self.get_node(nid) for nid in cluster["node_ids"]]
        cluster_nodes = [n for n in cluster_nodes if n]
        top_nodes = sorted(cluster_nodes, key=lambda n: n.importance, reverse=True)[:10]

        # Get relationships between cluster nodes
        cluster_id_set = set(cluster["node_ids"])
        edges = []
        for nid in cluster["node_ids"][:20]:
            for edge in self.edges_of_node(nid):
                if edge.source in cluster_id_set and edge.target in cluster_id_set:
                    src = self.get_node(edge.source)
                    tgt = self.get_node(edge.target)
                    if src and tgt:
                        edges.append(f"{src.label} --[{edge.relationship}]--> {tgt.label}")

        # Sample document chunks if doc_store available
        doc_context = ""
        if doc_store:
            source_files = set()
            for n in top_nodes[:5]:
                source_files.update(n.metadata.get("source_files", []))
            if source_files:
                chunks = doc_store.search_by_sources(
                    cluster["label"], list(source_files)[:3], top_k=2
                )
                if chunks:
                    doc_context = "Sample document excerpts:\n" + "\n".join(
                        c["content"][:200] for c in chunks
                    )

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            label=cluster["label"],
            concepts="\n".join(f"- {n.label} (importance: {n.importance:.0%})" for n in top_nodes),
            relationships="\n".join(edges[:10]) if edges else "No direct relationships found.",
            document_context=doc_context,
        )

        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        cluster["summary"] = response.content.strip()
```

**Step 3: Cache summaries**

Store summaries in `data/users/{user_id}/cluster_summaries.json`. Regenerate only when:
- Cluster composition changes (compare node_ids hash)
- User explicitly requests regeneration
- After deduplication (clusters change)

**Step 4: Use in agent context**

In `build_context()`, prepend cluster summaries before individual concepts:

```python
## Knowledge Overview (Cluster Summaries)
- **AI Systems (373 nodes)**: This cluster covers the technical foundations of AI systems...
- **Equal Opportunity (75 nodes)**: Focuses on fairness metrics and measurement approaches...

## Relevant Knowledge Graph Context
### Concepts:
...
```

This gives the LLM a "30,000 foot view" before diving into specific nodes.

**Step 5: Frontend — show summary in cluster legend tooltip**

In `BrainGraph.tsx`, modify the cluster legend to show summary on hover:

```tsx
<div title={cluster.summary || ''}>
  {cluster.label} {cluster.size}
</div>
```

Or an expandable section below each cluster in the legend.

#### Cost

| Metric | Value |
|---|---|
| LLM calls | 1 per cluster, ~200 tokens input + 60 tokens output |
| For 10 clusters | ~2600 tokens ≈ $0.01 |
| Regeneration | Only on cluster change (cached) |

---

### Plan C: Response Visualization in the Graph

#### Problem

When the agent responds, the user can't see which nodes/edges were used as context. The 3D graph is static — there's no visual connection between the chat response and the graph. The brain metaphor breaks because you can't see "which neurons fired."

#### Solution: Highlight Activated Nodes on Each Response

When the agent responds, visually highlight the nodes and edges that were used to build the context. Show a pulse/glow animation that fades over time, making the graph feel alive.

#### Current State (already exists but unused)

- **Backend sends `activated_nodes`** in `GraphDiff.activated_nodes` (list of node IDs)
- **Backend sends `spreading_activation`** with `waves` (list of node ID lists per depth level)
- **Frontend stores `activatedNodes`** in Zustand as `Set<string>`
- **Frontend stores `spreadingWaves`** in Zustand
- **BUT**: `nodeThreeObject` callback does NOT check `activatedNodes` — all nodes look the same regardless of activation state

#### Implementation — 3 Changes in BrainGraph.tsx

**Change 1: Create activated pill texture variant**

New function `createActivatedPillTexture` — same as `createPillTexture` but with:
- Brighter background (opacity 1.0 instead of 0.92)
- Glowing border: 2px solid with the node's color at full opacity
- Outer glow: larger rounded rect with semi-transparent color behind

```typescript
function createActivatedPillTexture(label: string, color: string): THREE.CanvasTexture {
  // Same as createPillTexture but:
  // - Outer glow: ctx.shadowColor = color, ctx.shadowBlur = 15
  // - Brighter fill: globalAlpha = 1.0
  // - Thicker border: lineWidth = 2, strokeStyle = color at full alpha
  // Cache separately: key = `${label}|${color}|activated`
}
```

**Change 2: Modify `nodeThreeObject` to check activation**

```typescript
const nodeThreeObject = useCallback((node: GraphNode) => {
  const color = getNodeColor(node.id);
  const isActivated = activatedNodes.has(node.id);

  const texture = isActivated
    ? createActivatedPillTexture(node.label, color)
    : createPillTexture(node.label, color, node.memory_type === 'stm');

  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);

  const canvas = texture.image as HTMLCanvasElement;
  const aspect = canvas.width / canvas.height;
  // Activated nodes are slightly larger
  const scale = isActivated ? (3 + node.importance * 1.5) : (2 + node.importance * 1);
  sprite.scale.set(scale * aspect, scale, 1);

  return sprite;
}, [getNodeColor, activatedNodes]);  // Add activatedNodes to deps
```

**Important**: Adding `activatedNodes` to the deps of `nodeThreeObject` means all node objects are recreated when activations change. For 500 nodes this is acceptable (Three.js sprite creation is fast). The texture cache prevents re-drawing canvases.

**Change 3: Auto-clear activations after delay**

```typescript
// Clear activation highlighting after 8 seconds
useEffect(() => {
  if (activatedNodes.size === 0) return;
  const timer = setTimeout(() => {
    useGraphStore.setState({ activatedNodes: new Set() });
  }, 8000);
  return () => clearTimeout(timer);
}, [activatedNodes]);
```

**Change 4: Camera fly-to activated region**

When new activations arrive, optionally fly the camera to the center of the activated nodes:

```typescript
useEffect(() => {
  if (activatedNodes.size === 0 || !fgRef.current) return;

  // Compute center of activated nodes
  const activated = graphData.nodes.filter(n => activatedNodes.has(n.id));
  if (activated.length === 0) return;

  let cx = 0, cy = 0, cz = 0;
  for (const n of activated) { cx += (n.x || 0); cy += (n.y || 0); cz += (n.z || 0); }
  cx /= activated.length; cy /= activated.length; cz /= activated.length;

  // Fly camera to see the activation
  const dist = Math.max(50, activated.length * 3);
  fgRef.current.cameraPosition(
    { x: cx + dist * 0.5, y: cy + dist * 0.3, z: cz + dist },
    { x: cx, y: cy, z: cz },
    1000
  );
}, [activatedNodes, graphData.nodes]);
```

#### Visual Effect Timeline

```
t=0s  User sends message
t=1-3s Agent processes (loading dots in chat)
t=3s  Response arrives:
      → Activated nodes GLOW (bright border, larger scale)
      → Camera flies to activated region
      → Spreading waves animate outward
t=8s  Glow fades (activatedNodes cleared)
      → Nodes return to normal appearance
```

#### Performance Considerations

| Concern | Mitigation |
|---|---|
| Re-creating all node sprites on activation | Texture cache: only new textures for ~20 activated nodes. Existing cached textures reused for the other ~480 |
| Camera animation during interaction | Only flies to activation region, not a full reset. User can still rotate/zoom during animation |
| Memory from activated textures | Activated textures are cached with `|activated` key suffix. Max ~50 extra textures = ~5MB |

#### Files to Modify

| File | Change |
|---|---|
| `frontend/src/components/BrainGraph/BrainGraph.tsx` | `createActivatedPillTexture`, modify `nodeThreeObject`, add activation clear timer, add camera fly-to |

No backend changes needed — the activation data is already sent via WebSocket.

---

### Plan D: Contextual Spreading Activation

#### Problem

The current spreading activation treats all edges equally regardless of what the user asked. When the user asks "How does fairness impact hiring practices?", activation spreads from "fairness" to ALL neighbors with the same formula:

```
spread_value = parent_activation × edge_weight × SPREADING_FACTOR
```

This means "fairness → EU AI Act" gets the same treatment as "fairness → hiring discrimination", even though only the latter is relevant to the query. The agent ends up with a context full of tangentially related concepts that dilute the answer quality.

In the brain, attention modulates which neural pathways activate. The same neuron (concept) can activate different downstream pathways depending on the current cognitive task. This is what's missing.

#### Solution: Query-Weighted Spreading

Modulate the spreading factor by the **semantic similarity between the user's query and each candidate neighbor**. Nodes that are semantically related to the query get stronger activation; irrelevant neighbors get dampened.

**New formula:**

```
context_factor = 0.3 + 0.7 × cosine_similarity(query_embedding, neighbor_embedding)
spread_value = parent_activation × edge_weight × SPREADING_FACTOR × context_factor
```

The `0.3 + 0.7 × similarity` mapping ensures:
- Minimum factor = 0.3 (even irrelevant neighbors get some activation — preserves serendipitous discovery)
- Maximum factor = 1.0 (highly relevant neighbors get full activation)
- The query "steers" which pathways light up

**Example — Query: "How does fairness impact hiring?"**

From seed "fairness" (activation=1.0):

| Neighbor | edge_weight | query_similarity | context_factor | spread_value |
|---|---|---|---|---|
| hiring discrimination | 0.4 | 0.92 | 0.94 | 1.0 × 0.4 × 0.5 × 0.94 = **0.188** |
| demographic parity | 0.5 | 0.65 | 0.76 | 1.0 × 0.5 × 0.5 × 0.76 = **0.190** |
| bias | 0.4 | 0.70 | 0.79 | 1.0 × 0.4 × 0.5 × 0.79 = **0.158** |
| EU AI Act | 0.3 | 0.30 | 0.51 | 1.0 × 0.3 × 0.5 × 0.51 = **0.077** |
| machine learning | 0.2 | 0.25 | 0.48 | 1.0 × 0.2 × 0.5 × 0.48 = **0.048** (below threshold!) |

Result: "machine learning" gets pruned, "EU AI Act" barely activates, but "hiring discrimination" and "demographic parity" get strong activation. The context sent to the LLM is focused on what matters for THIS specific question.

#### Architecture: Node Embedding Cache

To avoid calling the embedding API for every node during every query, pre-compute and cache all node label embeddings:

```
data/users/{user_id}/
├── graph.json              # Knowledge graph
├── chroma/                 # Document chunks (RAG)
└── node_embeddings.npz     # Cached node label embeddings (NEW)
```

**Cache structure** (numpy .npz file):
```python
{
    "ids": np.array(["fairness", "bias", "hiring", ...]),      # node IDs, shape (N,)
    "embeddings": np.array([[0.012, -0.034, ...], ...]),        # shape (N, 1536)
    "version": hash(sorted(node_ids))                           # cache invalidation key
}
```

**Cache invalidation**: Rebuild when `hash(sorted(current_node_ids)) != cached_version`. This happens when nodes are added, removed, or merged (deduplication). The rebuild is one batch API call (~500 labels = ~2500 tokens ≈ $0.00005).

**At query time**: Only ONE embedding call (the user's query). All similarity computations are numpy dot products (microseconds for 500 nodes).

#### Implementation — 5 Steps

**Step 1: Create `NodeEmbeddingCache`**

**New file**: `backend/app/graph_kb/embedding_cache.py`

```python
class NodeEmbeddingCache:
    """Persistent cache for node label embeddings."""

    def __init__(self, user_id: str, data_dir: Path):
        self._path = data_dir / "users" / user_id / "node_embeddings.npz"
        self._embeddings_model = OpenAIEmbeddings(
            model=settings.rag_embedding_model,
            api_key=settings.openai_api_key,
        )
        self._ids: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._version: str = ""
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if self._path.exists():
            data = np.load(self._path, allow_pickle=True)
            self._ids = data["ids"]
            self._embeddings = data["embeddings"]
            self._version = str(data["version"])

    def _save(self) -> None:
        """Persist cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._path,
            ids=self._ids,
            embeddings=self._embeddings,
            version=self._version,
        )

    def _compute_version(self, node_ids: list[str]) -> str:
        """Hash of sorted node IDs for cache invalidation."""
        import hashlib
        return hashlib.md5(",".join(sorted(node_ids)).encode()).hexdigest()

    def ensure_fresh(self, graph: KnowledgeGraph) -> None:
        """Rebuild cache if stale (nodes added/removed since last build)."""
        all_nodes = graph.all_nodes()
        current_ids = [n.id for n in all_nodes if not n.metadata.get("is_folder")]
        current_version = self._compute_version(current_ids)

        if current_version == self._version and self._embeddings is not None:
            return  # Cache is fresh

        # Rebuild: batch embed all node labels
        labels = []
        ids = []
        for n in all_nodes:
            if not n.metadata.get("is_folder") and len(n.label) > 1:
                labels.append(n.label)
                ids.append(n.id)

        if not labels:
            self._ids = np.array([])
            self._embeddings = np.zeros((0, 1536))
            self._version = current_version
            self._save()
            return

        raw_embeddings = self._embeddings_model.embed_documents(labels)
        self._ids = np.array(ids)
        self._embeddings = np.array(raw_embeddings)
        self._version = current_version
        self._save()
        logger.info("Rebuilt embedding cache: %d nodes for version %s", len(ids), current_version[:8])

    def get_similarities(self, query: str) -> dict[str, float]:
        """Compute cosine similarity between query and all cached node labels.

        Returns: {node_id: similarity} for all cached nodes.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return {}

        query_embedding = np.array(self._embeddings_model.embed_query(query))

        # Cosine similarity: dot(q, E) / (||q|| * ||E||)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return {}

        similarities = (self._embeddings @ query_embedding) / (norms * query_norm + 1e-10)

        return {self._ids[i]: float(similarities[i]) for i in range(len(self._ids))}
```

**Step 2: Modify `SpreadingActivation` to accept query similarities**

**Modify**: `backend/app/graph_kb/spreading.py`

Add an optional `query_similarities` parameter to `spread()`:

```python
def spread(
    self,
    seed_ids: list[str],
    max_depth: int = 2,
    min_activation: float = 0.05,
    query_similarities: dict[str, float] | None = None,  # NEW
) -> tuple[dict[str, float], list[list[str]]]:

    # ... inside the loop:
    for edge in self.graph.edges_of_node(node_id):
        neighbor_id = edge.target if edge.source == node_id else edge.source
        if neighbor_id in visited:
            continue

        # Query-weighted context factor
        if query_similarities and neighbor_id in query_similarities:
            sim = query_similarities[neighbor_id]
            context_factor = 0.3 + 0.7 * max(0.0, sim)
        else:
            context_factor = 1.0  # No query info → uniform spreading

        spread_value = parent_activation * edge.weight * settings.spreading_factor * context_factor
        # ...
```

When `query_similarities` is `None` (e.g., during ingestion or non-query spreading), the behavior is identical to the current implementation (`context_factor = 1.0`). This ensures backward compatibility.

**Step 3: Thread the query through the pipeline**

The user's query needs to flow from `build_context` → `get_context_for_prompt` → `spread_and_collect_context` → `spread()`.

**Modify**: `backend/app/graph_kb/memory.py`

```python
def spread_and_collect_context(
    self, seed_ids: list[str], max_depth: int = 2, query: str | None = None,
) -> tuple[list[NodeData], list[EdgeData], list[list[str]]]:
    """Spread activation from seeds, optionally weighted by query relevance."""

    query_similarities = None
    if query:
        from .embedding_cache import get_embedding_cache
        cache = get_embedding_cache(self.user_id)
        cache.ensure_fresh(self.graph)
        query_similarities = cache.get_similarities(query)

    activations, waves = self.spreading.spread(
        seed_ids, max_depth=max_depth, query_similarities=query_similarities,
    )
    relevant_ids = list(activations.keys())
    sub = self.graph.subgraph(relevant_ids)
    return sub.nodes, sub.edges, waves


def get_context_for_prompt(self, seed_ids: list[str], query: str | None = None) -> str:
    """Build textual context, optionally using query-weighted spreading."""
    nodes, edges, _ = self.spread_and_collect_context(seed_ids, max_depth=2, query=query)
    # ... rest unchanged
```

**Modify**: `backend/app/agent/nodes.py` — `build_context`

```python
def build_context(state: AgentState) -> dict[str, Any]:
    user_id = state["user_id"]
    activated_ids = state.get("activated_ids", [])

    # Extract user query for contextual spreading
    query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.role == "user":
            query = msg.content
            break

    context = ""
    if activated_ids:
        mm = _get_memory_manager(user_id)
        context = mm.get_context_for_prompt(activated_ids, query=query)  # Pass query

    # ... rest unchanged (append document chunks)
```

**Step 4: Add `get_embedding_cache` factory**

In `embedding_cache.py`, add module-level cache (same pattern as `get_document_store`):

```python
_caches: dict[str, NodeEmbeddingCache] = {}

def get_embedding_cache(user_id: str) -> NodeEmbeddingCache:
    if user_id not in _caches:
        _caches[user_id] = NodeEmbeddingCache(user_id, settings.data_dir)
    return _caches[user_id]
```

**Step 5: Update AgentState (optional)**

No change needed — the query is already available via `state["messages"]`. No new state fields required.

#### Data Flow Diagram

```
User: "How does fairness impact hiring?"
     │
     ▼
extract_concepts → ["fairness", "hiring"]
     │
     ▼
activate_memory → activated_ids = ["fairness", "hiring", "bias", ...]
     │
     ▼
retrieve_documents → chunks from ChromaDB (semantic + graph-guided)
     │
     ▼
build_context:
  │
  ├── query = "How does fairness impact hiring?"
  │
  ├── NodeEmbeddingCache.ensure_fresh(graph)
  │     └── if stale: batch embed 500 labels → save .npz
  │
  ├── NodeEmbeddingCache.get_similarities(query)
  │     └── 1 API call (query embedding) + numpy dot products
  │     └── returns {node_id: similarity} for all 500 nodes
  │
  ├── SpreadingActivation.spread(seeds, query_similarities=sims)
  │     └── context_factor = 0.3 + 0.7 × similarity
  │     └── "hiring discrimination" gets 0.94 factor (strong)
  │     └── "EU AI Act" gets 0.51 factor (dampened)
  │     └── "machine learning" gets 0.48 factor → below threshold → pruned
  │
  └── Build text context from filtered subgraph + document chunks
     │
     ▼
generate_response → LLM receives focused, relevant context
```

#### Performance Analysis

| Operation | Cost | When |
|---|---|---|
| Embed all node labels (500) | ~$0.00005, ~1s | Only when cache is stale (nodes added/removed) |
| Embed user query | ~$0.000001, ~200ms | Every query |
| Cosine similarity (500 nodes) | ~0.01ms (numpy) | Every query |
| Cache disk I/O (.npz) | ~5ms | Load on first query, save on rebuild |

**Total overhead per query**: ~200ms (dominated by query embedding API call).
**Total overhead for cache rebuild**: ~1s + $0.00005 (rare — only after ingestion or dedup).

#### Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `backend/app/graph_kb/embedding_cache.py` | **Create** | NodeEmbeddingCache: build, persist, query similarities |
| `backend/app/graph_kb/spreading.py` | Modify | Add `query_similarities` parameter, compute `context_factor` |
| `backend/app/graph_kb/memory.py` | Modify | Thread `query` through `spread_and_collect_context` and `get_context_for_prompt` |
| `backend/app/agent/nodes.py` | Modify | Extract user query in `build_context`, pass to `get_context_for_prompt` |

#### Backward Compatibility

All changes are additive:
- `spreading.spread()` without `query_similarities` behaves identically to current code
- `get_context_for_prompt()` without `query` behaves identically to current code
- Ingestion pipeline is unaffected (doesn't pass query)
- Cache is lazily built — no startup cost if never used

#### Edge Cases

| Case | Handling |
|---|---|
| New node not in cache | `context_factor = 1.0` (uniform, same as current behavior) |
| Empty query | `query_similarities = None` → uniform spreading |
| Cache file missing | Rebuilds automatically on first `ensure_fresh()` call |
| Node label too short (e.g., "AI") | Included in cache; embedding model handles short strings fine |
| Deduplication merges nodes | Cache version changes → auto-rebuild on next query |

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
