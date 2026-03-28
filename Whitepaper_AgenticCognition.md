# Agentic Cognition Through Brain-Inspired Knowledge Graphs

## A Whitepaper on the Cognitive Architecture of NeuroGraph

**Abstract.** This paper presents NeuroGraph, a cognitive architecture that reconstructs the fundamental mechanisms of biological cognition — memory formation, synaptic plasticity, spreading activation, and attentional gating — using modern software primitives. Rather than treating an AI agent as a stateless query-response system, NeuroGraph endows it with a persistent, evolving knowledge graph that learns, forgets, consolidates, and reasons in ways that parallel the human brain. We describe the theoretical foundations, implementation decisions, and empirical results of each cognitive subsystem, detailing the specific libraries and techniques employed to achieve biologically plausible yet computationally tractable artificial cognition.

---

## 1. Introduction: The Case for Cognitive Architecture

Contemporary Large Language Models (LLMs) exhibit remarkable linguistic capabilities but lack persistent memory, structured knowledge, and the ability to learn from interaction. Each conversation begins tabula rasa. The model cannot recall what it discussed yesterday, cannot strengthen its understanding of frequently encountered concepts, and cannot prune knowledge that has become irrelevant.

NeuroGraph addresses this gap by constructing a **cognitive architecture** around the LLM — a system of interconnected modules that collectively simulate the memory, learning, and reasoning processes of a biological brain. The LLM serves as the language cortex; the surrounding architecture provides the hippocampus (short-term memory), the neocortex (long-term consolidated knowledge), the synaptic plasticity engine (learning), and the attentional system (context selection).

The guiding metaphor is not merely decorative. Each architectural decision maps to a specific neuroscientific mechanism, and each mechanism is implemented using a specific library or technique chosen for its alignment with the biological analogy.

---

## 2. The Structural Substrate: Knowledge Graph as Neural Tissue

### 2.1 Biological Analogy

The brain stores information not in isolated cells but in **networks of interconnected neurons**. Knowledge is distributed across connections (synapses), and the strength of each connection encodes the association between concepts. There is no central database — meaning emerges from topology.

### 2.2 Implementation

NeuroGraph implements its neural tissue as a directed graph using **NetworkX** (`networkx>=3.3`), a Python library for graph data structures and algorithms. Each node represents a concept (analogous to a neuron or neural assembly), and each directed edge represents a semantic relationship (analogous to a synapse).

**Node schema** (concept / neural assembly):
```
id: str                    # Deterministic identifier from label
label: str                 # Human-readable concept name
category: NodeCategory     # person | place | concept | action | object | event
activation: float [0, 1]   # Current activation level (membrane potential)
importance: float [0, 1]   # Long-term potentiation accumulation
memory_type: MemoryType    # STM (hippocampal) | LTM (neocortical)
activation_count: int      # Total lifetime activations
metadata: dict             # Provenance: source_files, is_folder, etc.
```

**Edge schema** (synapse):
```
source, target: str            # Pre- and post-synaptic node IDs
weight: float [0, 1]          # Synaptic strength
relationship: RelationshipType # Semantic type (10 categories)
co_activation_count: int      # Hebbian co-occurrence count
```

**Design choice**: NetworkX over Neo4j. The knowledge graph of a single user is small enough (~300-500 nodes) to reside entirely in memory, eliminating network latency and external dependencies. Persistence is achieved through JSON serialization to `data/users/{user_id}/graph.json`. This mirrors the brain's self-contained nature — all knowledge is local.

**Data validation**: **Pydantic v2** (`pydantic>=2.9.0`) enforces strict typing on all graph data models, preventing structural corruption. Relationship types are constrained to a validated enum of 10 semantic categories (`related_to`, `is_a`, `part_of`, `causes`, `has_property`, `used_for`, `located_in`, `created_by`, `depends_on`, `opposite_of`), with automatic fallback for LLM-generated types outside this vocabulary.

---

## 3. Memory Systems: Hippocampus and Neocortex

### 3.1 Biological Analogy

The brain maintains at least two distinct memory systems. The **hippocampus** handles short-term, episodic memory — rapidly encoding new experiences with high plasticity but limited capacity. The **neocortex** stores long-term, semantic memory — slowly consolidated, highly stable, and effectively unlimited in capacity. During sleep, the hippocampus replays recent experiences, transferring significant ones to the neocortex through a process called **memory consolidation**.

### 3.2 Implementation

NeuroGraph implements this dual memory system in `MemoryManager` (`backend/app/graph_kb/memory.py`):

**Short-Term Memory (STM)** — hippocampal analog:
- Capacity: 7 concepts (Miller's Law, 1956)
- Decay rate: 0.3 per turn (rapid forgetting)
- New concepts enter STM with activation = 1.0
- When capacity is exceeded, lowest-activation nodes are evicted or, if sufficiently activated (≥3 uses), promoted to LTM

**Long-Term Memory (LTM)** — neocortical analog:
- Capacity: unlimited
- Decay rate: 0.01 per turn (slow forgetting)
- Persistence: survives across sessions, serialized to JSON
- Consolidation threshold: importance ≥ 0.7 OR activation_count ≥ 5

**Consolidation process**:
```python
for node in stm_nodes:
    if node.importance >= CONSOLIDATION_THRESHOLD:  # 0.7
        promote_to_ltm(node)
    elif node.activation_count >= 5:
        promote_to_ltm(node)  # Repeated use = consolidation
```

This mechanism ensures that concepts mentioned once in passing remain transient (STM), while concepts that recur across conversations accumulate importance and consolidate into permanent knowledge (LTM) — precisely mirroring hippocampal-neocortical transfer.

---

## 4. Synaptic Plasticity: Learning Through Use

### 4.1 Biological Analogy

The brain learns through **synaptic plasticity** — the ability of synapses to strengthen or weaken based on activity. Donald Hebb's foundational principle (1949) states: *"Neurons that fire together wire together."* This Hebbian learning is complemented by **Long-Term Potentiation** (LTP, strengthening through repeated use), **Long-Term Depression** (LTD, weakening through disuse), and **synaptic pruning** (elimination of very weak connections).

### 4.2 Implementation

The `PlasticityEngine` (`backend/app/graph_kb/plasticity.py`) implements four plasticity mechanisms:

| Mechanism | Biological Process | Implementation | Parameter |
|---|---|---|---|
| **Hebbian Learning** | Co-activation strengthens synapses | Edge weight += 0.1 when both endpoints activate in the same turn | `HEBBIAN_LEARNING_RATE = 0.1` |
| **LTP** | Repeated stimulation strengthens neurons | Node importance += 0.05 per activation | `LTP_RATE = 0.05` |
| **LTD** | Disuse weakens connections | Edge weight -= 0.01 per turn for inactive edges | `LTD_DECAY_RATE = 0.01` |
| **Synaptic Pruning** | Very weak synapses are eliminated | Edges with weight < 0.05 are removed | `PRUNING_THRESHOLD = 0.05` |

These mechanisms operate after every conversation turn, applied by `PlasticityEngine.apply_plasticity(activated_ids)`. The result is a knowledge graph that continuously reshapes itself — strengthening pathways that are used, weakening those that are not, and pruning connections that fall below the threshold of relevance.

**Concept sanitization**: Before plasticity is applied, extracted concepts are filtered against a blocklist of sensitive terms (passwords, credentials, API keys) to prevent contamination of the knowledge substrate — analogous to the blood-brain barrier filtering harmful substances.

---

## 5. Spreading Activation: Neural Signal Propagation

### 5.1 Biological Analogy

When a neuron fires, it doesn't activate in isolation. The signal propagates through connected synapses to neighboring neurons, which may in turn fire and propagate further. This **spreading activation** (Collins & Loftus, 1975) is how the brain retrieves associated concepts: thinking of "doctor" activates "hospital," "medicine," "stethoscope" — each with decreasing activation as distance from the source increases.

### 5.2 Implementation

The `SpreadingActivation` class (`backend/app/graph_kb/spreading.py`) implements a BFS-based propagation algorithm:

```
activation(neighbor) = activation(parent) × edge_weight × SPREADING_FACTOR × context_factor
```

- **Depth**: 2 hops from seed nodes
- **Minimum activation**: 0.05 (propagation stops below this threshold)
- **Spreading factor**: 0.5 (each hop halves the signal)
- **Hub limiting**: Maximum 30 edges traversed per node (prevents combinatorial explosion from hub nodes like "Fairness" with 150+ connections)

### 5.3 Contextual Spreading: Attentional Gating

A critical enhancement over classical spreading activation is **contextual modulation** — the ability to steer propagation based on the current query. In the brain, **attention** modulates which neural pathways activate: the same word "bank" activates financial concepts when thinking about money, but river-related concepts when thinking about nature.

NeuroGraph implements this through `NodeEmbeddingCache` (`backend/app/graph_kb/embedding_cache.py`):

1. **Pre-computation**: All node labels are embedded using **OpenAI text-embedding-3-small** (1536 dimensions) via **LangChain OpenAI** (`langchain-openai>=0.2.0`). Embeddings are cached as a **NumPy** `.npz` file, rebuilding only when LTM nodes change.

2. **Query-time modulation**: The user's query is embedded, and cosine similarity is computed against all cached node embeddings (a single matrix multiplication via **NumPy**):

```
context_factor = 0.3 + 0.7 × cosine_similarity(query_embedding, neighbor_embedding)
```

This maps to a range of [0.3, 1.0]: even irrelevant neighbors receive some activation (preserving serendipitous discovery), but relevant neighbors receive substantially more. The result is that the same graph, activated by the same seed nodes, produces different subgraphs depending on the question — exactly as attention modulates neural processing.

**Performance**: Pre-computation is ~1 second for 500 nodes (one batch API call). Query-time modulation is <1ms (NumPy dot product). Cache invalidation is version-based, using an MD5 hash of sorted LTM node IDs — STM changes (which occur every turn) do not trigger expensive rebuilds.

---

## 6. Perception: Concept Extraction as Sensory Processing

### 6.1 Biological Analogy

The brain's sensory cortices transform raw stimuli (photons, sound waves) into meaningful representations (objects, words, faces). Similarly, NeuroGraph must transform raw text into structured concepts and relationships that can enter the knowledge graph.

### 6.2 Implementation

Concept extraction is performed by **GPT-4o** via **LangChain** (`langchain>=0.3.0`) and **langchain-openai** (`langchain-openai>=0.2.0`). The LLM receives the user's message along with a structured extraction prompt and returns JSON:

```json
{
  "concepts": [
    {
      "label": "Demographic Parity",
      "category": "concept",
      "relationships": [
        {"target_label": "Fairness Metrics", "relationship": "is_a"}
      ]
    }
  ]
}
```

**Fuzzy concept matching**: After extraction, the system searches existing graph nodes for semantically related concepts using substring and word-overlap matching (`_find_related_nodes()`). This prevents the creation of redundant stub nodes and ensures that new information connects to existing knowledge — analogous to how the brain integrates new experiences with prior memories rather than creating isolated traces.

**Semantic deduplication**: Over time, concept extraction generates quasi-duplicates ("AI Fairness," "Fairness in AI," "Algorithmic Fairness"). The `NodeDeduplicator` (`backend/app/graph_kb/deduplication.py`) periodically merges these using embedding-based cosine similarity (threshold 0.85), consolidating fragmented knowledge into unified representations. This mirrors the brain's memory reconsolidation process, where related memories are merged during sleep.

---

## 7. Episodic Memory: Document-Grounded Retrieval (RAG)

### 7.1 Biological Analogy

The hippocampus stores not just abstract concepts but **episodic memories** — specific experiences tied to context (time, place, sensory details). When recalling a concept, the brain can retrieve both the abstract meaning (semantic memory) and specific episodes where it was encountered (episodic memory).

### 7.2 Implementation

NeuroGraph implements episodic memory through **Retrieval-Augmented Generation (RAG)**, storing the original document text alongside the extracted concepts:

- **Vector store**: **ChromaDB** (`chromadb>=1.0.0`) with cosine similarity HNSW indexing
- **Embeddings**: **OpenAI text-embedding-3-small** (1536 dimensions)
- **Text splitting**: **LangChain Text Splitters** (`langchain-text-splitters>=0.3.0`) with paragraph-aware chunking (800 chars, 150 overlap, 7-level separator hierarchy)
- **Document summaries**: Each ingested document generates a 2-3 sentence LLM summary stored as a special "summary chunk" — providing a holistic view alongside fragmentary episodic details

**Current state**: 4,844 chunks + 60 document summaries from 84 ingested files.

### 7.3 Dual Retrieval Strategy

At query time, two retrieval strategies operate in parallel:

1. **Semantic retrieval**: The query is embedded and compared against all chunk embeddings via cosine similarity (top 7 results)
2. **Graph-guided retrieval**: Activated graph nodes' `source_files` metadata is used to search within specific documents — combining structural knowledge (which documents are relevant) with semantic search (which passages within those documents match)

This dual strategy mirrors how the brain combines semantic memory (knowing that a concept is associated with a particular context) with episodic recall (retrieving the specific details from that context).

### 7.4 Self-Hallucination Loop Prevention

A critical discovery during development: conversation transcripts saved by the agent were being ingested as RAG documents. When the agent previously hallucinated a definition, that hallucination was saved, chunked, embedded, and later retrieved as "evidence" — creating a self-reinforcing confabulation loop. The fix: conversation files (`chat_*.md`) are excluded from RAG ingestion. Document excerpts are designated as the **primary source of truth**, overriding graph label inferences in the system prompt.

---

## 8. Executive Function: Query Routing as Attentional Control

### 8.1 Biological Analogy

The prefrontal cortex implements **executive function** — deciding how to allocate cognitive resources based on the nature of the task. A simple reflex (touching a hot surface) doesn't require deep reasoning; a complex decision (career change) engages the full cognitive apparatus. The brain dynamically allocates processing depth based on task demands.

### 8.2 Implementation

NeuroGraph implements executive function through **query routing** using **GPT-4o-mini** (a fast, cheap classifier):

| Route | Cognitive Analog | Pipeline | Latency |
|---|---|---|---|
| `greeting` | Reflex arc | Skip all processing → direct LLM response | ~2s |
| `simple` | Familiar recall | RAG search only, no graph activation | ~6s |
| `complex` | Deep reasoning | Full pipeline: extraction, activation, spreading, RAG, grounding | ~8s |

The classifier achieves **10/10 accuracy** on test cases, correctly routing "Hola" (greeting), "What is fairness?" (simple), and multi-concept analytical questions (complex).

**Cost**: One GPT-4o-mini call per query (~10 tokens, ~$0.000002). The savings from skipping pipeline stages for greetings far exceed this cost.

---

## 9. Language Processing: Multilingual Comprehension

### 9.1 Implementation

NeuroGraph detects the query language using heuristic analysis (`_detect_language()`) and injects a response-language instruction into the context. This ensures the agent responds in the user's language regardless of the language of the source documents (which are predominantly English).

Supported languages: Spanish, English, Chinese, Arabic, French, Portuguese, German. Detection accuracy: **10/10** in testing.

---

## 10. Metacognition: Grounding and Self-Monitoring

### 10.1 Biological Analogy

Metacognition is the brain's ability to monitor its own cognitive processes — knowing what it knows, recognizing uncertainty, and calibrating confidence. Humans naturally say "I'm not sure about this" or "I remember reading about that." LLMs, by contrast, tend to confabulate with high confidence.

### 10.2 Implementation

NeuroGraph implements three metacognitive mechanisms:

**A. Grounding check** — post-processing verification:

After the LLM generates a response, a deterministic scanner (`_grounding_check()`) extracts specific claims using regex patterns:
- Article numbers: `Article 52`, `Section 3.2`
- Statistics: `73%`, `42.5%`
- Author citations: `(Smith et al., 2023)`
- Study references: `study by Holstein`
- Specific years: `in 2019`

Each claim is verified against the context that was provided to the LLM. If >2 claims cannot be found in the context, a transparency disclaimer is appended. This mechanism is **deterministic** (zero extra LLM calls) and catches the most dangerous form of hallucination: fabrication of specific, verifiable-sounding details.

**B. Out-of-domain detection**:

In `build_context()`, the system checks whether any activated nodes are established LTM knowledge (importance > 0.2). If only ephemeral STM stubs exist (created from the current query's concept extraction) and no document chunks were retrieved, a disclaimer is injected: *"This topic is not in my knowledge base."*

**C. Confidence visualization**:

Each node's visual opacity in the 3D graph encodes its groundedness:
```
confidence = (source_files_count / 3) × 0.6 + (activation_count / 5) × 0.4
opacity = 0.4 + confidence × 0.55
```

Nodes backed by multiple source documents and frequent activation render solidly; stub nodes with no provenance render semi-transparently. This provides the user with an intuitive visual signal of knowledge reliability.

---

## 11. The Agent Pipeline: Orchestrated Cognition

### 11.1 Orchestration Framework

The cognitive pipeline is orchestrated by **LangGraph** (`langgraph>=0.2.0`), a stateful graph-based workflow engine. Each cognitive subsystem is implemented as a **node** in a `StateGraph`, with typed state flowing between them:

```
User Message
     │
     ▼
┌─────────────────┐
│ Query Routing    │  ← Executive function (GPT-4o-mini)
│ (classify_query) │
└────────┬────────┘
         │
    ┌────┴────┐
    │ greeting │──→ Direct response (skip pipeline)
    └────┬────┘
         │ simple / complex
         ▼
┌─────────────────┐
│ Extract Concepts │  ← Sensory processing (GPT-4o)
│ + Fuzzy Match    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Activate Memory  │  ← Hebbian learning, LTP, consolidation
│ + Plasticity     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Docs    │  ← Episodic memory (ChromaDB)
│ Semantic + Graph │     Dual retrieval strategy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Build Context    │  ← Attentional selection
│ + Language Detect│     Contextual spreading activation
│ + Out-of-Domain  │     Mode D: chunks-only (primary)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate Response│  ← Language cortex (GPT-4o, streaming)
│ + Grounding Check│     Post-processing verification
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update Graph     │  ← Memory consolidation
│ + Save + Broadcast│    WebSocket graph diff to frontend
└─────────────────┘
```

### 11.2 Streaming: Real-Time Cognitive Output

The generation phase uses **token-level WebSocket streaming**: the pre-generation pipeline (extraction, activation, retrieval, context building) runs in ~1.5 seconds, after which LLM tokens stream to the frontend as they are generated. The user begins reading the response while it is still being produced — reducing perceived latency by ~70%.

**Backend**: **FastAPI** (`fastapi>=0.115.0`) with **uvicorn** (`uvicorn[standard]>=0.30.0`) provides async WebSocket support. **LangChain's `astream()`** enables async token iteration from OpenAI.

**Frontend**: **React 18** with **Zustand** (`zustand>=5.0`) for reactive state management. Each `response_chunk` WebSocket message appends to the displayed message via `appendToLastMessage()`.

---

## 12. Visualization: The Brain Scanner

### 12.1 3D Neural Visualization

The knowledge graph is rendered as an interactive 3D force-directed graph using **react-force-graph-3d** (Three.js-based) with **d3-force-3d** for physics simulation.

- **Node rendering**: Canvas2D pill-shaped textures cached as **Three.js** `Sprite` objects (billboard rendering — always face the camera)
- **Initial layout**: Fibonacci sphere distribution for even 3D placement
- **Force system**: Charge repulsion (-15), link distance (8), cluster attraction (0.3), gravity (0.15)
- **Cluster detection**: **NetworkX Louvain community detection** (resolution=1.5) with LLM-generated cluster summaries
- **Activation glow**: When the agent responds, activated nodes render with a bright glow effect (`createActivatedPillTexture`) and the camera automatically flies to the activation region

### 12.2 Confidence Encoding

Node opacity encodes groundedness: nodes with multiple source documents and high activation counts render solidly, while stub nodes render semi-transparently. This provides an immediate visual indicator of knowledge reliability across the entire graph.

---

## 13. Immune System: Security and Integrity

### 13.1 Prompt Injection Defense (Blood-Brain Barrier)

An **immutable identity block** at the top of the system prompt prevents persona override attacks. Input sanitization detects 15+ common injection patterns and logs warnings without blocking (to avoid false positives on legitimate queries).

### 13.2 Concept Sanitization (Immune Response)

A blocklist filters sensitive terms (passwords, credentials, API keys) from being created as graph nodes, preventing contamination of the knowledge substrate.

### 13.3 User ID Validation (Cell Membrane)

Regex validation `[a-zA-Z0-9_-]{1,64}` on all endpoints prevents path traversal and directory escape attacks.

---

## 14. Empirical Results

### 14.1 Stress Test Summary

| Metric | Result |
|---|---|
| Query routing accuracy | 10/10 |
| Language detection accuracy | 10/10 |
| Prompt injection blocked | Yes (refused pirate persona) |
| Concept sanitization | Yes (0 sensitive nodes created) |
| Hallucination: specific claims | 0 fabricated statistics, 0 fake citations |
| Concurrent queries (5 simultaneous) | All responded, 7s total |
| Graph integrity | 0 self-loops, 0 dangling edges, 0 duplicates |
| Query latency (greeting) | ~2.0s |
| Query latency (simple) | ~6.0s |
| Query latency (complex) | ~7.5s |
| Cost per query | ~$0.019 |

### 14.2 A/B/C/D Context Mode Testing

Empirical testing revealed that the graph context (node labels + relationships) can **introduce noise** when labels are ambiguous. The concept "EquiHire" was consistently misidentified as "acqui-hire" (acquiring a company for its talent) when graph context was included, because the node's label and `related_to → acquisition` edge misled the LLM. With document chunks only (Mode D), the correct definition ("recruitment startup in the EU") was consistently returned.

| Mode | Context | Accuracy | Latency |
|---|---|---|---|
| A: Graph + Chunks (3) | 1,305 tokens | 6/8 | 2.0s |
| B: Chunks only (5) | 563 tokens | 6/8 | 2.4s |
| C: Chunks only (10) | 1,126 tokens | 7/8 | 1.8s |
| **D: Chunks only (7)** | **793 tokens** | **7/8** | **1.9s** |

Mode D was selected as the default: best accuracy, minimal tokens, fast latency. Mode A is preserved and can be reactivated via `CONTEXT_MODE=graph_and_chunks` in `.env`.

---

## 15. Technology Stack Summary

| Cognitive Function | Biological Analog | Library/Tool |
|---|---|---|
| Neural tissue (graph) | Connectome | NetworkX |
| Data integrity | DNA error correction | Pydantic v2 |
| Short/Long-term memory | Hippocampus / Neocortex | MemoryManager (custom) |
| Synaptic plasticity | Hebbian learning, LTP, LTD | PlasticityEngine (custom) |
| Signal propagation | Spreading activation | SpreadingActivation (custom) |
| Attentional gating | Prefrontal attention | NodeEmbeddingCache + NumPy |
| Sensory processing | Visual/auditory cortex | GPT-4o via LangChain |
| Episodic memory | Hippocampal replay | ChromaDB + OpenAI Embeddings |
| Executive function | Prefrontal routing | GPT-4o-mini classifier |
| Language processing | Broca's / Wernicke's areas | GPT-4o (streaming) |
| Metacognition | Self-monitoring | Grounding check (regex) |
| Memory consolidation | Sleep replay | Deduplication + pruning |
| Immune system | Blood-brain barrier | Prompt defense + sanitization |
| Brain scanner | fMRI / EEG | react-force-graph-3d + Three.js |
| Orchestration | Neural pathways | LangGraph StateGraph |
| Real-time signaling | Neurotransmitter release | WebSocket (FastAPI) |
| State management | Working memory buffer | Zustand (React) |

---

## 16. Conclusion

NeuroGraph demonstrates that biologically-inspired cognitive architecture can meaningfully enhance LLM-based agents. By implementing persistent memory, synaptic plasticity, spreading activation, attentional gating, and metacognitive self-monitoring, the system produces responses that are more grounded, more contextually relevant, and more transparent about their provenance than a stateless LLM alone.

The key insight from empirical testing is that **structure and content serve different roles**: the knowledge graph excels at capturing relationships and guiding retrieval, while document chunks provide the factual evidence that grounds responses. The optimal configuration (Mode D) uses the graph as a silent guide — directing which documents to search — while letting the original text speak for itself in the LLM prompt.

Future directions include temporal decay (Ebbinghaus forgetting curve), spaced repetition bonuses, and competitive inhibition between nodes — mechanisms that would further close the gap between artificial and biological cognition.

---

*NeuroGraph — Built with neuroscience-inspired algorithms, validated through adversarial stress testing.*
