import { create } from 'zustand';
import type { NodeData, EdgeData, GraphDiff, GraphSnapshot, GraphStats, ClusterInfo } from '../types/graph';
import type { ChatMessage } from '../types/messages';

interface GraphState {
  nodes: NodeData[];
  edges: EdgeData[];
  activatedNodes: Set<string>;
  spreadingWaves: string[][];
  messages: ChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
  stats: GraphStats | null;
  clusters: ClusterInfo[];
  nodeClusterMap: Record<string, number>;

  applyDiff: (diff: GraphDiff) => void;
  setFullGraph: (snapshot: GraphSnapshot) => void;
  addMessage: (msg: ChatMessage) => void;
  setConnected: (val: boolean) => void;
  setLoading: (val: boolean) => void;
  setStats: (stats: GraphStats) => void;
  clearGraph: () => void;
}

export const useGraphStore = create<GraphState>((set) => ({
  nodes: [],
  edges: [],
  activatedNodes: new Set<string>(),
  spreadingWaves: [],
  messages: [],
  isConnected: false,
  isLoading: false,
  stats: null,
  clusters: [],
  nodeClusterMap: {},

  applyDiff: (diff: GraphDiff) =>
    set((state) => {
      let newNodes = [...state.nodes];

      for (const nd of diff.node_diffs) {
        if (nd.action === 'added') {
          if (!newNodes.find((n) => n.id === nd.node.id)) {
            newNodes.push(nd.node);
          }
        } else if (nd.action === 'updated') {
          const idx = newNodes.findIndex((n) => n.id === nd.node.id);
          if (idx !== -1) {
            newNodes[idx] = nd.node;
          } else {
            newNodes.push(nd.node);
          }
        } else if (nd.action === 'removed') {
          newNodes = newNodes.filter((n) => n.id !== nd.node.id);
        }
      }

      let newEdges = [...state.edges];

      for (const ed of diff.edge_diffs) {
        if (ed.action === 'added') {
          if (
            !newEdges.find(
              (e) => e.source === ed.edge.source && e.target === ed.edge.target
            )
          ) {
            newEdges.push(ed.edge);
          }
        } else if (ed.action === 'updated') {
          const idx = newEdges.findIndex(
            (e) => e.source === ed.edge.source && e.target === ed.edge.target
          );
          if (idx !== -1) {
            newEdges[idx] = ed.edge;
          } else {
            newEdges.push(ed.edge);
          }
        } else if (ed.action === 'removed') {
          newEdges = newEdges.filter(
            (e) =>
              !(e.source === ed.edge.source && e.target === ed.edge.target)
          );
        }
      }

      return {
        nodes: newNodes,
        edges: newEdges,
        activatedNodes: new Set(diff.activated_nodes),
        spreadingWaves: diff.spreading_waves,
      };
    }),

  setFullGraph: (snapshot: GraphSnapshot) =>
    set({
      nodes: snapshot.nodes,
      edges: snapshot.edges,
      clusters: snapshot.clusters || [],
      nodeClusterMap: snapshot.node_cluster_map || {},
    }),

  addMessage: (msg: ChatMessage) =>
    set((state) => ({
      messages: [...state.messages, msg],
    })),

  appendToLastMessage: (chunk: string) =>
    set((state) => {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'assistant') {
        msgs[msgs.length - 1] = { ...last, content: last.content + chunk };
      } else {
        msgs.push({ role: 'assistant', content: chunk, timestamp: new Date().toISOString() });
      }
      return { messages: msgs, isLoading: false };
    }),

  setConnected: (val: boolean) =>
    set({ isConnected: val }),

  setLoading: (val: boolean) =>
    set({ isLoading: val }),

  setStats: (stats: GraphStats) =>
    set({ stats }),

  clearGraph: () =>
    set({
      nodes: [],
      edges: [],
      messages: [],
      activatedNodes: new Set<string>(),
      spreadingWaves: [],
      clusters: [],
      nodeClusterMap: {},
    }),
}));
