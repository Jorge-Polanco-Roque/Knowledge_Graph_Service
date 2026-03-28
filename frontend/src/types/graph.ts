export type NodeCategory = 'person' | 'place' | 'concept' | 'action' | 'object' | 'event';
export type MemoryType = 'stm' | 'ltm';
export type DiffAction = 'added' | 'updated' | 'removed';

export interface NodeData {
  id: string;
  label: string;
  category: NodeCategory;
  activation: number;
  importance: number;
  memory_type: MemoryType;
  first_seen: string;
  last_activated: string;
  activation_count: number;
  metadata: Record<string, any>;
}

export interface EdgeData {
  source: string;
  target: string;
  weight: number;
  relationship: string;
  co_activation_count: number;
  last_co_activated: string;
}

export interface NodeDiff {
  action: DiffAction;
  node: NodeData;
}

export interface EdgeDiff {
  action: DiffAction;
  edge: EdgeData;
}

export interface GraphDiff {
  node_diffs: NodeDiff[];
  edge_diffs: EdgeDiff[];
  activated_nodes: string[];
  spreading_waves: string[][];
}

export interface ClusterInfo {
  id: number;
  label: string;
  node_ids: string[];
  size: number;
  summary?: string;
}

export interface GraphSnapshot {
  nodes: NodeData[];
  edges: EdgeData[];
  clusters?: ClusterInfo[];
  node_cluster_map?: Record<string, number>;
}

export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  stm_nodes: number;
  ltm_nodes: number;
  stm_capacity: string;
  top_concepts: { label: string; importance: number }[];
}
