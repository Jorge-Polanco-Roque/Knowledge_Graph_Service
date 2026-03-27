import { useCallback, useRef, useEffect, useMemo, useState } from 'react';
import ForceGraph3D, { type ForceGraphMethods } from 'react-force-graph-3d';
import * as THREE from 'three';
import { useGraphStore } from '../../hooks/useGraphStore';
import { useThemeStore } from '../../hooks/useThemeStore';
import type { NodeData } from '../../types/graph';

const CLUSTER_COLORS = [
  '#6366f1', '#f43f5e', '#14b8a6', '#f97316',
  '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16',
  '#eab308', '#3b82f6', '#10b981', '#ef4444',
  '#a855f7', '#22d3ee', '#f59e0b', '#64748b',
];

interface GraphNode extends NodeData {
  x?: number; y?: number; z?: number;
}
interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  weight: number;
  relationship: string;
}

// Cache textures so we don't recreate every frame
const textureCache = new Map<string, THREE.CanvasTexture>();

function createPillTexture(label: string, color: string, isSTM: boolean): THREE.CanvasTexture {
  const key = `${label}|${color}|${isSTM}`;
  const cached = textureCache.get(key);
  if (cached) return cached;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const fontSize = 28;
  ctx.font = `600 ${fontSize}px Inter, -apple-system, system-ui, sans-serif`;
  const displayLabel = label.length > 18 ? label.slice(0, 16) + '\u2026' : label;
  const textWidth = ctx.measureText(displayLabel).width;
  const padX = 20;
  const padY = 12;
  const w = textWidth + padX * 2;
  const h = fontSize + padY * 2;

  canvas.width = w * 2;
  canvas.height = h * 2;
  ctx.scale(2, 2);

  const radius = h / 2;

  // Pill background
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.92;
  ctx.beginPath();
  ctx.roundRect(2, 2, w - 4, h - 4, radius - 2);
  ctx.fill();

  // Border
  ctx.globalAlpha = 0.3;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(2, 2, w - 4, h - 4, radius - 2);
  ctx.stroke();

  // STM dashed
  if (isSTM) {
    ctx.globalAlpha = 0.6;
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, radius);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Text
  ctx.globalAlpha = 1;
  ctx.font = `600 ${fontSize}px Inter, -apple-system, system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ffffff';
  ctx.fillText(displayLabel, w / 2, h / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  textureCache.set(key, texture);
  return texture;
}

export function BrainGraph() {
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const { colors: themeColors, theme } = useThemeStore();
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const activatedNodes = useGraphStore((s) => s.activatedNodes);
  const clusters = useGraphStore((s) => s.clusters);
  const nodeClusterMap = useGraphStore((s) => s.nodeClusterMap);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDimensions({ width: Math.floor(width), height: Math.floor(height) });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const getNodeColor = useCallback((nodeId: string): string => {
    const cid = nodeClusterMap[nodeId];
    if (cid !== undefined) return CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
    return '#8b5cf6';
  }, [nodeClusterMap]);

  const graphData = useMemo(() => ({
    nodes: nodes.map((n) => ({ ...n })) as GraphNode[],
    links: edges.map((e) => ({ source: e.source, target: e.target, weight: e.weight, relationship: e.relationship })),
  }), [nodes, edges]);

  // Auto-fit
  const prevCountRef = useRef(0);
  useEffect(() => {
    if (nodes.length > 0 && nodes.length !== prevCountRef.current) {
      prevCountRef.current = nodes.length;
      const t1 = setTimeout(() => fgRef.current?.zoomToFit(600, 80), 2000);
      const t2 = setTimeout(() => fgRef.current?.zoomToFit(600, 80), 5000);
      return () => { clearTimeout(t1); clearTimeout(t2); };
    }
  }, [nodes.length]);

  // Forces
  useEffect(() => {
    if (!fgRef.current) return;
    fgRef.current.d3Force('charge')?.strength(-30);
    fgRef.current.d3Force('link')?.distance(12);

    // Custom gravity: pull all nodes toward origin
    const gravityForce = (alpha: number) => {
      const str = 0.08 * alpha;
      for (const node of graphData.nodes as any[]) {
        node.vx = (node.vx || 0) - (node.x || 0) * str;
        node.vy = (node.vy || 0) - (node.y || 0) * str;
        node.vz = (node.vz || 0) - (node.z || 0) * str;
      }
    };
    fgRef.current.d3Force('gravity', gravityForce);
  }, [graphData]);

  // Cluster force
  useEffect(() => {
    if (!fgRef.current || Object.keys(nodeClusterMap).length === 0) return;
    const fg = fgRef.current;
    const clusterForce = (alpha: number) => {
      const centers: Record<number, { x: number; y: number; z: number; count: number }> = {};
      for (const node of graphData.nodes) {
        const cid = nodeClusterMap[node.id];
        if (cid === undefined) continue;
        if (!centers[cid]) centers[cid] = { x: 0, y: 0, z: 0, count: 0 };
        centers[cid].x += (node.x || 0);
        centers[cid].y += (node.y || 0);
        centers[cid].z += (node.z || 0);
        centers[cid].count++;
      }
      for (const cid in centers) { const c = centers[cid]; c.x /= c.count; c.y /= c.count; c.z /= c.count; }
      const strength = 0.6 * alpha;
      for (const node of graphData.nodes as any[]) {
        const cid = nodeClusterMap[node.id];
        if (cid === undefined || !centers[cid]) continue;
        const c = centers[cid];
        node.vx = (node.vx || 0) + (c.x - (node.x || 0)) * strength;
        node.vy = (node.vy || 0) + (c.y - (node.y || 0)) * strength;
        node.vz = (node.vz || 0) + (c.z - (node.z || 0)) * strength;
      }
    };
    fg.d3Force('cluster', clusterForce);
    return () => { fgRef.current?.d3Force('cluster', null); };
  }, [nodeClusterMap, graphData.nodes]);

  // Node Three.js object — billboard sprite with cached pill texture
  const nodeThreeObject = useCallback((node: GraphNode) => {
    const color = getNodeColor(node.id);
    const texture = createPillTexture(node.label, color, node.memory_type === 'stm');

    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);

    // Size proportional to canvas aspect ratio, importance scales slightly
    const canvas = texture.image as HTMLCanvasElement;
    const aspect = canvas.width / canvas.height;
    const scale = 5 + node.importance * 2;
    sprite.scale.set(scale * aspect, scale, 1);

    return sprite;
  }, [getNodeColor]);

  // Link styling
  const linkColor = useCallback((link: GraphLink) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id;
    const cid = nodeClusterMap[sourceId];
    const alpha = Math.max(0.06, Math.min(link.weight * 0.4, 0.4)) * (theme === 'light' ? 2.5 : 1);
    if (cid !== undefined) {
      const c = CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
      const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${Math.min(alpha, 0.7)})`;
    }
    return theme === 'light' ? `rgba(100,100,130,${alpha})` : `rgba(0,212,255,${alpha})`;
  }, [nodeClusterMap, theme]);

  const linkWidth = useCallback((link: GraphLink) => 0.3 + link.weight * 2, []);

  // Tooltip
  const nodeLabel = useCallback((node: GraphNode) => {
    const memColor = node.memory_type === 'stm' ? '#00d4ff' : '#10b981';
    const memLabel = node.memory_type === 'stm' ? 'Short-term' : 'Long-term';
    return `<div style="background:rgba(10,10,26,0.95);border:1px solid rgba(0,212,255,0.25);border-radius:10px;padding:10px 14px;font-family:Inter,system-ui;font-size:12px;color:#e6edf3;min-width:140px;">
      <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${getNodeColor(node.id)};">${node.label}</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;"><span style="color:#8b949e;">Category</span><span>${node.category}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;"><span style="color:#8b949e;">Memory</span><span style="color:${memColor};">${memLabel}</span></div>
      <div style="display:flex;justify-content:space-between;"><span style="color:#8b949e;">Importance</span><span>${(node.importance * 100).toFixed(0)}%</span></div>
    </div>`;
  }, [getNodeColor]);

  // Zoom to a specific cluster
  const handleZoomToCluster = useCallback((clusterId: number) => {
    if (!fgRef.current) return;
    const cluster = clusters.find((c) => c.id === clusterId);
    if (!cluster) return;

    // Find positions of all nodes in this cluster
    const clusterNodes = graphData.nodes.filter((n) => cluster.node_ids.includes(n.id));
    if (clusterNodes.length === 0) return;

    // Compute center
    let cx = 0, cy = 0, cz = 0;
    for (const n of clusterNodes) { cx += (n.x || 0); cy += (n.y || 0); cz += (n.z || 0); }
    cx /= clusterNodes.length; cy /= clusterNodes.length; cz /= clusterNodes.length;

    // Compute radius of cluster
    let maxDist = 0;
    for (const n of clusterNodes) {
      const d = Math.sqrt(((n.x || 0) - cx) ** 2 + ((n.y || 0) - cy) ** 2 + ((n.z || 0) - cz) ** 2);
      if (d > maxDist) maxDist = d;
    }

    // Position camera at distance proportional to cluster size
    const dist = Math.max(maxDist * 2.5, 80);
    fgRef.current.cameraPosition(
      { x: cx, y: cy, z: cz + dist },
      { x: cx, y: cy, z: cz },
      1000
    );
  }, [clusters, graphData.nodes]);

  // Controls
  const handleZoomToFit = useCallback(() => fgRef.current?.zoomToFit(600, 80), []);
  const handleResetCamera = useCallback(() => fgRef.current?.cameraPosition({ x: 0, y: 0, z: 300 }, { x: 0, y: 0, z: 0 }, 800), []);
  const handleZoomIn = useCallback(() => {
    const cam = fgRef.current?.camera();
    if (cam) { const d = cam.position.clone().normalize(); fgRef.current?.cameraPosition(cam.position.clone().sub(d.multiplyScalar(50)), undefined, 300); }
  }, []);
  const handleZoomOut = useCallback(() => {
    const cam = fgRef.current?.camera();
    if (cam) { const d = cam.position.clone().normalize(); fgRef.current?.cameraPosition(cam.position.clone().add(d.multiplyScalar(50)), undefined, 300); }
  }, []);

  const btnStyle = {
    width: 34, height: 34, borderRadius: '8px',
    border: `1px solid ${theme === 'dark' ? 'rgba(0,212,255,0.2)' : 'rgba(0,0,0,0.15)'}`,
    background: theme === 'dark' ? 'rgba(10,10,26,0.7)' : 'rgba(255,255,255,0.8)',
    color: theme === 'dark' ? '#00d4ff' : '#6366f1',
    cursor: 'pointer' as const, display: 'flex' as const,
    alignItems: 'center' as const, justifyContent: 'center' as const,
    backdropFilter: 'blur(4px)', transition: 'all 0.15s', fontSize: '14px',
  };

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative', background: themeColors.bgCanvas }}>
      {/* Header */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, padding: '12px 16px',
        background: theme === 'dark' ? 'linear-gradient(180deg,rgba(10,10,26,0.8) 0%,transparent 100%)' : 'linear-gradient(180deg,rgba(240,242,245,0.9) 0%,transparent 100%)',
        zIndex: 10, pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '20px' }}>🧠</span>
          <span style={{ fontSize: '13px', fontWeight: 600, background: 'linear-gradient(90deg,#00d4ff,#8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Knowledge Graph</span>
        </div>
        <div style={{ display: 'flex', gap: '12px', fontSize: '11px', color: themeColors.textSecondary }}>
          <span>{nodes.length} nodes</span><span>{edges.length} edges</span>
        </div>
      </div>

      {/* Controls */}
      <div style={{ position: 'absolute', top: 50, right: 12, display: 'flex', flexDirection: 'column', gap: '6px', zIndex: 10 }}>
        <button onClick={useThemeStore.getState().toggleTheme} title={theme === 'dark' ? 'Light' : 'Dark'} style={{ ...btnStyle, color: theme === 'dark' ? '#f59e0b' : '#6366f1', fontSize: '16px' }}>{theme === 'dark' ? '\u2600' : '\u263E'}</button>
        <button onClick={handleZoomToFit} title="Fit" style={btnStyle}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg></button>
        <button onClick={handleResetCamera} title="Reset" style={btnStyle}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg></button>
        <div style={{ height: 4 }} />
        <button onClick={handleZoomIn} title="Zoom in" style={{ ...btnStyle, fontSize: '18px', fontWeight: 700 }}>+</button>
        <button onClick={handleZoomOut} title="Zoom out" style={{ ...btnStyle, fontSize: '18px', fontWeight: 700 }}>&minus;</button>
      </div>

      {/* Cluster legend */}
      {clusters.length > 0 && (
        <div style={{ position: 'absolute', bottom: 12, left: 12, display: 'flex', flexDirection: 'column', gap: '4px', zIndex: 10, maxHeight: '200px', overflowY: 'auto' }}>
          <div style={{ fontSize: '10px', color: themeColors.textSecondary, fontWeight: 600, textTransform: 'uppercase' as const, letterSpacing: '0.5px', marginBottom: '2px' }}>Clusters</div>
          {clusters.filter((c) => c.size > 1).sort((a, b) => b.size - a.size).slice(0, 12).map((cluster) => {
            const col = CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length];
            return (
              <div key={cluster.id} onClick={() => handleZoomToCluster(cluster.id)} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '3px 10px', borderRadius: '10px', background: `${col}15`, border: `1px solid ${col}40`, fontSize: '11px', color: col, cursor: 'pointer', transition: 'all 0.15s' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = `${col}30`; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = `${col}15`; }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: col, flexShrink: 0 }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const, maxWidth: '160px' }}>{cluster.label}</span>
                <span style={{ color: themeColors.textSecondary, fontSize: '9px', flexShrink: 0 }}>{cluster.size}</span>
              </div>
            );
          })}
        </div>
      )}

      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor={themeColors.bgCanvas}
        showNavInfo={false}
        nodeThreeObject={nodeThreeObject as any}
        nodeLabel={nodeLabel as any}
        linkColor={linkColor as any}
        linkWidth={linkWidth as any}
        linkOpacity={theme === 'light' ? 0.8 : 0.5}
        linkDirectionalParticles={1}
        linkDirectionalParticleWidth={1.5}
        linkDirectionalParticleSpeed={0.004}
        warmupTicks={200}
        cooldownTicks={300}
        d3AlphaDecay={0.01}
        d3VelocityDecay={0.2}
        onNodeDragEnd={(node: any) => { node.fx = node.x; node.fy = node.y; node.fz = node.z; }}
        onEngineStop={() => { if (fgRef.current && nodes.length > 0) fgRef.current.zoomToFit(600, 80); }}
      />

      {nodes.length === 0 && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', textAlign: 'center', pointerEvents: 'none' }}>
          <div style={{ fontSize: '64px', marginBottom: '16px', opacity: 0.15 }}>🧠</div>
          <div style={{ fontSize: '16px', fontWeight: 600, color: themeColors.textSecondary }}>Neural Space Empty</div>
          <div style={{ fontSize: '12px', color: themeColors.textSecondary, opacity: 0.5 }}>Start a conversation to grow your knowledge graph</div>
        </div>
      )}
    </div>
  );
}
