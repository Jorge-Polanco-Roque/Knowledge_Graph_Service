import { motion, AnimatePresence } from 'framer-motion';
import { useGraphStore } from '../../hooks/useGraphStore';

const CATEGORY_COLORS: Record<string, string> = {
  person: '#3b82f6',
  place: '#10b981',
  concept: '#8b5cf6',
  action: '#f97316',
  object: '#ec4899',
  event: '#f59e0b',
};

export function MetricsBar() {
  const stats = useGraphStore((s) => s.stats);
  const nodes = useGraphStore((s) => s.nodes);
  const isConnected = useGraphStore((s) => s.isConnected);

  // Count STM vs LTM from live nodes
  const stmCount = nodes.filter((n) => n.memory_type === 'stm').length;
  const ltmCount = nodes.filter((n) => n.memory_type === 'ltm').length;

  return (
    <div
      style={{
        padding: '8px 20px',
        background: 'rgba(10, 10, 26, 0.95)',
        borderTop: '1px solid rgba(0, 212, 255, 0.1)',
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        fontSize: '12px',
        color: '#8b949e',
        flexShrink: 0,
        overflowX: 'auto',
        whiteSpace: 'nowrap',
        backdropFilter: 'blur(8px)',
      }}
    >
      {!stats && !isConnected ? (
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <motion.div
            style={{ width: 6, height: 6, borderRadius: '50%', background: '#f97316' }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          Connecting to NeuroGraph...
        </span>
      ) : (
        <>
          <Metric icon="🧠" label="Nodes" value={nodes.length} color="#00d4ff" />
          <Divider />
          <Metric icon="🔗" label="Edges" value={stats?.total_edges ?? 0} color="#8b5cf6" />
          <Divider />
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span>⚡</span>
            <span>STM</span>
            <span style={{ color: '#00d4ff', fontWeight: 600 }}>{stmCount}</span>
            <span style={{ color: '#4a5568' }}>/</span>
            <span>7</span>
          </span>
          <Divider />
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span>🏛️</span>
            <span>LTM</span>
            <span style={{ color: '#10b981', fontWeight: 600 }}>{ltmCount}</span>
          </span>
          <Divider />
          {stats && stats.top_concepts.length > 0 && (
            <>
              <span>🏆 Top:</span>
              <AnimatePresence mode="popLayout">
                {stats.top_concepts.map((concept) => {
                  // Find the node to get its category color
                  const node = nodes.find(
                    (n) => n.label.toLowerCase() === concept.label.toLowerCase()
                  );
                  const color = node
                    ? CATEGORY_COLORS[node.category] || '#8b5cf6'
                    : '#8b5cf6';
                  return (
                    <motion.span
                      key={concept.label}
                      initial={{ opacity: 0, scale: 0.8, x: -4 }}
                      animate={{ opacity: 1, scale: 1, x: 0 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        padding: '1px 8px',
                        borderRadius: '10px',
                        background: `${color}18`,
                        border: `1px solid ${color}35`,
                        color,
                        fontSize: '11px',
                        fontWeight: 500,
                      }}
                    >
                      {concept.label}
                      <span style={{ marginLeft: '4px', opacity: 0.5, fontSize: '10px' }}>
                        {(concept.importance * 100).toFixed(0)}%
                      </span>
                    </motion.span>
                  );
                })}
              </AnimatePresence>
            </>
          )}
        </>
      )}
    </div>
  );
}

function Metric({ icon, label, value, color }: { icon: string; label: string; value: number; color: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
      <span>{icon}</span>
      <span>{label}</span>
      <motion.span
        key={value}
        initial={{ scale: 1.3, color }}
        animate={{ scale: 1, color: '#e6edf3' }}
        transition={{ duration: 0.4 }}
        style={{ fontWeight: 600, minWidth: '16px', textAlign: 'center' }}
      >
        {value}
      </motion.span>
    </span>
  );
}

function Divider() {
  return <span style={{ color: 'rgba(0, 212, 255, 0.15)' }}>|</span>;
}
