import { useEffect } from 'react';
import { SplitView } from './components/Layout/SplitView';
import { BrainGraph } from './components/BrainGraph/BrainGraph';
import { ChatPanel } from './components/Chat/ChatPanel';
import { InputBar } from './components/Chat/InputBar';
import { IngestButton } from './components/Chat/IngestButton';
import { MetricsBar } from './components/Metrics/MetricsBar';
import { useWebSocket } from './hooks/useWebSocket';
import { useGraphStore } from './hooks/useGraphStore';
import { useThemeStore } from './hooks/useThemeStore';

const USER_ID = 'default-user';

function App() {
  const { sendMessage, isConnected } = useWebSocket(USER_ID);
  const setFullGraph = useGraphStore((s) => s.setFullGraph);
  const setStats = useGraphStore((s) => s.setStats);
  const isLoading = useGraphStore((s) => s.isLoading);
  const { colors: tc, theme } = useThemeStore();

  // Fetch initial graph state on mount
  useEffect(() => {
    fetch(`/api/graph/${USER_ID}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setFullGraph(data);
      })
      .catch(() => {});

    fetch(`/api/graph/${USER_ID}/stats`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setStats(data);
      })
      .catch(() => {});
  }, [setFullGraph, setStats]);

  const chatRight = (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: theme === 'dark' ? 'rgba(5, 5, 16, 0.6)' : 'rgba(255, 255, 255, 0.95)',
      color: tc.text,
    }}>
      {/* Header */}
      <div
        style={{
          padding: '14px 18px',
          borderBottom: `1px solid ${tc.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: theme === 'dark' ? 'rgba(10, 10, 26, 0.5)' : 'rgba(245, 247, 250, 0.9)',
          backdropFilter: 'blur(8px)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #00d4ff22, #8b5cf622)',
            border: '1px solid rgba(0, 212, 255, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '16px',
          }}>
            🧠
          </div>
          <div>
            <h1
              style={{
                fontSize: '16px',
                fontWeight: 700,
                background: 'linear-gradient(90deg, #00d4ff, #8b5cf6)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                lineHeight: 1.2,
              }}
            >
              NeuroGraph
            </h1>
            <div style={{ fontSize: '10px', color: tc.textSecondary, marginTop: '1px' }}>
              Neural Knowledge Agent
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <IngestButton userId={USER_ID} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: isConnected ? '#10b981' : '#ef4444',
                boxShadow: isConnected
                  ? '0 0 8px rgba(16, 185, 129, 0.5)'
                  : '0 0 8px rgba(239, 68, 68, 0.5)',
                transition: 'all 0.3s',
              }}
            />
            <span style={{ fontSize: '11px', color: '#8b949e' }}>
              {isConnected ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
      {/* Messages */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <ChatPanel />
      </div>
      {/* Input */}
      <InputBar onSend={sendMessage} disabled={!isConnected || isLoading} />
    </div>
  );

  return (
    <div className="flex flex-col h-screen" style={{ background: tc.bg, transition: 'background 0.3s' }}>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <SplitView left={<BrainGraph />} right={chatRight} />
      </div>
      <MetricsBar />
    </div>
  );
}

export default App;
