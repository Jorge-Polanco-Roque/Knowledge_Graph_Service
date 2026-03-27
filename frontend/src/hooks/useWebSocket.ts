import { useEffect, useRef, useCallback } from 'react';
import { useGraphStore } from './useGraphStore';
import type { WSMessage } from '../types/messages';

export function useWebSocket(userId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const store = useGraphStore;
  const isConnected = useGraphStore((s) => s.isConnected);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`ws://localhost:8899/ws/${userId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      store.getState().setConnected(true);
    };

    ws.onclose = () => {
      store.getState().setConnected(false);
      // Reconnect after 3 seconds
      reconnectTimerRef.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.onerror = () => {
      // Will trigger onclose
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        const { type, data } = msg;

        switch (type) {
          case 'agent_response': {
            // Snapshot activated concepts at response time so tags are static per message
            const concepts = (data.extracted_concepts || []).map((c: any) => ({
              id: c.label?.toLowerCase().replace(/\s+/g, '_') || '',
              label: c.label || '',
              category: c.category || 'concept',
            }));
            store.getState().addMessage({
              role: 'assistant',
              content: data.content,
              timestamp: new Date().toISOString(),
              concepts,
            });
            store.getState().setLoading(false);
            if (data.graph_diff) {
              store.getState().applyDiff(data.graph_diff);
            }
            // Auto-refresh stats after agent response
            fetch(`/api/graph/${userId}/stats`)
              .then((r) => r.ok ? r.json() : null)
              .then((stats) => { if (stats) store.getState().setStats(stats); })
              .catch(() => {});
            break;
          }

          case 'graph_diff':
            store.getState().applyDiff(data as any);
            break;

          case 'spreading_activation':
            store.setState({ spreadingWaves: data.waves || [] });
            break;

          case 'error':
            console.error('WebSocket error message:', data);
            store.getState().setLoading(false);
            break;

          default:
            console.warn('Unknown WS message type:', type);
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };
  }, [userId, store]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on intentional close
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback(
    (content: string) => {
      if (!content.trim()) return;

      store.getState().addMessage({
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
      });
      store.getState().setLoading(true);

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: 'user_message',
            data: { content },
          })
        );
      }
    },
    [store]
  );

  return { sendMessage, isConnected };
}
