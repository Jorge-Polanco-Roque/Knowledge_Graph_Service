import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGraphStore } from '../../hooks/useGraphStore';
import type { ChatMessage } from '../../types/messages';

const CATEGORY_COLORS: Record<string, string> = {
  person: '#3b82f6',
  place: '#10b981',
  concept: '#8b5cf6',
  action: '#f97316',
  object: '#ec4899',
  event: '#f59e0b',
};

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const activatedNodes = useGraphStore((s) => s.activatedNodes);
  const nodes = useGraphStore((s) => s.nodes);

  // For assistant messages, show recently activated concept tags
  const conceptTags = !isUser
    ? nodes.filter((n) => activatedNodes.has(n.id)).slice(0, 6)
    : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1], delay: 0.05 }}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: '14px',
        paddingLeft: '16px',
        paddingRight: '16px',
      }}
    >
      {/* Role label */}
      <div style={{
        fontSize: '10px',
        color: '#8b949e',
        marginBottom: '4px',
        paddingLeft: isUser ? 0 : '4px',
        paddingRight: isUser ? '4px' : 0,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        {isUser ? 'You' : 'NeuroGraph'}
      </div>

      {/* Message bubble */}
      <div
        style={{
          maxWidth: '85%',
          padding: '12px 16px',
          borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
          background: isUser
            ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)'
            : 'rgba(13, 17, 23, 0.8)',
          border: isUser ? 'none' : '1px solid rgba(0, 212, 255, 0.12)',
          fontSize: '14px',
          lineHeight: '1.6',
          color: '#e6edf3',
          wordBreak: 'break-word',
          whiteSpace: 'pre-wrap',
          boxShadow: isUser
            ? '0 2px 12px rgba(59, 130, 246, 0.2)'
            : '0 2px 8px rgba(0, 0, 0, 0.2)',
        }}
      >
        {message.content}
      </div>

      {/* Concept tags for assistant messages */}
      {conceptTags.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '4px',
            marginTop: '6px',
            maxWidth: '85%',
          }}
        >
          {conceptTags.map((node) => (
            <span
              key={node.id}
              style={{
                fontSize: '10px',
                padding: '2px 8px',
                borderRadius: '10px',
                background: `${CATEGORY_COLORS[node.category] || '#8b5cf6'}18`,
                border: `1px solid ${CATEGORY_COLORS[node.category] || '#8b5cf6'}40`,
                color: CATEGORY_COLORS[node.category] || '#8b5cf6',
                fontWeight: 500,
              }}
            >
              {node.label}
            </span>
          ))}
        </motion.div>
      )}
    </motion.div>
  );
}

function LoadingDots() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        display: 'flex',
        justifyContent: 'flex-start',
        paddingLeft: '16px',
        paddingRight: '16px',
        marginBottom: '12px',
        flexDirection: 'column',
        alignItems: 'flex-start',
      }}
    >
      <div style={{
        fontSize: '10px',
        color: '#8b949e',
        marginBottom: '4px',
        paddingLeft: '4px',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        NeuroGraph
      </div>
      <div
        style={{
          padding: '12px 18px',
          borderRadius: '16px 16px 16px 4px',
          background: 'rgba(13, 17, 23, 0.8)',
          border: '1px solid rgba(0, 212, 255, 0.12)',
          display: 'flex',
          gap: '5px',
          alignItems: 'center',
        }}
      >
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #00d4ff, #8b5cf6)',
            }}
            animate={{
              scale: [0.8, 1.2, 0.8],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 1.2,
              repeat: Infinity,
              delay: i * 0.2,
              ease: 'easeInOut',
            }}
          />
        ))}
        <span style={{ fontSize: '11px', color: '#8b949e', marginLeft: '6px' }}>
          Processing neural pathways...
        </span>
      </div>
    </motion.div>
  );
}

export function ChatPanel() {
  const messages = useGraphStore((s) => s.messages);
  const isLoading = useGraphStore((s) => s.isLoading);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, isLoading]);

  return (
    <div
      ref={scrollRef}
      style={{
        height: '100%',
        overflowY: 'auto',
        paddingTop: '16px',
        paddingBottom: '16px',
      }}
    >
      {messages.length === 0 && !isLoading && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            padding: '32px',
            textAlign: 'center',
            gap: '12px',
          }}
        >
          <div style={{ fontSize: '36px', opacity: 0.2 }}>💬</div>
          <p style={{ color: '#8b949e', fontSize: '14px', lineHeight: '1.6', maxWidth: '280px' }}>
            Start a conversation to see your knowledge graph grow in real-time
          </p>
          <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
            {['concept', 'person', 'place'].map((cat) => (
              <span
                key={cat}
                style={{
                  fontSize: '10px',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  background: `${CATEGORY_COLORS[cat]}15`,
                  border: `1px solid ${CATEGORY_COLORS[cat]}30`,
                  color: `${CATEGORY_COLORS[cat]}88`,
                }}
              >
                {cat}
              </span>
            ))}
          </div>
        </div>
      )}

      <AnimatePresence>
        {messages.map((msg, idx) => (
          <MessageBubble key={`${idx}-${msg.timestamp}`} message={msg} />
        ))}
      </AnimatePresence>

      {isLoading && <LoadingDots />}
    </div>
  );
}
