export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
  /** Concepts activated at the time this message was generated (assistant only) */
  concepts?: { id: string; label: string; category: string }[];
}

export interface WSMessage {
  type: string;
  data: Record<string, any>;
}
