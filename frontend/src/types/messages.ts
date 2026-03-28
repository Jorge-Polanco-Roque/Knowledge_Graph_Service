export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
  /** Concepts activated at the time this message was generated (assistant only) */
  concepts?: { id: string; label: string; category: string }[];
  /** Unique message ID for feedback tracking */
  id?: string;
  /** User feedback: 1 = thumbs up, -1 = thumbs down */
  feedback?: 1 | -1;
}

export interface WSMessage {
  type: string;
  data: Record<string, any>;
}
