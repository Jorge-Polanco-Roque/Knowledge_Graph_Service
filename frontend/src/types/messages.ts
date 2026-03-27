export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
}

export interface WSMessage {
  type: string;
  data: Record<string, any>;
}
