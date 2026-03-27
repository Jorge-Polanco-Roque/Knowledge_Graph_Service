import { useState, useCallback, type KeyboardEvent } from 'react';

interface InputBarProps {
  onSend: (msg: string) => void;
  disabled: boolean;
}

export function InputBar({ onSend, disabled }: InputBarProps) {
  const [value, setValue] = useState('');

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div
      style={{
        padding: '12px 16px',
        borderTop: '1px solid rgba(0, 212, 255, 0.1)',
        display: 'flex',
        gap: '8px',
        alignItems: 'center',
        background: 'rgba(10, 10, 26, 0.4)',
      }}
    >
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Ask NeuroGraph..."
        style={{
          flex: 1,
          padding: '11px 16px',
          borderRadius: '14px',
          border: '1px solid rgba(0, 212, 255, 0.12)',
          background: 'rgba(13, 17, 23, 0.5)',
          color: '#e6edf3',
          fontSize: '14px',
          outline: 'none',
          opacity: disabled ? 0.5 : 1,
          transition: 'border-color 0.2s, box-shadow 0.2s',
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.35)';
          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0, 212, 255, 0.08)';
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.12)';
          e.currentTarget.style.boxShadow = 'none';
        }}
      />
      <button
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        style={{
          width: '42px',
          height: '42px',
          borderRadius: '14px',
          border: 'none',
          background:
            disabled || !value.trim()
              ? 'rgba(139, 92, 246, 0.15)'
              : 'linear-gradient(135deg, #00d4ff, #8b5cf6)',
          color: '#ffffff',
          fontSize: '18px',
          cursor: disabled || !value.trim() ? 'not-allowed' : 'pointer',
          opacity: disabled || !value.trim() ? 0.4 : 1,
          transition: 'opacity 0.2s, box-shadow 0.2s, transform 0.1s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          if (!disabled && value.trim()) {
            e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.35)';
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.boxShadow = 'none';
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="22" y1="2" x2="11" y2="13" />
          <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
      </button>
    </div>
  );
}
