import { useState, useCallback, useRef, useEffect, type ReactNode } from 'react';

interface SplitViewProps {
  left: ReactNode;
  right: ReactNode;
}

export function SplitView({ left, right }: SplitViewProps) {
  const [splitPos, setSplitPos] = useState(60);
  const isDragging = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pos = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitPos(Math.min(Math.max(pos, 20), 80));
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div ref={containerRef} className="flex h-full w-full">
      {/* Left panel */}
      <div
        style={{ width: `${splitPos}%` }}
        className="h-full overflow-hidden"
      >
        {left}
      </div>

      {/* Divider */}
      <div
        onMouseDown={handleMouseDown}
        style={{
          width: '4px',
          cursor: 'col-resize',
          background: 'rgba(0, 212, 255, 0.15)',
          transition: 'background 0.2s',
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background =
            'rgba(0, 212, 255, 0.5)';
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background =
            'rgba(0, 212, 255, 0.15)';
        }}
      />

      {/* Right panel */}
      <div
        style={{ width: `${100 - splitPos}%` }}
        className="h-full overflow-hidden"
      >
        {right}
      </div>
    </div>
  );
}
