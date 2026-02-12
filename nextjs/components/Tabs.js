import { useRef, useCallback } from 'react';
import { cn } from '@/lib/cn';

export default function Tabs({ tabs = [], active, onChange }) {
  const tabListRef = useRef(null);

  const handleKeyDown = useCallback((e) => {
    const currentIndex = tabs.findIndex(t => t.key === active);
    let nextIndex = -1;

    if (e.key === 'ArrowRight') {
      e.preventDefault();
      nextIndex = (currentIndex + 1) % tabs.length;
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    } else if (e.key === 'Home') {
      e.preventDefault();
      nextIndex = 0;
    } else if (e.key === 'End') {
      e.preventDefault();
      nextIndex = tabs.length - 1;
    }

    if (nextIndex >= 0) {
      onChange(tabs[nextIndex].key);
      const buttons = tabListRef.current?.querySelectorAll('[role="tab"]');
      buttons?.[nextIndex]?.focus();
    }
  }, [tabs, active, onChange]);

  return (
    <div className="mb-4">
      <div
        ref={tabListRef}
        role="tablist"
        aria-label="탭 목록"
        className="flex flex-wrap gap-2 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-2 shadow-sm backdrop-blur"
      >
        {tabs.map((t) => {
          const isActive = t.key === active;
          return (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={isActive}
              tabIndex={isActive ? 0 : -1}
              onClick={() => onChange(t.key)}
              onKeyDown={handleKeyDown}
              className={cn(
                'rounded-2xl px-4 py-2 text-sm font-black transition active:translate-y-[1px]',
                isActive
                  ? 'bg-gradient-to-br from-cafe24-yellow via-cafe24-orange to-cafe24-yellow text-cafe24-brown shadow-cafe24-sm'
                  : 'bg-white/70 text-cafe24-brown/60 hover:bg-cafe24-yellow/20'
              )}
            >
              {t.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
