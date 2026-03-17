// components/common/ToolExplorer.js
// CAFE24 AI 운영 플랫폼 - 도구 탐색기 (인터랙티브 아코디언 UI)

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronDown, Wrench, Store, UserSearch, BrainCircuit,
  Headphones, LayoutDashboard, ShieldCheck, Search, BarChart3,
} from 'lucide-react';
import { TOOL_CATEGORIES } from './toolRegistry';

// 문자열 아이콘명 → lucide-react 컴포넌트 매핑
const ICON_MAP = {
  Store, UserSearch, BrainCircuit, Headphones, LayoutDashboard,
  ShieldCheck, Search, BarChart3, Wrench,
};
const resolveIcon = (icon) => (typeof icon === 'string' ? ICON_MAP[icon] || Wrench : icon || Wrench);

// 카테고리별 accent 색상 매핑
const ACCENT_MAP = {
  blue:    { header: 'bg-blue-50/80 border-blue-200/60 hover:bg-blue-100/70',    badge: 'bg-blue-100 text-blue-700',    dot: 'bg-blue-400' },
  emerald: { header: 'bg-emerald-50/80 border-emerald-200/60 hover:bg-emerald-100/70', badge: 'bg-emerald-100 text-emerald-700', dot: 'bg-emerald-400' },
  purple:  { header: 'bg-purple-50/80 border-purple-200/60 hover:bg-purple-100/70',  badge: 'bg-purple-100 text-purple-700',  dot: 'bg-purple-400' },
  pink:    { header: 'bg-pink-50/80 border-pink-200/60 hover:bg-pink-100/70',    badge: 'bg-pink-100 text-pink-700',    dot: 'bg-pink-400' },
  amber:   { header: 'bg-amber-50/80 border-amber-200/60 hover:bg-amber-100/70',   badge: 'bg-amber-100 text-amber-700',   dot: 'bg-amber-400' },
  orange:  { header: 'bg-orange-50/80 border-orange-200/60 hover:bg-orange-100/70',  badge: 'bg-orange-100 text-orange-700',  dot: 'bg-orange-400' },
  teal:    { header: 'bg-teal-50/80 border-teal-200/60 hover:bg-teal-100/70',    badge: 'bg-teal-100 text-teal-700',    dot: 'bg-teal-400' },
  red:     { header: 'bg-red-50/80 border-red-200/60 hover:bg-red-100/70',      badge: 'bg-red-100 text-red-700',      dot: 'bg-red-400' },
  indigo:  { header: 'bg-indigo-50/80 border-indigo-200/60 hover:bg-indigo-100/70',  badge: 'bg-indigo-100 text-indigo-700',  dot: 'bg-indigo-400' },
  violet:  { header: 'bg-violet-50/80 border-violet-200/60 hover:bg-violet-100/70',  badge: 'bg-violet-100 text-violet-700',  dot: 'bg-violet-400' },
  sky:     { header: 'bg-sky-50/80 border-sky-200/60 hover:bg-sky-100/70',       badge: 'bg-sky-100 text-sky-700',       dot: 'bg-sky-400' },
  rose:    { header: 'bg-rose-50/80 border-rose-200/60 hover:bg-rose-100/70',     badge: 'bg-rose-100 text-rose-700',     dot: 'bg-rose-400' },
};

// 에이전트 이름 → 뱃지 색상
const AGENT_COLOR = {
  search_agent:   'bg-blue-100 text-blue-700 border-blue-200',
  analysis_agent: 'bg-purple-100 text-purple-700 border-purple-200',
  cs_agent:       'bg-pink-100 text-pink-700 border-pink-200',
  retention_agent:'bg-orange-100 text-orange-700 border-orange-200',
  coordinator:    'bg-gray-200 text-gray-700 border-gray-300',
  all:            'bg-gray-100 text-gray-600 border-gray-200',
};

const getAgentColor = (agent) => AGENT_COLOR[agent] || 'bg-gray-100 text-gray-600 border-gray-200';

// 도구 카드 — 호버 시 아래로 설명 스르륵 펼침
const ToolCard = React.memo(function ToolCard({ tool, accent }) {
  const colors = ACCENT_MAP[accent] || ACCENT_MAP.blue;

  return (
    <div className="group rounded-xl border border-gray-200/80 bg-white/90 px-3 py-2 cursor-default transition-all duration-200 hover:border-cafe24-orange/30 hover:shadow-sm">
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 transition-transform duration-300 group-hover:scale-150 ${colors.dot}`} />
        <span className="text-[11px] font-bold text-cafe24-brown">{tool.label}</span>
      </div>

      <div className="grid grid-rows-[0fr] group-hover:grid-rows-[1fr] transition-[grid-template-rows] duration-300 ease-out">
        <div className="overflow-hidden">
          <div className="pt-2 mt-1 border-t border-gray-100 space-y-2">
            <p className="text-[11px] text-cafe24-brown/70 leading-relaxed">{tool.description}</p>
            {tool.params && tool.params.length > 0 && (
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-[9px] text-cafe24-brown/40">매개변수:</span>
                {tool.params.map((p) => (
                  <span key={p} className={`inline-block rounded-md px-1.5 py-0.5 text-[9px] font-semibold ${colors.badge}`}>{p}</span>
                ))}
              </div>
            )}
            {tool.returns && (
              <div className="flex items-start gap-1.5">
                <span className="text-[9px] text-cafe24-brown/40 flex-shrink-0 mt-px">→ 반환:</span>
                <span className="text-[10px] text-emerald-700/80 leading-snug">{tool.returns}</span>
              </div>
            )}
            {tool.example && (
              <div className="rounded-lg bg-cafe24-yellow/10 px-2.5 py-1.5 flex items-start gap-1.5">
                <span className="text-[10px] mt-px">💬</span>
                <span className="text-[11px] text-cafe24-brown/80 font-medium leading-snug">"{tool.example}"</span>
              </div>
            )}
            <code className="inline-block text-[10px] text-cafe24-brown/60 font-mono bg-gray-50 rounded px-1.5 py-0.5">{tool.name}({tool.params?.join(', ') || ''})</code>
          </div>
        </div>
      </div>
    </div>
  );
});

// 카테고리 아코디언 (개별)
const CategoryAccordion = React.memo(function CategoryAccordion({ category, isOpen, onToggle }) {
  const colors = ACCENT_MAP[category.color] || ACCENT_MAP.blue;
  const Icon = resolveIcon(category.icon);

  return (
    <div className="rounded-2xl border border-gray-200/60 bg-white/60 backdrop-blur">
      {/* 헤더 */}
      <button
        type="button"
        onClick={onToggle}
        className={`w-full flex items-center gap-2.5 px-3 py-2.5 border-b transition-colors ${colors.header} ${
          isOpen ? 'border-gray-200/60' : 'border-transparent'
        }`}
      >
        <span className={`flex items-center justify-center w-6 h-6 rounded-lg ${colors.badge}`}>
          <Icon size={13} />
        </span>
        <span className="text-xs font-extrabold text-cafe24-brown flex-1 text-left">
          {category.name}
        </span>
        <span className={`inline-flex items-center justify-center min-w-[20px] h-5 rounded-full px-1.5 text-[10px] font-bold ${colors.badge}`}>
          {category.tools.length}
        </span>
        <motion.span
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-cafe24-brown/40"
        >
          <ChevronDown size={14} />
        </motion.span>
      </button>

      {/* 도구 목록 */}
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="p-2 space-y-1.5">
              {category.tools.map((tool) => (
                <ToolCard key={tool.name} tool={tool} accent={category.color} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

// 메인 컴포넌트
export default React.memo(function ToolExplorer() {
  const [openIds, setOpenIds] = useState(new Set());

  const handleToggle = useCallback((categoryId) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(categoryId)) {
        next.delete(categoryId);
      } else {
        next.add(categoryId);
      }
      return next;
    });
  }, []);

  const totalTools = TOOL_CATEGORIES.reduce((sum, c) => sum + c.tools.length, 0);

  return (
    <div className="space-y-2">
      {/* 타이틀 */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <Wrench size={14} className="text-cafe24-brown/60" />
          <span className="text-xs font-extrabold text-cafe24-brown">에이전트 도구</span>
        </div>
        <span className="text-[10px] font-bold text-cafe24-brown/40">
          {TOOL_CATEGORIES.length}개 카테고리 / {totalTools}개 도구
        </span>
      </div>

      {/* 아코디언 리스트 */}
      <div className="space-y-1.5">
        {TOOL_CATEGORIES.map((category) => (
          <CategoryAccordion
            key={category.id}
            category={category}
            isOpen={openIds.has(category.id)}
            onToggle={() => handleToggle(category.id)}
          />
        ))}
      </div>
    </div>
  );
});
