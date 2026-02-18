// guardian/common/ToolStep.js — 가디언 도구 실행 단계 표시 (MonitorTab + RecoverTab 공통)

import { useState } from 'react';
import {
  ChevronDown, ChevronRight,
  Database, User, Search, Shield, FileText,
} from 'lucide-react';

const TOOL_ICONS = {
  analyze_impact: Database,
  get_user_pattern: User,
  search_similar: Search,
  execute_decision: Shield,
};

const TOOL_NAMES = {
  analyze_impact: '영향도 분석',
  get_user_pattern: '사용자 패턴',
  search_similar: '유사 사례 검색',
  execute_decision: '판단 실행',
};

export default function ToolStep({ index, step }) {
  const [open, setOpen] = useState(false);
  const Icon = TOOL_ICONS[step.tool] || FileText;

  return (
    <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors">
        {open ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <Icon size={14} className="text-indigo-500" />
        <span className="text-xs font-semibold text-gray-700">Tool {index}: {TOOL_NAMES[step.tool] || step.tool}</span>
      </button>
      {open && (
        <div className="border-t border-gray-100 bg-gray-50/50 px-3 py-2">
          <pre className="whitespace-pre-wrap text-xs text-gray-600 leading-relaxed">{step.output}</pre>
        </div>
      )}
    </div>
  );
}
