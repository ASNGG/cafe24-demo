// guardian/RecoverTab.js — 복구 요청 탭

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import {
  RotateCcw, Loader2, ChevronDown, ChevronRight,
  Database, User, Search, Shield, FileText,
} from 'lucide-react';

const EXAMPLES = [
  '아까 실수로 삭제한 12월 주문 데이터 347건 복구해주세요',
  'kim 사용자가 삭제한 payments 데이터 복구해줘',
  '오늘 차단된 orders 삭제 건 복구 SQL 만들어줘',
];

// L19: ToolStep 기본 open=false
function ToolStep({ index, step }) {
  const [open, setOpen] = useState(false);
  const toolIcons = { analyze_impact: Database, get_user_pattern: User, search_similar: Search, execute_decision: Shield };
  const Icon = toolIcons[step.tool] || FileText;
  const toolNames = { analyze_impact: '영향도 분석', get_user_pattern: '사용자 패턴', search_similar: '유사 사례 검색', execute_decision: '판단 실행' };

  return (
    <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors">
        {open ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <Icon size={14} className="text-indigo-500" />
        <span className="text-xs font-semibold text-gray-700">Tool {index}: {toolNames[step.tool] || step.tool}</span>
      </button>
      {open && (
        <div className="border-t border-gray-100 bg-gray-50/50 px-3 py-2">
          <pre className="whitespace-pre-wrap text-xs text-gray-600 leading-relaxed">{step.output}</pre>
        </div>
      )}
    </div>
  );
}

export default function RecoverTab({ auth, apiCall }) {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const runRecovery = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/guardian/recover', auth, method: 'POST',
        data: { message }, timeoutMs: 30000,
      });
      if (res?.status === 'success') setResult(res);
      else toast.error(res?.message || '복구 요청 실패');
    } catch (e) {
      toast.error('복구 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <p className="mb-1 text-sm font-bold text-gray-700">자연어 복구 요청</p>
        <p className="mb-3 text-xs text-gray-400">삭제된 데이터를 자연어로 설명하면 Agent가 복구 SQL을 생성합니다. (DBA 승인 필요)</p>

        <div className="mb-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex, i) => (
            <button key={i} onClick={() => setMessage(ex)}
              className="rounded-lg border border-gray-200 bg-gray-50 px-2.5 py-1 text-xs text-gray-600 hover:bg-gray-100 transition-colors">
              {ex.length > 35 ? ex.slice(0, 35) + '...' : ex}
            </button>
          ))}
        </div>

        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="예: 아까 실수로 삭제한 12월 주문 데이터 복구해주세요"
          className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm focus:border-indigo-400 focus:outline-none resize-none"
          rows={3}
        />

        <button
          onClick={runRecovery}
          disabled={loading || !message.trim()}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-emerald-200 transition-all hover:shadow-xl disabled:opacity-50"
        >
          {loading ? <><Loader2 size={16} className="animate-spin" /> 복구 계획 수립 중...</> : <><RotateCcw size={16} /> 복구 요청</>}
        </button>
      </div>

      {result && (
        <div className="rounded-2xl border-2 border-emerald-200 bg-emerald-50/30 p-4">
          <div className="flex items-center gap-2 mb-3">
            <RotateCcw size={16} className="text-emerald-600" />
            <span className="text-sm font-bold text-gray-800">복구 계획</span>
          </div>

          {result.steps?.length > 0 && (
            <div className="mb-4 space-y-2">
              {result.steps.map((step, i) => (
                <ToolStep key={i} index={i + 1} step={step} />
              ))}
            </div>
          )}

          <div className="rounded-xl border border-emerald-200 bg-white/80 p-4">
            <div className="prose prose-sm max-w-none text-gray-700">
              <ReactMarkdown>{result.output}</ReactMarkdown>
            </div>
          </div>

          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
            <p className="text-xs font-medium text-amber-700">
              ⚠️ 복구 SQL은 DBA 승인 후 실행됩니다. 승인 요청이 자동 발송됩니다.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
