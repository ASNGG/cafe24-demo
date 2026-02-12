// process-miner/RecommendTab.js — AI 자동화 추천 탭

import { useState } from 'react';
import toast from 'react-hot-toast';
import {
  Play, Loader2, ChevronDown, ChevronRight,
  Sparkles, Zap, Brain, Bot, Clock, TrendingUp, FileText, Target,
} from 'lucide-react';
import { isSuccess, getErrorMsg, formatMinutes, formatPercent } from './utils';
import { SimpleMarkdown } from './SharedComponents';

const automationIcons = {
  rule_based: Zap,
  ml_based: Brain,
  llm_based: Bot,
};

const priorityBadge = {
  high: 'bg-red-100 text-red-700',
  medium: 'bg-amber-100 text-amber-700',
  low: 'bg-emerald-100 text-emerald-700',
};

export default function RecommendTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showSop, setShowSop] = useState(false);

  const runRecommend = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/recommend',
        auth, method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) setResult(res.data || res);
      else toast.error(getErrorMsg(res));
    } catch (e) {
      toast.error('분석 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <button onClick={runRecommend} disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-teal-200 transition-all hover:shadow-xl disabled:opacity-50">
        {loading ? <><Loader2 size={16} className="animate-spin" /> AI 추천 생성 중...</> : <><Play size={16} /> 분석 시작</>}
      </button>

      {result && (
        <>
          <div className="rounded-2xl border-2 border-teal-200 bg-gradient-to-br from-teal-50 to-emerald-50 p-6 text-center">
            <TrendingUp size={28} className="mx-auto mb-2 text-teal-600" />
            {(result.estimated_time_saving_percent != null || result.estimated_improvement != null) && (
              <p className="text-2xl font-black text-teal-700 mb-1">
                약 {result.estimated_time_saving_percent != null
                  ? `${result.estimated_time_saving_percent}%`
                  : formatPercent(result.estimated_improvement)} 효율 개선 가능
              </p>
            )}
            {result.summary && <p className="text-sm text-gray-600">{result.summary}</p>}
          </div>

          {result.recommendations?.length > 0 && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles size={16} className="text-violet-600" />
                <span className="text-sm font-bold text-gray-700">자동화 추천</span>
                <span className="ml-auto text-xs text-gray-400">{result.recommendations.length}건</span>
              </div>
              <div className="space-y-3">
                {result.recommendations.map((rec, i) => {
                  const AutoIcon = automationIcons[rec.automation_type] || Zap;
                  const iconColor = rec.automation_type === 'llm_based'
                    ? 'text-violet-600 bg-violet-100'
                    : rec.automation_type === 'ml_based'
                      ? 'text-indigo-600 bg-indigo-100'
                      : 'text-amber-600 bg-amber-100';
                  return (
                    <div key={i} className="rounded-xl border border-gray-200 bg-gray-50/50 p-4">
                      <div className="flex items-start gap-3">
                        <div className={`rounded-lg p-2 ${iconColor}`}><AutoIcon size={18} /></div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-bold text-gray-800">{rec.target_step}</span>
                            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${priorityBadge[rec.priority] || priorityBadge.medium}`}>{rec.priority}</span>
                            {rec.automation_type && (
                              <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-medium text-gray-500">{rec.automation_type.replace('_', ' ')}</span>
                            )}
                          </div>
                          <p className="text-xs text-gray-600 mb-2">{rec.description}</p>
                          <div className="flex flex-wrap items-center gap-3">
                            {rec.current_avg_minutes != null && (
                              <div className="flex items-center gap-1">
                                <Clock size={12} className="text-gray-400" />
                                <span className="text-xs text-gray-500">현재 평균 {formatMinutes(rec.current_avg_minutes)}</span>
                              </div>
                            )}
                            {rec.expected_improvement && (
                              <div className="flex items-center gap-1">
                                <TrendingUp size={12} className="text-emerald-500" />
                                <span className="text-xs font-semibold text-emerald-700">{rec.expected_improvement}</span>
                              </div>
                            )}
                            {rec.implementation_effort && (
                              <div className="flex items-center gap-1">
                                <Target size={12} className="text-gray-400" />
                                <span className="text-xs text-gray-500">난이도: {rec.implementation_effort}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {result.sop_document && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 backdrop-blur overflow-hidden">
              <button onClick={() => setShowSop(!showSop)}
                className="flex w-full items-center gap-2 p-4 text-left hover:bg-gray-50 transition-colors">
                {showSop ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
                <FileText size={16} className="text-teal-600" />
                <span className="text-sm font-bold text-gray-700">SOP 문서</span>
              </button>
              {showSop && (
                <div className="border-t border-gray-100 p-4">
                  <SimpleMarkdown text={result.sop_document} />
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
