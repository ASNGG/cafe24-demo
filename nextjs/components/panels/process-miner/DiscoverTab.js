// process-miner/DiscoverTab.js — 프로세스 발견 탭

import { useState } from 'react';
import toast from 'react-hot-toast';
import StatCard from '@/components/common/StatCard';
import {
  GitBranch, Play, Loader2, ChevronDown, ChevronRight,
  ArrowRight, Brain, BarChart3, Clock, Activity, TrendingUp, Target,
} from 'lucide-react';
import { isSuccess, getErrorMsg, formatMinutes, formatNumber, formatPercent } from './utils';
import { FeatureImportanceChart, TransitionMatrix } from './SharedComponents';

// 프로세스 플로우 다이어그램
function ProcessFlowDiagram({ transitions, selectedNode, onNodeClick }) {
  if (!transitions?.length) return null;

  const nodesSet = new Set();
  transitions.forEach(t => { nodesSet.add(t.from); nodesSet.add(t.to); });
  const nodes = Array.from(nodesSet);

  const outCount = {};
  const inCount = {};
  nodes.forEach(n => { outCount[n] = 0; inCount[n] = 0; });
  transitions.forEach(t => {
    outCount[t.from] = (outCount[t.from] || 0) + t.count;
    inCount[t.to] = (inCount[t.to] || 0) + t.count;
  });

  const startNodes = nodes.filter(n => inCount[n] === 0);
  const endNodes = nodes.filter(n => outCount[n] === 0);

  const levels = {};
  const visited = new Set();
  const queue = startNodes.length > 0 ? [...startNodes] : [nodes[0]];
  queue.forEach(n => { levels[n] = 0; visited.add(n); });

  while (queue.length > 0) {
    const curr = queue.shift();
    transitions
      .filter(t => t.from === curr)
      .forEach(t => {
        if (!visited.has(t.to)) {
          visited.add(t.to);
          levels[t.to] = (levels[curr] || 0) + 1;
          queue.push(t.to);
        }
      });
  }

  nodes.forEach(n => { if (levels[n] === undefined) levels[n] = 999; });

  const levelGroups = {};
  nodes.forEach(n => {
    const lv = levels[n];
    if (!levelGroups[lv]) levelGroups[lv] = [];
    levelGroups[lv].push(n);
  });
  const sortedLevels = Object.keys(levelGroups).map(Number).sort((a, b) => a - b);
  const maxCount = Math.max(...transitions.map(t => t.count), 1);

  return (
    <div className="overflow-x-auto">
      <div className="flex items-start gap-3 min-w-fit py-2">
        {sortedLevels.map((lv, li) => (
          <div key={lv} className="flex items-center gap-3">
            <div className="flex flex-col gap-2 items-center">
              {levelGroups[lv].map(node => {
                const isStart = startNodes.includes(node);
                const isEnd = endNodes.includes(node);
                const isSelected = selectedNode === node;
                return (
                  <button
                    key={node}
                    onClick={() => onNodeClick(isSelected ? null : node)}
                    className={`rounded-xl px-3 py-2 text-xs font-semibold transition-all border-2 min-w-[80px] text-center ${
                      isSelected
                        ? 'border-teal-500 bg-teal-500 text-white shadow-lg shadow-teal-200 scale-105'
                        : isStart
                          ? 'border-emerald-300 bg-gradient-to-br from-emerald-50 to-emerald-100 text-emerald-800 hover:shadow-md'
                          : isEnd
                            ? 'border-indigo-300 bg-gradient-to-br from-indigo-50 to-indigo-100 text-indigo-800 hover:shadow-md'
                            : 'border-gray-200 bg-gradient-to-br from-white to-gray-50 text-gray-700 hover:shadow-md hover:border-teal-300'
                    }`}
                  >
                    {node}
                  </button>
                );
              })}
            </div>
            {li < sortedLevels.length - 1 && (
              <div className="flex flex-col gap-1 items-center min-w-[40px]">
                {transitions
                  .filter(t => levelGroups[lv].includes(t.from))
                  .slice(0, 3)
                  .map((t, ti) => {
                    const thickness = Math.max(1, Math.round((t.count / maxCount) * 4));
                    return (
                      <div key={ti} className="flex items-center gap-1">
                        <div className="bg-teal-400 rounded-full" style={{ height: `${thickness}px`, width: '24px' }} />
                        <ArrowRight size={12} className="text-teal-400 flex-shrink-0" />
                        <span className="text-[9px] text-gray-400 whitespace-nowrap">{t.count}</span>
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DiscoverTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showMermaid, setShowMermaid] = useState(false);

  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult] = useState(null);
  const [predictCaseId, setPredictCaseId] = useState('');
  const [showPredictFeatures, setShowPredictFeatures] = useState(false);

  const runPredict = async () => {
    setPredictLoading(true);
    setPredictResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/predict',
        auth, method: 'POST',
        data: { process_type: processType, n_cases: nCases, case_id: predictCaseId || '' },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) setPredictResult(res.data || res);
      else toast.error(getErrorMsg(res));
    } catch (e) {
      toast.error('예측 요청 실패: ' + (e.message || ''));
    } finally {
      setPredictLoading(false);
    }
  };

  const runDiscover = async () => {
    setLoading(true);
    setResult(null);
    setSelectedNode(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/discover',
        auth, method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        const d = res.data || res;
        if (d.patterns && !d.top_patterns) {
          d.top_patterns = d.patterns.map(p => ({
            sequence: p.path || p.sequence,
            count: p.count,
            ratio: p.percentage != null ? p.percentage / 100 : p.ratio,
            avg_duration_hours: p.avg_duration_hours,
          }));
          if (!d.unique_patterns) d.unique_patterns = d.patterns.length;
        }
        setResult(d);
      } else {
        toast.error(getErrorMsg(res));
      }
    } catch (e) {
      toast.error('분석 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <button
        onClick={runDiscover}
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-teal-200 transition-all hover:shadow-xl disabled:opacity-50"
      >
        {loading ? <><Loader2 size={16} className="animate-spin" /> 프로세스 분석 중...</> : <><Play size={16} /> 분석 시작</>}
      </button>

      {result && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <StatCard icon={BarChart3} label="총 케이스" value={formatNumber(result.total_cases)} color="teal" />
            <StatCard icon={GitBranch} label="고유 패턴" value={formatNumber(result.unique_patterns)} color="indigo" />
            <StatCard icon={Clock} label="평균 처리시간" value={result.avg_duration_minutes != null ? formatMinutes(result.avg_duration_minutes) : result.total_avg_process_hours != null ? formatMinutes(result.total_avg_process_hours * 60) : '-'} color="amber" />
          </div>

          {result.transitions?.length > 0 && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-teal-600" />
                  <span className="text-sm font-bold text-gray-700">프로세스 플로우</span>
                </div>
                {result.mermaid_diagram && (
                  <button onClick={() => setShowMermaid(!showMermaid)} className="text-xs text-gray-400 hover:text-gray-600 transition-colors">
                    {showMermaid ? 'Flow 보기' : 'Mermaid 코드'}
                  </button>
                )}
              </div>
              {showMermaid && result.mermaid_diagram ? (
                <pre className="rounded-lg bg-gray-900 p-4 text-xs text-gray-200 overflow-x-auto whitespace-pre-wrap">{result.mermaid_diagram}</pre>
              ) : (
                <ProcessFlowDiagram transitions={result.transitions} selectedNode={selectedNode} onNodeClick={setSelectedNode} />
              )}
              {selectedNode && (
                <div className="mt-3 rounded-xl border border-teal-200 bg-teal-50/50 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Target size={14} className="text-teal-600" />
                    <span className="text-xs font-bold text-teal-800">{selectedNode} 단계 상세</span>
                  </div>
                  <div className="text-xs text-gray-600 space-y-1">
                    {result.transitions
                      .filter(t => t.from === selectedNode || t.to === selectedNode)
                      .map((t, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="font-medium text-gray-800">{t.from}</span>
                          <ArrowRight size={12} className="text-gray-400" />
                          <span className="font-medium text-gray-800">{t.to}</span>
                          <span className="ml-auto text-gray-500">빈도 {t.count}회</span>
                          {t.probability != null && <span className="text-gray-400">({formatPercent(t.probability)})</span>}
                          {t.avg_minutes != null && <span className="text-gray-400">{formatMinutes(t.avg_minutes)}</span>}
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {result.top_patterns?.length > 0 && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp size={16} className="text-indigo-600" />
                <span className="text-sm font-bold text-gray-700">Top 프로세스 패턴</span>
              </div>
              <div className="space-y-2">
                {result.top_patterns.map((pattern, i) => (
                  <div key={i} className="rounded-xl border border-gray-100 bg-gray-50/50 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-gray-600">패턴 #{i + 1}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-500">{pattern.count}건</span>
                        <span className="rounded-full bg-teal-100 px-2 py-0.5 text-[10px] font-bold text-teal-700">{formatPercent(pattern.ratio)}</span>
                        {(pattern.avg_duration_minutes != null || pattern.avg_duration_hours != null) && (
                          <span className="text-xs text-gray-400">
                            {pattern.avg_duration_minutes != null ? formatMinutes(pattern.avg_duration_minutes) : formatMinutes(pattern.avg_duration_hours * 60)}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-1">
                      {(pattern.sequence || []).map((step, j) => (
                        <span key={j} className="flex items-center gap-1">
                          <span className="rounded-lg bg-gradient-to-r from-teal-50 to-teal-100 border border-teal-200 px-2 py-1 text-[11px] font-semibold text-teal-800">{step}</span>
                          {j < pattern.sequence.length - 1 && <ArrowRight size={12} className="text-gray-300" />}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.transitions?.length > 0 && <TransitionMatrix transitions={result.transitions} />}
        </>
      )}

      {/* 다음 활동 예측 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <Brain size={16} className="text-indigo-600" />
          <span className="text-sm font-bold text-gray-700">다음 활동 예측</span>
        </div>
        <div className="flex items-end gap-3">
          <div className="flex-1 min-w-[160px]">
            <label className="mb-1 block text-xs text-gray-500">케이스 ID (비워두면 랜덤 선택)</label>
            <input type="text" value={predictCaseId} onChange={e => setPredictCaseId(e.target.value)}
              placeholder="예: ORD-001" className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none" />
          </div>
          <button onClick={runPredict} disabled={predictLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-indigo-200 transition-all hover:shadow-xl disabled:opacity-50">
            {predictLoading ? <><Loader2 size={14} className="animate-spin" /> 예측 중...</> : <><Brain size={14} /> 예측 실행</>}
          </button>
        </div>

        {predictResult && (
          <div className="space-y-3">
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">케이스:</span>
                <span className="rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-bold text-indigo-700">{predictResult.case_id}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">현재 활동:</span>
                <span className="rounded-lg bg-gradient-to-r from-indigo-50 to-indigo-100 border border-indigo-200 px-2 py-1 text-[11px] font-semibold text-indigo-800">{predictResult.current_activity}</span>
              </div>
              {predictResult.model_accuracy != null && (
                <span className="ml-auto rounded-full bg-emerald-100 px-2.5 py-0.5 text-[10px] font-bold text-emerald-700">모델 정확도 {formatPercent(predictResult.model_accuracy)}</span>
              )}
            </div>
            {predictResult.predictions?.length > 0 && (
              <div className="rounded-xl border border-indigo-100 bg-indigo-50/30 p-3 space-y-2">
                <span className="text-xs font-bold text-gray-600">예측 결과</span>
                {predictResult.predictions.map((pred, i) => {
                  const barWidth = Math.max(pred.probability * 100, 2);
                  const barColor = i === 0 ? 'bg-gradient-to-r from-indigo-500 to-indigo-400' : i === 1 ? 'bg-gradient-to-r from-indigo-400 to-indigo-300' : 'bg-gradient-to-r from-indigo-300 to-indigo-200';
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs font-semibold text-gray-700 min-w-[80px] truncate">{pred.activity}</span>
                      <div className="flex-1 h-5 rounded-full bg-gray-100 overflow-hidden">
                        <div className={`h-5 rounded-full ${barColor} flex items-center justify-end pr-2 transition-all`} style={{ width: `${barWidth}%` }}>
                          {barWidth > 15 && <span className="text-[10px] font-bold text-white">{formatPercent(pred.probability)}</span>}
                        </div>
                      </div>
                      {barWidth <= 15 && <span className="text-[10px] font-bold text-gray-500">{formatPercent(pred.probability)}</span>}
                    </div>
                  );
                })}
              </div>
            )}
            {predictResult.feature_importance && (
              <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
                <button onClick={() => setShowPredictFeatures(!showPredictFeatures)}
                  className="flex w-full items-center gap-2 p-3 text-left hover:bg-gray-50 transition-colors">
                  {showPredictFeatures ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
                  <BarChart3 size={14} className="text-indigo-500" />
                  <span className="text-xs font-bold text-gray-600">피처 중요도</span>
                </button>
                {showPredictFeatures && (
                  <div className="border-t border-gray-100 p-3">
                    <FeatureImportanceChart features={predictResult.feature_importance} color="indigo" />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
