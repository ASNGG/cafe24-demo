// components/panels/ProcessMinerPanel.js
// Process Miner — AI 기반 프로세스 마이닝 & 자동화 추천

import { useState } from 'react';
import toast from 'react-hot-toast';
import StatCard from '@/components/common/StatCard';
import {
  GitBranch, Timer, Sparkles, Play, Loader2, ChevronDown, ChevronRight,
  ArrowRight, AlertTriangle, CheckCircle2, Zap, Brain, Bot, BarChart3,
  Clock, Activity, TrendingUp, FileText, Target, Search, ShieldAlert,
  RefreshCw
} from 'lucide-react';

const TABS = [
  { key: 'discover', label: '프로세스 발견', icon: GitBranch },
  { key: 'bottleneck', label: '병목 분석', icon: Timer },
  { key: 'recommend', label: 'AI 자동화 추천', icon: Sparkles },
];

const PROCESS_TYPES = [
  { value: 'order', label: '주문 프로세스' },
  { value: 'cs', label: 'CS 문의 프로세스' },
  { value: 'settlement', label: '정산 프로세스' },
];

const CASE_OPTIONS = [100, 200, 500];

// API 응답 성공 여부 (SUCCESS 또는 OK)
function isSuccess(res) {
  return res?.status === 'success' || res?.status === 'ok';
}

function getErrorMsg(res) {
  return res?.message || res?.error || '분석 실패';
}

// 숫자 포맷 유틸
function formatMinutes(min) {
  if (min == null || isNaN(min)) return '-';
  if (min < 1) return `${(min * 60).toFixed(0)}초`;
  if (min < 60) return `${min.toFixed(1)}분`;
  if (min < 1440) return `${(min / 60).toFixed(1)}시간`;
  return `${(min / 1440).toFixed(1)}일`;
}

function formatNumber(n) {
  if (n == null || isNaN(n)) return '-';
  return Number(n).toLocaleString();
}

function formatPercent(n) {
  if (n == null || isNaN(n)) return '-';
  return `${(n * 100).toFixed(1)}%`;
}

// 간단한 마크다운 렌더러
function SimpleMarkdown({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        const trimmed = line.trimStart();
        if (trimmed.startsWith('### ')) {
          return <h3 key={i} className="text-sm font-bold text-gray-800 mt-3 mb-1">{trimmed.slice(4)}</h3>;
        }
        if (trimmed.startsWith('## ')) {
          return <h2 key={i} className="text-base font-bold text-gray-900 mt-4 mb-1">{trimmed.slice(3)}</h2>;
        }
        if (trimmed.startsWith('# ')) {
          return <h1 key={i} className="text-lg font-bold text-gray-900 mt-4 mb-2">{trimmed.slice(2)}</h1>;
        }
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
          return <li key={i} className="ml-4 text-xs text-gray-700 list-disc">{trimmed.slice(2)}</li>;
        }
        if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
          return <p key={i} className="text-xs font-bold text-gray-800">{trimmed.slice(2, -2)}</p>;
        }
        if (trimmed === '') return <div key={i} className="h-1" />;
        return <p key={i} className="text-xs text-gray-700 leading-relaxed">{line}</p>;
      })}
    </div>
  );
}

export default function ProcessMinerPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('discover');
  const [processType, setProcessType] = useState('order');
  const [nCases, setNCases] = useState(100);

  return (
    <div className="space-y-4">
      {/* 서브탭 */}
      <div className="flex gap-2" role="tablist" aria-label="Process Miner 탭">
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={active}
              tabIndex={active ? 0 : -1}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all ${
                active
                  ? 'bg-gradient-to-r from-teal-500 to-teal-600 text-white shadow-lg shadow-teal-200'
                  : 'bg-white/80 text-gray-600 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* 공통 컨트롤 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[160px]">
            <label className="mb-1 block text-xs text-gray-500">프로세스 유형</label>
            <select
              value={processType}
              onChange={e => setProcessType(e.target.value)}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-teal-400 focus:outline-none"
            >
              {PROCESS_TYPES.map(pt => (
                <option key={pt.value} value={pt.value}>{pt.label}</option>
              ))}
            </select>
          </div>
          <div className="min-w-[120px]">
            <label className="mb-1 block text-xs text-gray-500">케이스 수</label>
            <select
              value={nCases}
              onChange={e => setNCases(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-teal-400 focus:outline-none"
            >
              {CASE_OPTIONS.map(n => (
                <option key={n} value={n}>{n}건</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {activeTab === 'discover' && (
        <DiscoverTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
      {activeTab === 'bottleneck' && (
        <BottleneckTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
      {activeTab === 'recommend' && (
        <RecommendTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
    </div>
  );
}

// ═══════════════════════════════════════
// 탭 1: 프로세스 발견
// ═══════════════════════════════════════
function DiscoverTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showMermaid, setShowMermaid] = useState(false);

  // 다음 활동 예측 상태
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
        auth,
        method: 'POST',
        data: { process_type: processType, n_cases: nCases, case_id: predictCaseId || '' },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        setPredictResult(res.data || res);
      } else {
        toast.error(getErrorMsg(res));
      }
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
        auth,
        method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        const d = res.data || res;
        // Normalize: backend-2 uses patterns[].path/percentage, confirmed uses top_patterns[].sequence/ratio
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
          {/* 요약 카드 */}
          <div className="grid grid-cols-3 gap-3">
            <StatCard icon={BarChart3} label="총 케이스" value={formatNumber(result.total_cases)} color="teal" />
            <StatCard icon={GitBranch} label="고유 패턴" value={formatNumber(result.unique_patterns)} color="indigo" />
            <StatCard icon={Clock} label="평균 처리시간" value={result.avg_duration_minutes != null ? formatMinutes(result.avg_duration_minutes) : result.total_avg_process_hours != null ? formatMinutes(result.total_avg_process_hours * 60) : '-'} color="amber" />
          </div>

          {/* 프로세스 플로우 */}
          {result.transitions?.length > 0 && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-teal-600" />
                  <span className="text-sm font-bold text-gray-700">프로세스 플로우</span>
                </div>
                {result.mermaid_diagram && (
                  <button
                    onClick={() => setShowMermaid(!showMermaid)}
                    className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showMermaid ? 'Flow 보기' : 'Mermaid 코드'}
                  </button>
                )}
              </div>

              {showMermaid && result.mermaid_diagram ? (
                <pre className="rounded-lg bg-gray-900 p-4 text-xs text-gray-200 overflow-x-auto whitespace-pre-wrap">
                  {result.mermaid_diagram}
                </pre>
              ) : (
                <ProcessFlowDiagram
                  transitions={result.transitions}
                  selectedNode={selectedNode}
                  onNodeClick={setSelectedNode}
                />
              )}

              {/* 선택된 노드 상세 */}
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
                          {t.probability != null && (
                            <span className="text-gray-400">({formatPercent(t.probability)})</span>
                          )}
                          {t.avg_minutes != null && (
                            <span className="text-gray-400">{formatMinutes(t.avg_minutes)}</span>
                          )}
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Top 패턴 */}
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
                        <span className="rounded-full bg-teal-100 px-2 py-0.5 text-[10px] font-bold text-teal-700">
                          {formatPercent(pattern.ratio)}
                        </span>
                        {(pattern.avg_duration_minutes != null || pattern.avg_duration_hours != null) && (
                          <span className="text-xs text-gray-400">
                            {pattern.avg_duration_minutes != null
                              ? formatMinutes(pattern.avg_duration_minutes)
                              : formatMinutes(pattern.avg_duration_hours * 60)}
                          </span>
                        )}
                      </div>
                    </div>
                    {/* 시퀀스 */}
                    <div className="flex flex-wrap items-center gap-1">
                      {(pattern.sequence || []).map((step, j) => (
                        <span key={j} className="flex items-center gap-1">
                          <span className="rounded-lg bg-gradient-to-r from-teal-50 to-teal-100 border border-teal-200 px-2 py-1 text-[11px] font-semibold text-teal-800">
                            {step}
                          </span>
                          {j < pattern.sequence.length - 1 && (
                            <ArrowRight size={12} className="text-gray-300" />
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 전이 행렬 */}
          {result.transitions?.length > 0 && (
            <TransitionMatrix transitions={result.transitions} />
          )}
        </>
      )}

      {/* ── 다음 활동 예측 섹션 ── */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <Brain size={16} className="text-indigo-600" />
          <span className="text-sm font-bold text-gray-700">다음 활동 예측</span>
        </div>

        {/* 케이스 ID 입력 + 예측 버튼 */}
        <div className="flex items-end gap-3">
          <div className="flex-1 min-w-[160px]">
            <label className="mb-1 block text-xs text-gray-500">케이스 ID (비워두면 랜덤 선택)</label>
            <input
              type="text"
              value={predictCaseId}
              onChange={e => setPredictCaseId(e.target.value)}
              placeholder="예: ORD-001"
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none"
            />
          </div>
          <button
            onClick={runPredict}
            disabled={predictLoading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-indigo-200 transition-all hover:shadow-xl disabled:opacity-50"
          >
            {predictLoading
              ? <><Loader2 size={14} className="animate-spin" /> 예측 중...</>
              : <><Brain size={14} /> 예측 실행</>}
          </button>
        </div>

        {/* 예측 결과 */}
        {predictResult && (
          <div className="space-y-3">
            {/* 케이스 정보 + 모델 정확도 */}
            <div className="flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">케이스:</span>
                <span className="rounded-full bg-indigo-100 px-2.5 py-0.5 text-xs font-bold text-indigo-700">{predictResult.case_id}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">현재 활동:</span>
                <span className="rounded-lg bg-gradient-to-r from-indigo-50 to-indigo-100 border border-indigo-200 px-2 py-1 text-[11px] font-semibold text-indigo-800">
                  {predictResult.current_activity}
                </span>
              </div>
              {predictResult.model_accuracy != null && (
                <span className="ml-auto rounded-full bg-emerald-100 px-2.5 py-0.5 text-[10px] font-bold text-emerald-700">
                  모델 정확도 {formatPercent(predictResult.model_accuracy)}
                </span>
              )}
            </div>

            {/* 예측 확률 바 */}
            {predictResult.predictions?.length > 0 && (
              <div className="rounded-xl border border-indigo-100 bg-indigo-50/30 p-3 space-y-2">
                <span className="text-xs font-bold text-gray-600">예측 결과</span>
                {predictResult.predictions.map((pred, i) => {
                  const barWidth = Math.max(pred.probability * 100, 2);
                  const barColor = i === 0
                    ? 'bg-gradient-to-r from-indigo-500 to-indigo-400'
                    : i === 1
                      ? 'bg-gradient-to-r from-indigo-400 to-indigo-300'
                      : 'bg-gradient-to-r from-indigo-300 to-indigo-200';
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs font-semibold text-gray-700 min-w-[80px] truncate">{pred.activity}</span>
                      <div className="flex-1 h-5 rounded-full bg-gray-100 overflow-hidden">
                        <div
                          className={`h-5 rounded-full ${barColor} flex items-center justify-end pr-2 transition-all`}
                          style={{ width: `${barWidth}%` }}
                        >
                          {barWidth > 15 && (
                            <span className="text-[10px] font-bold text-white">{formatPercent(pred.probability)}</span>
                          )}
                        </div>
                      </div>
                      {barWidth <= 15 && (
                        <span className="text-[10px] font-bold text-gray-500">{formatPercent(pred.probability)}</span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* 피처 중요도 (접기/펼치기) */}
            {predictResult.feature_importance && (
              <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
                <button
                  onClick={() => setShowPredictFeatures(!showPredictFeatures)}
                  className="flex w-full items-center gap-2 p-3 text-left hover:bg-gray-50 transition-colors"
                >
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

// 프로세스 플로우 다이어그램 (transitions 기반 커스텀 렌더링)
function ProcessFlowDiagram({ transitions, selectedNode, onNodeClick }) {
  if (!transitions?.length) return null;

  // 노드 추출
  const nodesSet = new Set();
  transitions.forEach(t => {
    nodesSet.add(t.from);
    nodesSet.add(t.to);
  });
  const nodes = Array.from(nodesSet);

  // 각 노드별 outgoing 합계로 순서 결정
  const outCount = {};
  const inCount = {};
  nodes.forEach(n => { outCount[n] = 0; inCount[n] = 0; });
  transitions.forEach(t => {
    outCount[t.from] = (outCount[t.from] || 0) + t.count;
    inCount[t.to] = (inCount[t.to] || 0) + t.count;
  });

  // 시작 노드 (inCount가 0이거나 가장 적은 노드)
  const startNodes = nodes.filter(n => inCount[n] === 0);
  const endNodes = nodes.filter(n => outCount[n] === 0);

  // 위상 정렬 (간단 BFS)
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

  // 미방문 노드 처리
  nodes.forEach(n => {
    if (levels[n] === undefined) levels[n] = 999;
  });

  // 레벨별 그룹
  const levelGroups = {};
  nodes.forEach(n => {
    const lv = levels[n];
    if (!levelGroups[lv]) levelGroups[lv] = [];
    levelGroups[lv].push(n);
  });
  const sortedLevels = Object.keys(levelGroups).map(Number).sort((a, b) => a - b);

  // 최대 빈도 (엣지 굵기용)
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
            {/* 화살표 (마지막 레벨 제외) */}
            {li < sortedLevels.length - 1 && (
              <div className="flex flex-col gap-1 items-center min-w-[40px]">
                {transitions
                  .filter(t => levelGroups[lv].includes(t.from))
                  .slice(0, 3)
                  .map((t, ti) => {
                    const thickness = Math.max(1, Math.round((t.count / maxCount) * 4));
                    return (
                      <div key={ti} className="flex items-center gap-1">
                        <div
                          className="bg-teal-400 rounded-full"
                          style={{ height: `${thickness}px`, width: '24px' }}
                        />
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

// 전이 행렬
function TransitionMatrix({ transitions }) {
  const [open, setOpen] = useState(false);

  const nodesSet = new Set();
  transitions.forEach(t => { nodesSet.add(t.from); nodesSet.add(t.to); });
  const nodes = Array.from(nodesSet);

  const matrix = {};
  let maxProb = 0;
  transitions.forEach(t => {
    if (!matrix[t.from]) matrix[t.from] = {};
    matrix[t.from][t.to] = t.probability != null ? t.probability : t.count;
    if (t.probability != null && t.probability > maxProb) maxProb = t.probability;
  });
  if (maxProb === 0) maxProb = Math.max(...transitions.map(t => t.count), 1);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white/80 backdrop-blur overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 p-4 text-left hover:bg-gray-50 transition-colors"
      >
        {open ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <BarChart3 size={16} className="text-teal-600" />
        <span className="text-sm font-bold text-gray-700">전이 행렬</span>
        <span className="ml-auto text-xs text-gray-400">{nodes.length}개 상태</span>
      </button>
      {open && (
        <div className="border-t border-gray-100 p-4 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="pb-2 pr-2 text-left text-gray-400 font-medium">From \ To</th>
                {nodes.map(n => (
                  <th key={n} className="pb-2 px-1 text-center text-gray-500 font-medium min-w-[60px]">{n}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {nodes.map(from => (
                <tr key={from} className="border-t border-gray-50">
                  <td className="py-1.5 pr-2 font-semibold text-gray-700">{from}</td>
                  {nodes.map(to => {
                    const val = matrix[from]?.[to];
                    const intensity = val != null ? Math.min(val / maxProb, 1) : 0;
                    return (
                      <td key={to} className="py-1.5 px-1 text-center">
                        {val != null ? (
                          <span
                            className="inline-block rounded px-1.5 py-0.5 text-[10px] font-bold"
                            style={{
                              backgroundColor: `rgba(20, 184, 166, ${intensity * 0.3 + 0.05})`,
                              color: intensity > 0.5 ? '#0f766e' : '#6b7280',
                            }}
                          >
                            {typeof val === 'number' && val < 1 ? formatPercent(val) : val}
                          </span>
                        ) : (
                          <span className="text-gray-200">-</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════
// 탭 2: 병목 분석
// ═══════════════════════════════════════
function BottleneckTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [expandedBottleneck, setExpandedBottleneck] = useState(null);

  // 이상 프로세스 탐지 상태
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [showAnomalyFeatures, setShowAnomalyFeatures] = useState(false);

  const runAnomalyDetection = async () => {
    setAnomalyLoading(true);
    setAnomalyResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/anomalies',
        auth,
        method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        setAnomalyResult(res.data || res);
      } else {
        toast.error(getErrorMsg(res));
      }
    } catch (e) {
      toast.error('이상 탐지 요청 실패: ' + (e.message || ''));
    } finally {
      setAnomalyLoading(false);
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    setResult(null);
    setExpandedBottleneck(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/bottlenecks',
        auth,
        method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        const d = res.data || res;
        // Normalize: backend-2 uses transition string ("A → B"), confirmed uses from/to
        if (d.bottlenecks) {
          d.bottlenecks = d.bottlenecks.map(bn => {
            if (bn.transition && !bn.from) {
              const parts = bn.transition.split(/\s*→\s*/);
              return { ...bn, from: parts[0], to: parts[1] || '' };
            }
            return bn;
          });
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
        onClick={runAnalysis}
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-teal-200 transition-all hover:shadow-xl disabled:opacity-50"
      >
        {loading ? <><Loader2 size={16} className="animate-spin" /> 병목 분석 중...</> : <><Play size={16} /> 분석 시작</>}
      </button>

      {result && (
        <>
          {/* 효율성 점수 */}
          {result.efficiency_score != null && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-6 backdrop-blur flex items-center justify-center">
              <EfficiencyGauge score={result.efficiency_score} />
            </div>
          )}

          {/* 병목 구간 */}
          {result.bottlenecks?.length > 0 && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle size={16} className="text-amber-600" />
                <span className="text-sm font-bold text-gray-700">병목 구간</span>
                <span className="ml-auto text-xs text-gray-400">{result.bottlenecks.length}건 발견</span>
              </div>
              <div className="space-y-2">
                {result.bottlenecks.map((bn, i) => (
                  <BottleneckItem
                    key={i}
                    bn={bn}
                    maxDuration={Math.max(...result.bottlenecks.map(b => b.avg_duration_minutes || b.avg_minutes || 0), 1)}
                    expanded={expandedBottleneck === i}
                    onToggle={() => setExpandedBottleneck(expandedBottleneck === i ? null : i)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* 시간대별 분석 */}
          {(result.time_distribution || result.time_analysis) && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
              <div className="flex items-center gap-2 mb-3">
                <Clock size={16} className="text-indigo-600" />
                <span className="text-sm font-bold text-gray-700">시간대별 분석</span>
              </div>
              <TimeAnalysisChart data={result.time_distribution || result.time_analysis} />
            </div>
          )}
        </>
      )}

      {/* ── 이상 프로세스 탐지 섹션 ── */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <ShieldAlert size={16} className="text-red-500" />
          <span className="text-sm font-bold text-gray-700">이상 프로세스 탐지</span>
        </div>

        <button
          onClick={runAnomalyDetection}
          disabled={anomalyLoading}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-orange-500 px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-red-200 transition-all hover:shadow-xl disabled:opacity-50"
        >
          {anomalyLoading
            ? <><Loader2 size={14} className="animate-spin" /> 이상 탐지 중...</>
            : <><AlertTriangle size={14} /> 이상 프로세스 탐지</>}
        </button>

        {/* 이상 탐지 결과 */}
        {anomalyResult && (
          <div className="space-y-3">
            {/* 요약 카드 */}
            <div className="grid grid-cols-3 gap-3">
              <StatCard icon={BarChart3} label="전체 케이스" value={formatNumber(anomalyResult.total_cases)} color="teal" />
              <StatCard icon={AlertTriangle} label="이상 케이스" value={formatNumber(anomalyResult.anomaly_count)} color="amber" />
              <StatCard icon={ShieldAlert} label="이상 비율" value={formatPercent(anomalyResult.anomaly_ratio)} color="amber" />
            </div>

            {/* 정상 패턴 요약 */}
            {anomalyResult.normal_pattern_summary && (
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 size={14} className="text-emerald-600" />
                  <span className="text-xs font-bold text-emerald-800">정상 패턴 요약</span>
                </div>
                <p className="text-xs text-gray-600 leading-relaxed">{anomalyResult.normal_pattern_summary}</p>
              </div>
            )}

            {/* 이상 케이스 리스트 */}
            {anomalyResult.anomalies?.length > 0 && (
              <div className="rounded-xl border border-red-100 bg-red-50/30 p-3 space-y-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-red-500" />
                  <span className="text-xs font-bold text-gray-700">이상 케이스</span>
                  <span className="ml-auto text-[10px] text-gray-400">
                    상위 {Math.min(anomalyResult.anomalies.length, 10)}건 표시
                  </span>
                </div>
                <div className="space-y-2 max-h-[480px] overflow-y-auto">
                  {anomalyResult.anomalies.slice(0, 10).map((anomaly, i) => (
                    <div key={i} className="rounded-lg border border-red-200 bg-white/80 p-3 space-y-2">
                      {/* 헤더: case_id, score, 배지 */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-bold text-gray-800">{anomaly.case_id}</span>
                        <span className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold text-red-700">
                          점수 {anomaly.anomaly_score?.toFixed(4)}
                        </span>
                        {anomaly.has_loop && (
                          <span className="rounded-full bg-orange-100 px-2 py-0.5 text-[10px] font-bold text-orange-700 flex items-center gap-1">
                            <RefreshCw size={10} /> 루프
                          </span>
                        )}
                        <div className="ml-auto flex items-center gap-2">
                          <span className="text-[10px] text-gray-400">{anomaly.sequence_length}단계</span>
                          {anomaly.total_duration_minutes != null && (
                            <span className="text-[10px] text-gray-500 font-medium">
                              {formatMinutes(anomaly.total_duration_minutes)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* 시퀀스 (화살표 연결) */}
                      {anomaly.sequence?.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1">
                          {anomaly.sequence.map((step, j) => {
                            const isException = anomaly.exception_steps?.includes(step);
                            return (
                              <span key={j} className="flex items-center gap-1">
                                <span className={`rounded-lg px-2 py-1 text-[11px] font-semibold border ${
                                  isException
                                    ? 'bg-gradient-to-r from-red-50 to-red-100 border-red-300 text-red-800'
                                    : 'bg-gradient-to-r from-gray-50 to-gray-100 border-gray-200 text-gray-700'
                                }`}>
                                  {step}
                                </span>
                                {j < anomaly.sequence.length - 1 && (
                                  <ArrowRight size={10} className="text-gray-300" />
                                )}
                              </span>
                            );
                          })}
                        </div>
                      )}

                      {/* 예외 활동 배지 */}
                      {anomaly.exception_steps?.length > 0 && (
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] text-gray-400">예외 활동:</span>
                          {anomaly.exception_steps.map((step, j) => (
                            <span key={j} className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold text-red-600">
                              {step}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* 사유 */}
                      {anomaly.reason && (
                        <p className="text-[11px] text-gray-500 leading-relaxed bg-gray-50 rounded-lg px-2 py-1.5">
                          {anomaly.reason}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 피처 중요도 (접기/펼치기) */}
            {anomalyResult.feature_importance && (
              <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
                <button
                  onClick={() => setShowAnomalyFeatures(!showAnomalyFeatures)}
                  className="flex w-full items-center gap-2 p-3 text-left hover:bg-gray-50 transition-colors"
                >
                  {showAnomalyFeatures ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
                  <BarChart3 size={14} className="text-red-500" />
                  <span className="text-xs font-bold text-gray-600">피처 중요도</span>
                </button>
                {showAnomalyFeatures && (
                  <div className="border-t border-gray-100 p-3">
                    <FeatureImportanceChart features={anomalyResult.feature_importance} color="red" />
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

// 효율성 게이지
function EfficiencyGauge({ score }) {
  // score can be 0-1 (backend) or 0-100; normalize to 0-100
  const pct = Math.round(score <= 1 ? score * 100 : score);
  const color = pct >= 80 ? 'text-emerald-500' : pct >= 50 ? 'text-amber-500' : 'text-red-500';
  const bgColor = pct >= 80 ? 'stroke-emerald-500' : pct >= 50 ? 'stroke-amber-500' : 'stroke-red-500';
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-[160px] h-[160px]">
        <svg width="160" height="160" className="-rotate-90">
          <circle cx="80" cy="80" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="10" />
          <circle
            cx="80" cy="80" r={radius} fill="none"
            className={bgColor}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-black ${color}`}>{pct}</span>
          <span className="text-xs text-gray-400">/ 100</span>
        </div>
      </div>
      <span className="mt-2 text-sm font-semibold text-gray-600">프로세스 효율성 점수</span>
    </div>
  );
}

// 병목 아이템
function BottleneckItem({ bn, maxDuration, expanded, onToggle }) {
  const severityColor = {
    high: { border: 'border-red-200', bg: 'bg-red-50/50', badge: 'bg-red-100 text-red-700', bar: 'bg-red-400' },
    medium: { border: 'border-amber-200', bg: 'bg-amber-50/50', badge: 'bg-amber-100 text-amber-700', bar: 'bg-amber-400' },
    low: { border: 'border-emerald-200', bg: 'bg-emerald-50/50', badge: 'bg-emerald-100 text-emerald-700', bar: 'bg-emerald-400' },
  };
  const sc = severityColor[bn.severity] || severityColor.medium;
  const avgMin = bn.avg_duration_minutes || bn.avg_minutes || 0;
  const medianMin = bn.median_duration_minutes || bn.median_minutes || 0;
  const p95Min = bn.p95_duration_minutes || bn.p95_minutes || 0;
  const barWidth = Math.min((avgMin / maxDuration) * 100, 100);

  return (
    <div className={`rounded-xl border ${sc.border} ${sc.bg} overflow-hidden`}>
      <button
        onClick={onToggle}
        className="flex w-full items-center gap-2 p-3 text-left hover:bg-white/30 transition-colors"
      >
        {expanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <span className="text-xs font-bold text-gray-800">
          {bn.from_step || bn.from} <ArrowRight size={12} className="inline text-gray-400 mx-1" /> {bn.to_step || bn.to}
        </span>
        <span className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-bold ${sc.badge}`}>
          {bn.severity}
        </span>
      </button>

      <div className="px-3 pb-3">
        {/* 소요시간 바 */}
        <div className="mb-2">
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div className={`h-2 rounded-full ${sc.bar} transition-all`} style={{ width: `${barWidth}%` }} />
          </div>
        </div>

        {/* 통계 */}
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <MiniStat label="평균" value={formatMinutes(avgMin)} />
          <MiniStat label="중앙값" value={formatMinutes(medianMin)} />
          <MiniStat label="P95" value={formatMinutes(p95Min)} />
          <MiniStat label="케이스" value={`${formatNumber(bn.case_count)}건`} />
        </div>
        {bn.outlier_cases?.length > 0 && (
          <div className="mt-1 text-[10px] text-red-500 font-medium">
            이상치 {bn.outlier_cases.length}건 감지
          </div>
        )}

        {/* 이상치 케이스 아코디언 */}
        {expanded && bn.outlier_cases?.length > 0 && (
          <div className="mt-3 rounded-lg border border-gray-200 bg-white/60 p-2">
            <p className="text-[10px] font-bold text-gray-500 mb-1">이상치 케이스</p>
            <div className="space-y-1 max-h-[120px] overflow-y-auto">
              {bn.outlier_cases.map((oc, i) => {
                // outlier_cases can be string[] (case IDs) or object[]
                const isString = typeof oc === 'string';
                return (
                  <div key={i} className="flex items-center justify-between text-[10px] text-gray-600 px-1">
                    <span>{isString ? oc : `Case #${oc.case_id || i + 1}`}</span>
                    {!isString && oc.duration_minutes != null && (
                      <span className="font-semibold text-red-600">{formatMinutes(oc.duration_minutes)}</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// 미니 통계
function MiniStat({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-gray-400">{label}</div>
      <div className="text-xs font-bold text-gray-700">{value}</div>
    </div>
  );
}

// 시간대별 분석 차트
function TimeAnalysisChart({ data }) {
  const periods = [
    { key: 'morning', label: '아침', emoji: '06-12시' },
    { key: 'afternoon', label: '점심', emoji: '12-18시' },
    { key: 'evening', label: '저녁', emoji: '18-24시' },
    { key: 'night', label: '심야', emoji: '00-06시' },
  ];

  // data[key] can be a number (flat) or { avg_minutes, case_count } (nested)
  const values = periods.map(p => {
    const v = data[p.key];
    if (v == null) return 0;
    if (typeof v === 'object') return v.avg_minutes || 0;
    return v;
  });
  const maxVal = Math.max(...values, 1);

  return (
    <div className="flex items-end gap-3 h-[120px]">
      {periods.map((p, i) => {
        const val = values[i];
        const raw = data[p.key];
        const caseCount = typeof raw === 'object' ? raw?.case_count : null;
        const height = Math.max((val / maxVal) * 100, 4);
        return (
          <div key={p.key} className="flex-1 flex flex-col items-center justify-end h-full">
            <span className="text-[10px] font-bold text-gray-600 mb-1">{formatMinutes(val)}</span>
            <div
              className="w-full rounded-t-lg bg-gradient-to-t from-teal-500 to-teal-300 transition-all"
              style={{ height: `${height}%` }}
            />
            <div className="mt-2 text-center">
              <div className="text-xs font-semibold text-gray-700">{p.label}</div>
              <div className="text-[10px] text-gray-400">{p.emoji}</div>
              {caseCount != null && (
                <div className="text-[9px] text-gray-400">{caseCount}건</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ═══════════════════════════════════════
// 탭 3: AI 자동화 추천
// ═══════════════════════════════════════
function RecommendTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showSop, setShowSop] = useState(false);

  const runRecommend = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/recommend',
        auth,
        method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        setResult(res.data || res);
      } else {
        toast.error(getErrorMsg(res));
      }
    } catch (e) {
      toast.error('분석 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

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

  return (
    <div className="space-y-4">
      <button
        onClick={runRecommend}
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-teal-200 transition-all hover:shadow-xl disabled:opacity-50"
      >
        {loading ? <><Loader2 size={16} className="animate-spin" /> AI 추천 생성 중...</> : <><Play size={16} /> 분석 시작</>}
      </button>

      {result && (
        <>
          {/* 요약 */}
          <div className="rounded-2xl border-2 border-teal-200 bg-gradient-to-br from-teal-50 to-emerald-50 p-6 text-center">
            <TrendingUp size={28} className="mx-auto mb-2 text-teal-600" />
            {(result.estimated_time_saving_percent != null || result.estimated_improvement != null) && (
              <p className="text-2xl font-black text-teal-700 mb-1">
                약 {result.estimated_time_saving_percent != null
                  ? `${result.estimated_time_saving_percent}%`
                  : formatPercent(result.estimated_improvement)} 효율 개선 가능
              </p>
            )}
            {result.summary && (
              <p className="text-sm text-gray-600">{result.summary}</p>
            )}
          </div>

          {/* 추천 목록 */}
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
                        <div className={`rounded-lg p-2 ${iconColor}`}>
                          <AutoIcon size={18} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-bold text-gray-800">{rec.target_step}</span>
                            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${priorityBadge[rec.priority] || priorityBadge.medium}`}>
                              {rec.priority}
                            </span>
                            {rec.automation_type && (
                              <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-medium text-gray-500">
                                {rec.automation_type.replace('_', ' ')}
                              </span>
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

          {/* SOP 문서 */}
          {result.sop_document && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 backdrop-blur overflow-hidden">
              <button
                onClick={() => setShowSop(!showSop)}
                className="flex w-full items-center gap-2 p-4 text-left hover:bg-gray-50 transition-colors"
              >
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

// 피처 중요도 가로 막대 차트 (공통)
function FeatureImportanceChart({ features, color = 'indigo' }) {
  if (!features || typeof features !== 'object') return null;

  const entries = Object.entries(features).sort((a, b) => b[1] - a[1]);
  const maxVal = Math.max(...entries.map(e => e[1]), 0.01);

  const barColors = {
    indigo: 'bg-gradient-to-r from-indigo-500 to-indigo-400',
    red: 'bg-gradient-to-r from-red-500 to-orange-400',
    teal: 'bg-gradient-to-r from-teal-500 to-teal-400',
  };
  const barColor = barColors[color] || barColors.indigo;

  return (
    <div className="space-y-1.5">
      {entries.map(([key, val]) => {
        const barWidth = Math.max((val / maxVal) * 100, 2);
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500 min-w-[100px] truncate text-right">{key}</span>
            <div className="flex-1 h-3.5 rounded-full bg-gray-100 overflow-hidden">
              <div
                className={`h-3.5 rounded-full ${barColor} transition-all`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <span className="text-[10px] font-bold text-gray-600 min-w-[36px]">{(val * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

