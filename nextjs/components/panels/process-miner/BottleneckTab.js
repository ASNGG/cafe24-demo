// process-miner/BottleneckTab.js — 병목 분석 탭

import { useState, useMemo } from 'react';
import toast from 'react-hot-toast';
import StatCard from '@/components/common/StatCard';
import {
  Play, Loader2, ChevronDown, ChevronRight, ArrowRight,
  AlertTriangle, CheckCircle2, BarChart3, Clock, ShieldAlert, RefreshCw,
} from 'lucide-react';
import { isSuccess, getErrorMsg, formatMinutes, formatNumber, formatPercent } from './utils';
import { MiniStat, FeatureImportanceChart, EfficiencyGauge, TimeAnalysisChart } from './SharedComponents';

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
      <button onClick={onToggle} className="flex w-full items-center gap-2 p-3 text-left hover:bg-white/30 transition-colors">
        {expanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <span className="text-xs font-bold text-gray-800">
          {bn.from_step || bn.from} <ArrowRight size={12} className="inline text-gray-400 mx-1" /> {bn.to_step || bn.to}
        </span>
        <span className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-bold ${sc.badge}`}>{bn.severity}</span>
      </button>
      <div className="px-3 pb-3">
        <div className="mb-2">
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div className={`h-2 rounded-full ${sc.bar} transition-all`} style={{ width: `${barWidth}%` }} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <MiniStat label="평균" value={formatMinutes(avgMin)} />
          <MiniStat label="중앙값" value={formatMinutes(medianMin)} />
          <MiniStat label="P95" value={formatMinutes(p95Min)} />
          <MiniStat label="케이스" value={`${formatNumber(bn.case_count)}건`} />
        </div>
        {bn.outlier_cases?.length > 0 && (
          <div className="mt-1 text-[10px] text-red-500 font-medium">이상치 {bn.outlier_cases.length}건 감지</div>
        )}
        {expanded && bn.outlier_cases?.length > 0 && (
          <div className="mt-3 rounded-lg border border-gray-200 bg-white/60 p-2">
            <p className="text-[10px] font-bold text-gray-500 mb-1">이상치 케이스</p>
            <div className="space-y-1 max-h-[120px] overflow-y-auto">
              {bn.outlier_cases.map((oc, i) => {
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

// maxDuration useMemo: 매 렌더마다 O(N) 재계산 방지
function BottleneckList({ bottlenecks, expandedBottleneck, setExpandedBottleneck }) {
  const maxDuration = useMemo(
    () => Math.max(...bottlenecks.map(b => b.avg_duration_minutes || b.avg_minutes || 0), 1),
    [bottlenecks]
  );
  return (
    <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle size={16} className="text-amber-600" />
        <span className="text-sm font-bold text-gray-700">병목 구간</span>
        <span className="ml-auto text-xs text-gray-400">{bottlenecks.length}건 발견</span>
      </div>
      <div className="space-y-2">
        {bottlenecks.map((bn, i) => (
          <BottleneckItem
            key={i} bn={bn}
            maxDuration={maxDuration}
            expanded={expandedBottleneck === i}
            onToggle={() => setExpandedBottleneck(expandedBottleneck === i ? null : i)}
          />
        ))}
      </div>
    </div>
  );
}

export default function BottleneckTab({ auth, apiCall, processType, nCases }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [expandedBottleneck, setExpandedBottleneck] = useState(null);

  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [showAnomalyFeatures, setShowAnomalyFeatures] = useState(false);

  const runAnomalyDetection = async () => {
    setAnomalyLoading(true);
    setAnomalyResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/process-miner/anomalies',
        auth, method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) setAnomalyResult(res.data || res);
      else toast.error(getErrorMsg(res));
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
        auth, method: 'POST',
        data: { process_type: processType, n_cases: nCases },
        timeoutMs: 30000,
      });
      if (isSuccess(res)) {
        const d = res.data || res;
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
      <button onClick={runAnalysis} disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-teal-500 to-teal-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-teal-200 transition-all hover:shadow-xl disabled:opacity-50">
        {loading ? <><Loader2 size={16} className="animate-spin" /> 병목 분석 중...</> : <><Play size={16} /> 분석 시작</>}
      </button>

      {result && (
        <>
          {result.efficiency_score != null && (
            <div className="rounded-2xl border border-gray-200 bg-white/80 p-6 backdrop-blur flex items-center justify-center">
              <EfficiencyGauge score={result.efficiency_score} />
            </div>
          )}

          {result.bottlenecks?.length > 0 && (
            <BottleneckList
              bottlenecks={result.bottlenecks}
              expandedBottleneck={expandedBottleneck}
              setExpandedBottleneck={setExpandedBottleneck}
            />
          )}

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

      {/* 이상 프로세스 탐지 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur space-y-4">
        <div className="flex items-center gap-2 mb-1">
          <ShieldAlert size={16} className="text-red-500" />
          <span className="text-sm font-bold text-gray-700">이상 프로세스 탐지</span>
        </div>
        <button onClick={runAnomalyDetection} disabled={anomalyLoading}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-orange-500 px-5 py-2.5 text-sm font-bold text-white shadow-lg shadow-red-200 transition-all hover:shadow-xl disabled:opacity-50">
          {anomalyLoading ? <><Loader2 size={14} className="animate-spin" /> 이상 탐지 중...</> : <><AlertTriangle size={14} /> 이상 프로세스 탐지</>}
        </button>

        {anomalyResult && (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <StatCard icon={BarChart3} label="전체 케이스" value={formatNumber(anomalyResult.total_cases)} color="teal" />
              <StatCard icon={AlertTriangle} label="이상 케이스" value={formatNumber(anomalyResult.anomaly_count)} color="amber" />
              <StatCard icon={ShieldAlert} label="이상 비율" value={formatPercent(anomalyResult.anomaly_ratio)} color="amber" />
            </div>

            {anomalyResult.normal_pattern_summary && (
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircle2 size={14} className="text-emerald-600" />
                  <span className="text-xs font-bold text-emerald-800">정상 패턴 요약</span>
                </div>
                <p className="text-xs text-gray-600 leading-relaxed">{anomalyResult.normal_pattern_summary}</p>
              </div>
            )}

            {anomalyResult.anomalies?.length > 0 && (
              <div className="rounded-xl border border-red-100 bg-red-50/30 p-3 space-y-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={14} className="text-red-500" />
                  <span className="text-xs font-bold text-gray-700">이상 케이스</span>
                  <span className="ml-auto text-[10px] text-gray-400">상위 {Math.min(anomalyResult.anomalies.length, 10)}건 표시</span>
                </div>
                <div className="space-y-2 max-h-[480px] overflow-y-auto">
                  {anomalyResult.anomalies.slice(0, 10).map((anomaly, i) => (
                    <div key={i} className="rounded-lg border border-red-200 bg-white/80 p-3 space-y-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-bold text-gray-800">{anomaly.case_id}</span>
                        <span className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold text-red-700">점수 {anomaly.anomaly_score?.toFixed(4)}</span>
                        {anomaly.has_loop && (
                          <span className="rounded-full bg-orange-100 px-2 py-0.5 text-[10px] font-bold text-orange-700 flex items-center gap-1"><RefreshCw size={10} /> 루프</span>
                        )}
                        <div className="ml-auto flex items-center gap-2">
                          <span className="text-[10px] text-gray-400">{anomaly.sequence_length}단계</span>
                          {anomaly.total_duration_minutes != null && (
                            <span className="text-[10px] text-gray-500 font-medium">{formatMinutes(anomaly.total_duration_minutes)}</span>
                          )}
                        </div>
                      </div>
                      {anomaly.sequence?.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1">
                          {anomaly.sequence.map((step, j) => {
                            const isException = anomaly.exception_steps?.includes(step);
                            return (
                              <span key={j} className="flex items-center gap-1">
                                <span className={`rounded-lg px-2 py-1 text-[11px] font-semibold border ${isException ? 'bg-gradient-to-r from-red-50 to-red-100 border-red-300 text-red-800' : 'bg-gradient-to-r from-gray-50 to-gray-100 border-gray-200 text-gray-700'}`}>{step}</span>
                                {j < anomaly.sequence.length - 1 && <ArrowRight size={10} className="text-gray-300" />}
                              </span>
                            );
                          })}
                        </div>
                      )}
                      {anomaly.exception_steps?.length > 0 && (
                        <div className="flex items-center gap-1 flex-wrap">
                          <span className="text-[10px] text-gray-400">예외 활동:</span>
                          {anomaly.exception_steps.map((step, j) => (
                            <span key={j} className="rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold text-red-600">{step}</span>
                          ))}
                        </div>
                      )}
                      {anomaly.reason && (
                        <p className="text-[11px] text-gray-500 leading-relaxed bg-gray-50 rounded-lg px-2 py-1.5">{anomaly.reason}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {anomalyResult.feature_importance && (
              <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
                <button onClick={() => setShowAnomalyFeatures(!showAnomalyFeatures)}
                  className="flex w-full items-center gap-2 p-3 text-left hover:bg-gray-50 transition-colors">
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
