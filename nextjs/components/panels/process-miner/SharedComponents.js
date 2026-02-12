// process-miner/SharedComponents.js — 공유 UI 컴포넌트

import { BarChart3, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { formatMinutes, formatPercent } from './utils';

// 간단한 마크다운 렌더러
export function SimpleMarkdown({ text }) {
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

// 미니 통계
export function MiniStat({ label, value }) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-gray-400">{label}</div>
      <div className="text-xs font-bold text-gray-700">{value}</div>
    </div>
  );
}

// 피처 중요도 차트
export function FeatureImportanceChart({ features, color = 'indigo' }) {
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

// 전이 행렬
export function TransitionMatrix({ transitions }) {
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

// 효율성 게이지
export function EfficiencyGauge({ score }) {
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

// 시간대별 분석 차트
export function TimeAnalysisChart({ data }) {
  const periods = [
    { key: 'morning', label: '아침', emoji: '06-12시' },
    { key: 'afternoon', label: '점심', emoji: '12-18시' },
    { key: 'evening', label: '저녁', emoji: '18-24시' },
    { key: 'night', label: '심야', emoji: '00-06시' },
  ];

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
