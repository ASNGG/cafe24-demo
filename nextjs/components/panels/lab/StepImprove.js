// components/panels/lab/StepImprove.js - Step 5: 개선 - 피드백 & 대시보드
import { useState, useCallback, useEffect } from 'react';
import { TrendingUp, Inbox, Clock } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { PRIORITY_COLORS } from './constants';
import { KpiMini } from './utils';

const BAR_COLORS = ['#D97B4A', '#EAC54F', '#5C4A3D', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ef4444', '#6b7280'];

export default function StepImprove({ result, history, apiCall, auth }) {
  const [stats, setStats] = useState(result?.statistics || null);
  const [loadingStats, setLoadingStats] = useState(false);

  const loadStats = useCallback(async () => {
    setLoadingStats(true);
    try {
      const res = await apiCall({ endpoint: '/api/cs/statistics', auth });
      if (res?.status === 'success') {
        setStats(res);
      }
    } catch {}
    setLoadingStats(false);
  }, [apiCall, auth]);

  useEffect(() => {
    if (!stats) loadStats();
  }, [stats, loadStats]);

  const chartData = stats?.by_category
    ? Object.entries(stats.by_category).map(([name, val]) => ({
        name,
        tickets: val.total_tickets || 0,
        satisfaction: val.satisfaction_score || 0,
        hours: val.avg_resolution_hours || 0,
      }))
    : [];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-cafe24-brown font-semibold text-lg">
          <TrendingUp className="w-5 h-5 text-cafe24-orange" />
          Step 5. 개선 - 피드백 & 대시보드
        </div>
        <button
          onClick={loadStats}
          disabled={loadingStats}
          className="text-xs px-2.5 py-1 rounded-lg border border-cafe24-brown/20 hover:bg-cafe24-yellow/10 text-cafe24-brown/60"
        >
          {loadingStats ? '로딩...' : '새로고침'}
        </button>
      </div>

      {/* KPI 카드 */}
      {stats && (
        <div className="grid grid-cols-3 gap-3">
          <KpiMini label="총 티켓 수" value={stats.total_tickets?.toLocaleString() || '-'} icon={Inbox} />
          <KpiMini label="평균 만족도" value={stats.avg_satisfaction_score ? `${stats.avg_satisfaction_score.toFixed(1)} / 5` : '-'} icon={TrendingUp} />
          <KpiMini label="평균 처리 시간" value={stats.avg_resolution_hours ? `${stats.avg_resolution_hours.toFixed(1)}h` : '-'} icon={Clock} />
        </div>
      )}

      {/* 카테고리별 차트 */}
      {chartData.length > 0 && (
        <div>
          <span className="text-sm font-medium text-cafe24-brown/80 mb-2 block">카테고리별 티켓 수</span>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(val, name) => {
                    if (name === 'tickets') return [val, '티켓 수'];
                    return [val, name];
                  }}
                />
                <Bar dataKey="tickets" radius={[4, 4, 0, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 파이프라인 메타 */}
      {result?.pipeline_meta && (
        <div className="p-3 rounded-lg bg-blue-50 border border-blue-100 space-y-1">
          <span className="text-xs text-blue-600 font-medium">파이프라인 메타정보</span>
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
            <div>분류 모델 정확도: <strong>{(result.pipeline_meta.classification_model_accuracy * 100).toFixed(0)}%</strong></div>
            <div>자동 처리 기준: <strong>{result.pipeline_meta.auto_routing_rate}</strong></div>
          </div>
        </div>
      )}

      {/* 파이프라인 처리 이력 */}
      {history.length > 0 && (
        <div>
          <span className="text-sm font-medium text-cafe24-brown/80 mb-2 block">파이프라인 처리 이력</span>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500">
                  <th className="text-left py-2 px-2">시간</th>
                  <th className="text-left py-2 px-2">문의 내용</th>
                  <th className="text-left py-2 px-2">카테고리</th>
                  <th className="text-left py-2 px-2">라우팅</th>
                  <th className="text-left py-2 px-2">우선순위</th>
                </tr>
              </thead>
              <tbody>
                {history.slice().reverse().map((row, i) => {
                  const pColor = PRIORITY_COLORS[row.priority] || PRIORITY_COLORS.normal;
                  return (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-1.5 px-2 text-gray-500">{row.time}</td>
                      <td className="py-1.5 px-2 text-gray-700">{row.text}</td>
                      <td className="py-1.5 px-2">
                        <span className="px-1.5 py-0.5 rounded bg-cafe24-orange/10 text-cafe24-orange font-medium">{row.category}</span>
                      </td>
                      <td className="py-1.5 px-2">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${
                          row.routing === 'auto' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}>{row.routing === 'auto' ? '자동' : '수동'}</span>
                      </td>
                      <td className="py-1.5 px-2">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${pColor.bg} ${pColor.text}`}>{row.priority}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
