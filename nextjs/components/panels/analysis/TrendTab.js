// components/panels/analysis/TrendTab.js
// 트렌드 분석 탭

import {
  TrendingUp, ArrowUpRight, ArrowDownRight, Brain, BarChart3
} from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import {
  XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, LineChart, Line, AreaChart, Area
} from 'recharts';

export default function TrendTab({ trendData }) {
  return (
    <div className="space-y-6">
      {!trendData ? (
        <div className="text-center py-16 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80">
          <TrendingUp size={48} className="mx-auto mb-3 text-cafe24-brown/30" />
          <p className="text-sm font-semibold text-cafe24-brown/50">트렌드 데이터를 불러올 수 없습니다</p>
          <p className="text-xs text-cafe24-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
        </div>
      ) : (
      <>
      {/* KPI 요약 카드 */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {(trendData.kpis || []).map((kpi, idx) => (
          <div key={idx} className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-cafe24-brown/60">{kpi.name}</span>
              <span className={`flex items-center gap-1 text-xs font-bold ${
                kpi.change >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {kpi.change >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                {kpi.change >= 0 ? '+' : ''}{kpi.change}%
              </span>
            </div>
            <div className="text-2xl font-black text-cafe24-brown">
              {kpi.name.includes('ARPU') ? '₩' : ''}{typeof kpi.current === 'number' ? kpi.current.toLocaleString() : kpi.current}{kpi.name.includes('률') || kpi.name.includes('전환') ? '%' : ''}
            </div>
            <div className="text-xs text-cafe24-brown/50">이전: {kpi.previous.toLocaleString()}</div>
          </div>
        ))}
      </div>

      {/* 일별 메트릭 차트 */}
      {(trendData.daily_metrics?.length > 0) && (
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="mb-4 text-sm font-black text-cafe24-brown">일별 핵심 지표 추이</div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData.daily_metrics}>
            <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
            <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
            <YAxis yAxisId="left" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="dau" name="DAU" stroke="#FF8C42" strokeWidth={2} dot={{ r: 4 }} />
            <Line yAxisId="left" type="monotone" dataKey="new_users" name="신규가입" stroke="#4ADE80" strokeWidth={2} dot={{ r: 4 }} />
            <Line yAxisId="right" type="monotone" dataKey="sessions" name="세션수" stroke="#60A5FA" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      )}

      {/* 예측 & 상관관계 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="flex items-center gap-2 mb-4">
            <Brain size={18} className="text-cafe24-orange" />
            <span className="text-sm font-black text-cafe24-brown">DAU 예측 (5일)</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={trendData.forecast || []}>
              <defs>
                <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#A78BFA" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#A78BFA" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
              <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
              <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} domain={['dataMin - 20', 'dataMax + 20']} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="upper" name="상한" stroke="transparent" fill="#A78BFA" fillOpacity={0.2} />
              <Area type="monotone" dataKey="lower" name="하한" stroke="transparent" fill="transparent" />
              <Line type="monotone" dataKey="predicted_dau" name="예측 DAU" stroke="#A78BFA" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 size={18} className="text-cafe24-orange" />
            <span className="text-sm font-black text-cafe24-brown">지표 상관관계</span>
          </div>
          <div className="space-y-3">
            {(trendData.correlation || []).map((item, idx) => {
              const corr = item.correlation ?? 0;
              return (
              <div key={idx} className="flex items-center gap-3">
                <div className="flex-1">
                  <div className="flex justify-between mb-1">
                    <span className="text-xs font-semibold text-cafe24-brown">{item.var1 || item.metric1} ↔ {item.var2 || item.metric2}</span>
                    <span className={`text-xs font-bold ${
                      corr >= 0.8 ? 'text-green-600' :
                      corr >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {corr.toFixed(2)}
                    </span>
                  </div>
                  <div className="h-2 bg-cafe24-beige rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        corr >= 0.8 ? 'bg-green-500' :
                        corr >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${corr * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              );
            })}
          </div>
        </div>
      </div>
      </>
      )}
    </div>
  );
}
