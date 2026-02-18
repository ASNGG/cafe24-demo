// components/panels/analysis/AnomalyTab.js
// 이상탐지 분석 탭

import { AlertTriangle, Shield, Eye, Activity, Zap } from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import { getSeverityClasses } from '@/components/common/constants';
import AnalysisEmptyState from './common/EmptyState';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, AreaChart, Area, Cell
} from 'recharts';

export default function AnomalyTab({ selectedUser, anomalyData }) {
  return (
    <div className="space-y-6">
      {/* 선택된 셀러 이상탐지 결과 */}
      {selectedUser?.model_predictions?.fraud && (
        <div className={`rounded-3xl border-2 p-5 shadow-sm backdrop-blur ${
          selectedUser.model_predictions.fraud.is_anomaly ? 'border-red-300 bg-red-50/80' : 'border-green-300 bg-green-50/80'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <Shield size={18} className={selectedUser.model_predictions.fraud.is_anomaly ? 'text-red-600' : 'text-green-600'} />
            <span className="text-sm font-black text-cafe24-brown">{selectedUser.id} 이상거래 탐지 결과</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-black" style={{
                color: selectedUser.model_predictions.fraud.is_anomaly ? '#DC2626' : '#16A34A'
              }}>{(selectedUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%</div>
              <div className="text-xs text-cafe24-brown/60">이상 점수</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-black ${
                selectedUser.model_predictions.fraud.is_anomaly ? 'text-red-600' : 'text-green-600'
              }`}>
                {selectedUser.model_predictions.fraud.risk_level}
              </div>
              <div className="text-xs text-cafe24-brown/60">판정</div>
            </div>
          </div>
          <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.fraud.model}</div>
        </div>
      )}
      {!anomalyData ? (
        <AnalysisEmptyState
          icon={AlertTriangle}
          title="이상탐지 데이터를 불러올 수 없습니다"
          subtitle="백엔드 API 연결을 확인하세요"
        />
      ) : (
      <>
      {/* 이상탐지 요약 카드 */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl border-2 border-red-200 bg-red-50 p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={18} className="text-red-500" />
            <span className="text-xs font-bold text-red-700">고위험</span>
          </div>
          <div className="text-2xl font-black text-red-600">{anomalyData.summary?.high_risk || 0}</div>
          <div className="text-xs text-red-600/70">즉시 조치 필요</div>
        </div>
        <div className="rounded-2xl border-2 border-orange-200 bg-orange-50 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Shield size={18} className="text-orange-500" />
            <span className="text-xs font-bold text-orange-700">중위험</span>
          </div>
          <div className="text-2xl font-black text-orange-600">{anomalyData.summary?.medium_risk || 0}</div>
          <div className="text-xs text-orange-600/70">모니터링 필요</div>
        </div>
        <div className="rounded-2xl border-2 border-yellow-200 bg-yellow-50 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Eye size={18} className="text-yellow-600" />
            <span className="text-xs font-bold text-yellow-700">저위험</span>
          </div>
          <div className="text-2xl font-black text-yellow-600">{anomalyData.summary?.low_risk || 0}</div>
          <div className="text-xs text-yellow-600/70">관찰 대상</div>
        </div>
        <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity size={18} className="text-cafe24-orange" />
            <span className="text-xs font-bold text-cafe24-brown">탐지율</span>
          </div>
          <div className="text-2xl font-black text-cafe24-brown">{anomalyData.summary?.anomaly_rate || 0}%</div>
          <div className="text-xs text-cafe24-brown/60">{anomalyData.summary?.anomaly_count || 0}/{anomalyData.summary?.total_sellers || 0}</div>
        </div>
      </div>

      {/* 이상유형별 분포 & 트렌드 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="mb-4 text-sm font-black text-cafe24-brown">이상 유형별 분포</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={anomalyData.by_type || []} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
              <YAxis type="category" dataKey="type" tick={{ fill: '#5C4A3D', fontSize: 10 }} width={120} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" name="탐지 수" radius={[0, 4, 4, 0]}>
                {(anomalyData.by_type || []).map((entry, idx) => (
                  <Cell key={idx} fill={entry.severity === 'high' ? '#EF4444' : entry.severity === 'medium' ? '#F97316' : '#EAB308'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="mb-4 text-sm font-black text-cafe24-brown">일별 이상 탐지 트렌드</div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={anomalyData.trend || []}>
              <defs>
                <linearGradient id="anomaly-colorAnomaly" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
              <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
              <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="count" name="탐지 수" stroke="#EF4444" fill="url(#anomaly-colorAnomaly)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 최근 알림 */}
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <Zap size={18} className="text-red-500" />
          <span className="text-sm font-black text-cafe24-brown">실시간 이상 탐지 알림</span>
        </div>
        <div className="space-y-3">
          {(anomalyData.recent_alerts || []).map((alert, idx) => {
            const sc = getSeverityClasses(alert.severity);
            return (
            <div key={idx} className={`flex items-center gap-4 p-4 rounded-2xl border-2 ${sc.border} ${sc.bg}`}>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${sc.icon}`}>
                <AlertTriangle size={18} className="text-white" />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-cafe24-brown">{alert.id}</span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${sc.badge}`}>{alert.type}</span>
                </div>
                <p className="text-sm text-cafe24-brown/70">{alert.detail}</p>
              </div>
              <div className="text-xs text-cafe24-brown/50">{alert.time}</div>
            </div>
            );
          })}
        </div>
      </div>
      </>
      )}
    </div>
  );
}
