// components/panels/analysis/SellerTab.js
// 셀러 분석 탭

import { useMemo } from 'react';
import {
  User, Search, Brain, Shield, Users, MessageSquare,
  TrendingUp, UserMinus
} from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import { ANALYSIS_COLORS as COLORS } from '@/components/common/constants';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Line
} from 'recharts';

export default function SellerTab({
  loading, searchQuery, setSearchQuery, searchInputRef,
  handleUserSearch, quickSelectUsers, selectedUser,
}) {
  const userRadarData = useMemo(() => {
    if (!selectedUser?.stats) return [];
    return Object.entries(selectedUser.stats).map(([key, value]) => ({
      subject: key,
      value,
      fullMark: 100,
    }));
  }, [selectedUser]);

  return (
    <div className="space-y-6">
      {/* 셀러 검색 */}
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <Search size={18} className="text-cafe24-orange" />
          <span className="text-sm font-black text-cafe24-brown">셀러 검색</span>
        </div>
        <div className="flex gap-3">
          <div className="flex-1">
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleUserSearch();
                }
              }}
              placeholder="셀러 ID 입력 (예: SEL0001)"
              className="w-full px-4 py-2.5 rounded-xl border-2 border-cafe24-orange/20 bg-white text-sm text-cafe24-brown placeholder:text-cafe24-brown/40 outline-none focus:border-cafe24-orange transition"
            />
          </div>
          <button
            onClick={handleUserSearch}
            disabled={loading}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white font-bold text-sm shadow-md hover:shadow-lg transition disabled:opacity-50"
          >
            {loading ? '검색 중...' : '검색'}
          </button>
        </div>
        {/* 빠른 선택 */}
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="text-xs text-cafe24-brown/60">빠른 선택:</span>
          {quickSelectUsers.map(userId => (
            <button
              key={userId}
              onClick={() => { setSearchQuery(userId); }}
              className="px-2 py-1 rounded-lg bg-cafe24-beige text-xs font-semibold text-cafe24-brown hover:bg-cafe24-yellow/30 transition"
            >
              {userId}
            </button>
          ))}
        </div>
      </div>

      {/* 셀러 상세 정보 */}
      {selectedUser && (
        <>
          {/* 기본 데이터 */}
          <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-black text-cafe24-brown">{selectedUser.id}</h3>
                <p className="text-sm text-cafe24-brown/60">{selectedUser.segment} · {selectedUser.plan_tier} · {selectedUser.region}</p>
              </div>
              <div className="flex gap-2">
                <span className="px-3 py-1 rounded-full bg-cafe24-yellow/30 text-xs font-bold text-cafe24-brown">
                  매출 ₩{(selectedUser.monthly_revenue || 0).toLocaleString()}
                </span>
                {selectedUser.is_anomaly && (
                  <span className="px-3 py-1 rounded-full bg-red-100 text-xs font-bold text-red-600">이상감지</span>
                )}
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-3 rounded-2xl bg-cafe24-beige/50">
                <div className="text-2xl font-black text-cafe24-brown">{selectedUser.product_count || 0}</div>
                <div className="text-xs text-cafe24-brown/60">상품 수</div>
              </div>
              <div className="text-center p-3 rounded-2xl bg-cafe24-beige/50">
                <div className="text-2xl font-black text-cafe24-brown">{selectedUser.order_count || 0}</div>
                <div className="text-xs text-cafe24-brown/60">주문 수</div>
              </div>
              <div className="text-center p-3 rounded-2xl bg-cafe24-beige/50">
                <div className="text-2xl font-black text-cafe24-brown">{selectedUser.period_stats?.active_days || 0}</div>
                <div className="text-xs text-cafe24-brown/60">활동일수</div>
              </div>
              <div className="text-center p-3 rounded-2xl bg-cafe24-beige/50">
                <div className="text-2xl font-black text-cafe24-brown">{selectedUser.period_stats?.total_cs || 0}</div>
                <div className="text-xs text-cafe24-brown/60">CS건수</div>
              </div>
            </div>
          </div>

          {/* 차트 그리드 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 활동 트렌드 */}
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">일별 운영 트렌드</div>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={selectedUser.activity}>
                  <defs>
                    <linearGradient id="seller-colorProductCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#FFD93D" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#FFD93D" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Area type="monotone" dataKey="product_count" name="상품 수" stroke="#FFD93D" fill="url(#seller-colorProductCount)" />
                  <Line type="monotone" dataKey="orders" name="주문 수" stroke="#4ADE80" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* 셀러 스탯 레이더 */}
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">셀러 특성 분석</div>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={userRadarData}>
                  <PolarGrid stroke="#FFD93D60" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#5C4A3D', fontSize: 10 }} />
                  <Radar name="스탯" dataKey="value" stroke="#FF8C42" fill="#FF8C42" fillOpacity={0.5} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* ML 모델 예측 결과 */}
          {selectedUser.model_predictions && Object.keys(selectedUser.model_predictions).length > 0 && (
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-4">
                <Brain size={18} className="text-cafe24-orange" />
                <span className="text-sm font-black text-cafe24-brown">ML 모델 예측 결과</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* 이탈 예측 */}
                {selectedUser.model_predictions.churn && (
                  <div className={`rounded-2xl p-4 border-2 ${
                    selectedUser.model_predictions.churn.risk_code >= 2
                      ? 'border-red-300 bg-red-50'
                      : selectedUser.model_predictions.churn.risk_code === 1
                      ? 'border-orange-300 bg-orange-50'
                      : 'border-green-300 bg-green-50'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <UserMinus size={16} className={
                        selectedUser.model_predictions.churn.risk_code >= 2 ? 'text-red-600' :
                        selectedUser.model_predictions.churn.risk_code === 1 ? 'text-orange-600' : 'text-green-600'
                      } />
                      <span className="text-xs font-bold text-cafe24-brown">이탈 예측</span>
                    </div>
                    <div className="text-2xl font-black mb-1" style={{
                      color: selectedUser.model_predictions.churn.risk_code >= 2 ? '#DC2626' :
                             selectedUser.model_predictions.churn.risk_code === 1 ? '#EA580C' : '#16A34A'
                    }}>
                      {selectedUser.model_predictions.churn.probability}%
                    </div>
                    <div className="text-xs text-cafe24-brown/60 mb-2">
                      위험도: {selectedUser.model_predictions.churn.risk_level}
                    </div>
                    {selectedUser.model_predictions.churn.factors?.slice(0, 3).map((f, i) => (
                      <div key={i} className="flex justify-between text-xs mt-1">
                        <span className="text-cafe24-brown/70">{f.factor}</span>
                        <span className="font-semibold text-cafe24-brown">{(f.importance * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                    <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.churn.model}</div>
                  </div>
                )}

                {/* 이상거래 탐지 */}
                {selectedUser.model_predictions.fraud && (
                  <div className={`rounded-2xl p-4 border-2 ${
                    selectedUser.model_predictions.fraud.is_anomaly
                      ? 'border-red-300 bg-red-50'
                      : selectedUser.model_predictions.fraud.anomaly_score > 0.5
                      ? 'border-yellow-300 bg-yellow-50'
                      : 'border-green-300 bg-green-50'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <Shield size={16} className={
                        selectedUser.model_predictions.fraud.is_anomaly ? 'text-red-600' :
                        selectedUser.model_predictions.fraud.anomaly_score > 0.5 ? 'text-yellow-600' : 'text-green-600'
                      } />
                      <span className="text-xs font-bold text-cafe24-brown">이상거래 탐지</span>
                    </div>
                    <div className="text-2xl font-black mb-1" style={{
                      color: selectedUser.model_predictions.fraud.is_anomaly ? '#DC2626' :
                             selectedUser.model_predictions.fraud.anomaly_score > 0.5 ? '#CA8A04' : '#16A34A'
                    }}>
                      {(selectedUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-cafe24-brown/60">
                      상태: {selectedUser.model_predictions.fraud.risk_level}
                    </div>
                    <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.fraud.model}</div>
                  </div>
                )}

                {/* 세그먼트 */}
                {selectedUser.model_predictions.segment && (
                  <div className="rounded-2xl p-4 border-2 border-blue-300 bg-blue-50">
                    <div className="flex items-center gap-2 mb-2">
                      <Users size={16} className="text-blue-600" />
                      <span className="text-xs font-bold text-cafe24-brown">셀러 세그먼트</span>
                    </div>
                    <div className="text-lg font-black text-blue-600 mb-1">
                      {selectedUser.model_predictions.segment.segment_name}
                    </div>
                    <div className="text-xs text-cafe24-brown/60">
                      클러스터 #{selectedUser.model_predictions.segment.cluster}
                    </div>
                    <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.segment.model}</div>
                  </div>
                )}

                {/* CS 응답 품질 */}
                {selectedUser.model_predictions.cs_quality && (
                  <div className={`rounded-2xl p-4 border-2 ${
                    selectedUser.model_predictions.cs_quality.score >= 80
                      ? 'border-green-300 bg-green-50'
                      : selectedUser.model_predictions.cs_quality.score >= 50
                      ? 'border-yellow-300 bg-yellow-50'
                      : 'border-red-300 bg-red-50'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <MessageSquare size={16} className={
                        selectedUser.model_predictions.cs_quality.score >= 80 ? 'text-green-600' :
                        selectedUser.model_predictions.cs_quality.score >= 50 ? 'text-yellow-600' : 'text-red-600'
                      } />
                      <span className="text-xs font-bold text-cafe24-brown">CS 응답 품질</span>
                    </div>
                    <div className="text-2xl font-black mb-1" style={{
                      color: selectedUser.model_predictions.cs_quality.score >= 80 ? '#16A34A' :
                             selectedUser.model_predictions.cs_quality.score >= 50 ? '#CA8A04' : '#DC2626'
                    }}>
                      {selectedUser.model_predictions.cs_quality.score}점
                    </div>
                    <div className="text-xs text-cafe24-brown/60">
                      등급: {selectedUser.model_predictions.cs_quality.grade}
                    </div>
                    <div className="flex justify-between text-xs mt-1">
                      <span className="text-cafe24-brown/70">환불률</span>
                      <span className="font-semibold">{(selectedUser.model_predictions.cs_quality.refund_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs mt-1">
                      <span className="text-cafe24-brown/70">평균 응답</span>
                      <span className="font-semibold">{selectedUser.model_predictions.cs_quality.avg_response_time}시간</span>
                    </div>
                    <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.cs_quality.model}</div>
                  </div>
                )}

                {/* 매출 예측 */}
                {selectedUser.model_predictions.revenue && (
                  <div className="rounded-2xl p-4 border-2 border-purple-300 bg-purple-50">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp size={16} className="text-purple-600" />
                      <span className="text-xs font-bold text-cafe24-brown">매출 예측</span>
                    </div>
                    <div className="text-lg font-black text-purple-600 mb-1">
                      {selectedUser.model_predictions.revenue.predicted_next_month >= 10000
                        ? `₩${(selectedUser.model_predictions.revenue.predicted_next_month / 10000).toFixed(0)}만`
                        : `₩${selectedUser.model_predictions.revenue.predicted_next_month?.toLocaleString()}`
                      }
                    </div>
                    <div className="text-xs text-cafe24-brown/60">다음달 예상 매출</div>
                    <div className="flex justify-between text-xs mt-1">
                      <span className="text-cafe24-brown/70">성장률</span>
                      <span className={`font-semibold ${selectedUser.model_predictions.revenue.growth_rate >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {selectedUser.model_predictions.revenue.growth_rate >= 0 ? '+' : ''}{selectedUser.model_predictions.revenue.growth_rate}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs mt-1">
                      <span className="text-cafe24-brown/70">신뢰도</span>
                      <span className="font-semibold">{selectedUser.model_predictions.revenue.confidence}%</span>
                    </div>
                    <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.revenue.model}</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {!selectedUser && !loading && (
        <div className="text-center py-12 text-cafe24-brown/50">
          <User size={48} className="mx-auto mb-3 opacity-30" />
          <p className="text-sm">셀러 ID를 검색하여 상세 분석을 확인하세요</p>
        </div>
      )}
    </div>
  );
}
