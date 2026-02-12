// components/panels/analysis/PredictionTab.js
// 예측 분석 탭 (H29: 자체 상태 관리)

import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import {
  Brain, UserMinus, DollarSign, Activity,
  ArrowUpRight
} from 'lucide-react';
import SellerSearchInput from './common/SellerSearchInput';
import AnalysisEmptyState from './common/EmptyState';
import { DAYS_MAP } from './common/constants';

export default function PredictionTab({
  predictionData, apiCall, auth, dateRange,
}) {
  const [predictionTab, setPredictionTab] = useState('churn');
  const [predictionSearchQuery, setPredictionSearchQuery] = useState('');
  const [predictionUser, setPredictionUser] = useState(null);
  const [predictionUserLoading, setPredictionUserLoading] = useState(false);

  const handlePredictionSearch = useCallback(async (userId) => {
    const id = (userId || predictionSearchQuery).trim();
    if (!id) { toast.error('셀러 ID를 입력하세요'); return; }
    setPredictionUserLoading(true);
    const days = DAYS_MAP[dateRange] || 7;
    try {
      const res = await apiCall({
        endpoint: `/api/sellers/search?q=${encodeURIComponent(id)}&days=${days}`,
        auth,
        timeoutMs: 10000,
      });
      if (res?.status === 'success' && res.user) {
        setPredictionUser({
          id: res.user.id,
          segment: res.user.segment,
          plan_tier: res.user.plan_tier,
          monthly_revenue: res.user.monthly_revenue,
          model_predictions: res.user.model_predictions || {},
        });
        toast.success(`${res.user.id} 예측 결과를 불러왔습니다`);
      } else {
        toast.error('셀러를 찾을 수 없습니다');
        setPredictionUser(null);
      }
    } catch (e) {
      toast.error('셀러 검색에 실패했습니다');
      setPredictionUser(null);
    }
    setPredictionUserLoading(false);
  }, [apiCall, auth, dateRange, predictionSearchQuery]);

  return (
    <div className="space-y-6">
      {/* 개별 셀러 예측 검색 (M57: SellerSearchInput 공통 컴포넌트) */}
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-3">
          <Brain size={16} className="text-cafe24-orange" />
          <span className="text-sm font-black text-cafe24-brown">개별 셀러 예측 조회</span>
        </div>
        <SellerSearchInput
          value={predictionSearchQuery}
          onChange={setPredictionSearchQuery}
          onSearch={handlePredictionSearch}
          loading={predictionUserLoading}
          buttonLabel="예측"
          loadingLabel="조회중..."
        />
      </div>

      {/* 개별 셀러 예측 결과 */}
      {predictionUser?.model_predictions && (
        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Brain size={18} className="text-cafe24-orange" />
              <span className="text-sm font-black text-cafe24-brown">{predictionUser.id} ML 예측 결과</span>
              <span className="px-2 py-0.5 rounded-full bg-cafe24-beige text-xs font-semibold text-cafe24-brown">
                {predictionUser.segment} · {predictionUser.plan_tier}
              </span>
            </div>
            <button
              onClick={() => setPredictionUser(null)}
              className="text-xs text-cafe24-brown/50 hover:text-cafe24-brown transition-all"
            >
              닫기
            </button>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {predictionUser.model_predictions.churn && (
              <div className={`rounded-2xl p-4 border-2 ${
                predictionUser.model_predictions.churn.risk_code >= 2 ? 'border-red-300 bg-red-50' :
                predictionUser.model_predictions.churn.risk_code === 1 ? 'border-orange-300 bg-orange-50' :
                'border-green-300 bg-green-50'
              }`}>
                <div className="text-xs font-bold text-cafe24-brown mb-1">이탈 확률</div>
                <div className="text-2xl font-black" style={{
                  color: predictionUser.model_predictions.churn.risk_code >= 2 ? '#DC2626' :
                         predictionUser.model_predictions.churn.risk_code === 1 ? '#EA580C' : '#16A34A'
                }}>{predictionUser.model_predictions.churn.probability}%</div>
                <div className="text-xs text-cafe24-brown/60">{predictionUser.model_predictions.churn.risk_level}</div>
              </div>
            )}
            {predictionUser.model_predictions.revenue && (
              <div className="rounded-2xl p-4 border-2 border-purple-300 bg-purple-50">
                <div className="text-xs font-bold text-cafe24-brown mb-1">예상 월매출</div>
                <div className="text-2xl font-black text-purple-600">
                  {predictionUser.model_predictions.revenue.predicted_next_month >= 10000
                    ? `₩${(predictionUser.model_predictions.revenue.predicted_next_month / 10000).toFixed(0)}만`
                    : `₩${(predictionUser.model_predictions.revenue.predicted_next_month || 0).toLocaleString()}`}
                </div>
                <div className="text-xs text-cafe24-brown/60">성장률 {predictionUser.model_predictions.revenue.growth_rate}%</div>
              </div>
            )}
            {predictionUser.model_predictions.fraud && (
              <div className={`rounded-2xl p-4 border-2 ${
                predictionUser.model_predictions.fraud.is_anomaly ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'
              }`}>
                <div className="text-xs font-bold text-cafe24-brown mb-1">이상거래</div>
                <div className="text-2xl font-black" style={{
                  color: predictionUser.model_predictions.fraud.is_anomaly ? '#DC2626' : '#16A34A'
                }}>{predictionUser.model_predictions.fraud.risk_level}</div>
                <div className="text-xs text-cafe24-brown/60">점수 {(predictionUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%</div>
              </div>
            )}
            {predictionUser.model_predictions.cs_quality && (
              <div className={`rounded-2xl p-4 border-2 ${
                predictionUser.model_predictions.cs_quality.score >= 80 ? 'border-green-300 bg-green-50' :
                predictionUser.model_predictions.cs_quality.score >= 50 ? 'border-yellow-300 bg-yellow-50' :
                'border-red-300 bg-red-50'
              }`}>
                <div className="text-xs font-bold text-cafe24-brown mb-1">CS 품질</div>
                <div className="text-2xl font-black" style={{
                  color: predictionUser.model_predictions.cs_quality.score >= 80 ? '#16A34A' :
                         predictionUser.model_predictions.cs_quality.score >= 50 ? '#CA8A04' : '#DC2626'
                }}>{predictionUser.model_predictions.cs_quality.score}점</div>
                <div className="text-xs text-cafe24-brown/60">{predictionUser.model_predictions.cs_quality.grade}</div>
              </div>
            )}
            {predictionUser.model_predictions.segment && (
              <div className="rounded-2xl p-4 border-2 border-blue-300 bg-blue-50">
                <div className="text-xs font-bold text-cafe24-brown mb-1">셀러 세그먼트</div>
                <div className="text-lg font-black text-blue-600">{predictionUser.model_predictions.segment.segment_name}</div>
                <div className="text-xs text-cafe24-brown/60">클러스터 #{predictionUser.model_predictions.segment.cluster}</div>
              </div>
            )}
          </div>
          {/* SHAP 요인 */}
          {predictionUser.model_predictions.churn?.factors?.length > 0 && (
            <div className="mt-4">
              <div className="text-xs font-bold text-cafe24-brown mb-2">이탈 주요 요인 (SHAP)</div>
              <div className="space-y-2">
                {predictionUser.model_predictions.churn.factors.map((f, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-cafe24-orange text-white text-xs font-bold flex items-center justify-center shrink-0">
                      {i + 1}
                    </span>
                    <span className="text-xs font-semibold text-cafe24-brown w-24">{f.factor}</span>
                    <div className="flex-1 h-2 bg-cafe24-beige rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cafe24-yellow to-cafe24-orange"
                        style={{ width: `${Math.min(100, f.importance * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-bold text-cafe24-orange w-10 text-right">{(f.importance * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {!predictionData ? (
        <AnalysisEmptyState
          icon={Brain}
          title="예측 데이터를 불러올 수 없습니다"
          subtitle="백엔드 API 연결을 확인하세요"
        />
      ) : (
      <>
      {/* 예측 유형 선택 */}
      <div className="flex gap-2">
        {[
          { key: 'churn', label: '이탈 예측', icon: UserMinus },
          { key: 'revenue', label: '매출 예측', icon: DollarSign },
          { key: 'engagement', label: '참여도 예측', icon: Activity },
        ].map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              onClick={() => setPredictionTab(tab.key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                predictionTab === tab.key
                  ? 'bg-cafe24-brown text-white'
                  : 'bg-white border-2 border-cafe24-orange/20 text-cafe24-brown hover:bg-cafe24-beige'
              }`}
            >
              <Icon size={14} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* 이탈 예측 */}
      {predictionTab === 'churn' && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-2xl border-2 border-red-200 bg-red-50 p-4">
              <div className="text-xs font-bold text-red-700 mb-1">고위험 이탈</div>
              <div className="text-2xl font-black text-red-600">{predictionData.churn.high_risk_count}</div>
              <div className="text-xs text-red-600/70">셀러</div>
            </div>
            <div className="rounded-2xl border-2 border-orange-200 bg-orange-50 p-4">
              <div className="text-xs font-bold text-orange-700 mb-1">중위험 이탈</div>
              <div className="text-2xl font-black text-orange-600">{predictionData.churn.medium_risk_count}</div>
              <div className="text-xs text-orange-600/70">셀러</div>
            </div>
            <div className="rounded-2xl border-2 border-green-200 bg-green-50 p-4">
              <div className="text-xs font-bold text-green-700 mb-1">안전</div>
              <div className="text-2xl font-black text-green-600">{predictionData.churn.low_risk_count}</div>
              <div className="text-xs text-green-600/70">셀러</div>
            </div>
            <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4">
              <div className="text-xs font-bold text-cafe24-brown mb-1">모델 정확도</div>
              <div className="text-2xl font-black text-cafe24-brown">{predictionData.churn.model_accuracy}%</div>
              <div className="text-xs text-cafe24-brown/60">F1 Score</div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">이탈 예측 주요 요인</div>
              <div className="space-y-3">
                {predictionData.churn.top_factors.map((factor, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-cafe24-orange text-white text-xs font-bold flex items-center justify-center">
                      {idx + 1}
                    </span>
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-semibold text-cafe24-brown">{factor.factor}</span>
                        <span className="text-sm font-bold text-cafe24-orange">{(factor.importance * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-cafe24-beige rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-cafe24-yellow to-cafe24-orange"
                          style={{ width: `${factor.importance * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">이탈 고위험 셀러</div>
              <div className="space-y-3">
                {(predictionData.churn?.high_risk_users || []).map((user, idx) => (
                  <div key={idx} className="flex items-center gap-4 p-3 rounded-2xl bg-red-50 border border-red-200">
                    <div className="w-10 h-10 rounded-full bg-red-500 text-white font-bold flex items-center justify-center text-sm">
                      {user.probability}%
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-cafe24-brown">{user.id}</div>
                      <div className="text-xs text-cafe24-brown/60">{user.segment}</div>
                    </div>
                    <div className="text-xs text-red-600 font-semibold">{user.last_active}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* 매출 예측 */}
      {predictionTab === 'revenue' && predictionData?.revenue && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-2xl border-2 border-green-200 bg-green-50 p-4">
            <div className="text-xs font-bold text-green-700 mb-1">예상 월매출</div>
            <div className="text-xl font-black text-green-600">₩{((predictionData.revenue.predicted_monthly || 0) / 10000).toFixed(0)}만</div>
            <div className="flex items-center gap-1 text-xs text-green-600">
              <ArrowUpRight size={12} />+{predictionData.revenue.growth_rate || 0}%
            </div>
          </div>
          <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
            <div className="text-xs font-bold text-blue-700 mb-1">예상 ARPU</div>
            <div className="text-xl font-black text-blue-600">₩{(predictionData.revenue.predicted_arpu || 0).toLocaleString()}</div>
            <div className="text-xs text-blue-600/70">셀러당 평균</div>
          </div>
          <div className="rounded-2xl border-2 border-purple-200 bg-purple-50 p-4">
            <div className="text-xs font-bold text-purple-700 mb-1">예상 ARPPU</div>
            <div className="text-xl font-black text-purple-600">₩{(predictionData.revenue.predicted_arppu || 0).toLocaleString()}</div>
            <div className="text-xs text-purple-600/70">유료 셀러 평균</div>
          </div>
          <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4">
            <div className="text-xs font-bold text-cafe24-brown mb-1">신뢰도</div>
            <div className="text-xl font-black text-cafe24-brown">{predictionData.revenue.confidence || 0}%</div>
            <div className="text-xs text-cafe24-brown/60">예측 정확도</div>
          </div>
          <div className="rounded-2xl border-2 border-pink-200 bg-pink-50 p-4 col-span-2 lg:col-span-1">
            <div className="text-xs font-bold text-pink-700 mb-1">Enterprise</div>
            <div className="text-xl font-black text-pink-600">{predictionData.revenue.whale_count || 0}명</div>
            <div className="text-xs text-pink-600/70">대형 셀러</div>
          </div>
          <div className="rounded-2xl border-2 border-cyan-200 bg-cyan-50 p-4 col-span-2 lg:col-span-1">
            <div className="text-xs font-bold text-cyan-700 mb-1">Premium</div>
            <div className="text-xl font-black text-cyan-600">{predictionData.revenue.dolphin_count || 0}명</div>
            <div className="text-xs text-cyan-600/70">중형 셀러</div>
          </div>
          <div className="rounded-2xl border-2 border-teal-200 bg-teal-50 p-4 col-span-2">
            <div className="text-xs font-bold text-teal-700 mb-1">Standard</div>
            <div className="text-xl font-black text-teal-600">{predictionData.revenue.minnow_count || 0}명</div>
            <div className="text-xs text-teal-600/70">소형 셀러</div>
          </div>
        </div>
      )}

      {/* 참여도 예측 */}
      {predictionTab === 'engagement' && predictionData?.engagement && (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
            <div className="text-xs font-bold text-blue-700 mb-1">예상 DAU</div>
            <div className="text-2xl font-black text-blue-600">{predictionData.engagement.predicted_dau || 0}</div>
            <div className="text-xs text-blue-600/70">일일 활성 셀러</div>
          </div>
          <div className="rounded-2xl border-2 border-indigo-200 bg-indigo-50 p-4">
            <div className="text-xs font-bold text-indigo-700 mb-1">예상 MAU</div>
            <div className="text-2xl font-black text-indigo-600">{predictionData.engagement.predicted_mau || 0}</div>
            <div className="text-xs text-indigo-600/70">월간 활성 셀러</div>
          </div>
          <div className="rounded-2xl border-2 border-violet-200 bg-violet-50 p-4">
            <div className="text-xs font-bold text-violet-700 mb-1">Stickiness</div>
            <div className="text-2xl font-black text-violet-600">{predictionData.engagement.stickiness || 0}%</div>
            <div className="text-xs text-violet-600/70">DAU/MAU</div>
          </div>
          <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4">
            <div className="text-xs font-bold text-cafe24-brown mb-1">평균 세션</div>
            <div className="text-2xl font-black text-cafe24-brown">{predictionData.engagement.avg_session || 0}분</div>
            <div className="text-xs text-cafe24-brown/60">세션당 운영 시간</div>
          </div>
          <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-4 col-span-2">
            <div className="text-xs font-bold text-cafe24-brown mb-1">일일 세션 수</div>
            <div className="text-2xl font-black text-cafe24-brown">{predictionData.engagement.sessions_per_day || 0}</div>
            <div className="text-xs text-cafe24-brown/60">셀러당 평균 접속 횟수</div>
          </div>
        </div>
      )}
      </>
      )}
    </div>
  );
}
