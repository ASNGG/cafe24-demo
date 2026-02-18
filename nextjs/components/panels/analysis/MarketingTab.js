// components/panels/analysis/MarketingTab.js
// 마케팅 최적화 탭 (H29: 자체 상태 관리)

import { useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import {
  DollarSign, RefreshCw, Users, Target, ShoppingBag,
  TrendingUp, ArrowUpRight, Search
} from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';

// IIFE → 모듈 레벨 함수 추출: 광고비 포맷팅
function formatAdSpend(cost) {
  const formatNum = (n) => n >= 1000 ? `${(n / 1000).toFixed(0)}K` : `${n}`;
  if (!cost || typeof cost !== 'object') return '0';
  return `${formatNum(Number(cost.ad_spend || 0))}원`;
}

const MARKETING_EXAMPLE_USERS = [
  { id: 'SEL0001', description: 'Premium, 고매출' },
  { id: 'SEL0050', description: 'Standard, 중간 매출' },
  { id: 'SEL0100', description: 'Basic, 낮은 매출' },
];

export default function MarketingTab({ apiCall, auth }) {
  const [marketingUser, setMarketingUser] = useState('');
  const [marketingUserInput, setMarketingUserInput] = useState('');
  const [marketingUserStatus, setMarketingUserStatus] = useState(null);
  const [marketingResult, setMarketingResult] = useState(null);
  const [marketingOptimizing, setMarketingOptimizing] = useState(false);
  const [marketingLoading, setMarketingLoading] = useState(false);

  const shopChartData = useMemo(() => {
    if (!marketingUserStatus?.shops?.length) return [];
    return marketingUserStatus.shops.slice(0, 8).map(s => ({
      name: s.name || s.shop_id || '쇼핑몰',
      전환율: s.cvr || s.conversion_rate || 0,
      매출: Math.round((s.monthly_revenue || 0) / 10000),
    }));
  }, [marketingUserStatus]);

  const loadMarketingUserStatus = useCallback(async (userId) => {
    setMarketingLoading(true);
    try {
      const res = await apiCall({
        endpoint: `/api/marketing/seller/${userId}`,
        method: 'GET',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setMarketingUserStatus(res.data);
        toast.success(`${userId} 셀러 정보를 불러왔습니다`);
      } else {
        setMarketingUserStatus(null);
        toast.error('셀러 정보를 불러올 수 없습니다');
      }
    } catch (error) {
      console.error('Failed to load seller status:', error);
      setMarketingUserStatus(null);
      toast.error('셀러 정보 조회에 실패했습니다. 백엔드 연결을 확인하세요.');
    } finally {
      setMarketingLoading(false);
    }
  }, [apiCall, auth]);

  const runMarketingOptimization = useCallback(async () => {
    if (!marketingUserStatus) return;
    setMarketingOptimizing(true);
    try {
      const res = await apiCall({
        endpoint: '/api/marketing/optimize',
        method: 'POST',
        auth,
        data: { seller_id: marketingUser, top_n: 10 },
        timeoutMs: 60000,
      });
      if (res?.status === 'success') {
        setMarketingResult(res.data);
        toast.success('마케팅 최적화가 완료되었습니다!');
      } else {
        setMarketingResult(null);
        toast.error('최적화에 실패했습니다');
      }
    } catch (error) {
      console.error('Optimization failed:', error);
      setMarketingResult(null);
      toast.error('최적화 실행에 실패했습니다. 백엔드 연결을 확인하세요.');
    } finally {
      setMarketingOptimizing(false);
    }
  }, [apiCall, auth, marketingUser, marketingUserStatus]);

  const handleMarketingExampleSelect = useCallback((userId) => {
    setMarketingUser(userId);
    setMarketingUserInput('');
    setMarketingResult(null);
    loadMarketingUserStatus(userId);
  }, [loadMarketingUserStatus]);

  const handleMarketingDirectSearch = useCallback(() => {
    const trimmed = marketingUserInput.trim();
    if (!trimmed) {
      toast.error('셀러 ID를 입력해주세요');
      return;
    }
    setMarketingUser(trimmed);
    setMarketingResult(null);
    loadMarketingUserStatus(trimmed);
  }, [marketingUserInput, loadMarketingUserStatus]);

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <DollarSign size={20} className="text-cafe24-orange" />
          <span className="text-lg font-black text-cafe24-brown">마케팅 최적화</span>
        </div>
        <button
          onClick={() => loadMarketingUserStatus(marketingUser)}
          disabled={marketingLoading || !marketingUser}
          className="p-2.5 rounded-xl border-2 border-cafe24-orange/20 hover:border-cafe24-orange hover:bg-cafe24-orange/10 transition-all disabled:opacity-50"
          aria-label="새로고침"
        >
          <RefreshCw className={`w-5 h-5 text-cafe24-orange ${marketingLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* 셀러 선택 UI */}
      <div className="bg-white rounded-2xl p-5 border-2 border-cafe24-orange/10 shadow-sm">
        <h3 className="text-sm font-bold text-cafe24-brown mb-4 flex items-center gap-2">
          <Users size={16} className="text-cafe24-orange" />
          셀러 선택
        </h3>

        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-3">
            <span className="text-sm text-cafe24-brown/70 self-center mr-2">예시:</span>
            {MARKETING_EXAMPLE_USERS.map((user) => (
              <button
                key={user.id}
                onClick={() => handleMarketingExampleSelect(user.id)}
                className={`px-4 py-2.5 rounded-xl border-2 transition-all flex flex-col items-start ${
                  marketingUser === user.id
                    ? 'border-cafe24-orange bg-cafe24-orange/10 text-cafe24-brown'
                    : 'border-cafe24-orange/20 hover:border-cafe24-orange/40 bg-white text-cafe24-brown/80'
                }`}
              >
                <span className="font-bold text-sm">{user.id}</span>
                <span className="text-xs text-cafe24-brown/60">{user.description}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-cafe24-orange/20" />
            <span className="text-sm text-cafe24-brown/50">또는 직접 입력</span>
            <div className="flex-1 h-px bg-cafe24-orange/20" />
          </div>

          <div className="flex items-center gap-3">
            <input
              type="text"
              value={marketingUserInput}
              onChange={(e) => setMarketingUserInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleMarketingDirectSearch(); }}
              placeholder="셀러 ID 입력 (예: SEL0001)"
              className="flex-1 px-4 py-3 rounded-xl border-2 border-cafe24-orange/20 bg-white text-cafe24-brown font-medium placeholder:text-cafe24-brown/40 focus:border-cafe24-orange focus:ring-2 focus:ring-cafe24-orange/20 outline-none transition-all"
            />
            <button
              onClick={handleMarketingDirectSearch}
              disabled={marketingLoading || !marketingUserInput.trim()}
              className="px-6 py-3 rounded-xl bg-cafe24-orange text-white font-bold hover:bg-cafe24-orange/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Search size={18} />
              조회
            </button>
          </div>

          {marketingUser && (
            <div className="flex items-center gap-2 text-sm text-cafe24-brown/70 bg-cafe24-yellow/10 px-4 py-2 rounded-xl">
              <Target size={16} className="text-cafe24-orange" />
              현재 선택: <span className="font-bold text-cafe24-brown">{marketingUser}</span>
            </div>
          )}
        </div>
      </div>

      {/* 셀러 현황 */}
      {marketingUserStatus && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white rounded-2xl p-5 border-2 border-cafe24-orange/10 shadow-sm">
            <h3 className="text-sm font-bold text-cafe24-brown mb-4 flex items-center gap-2">
              <DollarSign size={16} className="text-cafe24-orange" />
              운영 현황
            </h3>
            <div className="space-y-3">
              {Object.entries(marketingUserStatus.resources || {}).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-2 rounded-lg bg-gray-50">
                  <span className="text-sm text-cafe24-brown">{key === 'ad_budget' ? '광고 예산' : key === 'monthly_revenue' ? '월 매출' : key === 'product_count' ? '상품 수' : key === 'order_count' ? '주문 수' : key}</span>
                  <span className="font-bold text-cafe24-brown">{value?.toLocaleString() || 0}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-2 bg-white rounded-2xl p-5 border-2 border-cafe24-orange/10 shadow-sm">
            <h3 className="text-sm font-bold text-cafe24-brown mb-4 flex items-center gap-2">
              <ShoppingBag size={18} className="text-cafe24-orange" />
              운영 쇼핑몰 현황
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-gradient-to-br from-cafe24-orange/10 to-white border border-cafe24-orange/10">
                <div className="text-xs text-cafe24-brown/70">총 쇼핑몰</div>
                <div className="text-xl font-bold text-cafe24-brown">{marketingUserStatus.shops?.length || 0}</div>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-cafe24-yellow/10 to-white border border-cafe24-yellow/10">
                <div className="text-xs text-cafe24-brown/70">평균 전환율</div>
                <div className="text-xl font-bold text-cafe24-brown">
                  {(marketingUserStatus.shops?.reduce((sum, c) => sum + (c.cvr || 0), 0) / (marketingUserStatus.shops?.length || 1)).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-green-100 to-white border border-green-100">
                <div className="text-xs text-cafe24-brown/70">최대 전환율</div>
                <div className="text-xl font-bold text-cafe24-brown">
                  {Math.max(...(marketingUserStatus.shops?.map(c => c.cvr || 0) || [0])).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-white border border-purple-100">
                <div className="text-xs text-cafe24-brown/70">총 매출</div>
                <div className="text-xl font-bold text-cafe24-brown">{marketingUserStatus.total_revenue?.toLocaleString() || '계산중'}</div>
              </div>
            </div>
            {shopChartData.length > 0 && (
              <div className="mt-4">
                <h4 className="text-xs font-bold text-cafe24-brown/70 mb-2">쇼핑몰별 성과</h4>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={shopChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                    <XAxis dataKey="name" tick={{ fill: '#5C4A3D', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="전환율" fill="#FF8C42" radius={[4, 4, 0, 0]} barSize={18} />
                    <Bar dataKey="매출" name="매출(만원)" fill="#60A5FA" radius={[4, 4, 0, 0]} barSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 최적화 버튼 */}
      <div className="flex justify-center">
        <button
          onClick={runMarketingOptimization}
          disabled={marketingOptimizing || !marketingUserStatus}
          className="px-8 py-4 bg-gradient-to-r from-cafe24-orange to-cafe24-yellow text-white font-bold text-lg rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3"
        >
          {marketingOptimizing ? (
            <>
              <RefreshCw className="w-6 h-6 animate-spin" />
              마케팅 최적화 실행 중...
            </>
          ) : (
            <>
              <Target size={24} />
              마케팅 최적화 실행
            </>
          )}
        </button>
      </div>

      {/* 최적화 결과 */}
      {marketingResult && (
        <div className="bg-gradient-to-br from-cafe24-yellow/10 via-white to-cafe24-orange/10 rounded-2xl p-6 border-2 border-cafe24-orange/20 shadow-lg">
          <h3 className="text-lg font-bold text-cafe24-brown mb-4 flex items-center gap-2">
            <TrendingUp size={20} className="text-cafe24-orange" />
            최적화 결과 - 개인화된 마케팅 추천
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
            <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
              <div className="text-xs text-cafe24-brown/70 mb-1">예상 전환율 증가</div>
              <div className="text-xl font-bold text-green-600 flex items-center gap-1">
                +{Number(marketingResult.total_cvr_gain || 0).toFixed(1)}%
                <ArrowUpRight size={18} />
              </div>
            </div>
            <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
              <div className="text-xs text-cafe24-brown/70 mb-1">추천 개수</div>
              <div className="text-xl font-bold text-blue-600">{marketingResult.recommendations?.length || 0}개</div>
            </div>
            <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
              <div className="text-xs text-cafe24-brown/70 mb-1">평균 효율</div>
              <div className="text-xl font-bold text-pink-600">{marketingResult.recommendations?.length > 0
                ? (marketingResult.recommendations.reduce((sum, r) => sum + Number(r.efficiency || 0), 0) / marketingResult.recommendations.length * 100).toFixed(1)
                : 0}%</div>
            </div>
            <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
              <div className="text-xs text-cafe24-brown/70 mb-1">최적화 방식</div>
              <div className="text-xl font-bold text-purple-600">AI</div>
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 border border-cafe24-orange/10">
            <h4 className="font-bold text-cafe24-brown mb-4">우선순위별 마케팅 추천</h4>
            <div className="space-y-2">
              {marketingResult.recommendations?.slice(0, 8).map((rec, idx) => (
                <div key={idx} className="flex items-center gap-3 p-3 rounded-xl bg-gradient-to-r from-gray-50 to-white border border-gray-100 hover:border-cafe24-orange/30 transition-colors">
                  <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm ${
                    idx === 0 ? 'bg-gradient-to-br from-amber-500 to-yellow-500' :
                    idx === 1 ? 'bg-gradient-to-br from-gray-400 to-gray-500' :
                    idx === 2 ? 'bg-gradient-to-br from-orange-400 to-orange-500' :
                    'bg-gradient-to-br from-cafe24-orange to-cafe24-yellow'
                  }`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-cafe24-brown">{rec.channel_name}</div>
                    <div className="text-xs text-cafe24-brown/60">{rec.campaign_type} {rec.from_budget} → {rec.to_budget}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-green-600">+{Number(rec.cvr_gain || 0).toFixed(1)}%</div>
                    <div className="text-xs text-cafe24-brown/50">전환율 증가</div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium text-cafe24-brown text-sm">
                      {formatAdSpend(rec.cost)}
                    </div>
                    <div className="text-xs text-cafe24-brown/50">
                      {rec.campaign_type === 'cpc' ? '클릭당 비용' :
                       rec.campaign_type === 'display' ? '노출 비용' :
                       rec.campaign_type === 'social' ? 'SNS 비용' : '비용'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 로딩 중 */}
      {marketingLoading && !marketingUserStatus && (
        <div className="text-center py-16">
          <RefreshCw size={48} className="mx-auto mb-3 text-cafe24-orange animate-spin" />
          <p className="text-sm font-semibold text-cafe24-brown/50">셀러 정보를 불러오는 중...</p>
        </div>
      )}
    </div>
  );
}
