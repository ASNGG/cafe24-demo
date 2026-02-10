// components/panels/analysis/AnalysisPanel.js
// CAFE24 AI 운영 플랫폼 - 상세 분석 패널 (리팩토링)

import { useEffect, useState, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';
import { SkeletonCard } from '@/components/Skeleton';
import {
  Users, Search, Calendar, TrendingUp,
  RefreshCw, ChevronDown, User, ShoppingBag, MessageSquare,
  AlertTriangle, Brain, Target, DollarSign
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';

// 탭 컴포넌트
import SellerTab from './SellerTab';
import SegmentTab from './SegmentTab';
import AnomalyTab from './AnomalyTab';
import PredictionTab from './PredictionTab';
import CohortTab from './CohortTab';
import TrendTab from './TrendTab';
import ShopTab from './ShopTab';
import CsTab from './CsTab';
import MarketingTab from './MarketingTab';

// 분석 탭 정의
const ANALYSIS_TABS = [
  { key: 'seller', label: '셀러 분석', icon: User },
  { key: 'segment', label: '세그먼트', icon: Users },
  { key: 'anomaly', label: '이상탐지', icon: AlertTriangle },
  { key: 'prediction', label: '예측 분석', icon: Brain },
  { key: 'cohort', label: '코호트', icon: Target },
  { key: 'trend', label: '트렌드', icon: TrendingUp },
  { key: 'shop', label: '쇼핑몰 분석', icon: ShoppingBag },
  { key: 'cs', label: 'CS 분석', icon: MessageSquare },
  { key: 'marketing', label: '마케팅 최적화', icon: DollarSign },
];

// 기간 옵션
const DATE_OPTIONS = [
  { value: '7d', label: '최근 7일' },
  { value: '30d', label: '최근 30일' },
  { value: '90d', label: '최근 90일' },
];

export default function AnalysisPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('seller');
  const [dateRange, setDateRange] = useState('7d');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [showDateDropdown, setShowDateDropdown] = useState(false);

  // API 데이터 상태
  const [summaryData, setSummaryData] = useState(null);
  const [segmentsData, setSegmentsData] = useState(null);
  const [shopsData, setShopsData] = useState(null);
  const [csData, setCsData] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);

  // 빠른 선택용 샘플 셀러 ID
  const quickSelectUsers = ['SEL0001', 'SEL0025', 'SEL0100', 'SEL0250'];

  // 자동완성 관련 상태
  const searchInputRef = useRef(null);
  const autocompleteTimerRef = useRef(null);

  // 분석 데이터 상태
  const [anomalyData, setAnomalyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [predictionSearchQuery, setPredictionSearchQuery] = useState('');
  const [predictionUser, setPredictionUser] = useState(null);
  const [predictionUserLoading, setPredictionUserLoading] = useState(false);

  // 마케팅 최적화 상태
  const [marketingUser, setMarketingUser] = useState('');
  const [marketingUserInput, setMarketingUserInput] = useState('');
  const [marketingUserStatus, setMarketingUserStatus] = useState(null);
  const [marketingResult, setMarketingResult] = useState(null);
  const [marketingOptimizing, setMarketingOptimizing] = useState(false);
  const [marketingLoading, setMarketingLoading] = useState(false);

  // API 데이터 로드
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      try {
        const summaryRes = await apiCall({
          endpoint: `/api/stats/summary?days=${days}`,
          auth,
          timeoutMs: 10000,
        });

        if (summaryRes?.status === 'success') {
          setSummaryData(summaryRes);

          if (summaryRes.user_segments) {
            const segments = {};
            const metrics = summaryRes.segment_metrics || {};
            Object.entries(summaryRes.user_segments).forEach(([name, count]) => {
              const m = metrics[name] || {};
              segments[name] = {
                count,
                avg_monthly_revenue: m.avg_monthly_revenue || 0,
                avg_product_count: m.avg_product_count || 0,
                avg_order_count: m.avg_order_count || 0,
                retention: m.retention || 0,
              };
            });
            if (Object.keys(segments).length > 0) {
              setSegmentsData(segments);
            }
          }

          if (summaryRes.plan_tier_stats || summaryRes.shops_count > 0) {
            try {
              const shopsRes = await apiCall({
                endpoint: '/api/shops',
                auth,
                timeoutMs: 10000,
              });
              if (shopsRes?.status === 'success' && shopsRes.shops) {
                const transformed = shopsRes.shops.slice(0, 10).map(c => ({
                  name: c.name || c.shop_id,
                  plan_tier: c.plan_tier,
                  usage: c.usage ?? 0,
                  cvr: c.cvr ?? 0,
                  popularity: c.popularity ?? 0,
                }));
                if (transformed.length > 0) {
                  setShopsData(transformed);
                }
              }
            } catch (e) {
              console.log('쇼핑몰 API 실패');
            }
          }

          if (summaryRes.cs_stats_detail) {
            const channels = summaryRes.cs_stats_detail.map(stat => ({
              channel: stat.lang_name || stat.category || '기타',
              count: stat.total_count || 0,
              quality: stat.avg_quality?.toFixed(1) ?? '-',
              pending: stat.pending_count ?? 0,
            }));
            if (channels.length > 0) {
              setCsData({ channels, recent: [] });
            }
          }
        }

        // 이상탐지 API
        try {
          const anomalyRes = await apiCall({
            endpoint: `/api/analysis/anomaly?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (anomalyRes?.status === 'success') {
            setAnomalyData({
              summary: anomalyRes.summary || {},
              by_type: anomalyRes.by_type || [],
              recent_alerts: anomalyRes.recent_alerts || [],
              trend: anomalyRes.trend || [],
            });
          }
        } catch (e) {
          console.log('이상탐지 API 실패');
        }

        // 예측 분석 API
        try {
          const churnRes = await apiCall({
            endpoint: `/api/analysis/prediction/churn?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (churnRes?.status === 'success' && churnRes.churn) {
            setPredictionData({
              churn: churnRes.churn,
              revenue: churnRes.revenue || {},
              engagement: churnRes.engagement || {},
            });
          }
        } catch (e) {
          console.log('예측 API 실패');
        }

        // 코호트 API
        try {
          const cohortRes = await apiCall({
            endpoint: `/api/analysis/cohort/retention?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (cohortRes?.status === 'success' && cohortRes.retention) {
            setCohortData({
              retention: cohortRes.retention,
              ltv_by_cohort: cohortRes.ltv_by_cohort || [],
              conversion: cohortRes.conversion || [],
            });
          }
        } catch (e) {
          console.log('코호트 API 실패');
        }

        // 트렌드 KPI API
        try {
          const trendRes = await apiCall({
            endpoint: `/api/analysis/trend/kpis?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (trendRes?.status === 'success' && trendRes.kpis) {
            setTrendData({
              kpis: trendRes.kpis,
              daily_metrics: trendRes.daily_metrics || [],
              correlation: trendRes.correlation || [],
              forecast: trendRes.forecast || [],
            });
          }
        } catch (e) {
          console.log('트렌드 API 실패');
        }

      } catch (e) {
        console.log('API 호출 실패');
      }
      setDataLoaded(true);
      setLoading(false);
    }

    if (auth) {
      fetchData();
    }
  }, [auth, apiCall, dateRange]);

  // 마케팅 최적화: 셀러 상태 로드
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

  // 마케팅 최적화: 최적화 실행
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

  // 예측 분석: 개별 셀러 검색
  const handlePredictionSearch = useCallback(async (userId) => {
    const id = (userId || predictionSearchQuery).trim();
    if (!id) { toast.error('셀러 ID를 입력하세요'); return; }
    setPredictionUserLoading(true);
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;
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

  // 마케팅 최적화: 예시 셀러 선택
  const handleMarketingExampleSelect = useCallback((userId) => {
    setMarketingUser(userId);
    setMarketingUserInput('');
    setMarketingResult(null);
    loadMarketingUserStatus(userId);
  }, [loadMarketingUserStatus]);

  // 마케팅 최적화: 직접 입력 조회
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

  // 마케팅 최적화: Enter 키 처리
  const handleMarketingInputKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      handleMarketingDirectSearch();
    }
  }, [handleMarketingDirectSearch]);

  // 셀러 검색
  const handleUserSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      toast.error('셀러 ID를 입력하세요');
      return;
    }
    setLoading(true);
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;

    try {
      const res = await apiCall({
        endpoint: `/api/sellers/search?q=${encodeURIComponent(searchQuery)}&days=${days}`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'success' && res.user) {
        setSelectedUser({
          id: res.user.id,
          segment: res.user.segment,
          plan_tier: res.user.plan_tier,
          monthly_revenue: res.user.monthly_revenue,
          product_count: res.user.product_count,
          order_count: res.user.order_count,
          top_shops: res.user.top_shops || [],
          stats: res.user.stats || {},
          activity: res.user.activity || [],
          model_predictions: res.user.model_predictions || {},
          period_stats: res.user.period_stats || {},
          is_anomaly: res.user.is_anomaly,
          region: res.user.region,
        });
        toast.success(`${res.user.id} 셀러 데이터를 불러왔습니다`);
      } else {
        toast.error('셀러를 찾을 수 없습니다');
        setSelectedUser(null);
      }
    } catch (e) {
      console.log('셀러 검색 API 실패');
      toast.error('셀러 검색에 실패했습니다. 백엔드 연결을 확인하세요.');
      setSelectedUser(null);
    }
    setLoading(false);
  }, [apiCall, auth, dateRange, searchQuery]);

  // 기간 변경 시 선택된 셀러가 있으면 자동 재검색
  useEffect(() => {
    if (selectedUser?.id && auth) {
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      const refetchUser = async () => {
        try {
          const res = await apiCall({
            endpoint: `/api/sellers/search?q=${encodeURIComponent(selectedUser.id)}&days=${days}`,
            auth,
            timeoutMs: 10000,
          });

          if (res?.status === 'success' && res.user) {
            setSelectedUser({
              id: res.user.id,
              name: res.user.id,
              segment: res.user.segment || '알 수 없음',
              monthly_revenue: res.user.monthly_revenue || 0,
              product_count: res.user.product_count || 0,
              order_count: res.user.order_count || 0,
              top_shops: res.user.top_shops || [],
              stats: res.user.stats || {},
              activity: res.user.activity || [],
            });
          }
        } catch (e) {
          console.log('셀러 데이터 재조회 실패');
        }
      };

      refetchUser();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dateRange]);

  return (
    <div>
      <SectionHeader
        title="상세 분석"
        subtitle="셀러 · 세그먼트 · 쇼핑몰 · CS 데이터 심층 분석"
        right={
          <div className="flex items-center gap-2">
            {dataLoaded && (
              <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
                summaryData
                  ? 'border-green-400/50 bg-green-50 text-green-700'
                  : 'border-red-400/50 bg-red-50 text-red-700'
              }`}>
                {summaryData ? 'LIVE' : 'NO DATA'}
              </span>
            )}
            {['seller', 'anomaly', 'prediction', 'trend', 'cohort', 'shop', 'cs'].includes(activeTab) && (
            <div className="relative">
              <button
                onClick={() => setShowDateDropdown(!showDateDropdown)}
                className="flex items-center gap-1.5 rounded-full border-2 border-cookie-orange/20 bg-white/80 px-3 py-1.5 text-xs font-bold text-cookie-brown hover:bg-cookie-beige transition"
                aria-label="기간 선택"
              >
                <Calendar size={12} />
                {DATE_OPTIONS.find(d => d.value === dateRange)?.label}
                <ChevronDown size={12} />
              </button>
              {showDateDropdown && (
                <div className="absolute right-0 top-full mt-1 z-10 rounded-xl border-2 border-cookie-orange/20 bg-white shadow-lg overflow-hidden">
                  {DATE_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => { setDateRange(opt.value); setShowDateDropdown(false); }}
                      className={`block w-full px-4 py-2 text-left text-xs font-semibold hover:bg-cookie-beige transition ${
                        dateRange === opt.value ? 'bg-cookie-yellow/30 text-cookie-brown' : 'text-cookie-brown/70'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            )}
          </div>
        }
      />

      {/* 분석 유형 탭 */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2" role="tablist" aria-label="분석 유형">
        {ANALYSIS_TABS.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={isActive}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-cookie-yellow to-cookie-orange text-white shadow-md'
                  : 'bg-white/80 border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === 'seller' && (
        <SellerTab
          loading={loading}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          searchInputRef={searchInputRef}
          handleUserSearch={handleUserSearch}
          quickSelectUsers={quickSelectUsers}
          selectedUser={selectedUser}
        />
      )}

      {activeTab === 'segment' && (
        <SegmentTab selectedUser={selectedUser} segmentsData={segmentsData} />
      )}

      {activeTab === 'anomaly' && (
        <AnomalyTab selectedUser={selectedUser} anomalyData={anomalyData} />
      )}

      {activeTab === 'prediction' && (
        <PredictionTab
          predictionData={predictionData}
          predictionSearchQuery={predictionSearchQuery}
          setPredictionSearchQuery={setPredictionSearchQuery}
          predictionUser={predictionUser}
          setPredictionUser={setPredictionUser}
          predictionUserLoading={predictionUserLoading}
          handlePredictionSearch={handlePredictionSearch}
        />
      )}

      {activeTab === 'cohort' && (
        <CohortTab cohortData={cohortData} />
      )}

      {activeTab === 'trend' && (
        <TrendTab trendData={trendData} />
      )}

      {activeTab === 'shop' && (
        <ShopTab shopsData={shopsData} />
      )}

      {activeTab === 'cs' && (
        <CsTab csData={csData} />
      )}

      {activeTab === 'marketing' && (
        <MarketingTab
          marketingUser={marketingUser}
          marketingUserInput={marketingUserInput}
          setMarketingUserInput={setMarketingUserInput}
          marketingUserStatus={marketingUserStatus}
          marketingResult={marketingResult}
          marketingOptimizing={marketingOptimizing}
          marketingLoading={marketingLoading}
          loadMarketingUserStatus={loadMarketingUserStatus}
          runMarketingOptimization={runMarketingOptimization}
          handleMarketingExampleSelect={handleMarketingExampleSelect}
          handleMarketingDirectSearch={handleMarketingDirectSearch}
          handleMarketingInputKeyDown={handleMarketingInputKeyDown}
        />
      )}
    </div>
  );
}
