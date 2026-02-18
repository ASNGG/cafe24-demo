// components/panels/analysis/AnalysisPanel.js
// CAFE24 AI 운영 플랫폼 - 상세 분석 패널 (리팩토링)

import { useEffect, useState, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import toast from 'react-hot-toast';
import {
  Users, Calendar, TrendingUp,
  ChevronDown, User, ShoppingBag, MessageSquare,
  AlertTriangle, Brain, Target, DollarSign
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import { DAYS_MAP } from './common/constants';

// H28: recharts 사용 탭은 dynamic import (SSR 비활성화, 코드 스플리팅)
const SellerTab = dynamic(() => import('./SellerTab'), { ssr: false });
const SegmentTab = dynamic(() => import('./SegmentTab'), { ssr: false });
const AnomalyTab = dynamic(() => import('./AnomalyTab'), { ssr: false });
const PredictionTab = dynamic(() => import('./PredictionTab'), { ssr: false });
const CohortTab = dynamic(() => import('./CohortTab'), { ssr: false });
const TrendTab = dynamic(() => import('./TrendTab'), { ssr: false });
const ShopTab = dynamic(() => import('./ShopTab'), { ssr: false });
const MarketingTab = dynamic(() => import('./MarketingTab'), { ssr: false });
// CsTab은 recharts 미사용이므로 정적 import
import CsTab from './CsTab';

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

// setSelectedUser 매핑 중복 제거용 헬퍼
function mapUserData(user) {
  return {
    id: user.id,
    segment: user.segment || '알 수 없음',
    plan_tier: user.plan_tier,
    monthly_revenue: user.monthly_revenue || 0,
    product_count: user.product_count || 0,
    order_count: user.order_count || 0,
    top_shops: user.top_shops || [],
    stats: user.stats || {},
    activity: user.activity || [],
    model_predictions: user.model_predictions || {},
    period_stats: user.period_stats || {},
    is_anomaly: user.is_anomaly,
    region: user.region,
  };
}

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

  const searchInputRef = useRef(null);

  // 분석 데이터 상태
  const [anomalyData, setAnomalyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [trendData, setTrendData] = useState(null);

  // 선택된 셀러 ID ref (dateRange 변경 시 재검색용)
  const selectedUserIdRef = useRef(null);
  selectedUserIdRef.current = selectedUser?.id || null;

  // 쇼핑몰 목록은 dateRange 무관 → 별도 useEffect로 분리 (불변 API 재호출 방지)
  useEffect(() => {
    if (!auth) return;
    async function fetchShops() {
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
    fetchShops();
  }, [auth, apiCall]);

  // API 데이터 로드 (H30: Promise.all 병렬, M58: useEffect 통합)
  useEffect(() => {
    if (!auth) return;

    async function fetchData() {
      setLoading(true);
      const days = DAYS_MAP[dateRange] || 7;

      try {
        // H30: 독립 API 4개 병렬 호출
        const [summaryRes, anomalyRes, churnRes, cohortRes, trendRes] = await Promise.all([
          apiCall({ endpoint: `/api/stats/summary?days=${days}`, auth, timeoutMs: 10000 }).catch(() => null),
          apiCall({ endpoint: `/api/analysis/anomaly?days=${days}`, auth, timeoutMs: 10000 }).catch(() => null),
          apiCall({ endpoint: `/api/analysis/prediction/churn?days=${days}`, auth, timeoutMs: 10000 }).catch(() => null),
          apiCall({ endpoint: `/api/analysis/cohort/retention?days=${days}`, auth, timeoutMs: 10000 }).catch(() => null),
          apiCall({ endpoint: `/api/analysis/trend/kpis?days=${days}`, auth, timeoutMs: 10000 }).catch(() => null),
        ]);

        // summary 처리
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

        // 이상탐지 처리
        if (anomalyRes?.status === 'success') {
          setAnomalyData({
            summary: anomalyRes.summary || {},
            by_type: anomalyRes.by_type || [],
            recent_alerts: anomalyRes.recent_alerts || [],
            trend: anomalyRes.trend || [],
          });
        }

        // 예측 처리
        if (churnRes?.status === 'success' && churnRes.churn) {
          setPredictionData({
            churn: churnRes.churn,
            revenue: churnRes.revenue || {},
            engagement: churnRes.engagement || {},
          });
        }

        // 코호트 처리
        if (cohortRes?.status === 'success' && cohortRes.retention) {
          setCohortData({
            retention: cohortRes.retention,
            ltv_by_cohort: cohortRes.ltv_by_cohort || [],
            conversion: cohortRes.conversion || [],
          });
        }

        // 트렌드 처리
        if (trendRes?.status === 'success' && trendRes.kpis) {
          setTrendData({
            kpis: trendRes.kpis,
            daily_metrics: trendRes.daily_metrics || [],
            correlation: trendRes.correlation || [],
            forecast: trendRes.forecast || [],
          });
        }

        // M58: dateRange 변경 시 선택된 셀러 재검색 통합
        if (selectedUserIdRef.current) {
          try {
            const res = await apiCall({
              endpoint: `/api/sellers/search?q=${encodeURIComponent(selectedUserIdRef.current)}&days=${days}`,
              auth,
              timeoutMs: 10000,
            });
            if (res?.status === 'success' && res.user) {
              setSelectedUser(mapUserData(res.user));
            }
          } catch (e) {
            console.log('셀러 데이터 재조회 실패');
          }
        }
      } catch (e) {
        console.log('API 호출 실패');
      }
      setDataLoaded(true);
      setLoading(false);
    }

    fetchData();
  }, [auth, apiCall, dateRange]);

  // 셀러 검색
  const handleUserSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      toast.error('셀러 ID를 입력하세요');
      return;
    }
    setLoading(true);
    const days = DAYS_MAP[dateRange] || 7;

    try {
      const res = await apiCall({
        endpoint: `/api/sellers/search?q=${encodeURIComponent(searchQuery)}&days=${days}`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'success' && res.user) {
        setSelectedUser(mapUserData(res.user));
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
                className="flex items-center gap-1.5 rounded-full border-2 border-cafe24-orange/20 bg-white/80 px-3 py-1.5 text-xs font-bold text-cafe24-brown hover:bg-cafe24-beige transition"
                aria-label="기간 선택"
              >
                <Calendar size={12} />
                {DATE_OPTIONS.find(d => d.value === dateRange)?.label}
                <ChevronDown size={12} />
              </button>
              {showDateDropdown && (
                <div className="absolute right-0 top-full mt-1 z-10 rounded-xl border-2 border-cafe24-orange/20 bg-white shadow-lg overflow-hidden">
                  {DATE_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => { setDateRange(opt.value); setShowDateDropdown(false); }}
                      className={`block w-full px-4 py-2 text-left text-xs font-semibold hover:bg-cafe24-beige transition ${
                        dateRange === opt.value ? 'bg-cafe24-yellow/30 text-cafe24-brown' : 'text-cafe24-brown/70'
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
                  ? 'bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white shadow-md'
                  : 'bg-white/80 border-2 border-cafe24-orange/20 text-cafe24-brown hover:bg-cafe24-beige'
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
          apiCall={apiCall}
          auth={auth}
          dateRange={dateRange}
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
          apiCall={apiCall}
          auth={auth}
        />
      )}
    </div>
  );
}
