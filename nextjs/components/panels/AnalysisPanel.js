// components/panels/AnalysisPanel.js
// CAFE24 AI 운영 플랫폼 - 상세 분석 패널

import { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import toast from 'react-hot-toast';
import { SkeletonCard } from '@/components/Skeleton';
import {
  Users, Globe, Search, Calendar, Filter, TrendingUp,
  Crown, RefreshCw, ChevronDown, User, ShoppingBag, MessageSquare,
  AlertTriangle, Brain, Target, Activity, Zap, Shield,
  BarChart3, PieChartIcon, ArrowUpRight, ArrowDownRight,
  Clock, UserMinus, DollarSign, Repeat, Eye
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, LineChart, Line, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area,
  ComposedChart, Scatter
} from 'recharts';

// CAFE24 테마 색상
const COLORS = {
  primary: ['#FF8C42', '#FFD93D', '#4ADE80', '#60A5FA', '#F472B6', '#A78BFA'],
  tiers: {
    Enterprise: '#8B5CF6',
    Premium: '#F59E0B',
    Standard: '#3B82F6',
    Basic: '#6B7280',
  }
};

// 분석 탭 정의 (확장)
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

// 커스텀 툴팁
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border-2 border-cookie-orange/20 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <p className="text-xs font-bold text-cookie-brown">{label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm font-semibold" style={{ color: entry.color || entry.fill }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
        </p>
      ))}
    </div>
  );
};

export default function AnalysisPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('seller');
  const [dateRange, setDateRange] = useState('7d');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState('전체');
  const [showDateDropdown, setShowDateDropdown] = useState(false);

  // API 데이터 상태 (초기값 null - API 실패 시 데이터 없음 표시)
  const [summaryData, setSummaryData] = useState(null);
  const [segmentsData, setSegmentsData] = useState(null);
  const [shopsData, setShopsData] = useState(null);
  const [csData, setCsData] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);

  // 빠른 선택용 샘플 셀러 ID (UI용)
  const quickSelectUsers = ['SEL0001', 'SEL0025', 'SEL0100', 'SEL0250'];

  // 자동완성 관련 상태
  const [autocompleteResults, setAutocompleteResults] = useState([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompleteLoading, setAutocompleteLoading] = useState(false);
  const autocompleteRef = useRef(null);
  const searchInputRef = useRef(null);

  // 자동완성 debounce 타이머
  const autocompleteTimerRef = useRef(null);

  // 새로운 분석 데이터 상태
  const [anomalyData, setAnomalyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [predictionTab, setPredictionTab] = useState('churn'); // churn, revenue, engagement
  const [predictionSearchQuery, setPredictionSearchQuery] = useState('');
  const [predictionUser, setPredictionUser] = useState(null);
  const [predictionUserLoading, setPredictionUserLoading] = useState(false);
  const [cohortTab, setCohortTab] = useState('retention'); // retention, ltv, conversion

  // 마케팅 최적화 상태
  const [marketingUser, setMarketingUser] = useState('');
  const [marketingUserInput, setMarketingUserInput] = useState(''); // 직접 입력
  const [marketingUserStatus, setMarketingUserStatus] = useState(null);
  const [marketingResult, setMarketingResult] = useState(null);
  const [marketingOptimizing, setMarketingOptimizing] = useState(false);
  const [marketingLoading, setMarketingLoading] = useState(false);

  // 마케팅 최적화 예시 셀러 (3개)
  const MARKETING_EXAMPLE_USERS = [
    { id: 'SEL0001', description: 'Premium, 고매출' },
    { id: 'SEL0050', description: 'Standard, 중간 매출' },
    { id: 'SEL0100', description: 'Basic, 낮은 매출' },
  ];

  // API 데이터 로드
  useEffect(() => {
    async function fetchData() {
      setLoading(true);

      // 기간을 일수로 변환
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      try {
        // 통계 요약 API 호출
        const summaryRes = await apiCall({
          endpoint: `/api/stats/summary?days=${days}`,
          auth,
          timeoutMs: 10000,
        });

        if (summaryRes?.status === 'SUCCESS') {
          setSummaryData(summaryRes);

          // 세그먼트 데이터 변환 - API의 segment_metrics 사용
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

          // 쇼핑몰 데이터가 있으면 변환
          if (summaryRes.plan_tier_stats || summaryRes.shops_count > 0) {
            // 쇼핑몰 API 별도 호출
            try {
              const shopsRes = await apiCall({
                endpoint: '/api/shops',
                auth,
                timeoutMs: 10000,
              });
              if (shopsRes?.status === 'SUCCESS' && shopsRes.shops) {
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

          // CS 데이터 변환 - 상세 통계가 있으면 사용
          if (summaryRes.cs_stats_detail) {
            // 백엔드에서 제공하는 CS 상세 통계 사용
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

        // 이상탐지 API 호출
        try {
          const anomalyRes = await apiCall({
            endpoint: `/api/analysis/anomaly?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (anomalyRes?.status === 'SUCCESS') {
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

        // 예측 분석 API 호출
        try {
          const churnRes = await apiCall({
            endpoint: `/api/analysis/prediction/churn?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (churnRes?.status === 'SUCCESS' && churnRes.churn) {
            setPredictionData({
              churn: churnRes.churn,
              revenue: churnRes.revenue || {},
              engagement: churnRes.engagement || {},
            });
          }
        } catch (e) {
          console.log('예측 API 실패');
        }

        // 코호트 API 호출
        try {
          const cohortRes = await apiCall({
            endpoint: `/api/analysis/cohort/retention?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (cohortRes?.status === 'SUCCESS' && cohortRes.retention) {
            setCohortData({
              retention: cohortRes.retention,
              ltv_by_cohort: cohortRes.ltv_by_cohort || [],
              conversion: cohortRes.conversion || [],
            });
          }
        } catch (e) {
          console.log('코호트 API 실패');
        }

        // 트렌드 KPI API 호출
        try {
          const trendRes = await apiCall({
            endpoint: `/api/analysis/trend/kpis?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (trendRes?.status === 'SUCCESS' && trendRes.kpis) {
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
  const loadMarketingUserStatus = async (userId) => {
    setMarketingLoading(true);
    try {
      const res = await apiCall({
        endpoint: `/api/marketing/seller/${userId}`,
        method: 'GET',
        auth,
        timeoutMs: 30000,
      });

      if (res?.status === 'SUCCESS') {
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
  };

  // 마케팅 최적화: 최적화 실행
  const runMarketingOptimization = async () => {
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

      if (res?.status === 'SUCCESS') {
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
  };

  // 예측 분석: 개별 셀러 검색
  const handlePredictionSearch = async (userId) => {
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
      if (res?.status === 'SUCCESS' && res.user) {
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
  };

  // 마케팅 최적화: 예시 셀러 선택 핸들러
  const handleMarketingExampleSelect = (userId) => {
    setMarketingUser(userId);
    setMarketingUserInput('');
    setMarketingResult(null);
    loadMarketingUserStatus(userId);
  };

  // 마케팅 최적화: 직접 입력 조회 핸들러
  const handleMarketingDirectSearch = () => {
    const trimmed = marketingUserInput.trim();
    if (!trimmed) {
      toast.error('셀러 ID를 입력해주세요');
      return;
    }
    setMarketingUser(trimmed);
    setMarketingResult(null);
    loadMarketingUserStatus(trimmed);
  };

  // 마케팅 최적화: Enter 키 처리
  const handleMarketingInputKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleMarketingDirectSearch();
    }
  };

  // 자동완성 검색 (debounced)
  const fetchAutocomplete = useCallback(async (query) => {
    if (!query || query.length < 1) {
      setAutocompleteResults([]);
      setShowAutocomplete(false);
      return;
    }

    setAutocompleteLoading(true);
    try {
      const res = await apiCall({
        endpoint: `/api/sellers/autocomplete?q=${encodeURIComponent(query)}&limit=8`,
        auth,
        timeoutMs: 5000,
      });

      if (res?.status === 'SUCCESS' && res.users) {
        setAutocompleteResults(res.users);
        setShowAutocomplete(res.users.length > 0);
      } else {
        setAutocompleteResults([]);
        setShowAutocomplete(false);
      }
    } catch (e) {
      setAutocompleteResults([]);
      setShowAutocomplete(false);
    } finally {
      setAutocompleteLoading(false);
    }
  }, [apiCall, auth, quickSelectUsers]);

  // 자동완성 입력 핸들러 (debounce)
  const handleSearchInputChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);

    // 이전 타이머 취소
    if (autocompleteTimerRef.current) {
      clearTimeout(autocompleteTimerRef.current);
    }

    // 300ms 후 자동완성 검색
    autocompleteTimerRef.current = setTimeout(() => {
      fetchAutocomplete(value);
    }, 300);
  };

  // 자동완성 항목 선택
  const handleAutocompleteSelect = (user) => {
    setSearchQuery(user.id);
    setShowAutocomplete(false);
    // 선택 후 바로 검색 실행
    setTimeout(() => {
      searchInputRef.current?.blur();
      handleUserSearchDirect(user.id);
    }, 50);
  };

  // 직접 검색 (특정 ID로)
  const handleUserSearchDirect = async (userId) => {
    if (!userId?.trim()) return;
    setLoading(true);
    setShowAutocomplete(false);

    // 기간을 일수로 변환
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;

    try {
      const res = await apiCall({
        endpoint: `/api/sellers/search?q=${encodeURIComponent(userId)}&days=${days}`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'SUCCESS' && res.user) {
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
      toast.error('셀러 검색에 실패했습니다');
      setSelectedUser(null);
    }
    setLoading(false);
  };

  // 클릭 외부 감지 - 자동완성 닫기
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (autocompleteRef.current && !autocompleteRef.current.contains(e.target)) {
        setShowAutocomplete(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 셀러 검색
  const handleUserSearch = async () => {
    if (!searchQuery.trim()) {
      toast.error('셀러 ID를 입력하세요');
      return;
    }
    setLoading(true);

    // 기간을 일수로 변환
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;

    try {
      // API 호출 시도
      const res = await apiCall({
        endpoint: `/api/sellers/search?q=${encodeURIComponent(searchQuery)}&days=${days}`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'SUCCESS' && res.user) {
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
  };

  // 기간 변경 시 선택된 셀러가 있으면 자동 재검색
  useEffect(() => {
    if (selectedUser?.id && auth) {
      // 기간을 일수로 변환
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      const refetchUser = async () => {
        try {
          const res = await apiCall({
            endpoint: `/api/sellers/search?q=${encodeURIComponent(selectedUser.id)}&days=${days}`,
            auth,
            timeoutMs: 10000,
          });

          if (res?.status === 'SUCCESS' && res.user) {
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

  // 셀러 레이더 차트 데이터
  const userRadarData = useMemo(() => {
    if (!selectedUser?.stats) return [];
    return Object.entries(selectedUser.stats).map(([key, value]) => ({
      subject: key,
      value,
      fullMark: 100,
    }));
  }, [selectedUser]);

  // 세그먼트 비교 차트 데이터
  const segmentCompareData = useMemo(() => {
    if (!segmentsData) return [];
    return Object.entries(segmentsData).map(([name, data]) => ({
      name: name.replace(' ', '\n'),
      셀러수: data.count,
      평균매출: data.avg_monthly_revenue,
      리텐션: data.retention,
    }));
  }, [segmentsData]);

  // 쇼핑몰 운영 차트 데이터
  const shopUsageData = useMemo(() => {
    if (!shopsData) return [];
    return shopsData.map(shop => ({
      name: shop.name,
      운영점수: shop.usage,
      전환율: shop.cvr,
      인기도: shop.popularity,
      fill: COLORS.tiers[shop.plan_tier] || COLORS.primary[0],
    }));
  }, [shopsData]);

  return (
    <div>
      <SectionHeader
        title="상세 분석"
        subtitle="셀러 · 세그먼트 · 쇼핑몰 · CS 데이터 심층 분석"
        right={
          <div className="flex items-center gap-2">
            {/* 데이터 소스 배지 */}
            {dataLoaded && (
              <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
                summaryData
                  ? 'border-green-400/50 bg-green-50 text-green-700'
                  : 'border-red-400/50 bg-red-50 text-red-700'
              }`}>
                {summaryData ? 'LIVE' : 'NO DATA'}
              </span>
            )}
            {/* 기간 선택 - 기간 필터가 의미 있는 탭에서만 표시 */}
            {['seller', 'anomaly', 'prediction', 'trend'].includes(activeTab) && (
            <div className="relative">
              <button
                onClick={() => setShowDateDropdown(!showDateDropdown)}
                className="flex items-center gap-1.5 rounded-full border-2 border-cookie-orange/20 bg-white/80 px-3 py-1.5 text-xs font-bold text-cookie-brown hover:bg-cookie-beige transition"
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
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {ANALYSIS_TABS.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm whitespace-nowrap transition-all ${
                activeTab === tab.key
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

      {/* 셀러 분석 */}
      {activeTab === 'seller' && (
        <div className="space-y-6">
          {/* 셀러 검색 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Search size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">셀러 검색</span>
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
                  className="w-full px-4 py-2.5 rounded-xl border-2 border-cookie-orange/20 bg-white text-sm text-cookie-brown placeholder:text-cookie-brown/40 outline-none focus:border-cookie-orange transition"
                />
              </div>
              <button
                onClick={handleUserSearch}
                disabled={loading}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cookie-yellow to-cookie-orange text-white font-bold text-sm shadow-md hover:shadow-lg transition disabled:opacity-50"
              >
                {loading ? '검색 중...' : '검색'}
              </button>
            </div>
            {/* 빠른 선택 */}
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="text-xs text-cookie-brown/60">빠른 선택:</span>
              {quickSelectUsers.map(userId => (
                <button
                  key={userId}
                  onClick={() => { setSearchQuery(userId); }}
                  className="px-2 py-1 rounded-lg bg-cookie-beige text-xs font-semibold text-cookie-brown hover:bg-cookie-yellow/30 transition"
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
              <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-black text-cookie-brown">{selectedUser.id}</h3>
                    <p className="text-sm text-cookie-brown/60">{selectedUser.segment} · {selectedUser.plan_tier} · {selectedUser.region}</p>
                  </div>
                  <div className="flex gap-2">
                    <span className="px-3 py-1 rounded-full bg-cookie-yellow/30 text-xs font-bold text-cookie-brown">
                      매출 ₩{(selectedUser.monthly_revenue || 0).toLocaleString()}
                    </span>
                    {selectedUser.is_anomaly && (
                      <span className="px-3 py-1 rounded-full bg-red-100 text-xs font-bold text-red-600">이상감지</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.product_count || 0}</div>
                    <div className="text-xs text-cookie-brown/60">상품 수</div>
                  </div>
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.order_count || 0}</div>
                    <div className="text-xs text-cookie-brown/60">주문 수</div>
                  </div>
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.period_stats?.active_days || 0}</div>
                    <div className="text-xs text-cookie-brown/60">활동일수</div>
                  </div>
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.period_stats?.total_cs || 0}</div>
                    <div className="text-xs text-cookie-brown/60">CS건수</div>
                  </div>
                </div>
              </div>

              {/* 차트 그리드 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 활동 트렌드 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">일별 운영 트렌드</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={selectedUser.activity}>
                      <defs>
                        <linearGradient id="colorProductCount" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#FFD93D" stopOpacity={0.4}/>
                          <stop offset="95%" stopColor="#FFD93D" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                      <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Area type="monotone" dataKey="product_count" name="상품 수" stroke="#FFD93D" fill="url(#colorProductCount)" />
                      <Line type="monotone" dataKey="orders" name="주문 수" stroke="#4ADE80" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* 셀러 스탯 레이더 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">셀러 특성 분석</div>
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
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="flex items-center gap-2 mb-4">
                    <Brain size={18} className="text-cookie-orange" />
                    <span className="text-sm font-black text-cookie-brown">ML 모델 예측 결과</span>
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
                          <span className="text-xs font-bold text-cookie-brown">이탈 예측</span>
                        </div>
                        <div className="text-2xl font-black mb-1" style={{
                          color: selectedUser.model_predictions.churn.risk_code >= 2 ? '#DC2626' :
                                 selectedUser.model_predictions.churn.risk_code === 1 ? '#EA580C' : '#16A34A'
                        }}>
                          {selectedUser.model_predictions.churn.probability}%
                        </div>
                        <div className="text-xs text-cookie-brown/60 mb-2">
                          위험도: {selectedUser.model_predictions.churn.risk_level}
                        </div>
                        {selectedUser.model_predictions.churn.factors?.slice(0, 3).map((f, i) => (
                          <div key={i} className="flex justify-between text-xs mt-1">
                            <span className="text-cookie-brown/70">{f.factor}</span>
                            <span className="font-semibold text-cookie-brown">{(f.importance * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                        <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.churn.model}</div>
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
                          <span className="text-xs font-bold text-cookie-brown">이상거래 탐지</span>
                        </div>
                        <div className="text-2xl font-black mb-1" style={{
                          color: selectedUser.model_predictions.fraud.is_anomaly ? '#DC2626' :
                                 selectedUser.model_predictions.fraud.anomaly_score > 0.5 ? '#CA8A04' : '#16A34A'
                        }}>
                          {(selectedUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-cookie-brown/60">
                          상태: {selectedUser.model_predictions.fraud.risk_level}
                        </div>
                        <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.fraud.model}</div>
                      </div>
                    )}

                    {/* 세그먼트 */}
                    {selectedUser.model_predictions.segment && (
                      <div className="rounded-2xl p-4 border-2 border-blue-300 bg-blue-50">
                        <div className="flex items-center gap-2 mb-2">
                          <Users size={16} className="text-blue-600" />
                          <span className="text-xs font-bold text-cookie-brown">셀러 세그먼트</span>
                        </div>
                        <div className="text-lg font-black text-blue-600 mb-1">
                          {selectedUser.model_predictions.segment.segment_name}
                        </div>
                        <div className="text-xs text-cookie-brown/60">
                          클러스터 #{selectedUser.model_predictions.segment.cluster}
                        </div>
                        <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.segment.model}</div>
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
                          <span className="text-xs font-bold text-cookie-brown">CS 응답 품질</span>
                        </div>
                        <div className="text-2xl font-black mb-1" style={{
                          color: selectedUser.model_predictions.cs_quality.score >= 80 ? '#16A34A' :
                                 selectedUser.model_predictions.cs_quality.score >= 50 ? '#CA8A04' : '#DC2626'
                        }}>
                          {selectedUser.model_predictions.cs_quality.score}점
                        </div>
                        <div className="text-xs text-cookie-brown/60">
                          등급: {selectedUser.model_predictions.cs_quality.grade}
                        </div>
                        <div className="flex justify-between text-xs mt-1">
                          <span className="text-cookie-brown/70">환불률</span>
                          <span className="font-semibold">{(selectedUser.model_predictions.cs_quality.refund_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between text-xs mt-1">
                          <span className="text-cookie-brown/70">평균 응답</span>
                          <span className="font-semibold">{selectedUser.model_predictions.cs_quality.avg_response_time}시간</span>
                        </div>
                        <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.cs_quality.model}</div>
                      </div>
                    )}

                    {/* 매출 예측 */}
                    {selectedUser.model_predictions.revenue && (
                      <div className="rounded-2xl p-4 border-2 border-purple-300 bg-purple-50">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp size={16} className="text-purple-600" />
                          <span className="text-xs font-bold text-cookie-brown">매출 예측</span>
                        </div>
                        <div className="text-lg font-black text-purple-600 mb-1">
                          {selectedUser.model_predictions.revenue.predicted_next_month >= 10000
                            ? `₩${(selectedUser.model_predictions.revenue.predicted_next_month / 10000).toFixed(0)}만`
                            : `₩${selectedUser.model_predictions.revenue.predicted_next_month?.toLocaleString()}`
                          }
                        </div>
                        <div className="text-xs text-cookie-brown/60">다음달 예상 매출</div>
                        <div className="flex justify-between text-xs mt-1">
                          <span className="text-cookie-brown/70">성장률</span>
                          <span className={`font-semibold ${selectedUser.model_predictions.revenue.growth_rate >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {selectedUser.model_predictions.revenue.growth_rate >= 0 ? '+' : ''}{selectedUser.model_predictions.revenue.growth_rate}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs mt-1">
                          <span className="text-cookie-brown/70">신뢰도</span>
                          <span className="font-semibold">{selectedUser.model_predictions.revenue.confidence}%</span>
                        </div>
                        <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.revenue.model}</div>
                      </div>
                    )}

                  </div>
                </div>
              )}
            </>
          )}

          {!selectedUser && !loading && (
            <div className="text-center py-12 text-cookie-brown/50">
              <User size={48} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">셀러 ID를 검색하여 상세 분석을 확인하세요</p>
            </div>
          )}
        </div>
      )}

      {/* 세그먼트 분석 */}
      {activeTab === 'segment' && (
        <div className="space-y-6">
          {/* 선택된 셀러 세그먼트 */}
          {selectedUser?.model_predictions?.segment && (
            <div className="rounded-3xl border-2 border-blue-300 bg-blue-50/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-3">
                <Users size={18} className="text-blue-600" />
                <span className="text-sm font-black text-cookie-brown">{selectedUser.id} 세그먼트 분류</span>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-xl font-black text-blue-600">{selectedUser.model_predictions.segment.segment_name}</div>
                  <div className="text-xs text-cookie-brown/60">세그먼트</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-black text-cookie-brown">#{selectedUser.model_predictions.segment.cluster}</div>
                  <div className="text-xs text-cookie-brown/60">클러스터</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-black text-cookie-brown">{selectedUser.plan_tier}</div>
                  <div className="text-xs text-cookie-brown/60">플랜</div>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.segment.model}</div>
            </div>
          )}
          {!segmentsData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Users size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">세그먼트 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 세그먼트 비교 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Users size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">세그먼트 비교 분석</span>
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={segmentCompareData} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#5C4A3D', fontSize: 10 }}
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="셀러수" fill="#FF8C42" radius={[4, 4, 0, 0]} />
                <Bar dataKey="평균매출" fill="#FFD93D" radius={[4, 4, 0, 0]} />
                <Bar dataKey="리텐션" fill="#4ADE80" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 세그먼트 상세 테이블 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">세그먼트별 상세 지표</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cookie-orange/10">
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">세그먼트</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">셀러 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 월매출</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 상품 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 주문 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">리텐션</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(segmentsData).map(([name, data]) => (
                    <tr key={name} className="border-b border-cookie-orange/5 hover:bg-cookie-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cookie-brown">{name}</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.count.toLocaleString()}명</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.avg_monthly_revenue?.toLocaleString()}원</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.avg_product_count}개</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.avg_order_count}건</td>
                      <td className="py-3 px-2 text-right">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                          data.retention >= 70 ? 'bg-green-100 text-green-700' :
                          data.retention >= 40 ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {data.retention}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* 쇼핑몰 분석 */}
      {activeTab === 'shop' && (
        <div className="space-y-6">
          {!shopsData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <ShoppingBag size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">쇼핑몰 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 쇼핑몰 운영 차트 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <ShoppingBag size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">인기 쇼핑몰 분석</span>
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={shopUsageData} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#5C4A3D', fontSize: 11 }} width={90} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="운영점수" fill="#FF8C42" radius={[0, 4, 4, 0]} barSize={16} />
                <Bar dataKey="전환율" fill="#60A5FA" radius={[0, 4, 4, 0]} barSize={16} />
                <Bar dataKey="인기도" fill="#4ADE80" radius={[0, 4, 4, 0]} barSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 쇼핑몰 상세 리스트 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">쇼핑몰별 상세 통계</div>
            <div className="space-y-3">
              {shopsData.map((shop, idx) => (
                <div key={shop.name} className="flex items-center gap-4 p-3 rounded-2xl bg-cookie-beige/30 hover:bg-cookie-beige/50 transition">
                  <span
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm"
                    style={{ backgroundColor: COLORS.tiers[shop.plan_tier] }}
                  >
                    {idx + 1}
                  </span>
                  <div className="flex-1">
                    <div className="font-bold text-cookie-brown">{shop.name}</div>
                    <div className="text-xs text-cookie-brown/60">{shop.plan_tier}</div>
                  </div>
                  <div className="flex gap-4 text-sm">
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{shop.usage}%</div>
                      <div className="text-[10px] text-cookie-brown/50">운영점수</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{shop.cvr}%</div>
                      <div className="text-[10px] text-cookie-brown/50">전환율</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{shop.popularity}%</div>
                      <div className="text-[10px] text-cookie-brown/50">인기도</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* CS 분석 */}
      {activeTab === 'cs' && (
        <div className="space-y-6">
          {!csData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <MessageSquare size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">CS 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 채널별 CS 현황 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Globe size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">채널별 CS 현황</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cookie-orange/10">
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">채널</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">문의 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 품질</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">대기중</th>
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">품질 바</th>
                  </tr>
                </thead>
                <tbody>
                  {csData.channels.map(ch => (
                    <tr key={ch.channel} className="border-b border-cookie-orange/5 hover:bg-cookie-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cookie-brown">{ch.channel}</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{ch.count.toLocaleString()}</td>
                      <td className="py-3 px-2 text-right">
                        <span className={`font-bold ${parseFloat(ch.quality) >= 90 ? 'text-green-600' : 'text-yellow-600'}`}>
                          {ch.quality}%
                        </span>
                      </td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{ch.pending}</td>
                      <td className="py-3 px-2 w-40">
                        <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cookie-yellow to-cookie-orange"
                            style={{ width: `${ch.quality}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 최근 CS 샘플 */}
          {csData.recent && csData.recent.length > 0 && (
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">최근 CS 샘플</div>
            <div className="space-y-3">
              {csData.recent.map((item, idx) => (
                <div key={idx} className="p-4 rounded-2xl bg-cookie-beige/30">
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2 py-0.5 rounded-full bg-cookie-orange/20 text-xs font-bold text-cookie-brown">
                      {item.channel}
                    </span>
                    <span className={`text-sm font-bold ${item.quality >= 95 ? 'text-green-600' : 'text-yellow-600'}`}>
                      품질 {item.quality}%
                    </span>
                  </div>
                  <p className="text-sm text-cookie-brown">&ldquo;{item.text}&rdquo;</p>
                </div>
              ))}
            </div>
          </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 이상탐지 분석 */}
      {activeTab === 'anomaly' && (
        <div className="space-y-6">
          {/* 선택된 셀러 이상탐지 결과 */}
          {selectedUser?.model_predictions?.fraud && (
            <div className={`rounded-3xl border-2 p-5 shadow-sm backdrop-blur ${
              selectedUser.model_predictions.fraud.is_anomaly ? 'border-red-300 bg-red-50/80' : 'border-green-300 bg-green-50/80'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <Shield size={18} className={selectedUser.model_predictions.fraud.is_anomaly ? 'text-red-600' : 'text-green-600'} />
                <span className="text-sm font-black text-cookie-brown">{selectedUser.id} 이상거래 탐지 결과</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-black" style={{
                    color: selectedUser.model_predictions.fraud.is_anomaly ? '#DC2626' : '#16A34A'
                  }}>{(selectedUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%</div>
                  <div className="text-xs text-cookie-brown/60">이상 점수</div>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-black ${
                    selectedUser.model_predictions.fraud.is_anomaly ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {selectedUser.model_predictions.fraud.risk_level}
                  </div>
                  <div className="text-xs text-cookie-brown/60">판정</div>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-cookie-brown/40">{selectedUser.model_predictions.fraud.model}</div>
            </div>
          )}
          {!anomalyData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <AlertTriangle size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">이상탐지 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
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
            <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity size={18} className="text-cookie-orange" />
                <span className="text-xs font-bold text-cookie-brown">탐지율</span>
              </div>
              <div className="text-2xl font-black text-cookie-brown">{anomalyData.summary?.anomaly_rate || 0}%</div>
              <div className="text-xs text-cookie-brown/60">{anomalyData.summary?.anomaly_count || 0}/{anomalyData.summary?.total_sellers || 0}</div>
            </div>
          </div>

          {/* 이상유형별 분포 & 트렌드 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">이상 유형별 분포</div>
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

            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">일별 이상 탐지 트렌드</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={anomalyData.trend || []}>
                  <defs>
                    <linearGradient id="colorAnomaly" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="count" name="탐지 수" stroke="#EF4444" fill="url(#colorAnomaly)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 최근 알림 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Zap size={18} className="text-red-500" />
              <span className="text-sm font-black text-cookie-brown">실시간 이상 탐지 알림</span>
            </div>
            <div className="space-y-3">
              {(anomalyData.recent_alerts || []).map((alert, idx) => (
                <div key={idx} className={`flex items-center gap-4 p-4 rounded-2xl border-2 ${
                  alert.severity === 'high' ? 'border-red-200 bg-red-50' :
                  alert.severity === 'medium' ? 'border-orange-200 bg-orange-50' :
                  'border-yellow-200 bg-yellow-50'
                }`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    alert.severity === 'high' ? 'bg-red-500' :
                    alert.severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'
                  }`}>
                    <AlertTriangle size={18} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-cookie-brown">{alert.id}</span>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        alert.severity === 'high' ? 'bg-red-200 text-red-700' :
                        alert.severity === 'medium' ? 'bg-orange-200 text-orange-700' :
                        'bg-yellow-200 text-yellow-700'
                      }`}>{alert.type}</span>
                    </div>
                    <p className="text-sm text-cookie-brown/70">{alert.detail}</p>
                  </div>
                  <div className="text-xs text-cookie-brown/50">{alert.time}</div>
                </div>
              ))}
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* 예측 분석 */}
      {activeTab === 'prediction' && (
        <div className="space-y-6">
          {/* 개별 셀러 예측 검색 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-3">
              <Search size={16} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">개별 셀러 예측 조회</span>
            </div>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                placeholder="셀러 ID 입력 (예: SEL0001)"
                value={predictionSearchQuery}
                onChange={(e) => setPredictionSearchQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handlePredictionSearch(); }}
                className="flex-1 rounded-xl border-2 border-cookie-orange/20 bg-white px-4 py-2 text-sm font-semibold text-cookie-brown outline-none focus:border-cookie-orange transition-all"
              />
              <button
                onClick={() => handlePredictionSearch()}
                disabled={predictionUserLoading}
                className="rounded-xl bg-cookie-brown px-4 py-2 text-sm font-bold text-white hover:bg-cookie-brown/90 transition-all disabled:opacity-50"
              >
                {predictionUserLoading ? '조회중...' : '예측'}
              </button>
            </div>
            <div className="flex gap-2 flex-wrap">
              {['SEL0001', 'SEL0025', 'SEL0050', 'SEL0100'].map(id => (
                <button
                  key={id}
                  onClick={() => { setPredictionSearchQuery(id); handlePredictionSearch(id); }}
                  className="px-3 py-1 rounded-full bg-cookie-beige text-xs font-bold text-cookie-brown hover:bg-cookie-orange/20 transition-all"
                >
                  {id}
                </button>
              ))}
            </div>
          </div>

          {/* 개별 셀러 예측 결과 */}
          {predictionUser?.model_predictions && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Brain size={18} className="text-cookie-orange" />
                  <span className="text-sm font-black text-cookie-brown">{predictionUser.id} ML 예측 결과</span>
                  <span className="px-2 py-0.5 rounded-full bg-cookie-beige text-xs font-semibold text-cookie-brown">
                    {predictionUser.segment} · {predictionUser.plan_tier}
                  </span>
                </div>
                <button
                  onClick={() => setPredictionUser(null)}
                  className="text-xs text-cookie-brown/50 hover:text-cookie-brown transition-all"
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
                    <div className="text-xs font-bold text-cookie-brown mb-1">이탈 확률</div>
                    <div className="text-2xl font-black" style={{
                      color: predictionUser.model_predictions.churn.risk_code >= 2 ? '#DC2626' :
                             predictionUser.model_predictions.churn.risk_code === 1 ? '#EA580C' : '#16A34A'
                    }}>{predictionUser.model_predictions.churn.probability}%</div>
                    <div className="text-xs text-cookie-brown/60">{predictionUser.model_predictions.churn.risk_level}</div>
                  </div>
                )}
                {predictionUser.model_predictions.revenue && (
                  <div className="rounded-2xl p-4 border-2 border-purple-300 bg-purple-50">
                    <div className="text-xs font-bold text-cookie-brown mb-1">예상 월매출</div>
                    <div className="text-2xl font-black text-purple-600">
                      {predictionUser.model_predictions.revenue.predicted_next_month >= 10000
                        ? `₩${(predictionUser.model_predictions.revenue.predicted_next_month / 10000).toFixed(0)}만`
                        : `₩${(predictionUser.model_predictions.revenue.predicted_next_month || 0).toLocaleString()}`}
                    </div>
                    <div className="text-xs text-cookie-brown/60">성장률 {predictionUser.model_predictions.revenue.growth_rate}%</div>
                  </div>
                )}
                {predictionUser.model_predictions.fraud && (
                  <div className={`rounded-2xl p-4 border-2 ${
                    predictionUser.model_predictions.fraud.is_anomaly ? 'border-red-300 bg-red-50' : 'border-green-300 bg-green-50'
                  }`}>
                    <div className="text-xs font-bold text-cookie-brown mb-1">이상거래</div>
                    <div className="text-2xl font-black" style={{
                      color: predictionUser.model_predictions.fraud.is_anomaly ? '#DC2626' : '#16A34A'
                    }}>{predictionUser.model_predictions.fraud.risk_level}</div>
                    <div className="text-xs text-cookie-brown/60">점수 {(predictionUser.model_predictions.fraud.anomaly_score * 100).toFixed(1)}%</div>
                  </div>
                )}
                {predictionUser.model_predictions.cs_quality && (
                  <div className={`rounded-2xl p-4 border-2 ${
                    predictionUser.model_predictions.cs_quality.score >= 80 ? 'border-green-300 bg-green-50' :
                    predictionUser.model_predictions.cs_quality.score >= 50 ? 'border-yellow-300 bg-yellow-50' :
                    'border-red-300 bg-red-50'
                  }`}>
                    <div className="text-xs font-bold text-cookie-brown mb-1">CS 품질</div>
                    <div className="text-2xl font-black" style={{
                      color: predictionUser.model_predictions.cs_quality.score >= 80 ? '#16A34A' :
                             predictionUser.model_predictions.cs_quality.score >= 50 ? '#CA8A04' : '#DC2626'
                    }}>{predictionUser.model_predictions.cs_quality.score}점</div>
                    <div className="text-xs text-cookie-brown/60">{predictionUser.model_predictions.cs_quality.grade}</div>
                  </div>
                )}
                {predictionUser.model_predictions.segment && (
                  <div className="rounded-2xl p-4 border-2 border-blue-300 bg-blue-50">
                    <div className="text-xs font-bold text-cookie-brown mb-1">셀러 세그먼트</div>
                    <div className="text-lg font-black text-blue-600">{predictionUser.model_predictions.segment.segment_name}</div>
                    <div className="text-xs text-cookie-brown/60">클러스터 #{predictionUser.model_predictions.segment.cluster}</div>
                  </div>
                )}
              </div>
              {/* SHAP 요인 */}
              {predictionUser.model_predictions.churn?.factors?.length > 0 && (
                <div className="mt-4">
                  <div className="text-xs font-bold text-cookie-brown mb-2">이탈 주요 요인 (SHAP)</div>
                  <div className="space-y-2">
                    {predictionUser.model_predictions.churn.factors.map((f, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-full bg-cookie-orange text-white text-xs font-bold flex items-center justify-center shrink-0">
                          {i + 1}
                        </span>
                        <span className="text-xs font-semibold text-cookie-brown w-24">{f.factor}</span>
                        <div className="flex-1 h-2 bg-cookie-beige rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cookie-yellow to-cookie-orange"
                            style={{ width: `${Math.min(100, f.importance * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-cookie-orange w-10 text-right">{(f.importance * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {!predictionData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Brain size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">예측 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
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
                      ? 'bg-cookie-brown text-white'
                      : 'bg-white border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
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
                <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                  <div className="text-xs font-bold text-cookie-brown mb-1">모델 정확도</div>
                  <div className="text-2xl font-black text-cookie-brown">{predictionData.churn.model_accuracy}%</div>
                  <div className="text-xs text-cookie-brown/60">F1 Score</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 이탈 요인 분석 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">이탈 예측 주요 요인</div>
                  <div className="space-y-3">
                    {predictionData.churn.top_factors.map((factor, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-cookie-orange text-white text-xs font-bold flex items-center justify-center">
                          {idx + 1}
                        </span>
                        <div className="flex-1">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-semibold text-cookie-brown">{factor.factor}</span>
                            <span className="text-sm font-bold text-cookie-orange">{(factor.importance * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-cookie-yellow to-cookie-orange"
                              style={{ width: `${factor.importance * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 고위험 셀러 목록 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">이탈 고위험 셀러</div>
                  <div className="space-y-3">
                    {(predictionData.churn?.high_risk_users || []).map((user, idx) => (
                      <div key={idx} className="flex items-center gap-4 p-3 rounded-2xl bg-red-50 border border-red-200">
                        <div className="w-10 h-10 rounded-full bg-red-500 text-white font-bold flex items-center justify-center text-sm">
                          {user.probability}%
                        </div>
                        <div className="flex-1">
                          <div className="font-bold text-cookie-brown">{user.id}</div>
                          <div className="text-xs text-cookie-brown/60">{user.segment}</div>
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
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="text-xs font-bold text-cookie-brown mb-1">신뢰도</div>
                <div className="text-xl font-black text-cookie-brown">{predictionData.revenue.confidence || 0}%</div>
                <div className="text-xs text-cookie-brown/60">예측 정확도</div>
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
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="text-xs font-bold text-cookie-brown mb-1">평균 세션</div>
                <div className="text-2xl font-black text-cookie-brown">{predictionData.engagement.avg_session || 0}분</div>
                <div className="text-xs text-cookie-brown/60">세션당 운영 시간</div>
              </div>
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4 col-span-2">
                <div className="text-xs font-bold text-cookie-brown mb-1">일일 세션 수</div>
                <div className="text-2xl font-black text-cookie-brown">{predictionData.engagement.sessions_per_day || 0}</div>
                <div className="text-xs text-cookie-brown/60">셀러당 평균 접속 횟수</div>
              </div>
            </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 코호트 분석 */}
      {activeTab === 'cohort' && (
        <div className="space-y-6">
          {!cohortData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Target size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">코호트 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 코호트 유형 선택 */}
          <div className="flex gap-2">
            {[
              { key: 'retention', label: '리텐션', icon: Repeat },
              { key: 'ltv', label: 'LTV', icon: DollarSign },
              { key: 'conversion', label: '전환 퍼널', icon: Target },
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setCohortTab(tab.key)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                    cohortTab === tab.key
                      ? 'bg-cookie-brown text-white'
                      : 'bg-white border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
                  }`}
                >
                  <Icon size={14} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* 리텐션 히트맵 */}
          {cohortTab === 'retention' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">주간 리텐션 코호트</div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b-2 border-cookie-orange/10">
                      <th className="text-left py-3 px-3 font-bold text-cookie-brown">코호트</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 0</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 1</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 2</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 3</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 4</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(cohortData.retention || []).map((row, idx) => (
                      <tr key={idx} className="border-b border-cookie-orange/5">
                        <td className="py-3 px-3 font-semibold text-cookie-brown">{row.cohort}</td>
                        {['week0', 'week1', 'week2', 'week3', 'week4'].map((week) => (
                          <td key={week} className="py-3 px-3 text-center">
                            {row[week] !== null ? (
                              <span
                                className="inline-block px-3 py-1 rounded-lg text-xs font-bold"
                                style={{
                                  backgroundColor: `rgba(255, 140, 66, ${row[week] / 100})`,
                                  color: row[week] > 50 ? 'white' : '#5C4A3D'
                                }}
                              >
                                {row[week]}%
                              </span>
                            ) : (
                              <span className="text-cookie-brown/30">-</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* LTV 코호트 */}
          {cohortTab === 'ltv' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">월별 코호트 LTV</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={cohortData.ltv_by_cohort || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="cohort" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="ltv" name="LTV (원)" fill="#FF8C42" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="users" name="셀러 수" fill="#4ADE80" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* 전환 퍼널 */}
          {cohortTab === 'conversion' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">코호트별 전환 퍼널</div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={cohortData.conversion || []} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="cohort" tick={{ fill: '#5C4A3D', fontSize: 10 }} angle={-15} textAnchor="end" />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="registered" name="가입" fill="#60A5FA" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="activated" name="활성화" fill="#4ADE80" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="engaged" name="참여" fill="#FFD93D" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="converted" name="전환" fill="#F472B6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="retained" name="유지" fill="#A78BFA" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 트렌드 분석 */}
      {activeTab === 'trend' && (
        <div className="space-y-6">
          {!trendData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <TrendingUp size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">트렌드 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* KPI 요약 카드 */}
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
            {(trendData.kpis || []).map((kpi, idx) => (
              <div key={idx} className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-cookie-brown/60">{kpi.name}</span>
                  <span className={`flex items-center gap-1 text-xs font-bold ${
                    kpi.change >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {kpi.change >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                    {kpi.change >= 0 ? '+' : ''}{kpi.change}%
                  </span>
                </div>
                <div className="text-2xl font-black text-cookie-brown">
                  {kpi.name.includes('ARPU') ? '₩' : ''}{typeof kpi.current === 'number' ? kpi.current.toLocaleString() : kpi.current}{kpi.name.includes('률') || kpi.name.includes('전환') ? '%' : ''}
                </div>
                <div className="text-xs text-cookie-brown/50">이전: {kpi.previous.toLocaleString()}</div>
              </div>
            ))}
          </div>

          {/* 일별 메트릭 차트 */}
          {(trendData.daily_metrics?.length > 0) && (
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">일별 핵심 지표 추이</div>
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
            {/* DAU 예측 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-4">
                <Brain size={18} className="text-cookie-orange" />
                <span className="text-sm font-black text-cookie-brown">DAU 예측 (5일)</span>
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

            {/* 상관관계 분석 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={18} className="text-cookie-orange" />
                <span className="text-sm font-black text-cookie-brown">지표 상관관계</span>
              </div>
              <div className="space-y-3">
                {(trendData.correlation || []).map((item, idx) => {
                  const corr = item.correlation ?? 0;
                  return (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-xs font-semibold text-cookie-brown">{item.var1 || item.metric1} ↔ {item.var2 || item.metric2}</span>
                        <span className={`text-xs font-bold ${
                          corr >= 0.8 ? 'text-green-600' :
                          corr >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {corr.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
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
      )}

      {/* 마케팅 최적화 */}
      {activeTab === 'marketing' && (
        <div className="space-y-6">
          {/* 헤더 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <DollarSign size={20} className="text-cookie-orange" />
              <span className="text-lg font-black text-cookie-brown">마케팅 최적화</span>
            </div>
            <button
              onClick={() => loadMarketingUserStatus(marketingUser)}
              disabled={marketingLoading || !marketingUser}
              className="p-2.5 rounded-xl border-2 border-cookie-orange/20 hover:border-cookie-orange hover:bg-cookie-orange/10 transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-5 h-5 text-cookie-orange ${marketingLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* 셀러 선택 UI */}
          <div className="bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
            <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
              <Users size={16} className="text-cookie-orange" />
              셀러 선택
            </h3>

            <div className="flex flex-col gap-4">
              {/* 예시 셀러 버튼들 */}
              <div className="flex flex-wrap gap-3">
                <span className="text-sm text-cookie-brown/70 self-center mr-2">예시:</span>
                {MARKETING_EXAMPLE_USERS.map((user) => (
                  <button
                    key={user.id}
                    onClick={() => handleMarketingExampleSelect(user.id)}
                    className={`px-4 py-2.5 rounded-xl border-2 transition-all flex flex-col items-start ${
                      marketingUser === user.id
                        ? 'border-cookie-orange bg-cookie-orange/10 text-cookie-brown'
                        : 'border-cookie-orange/20 hover:border-cookie-orange/40 bg-white text-cookie-brown/80'
                    }`}
                  >
                    <span className="font-bold text-sm">{user.id}</span>
                    <span className="text-xs text-cookie-brown/60">{user.description}</span>
                  </button>
                ))}
              </div>

              {/* 구분선 */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-cookie-orange/20" />
                <span className="text-sm text-cookie-brown/50">또는 직접 입력</span>
                <div className="flex-1 h-px bg-cookie-orange/20" />
              </div>

              {/* 직접 입력 */}
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={marketingUserInput}
                  onChange={(e) => setMarketingUserInput(e.target.value)}
                  onKeyDown={handleMarketingInputKeyDown}
                  placeholder="셀러 ID 입력 (예: SEL0001)"
                  className="flex-1 px-4 py-3 rounded-xl border-2 border-cookie-orange/20 bg-white text-cookie-brown font-medium placeholder:text-cookie-brown/40 focus:border-cookie-orange focus:ring-2 focus:ring-cookie-orange/20 outline-none transition-all"
                />
                <button
                  onClick={handleMarketingDirectSearch}
                  disabled={marketingLoading || !marketingUserInput.trim()}
                  className="px-6 py-3 rounded-xl bg-cookie-orange text-white font-bold hover:bg-cookie-orange/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Search size={18} />
                  조회
                </button>
              </div>

              {/* 현재 선택된 셀러 표시 */}
              {marketingUser && (
                <div className="flex items-center gap-2 text-sm text-cookie-brown/70 bg-cookie-yellow/10 px-4 py-2 rounded-xl">
                  <Target size={16} className="text-cookie-orange" />
                  현재 선택: <span className="font-bold text-cookie-brown">{marketingUser}</span>
                </div>
              )}
            </div>
          </div>

          {/* 셀러 현황 */}
          {marketingUserStatus && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* 운영 현황 */}
              <div className="bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
                <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
                  <DollarSign size={16} className="text-cookie-orange" />
                  운영 현황
                </h3>
                <div className="space-y-3">
                  {Object.entries(marketingUserStatus.resources || {}).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between p-2 rounded-lg bg-gray-50">
                      <span className="text-sm text-cookie-brown">{key === 'ad_budget' ? '광고 예산' : key === 'monthly_revenue' ? '월 매출' : key === 'product_count' ? '상품 수' : key === 'order_count' ? '주문 수' : key}</span>
                      <span className="font-bold text-cookie-brown">{value?.toLocaleString() || 0}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 쇼핑몰 요약 */}
              <div className="lg:col-span-2 bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
                <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
                  <ShoppingBag size={18} className="text-cookie-orange" />
                  운영 쇼핑몰 현황
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-cookie-orange/10 to-white border border-cookie-orange/10">
                    <div className="text-xs text-cookie-brown/70">총 쇼핑몰</div>
                    <div className="text-xl font-bold text-cookie-brown">{marketingUserStatus.shops?.length || 0}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-cookie-yellow/10 to-white border border-cookie-yellow/10">
                    <div className="text-xs text-cookie-brown/70">평균 전환율</div>
                    <div className="text-xl font-bold text-cookie-brown">
                      {(marketingUserStatus.shops?.reduce((sum, c) => sum + (c.cvr || 0), 0) / (marketingUserStatus.shops?.length || 1)).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-green-100 to-white border border-green-100">
                    <div className="text-xs text-cookie-brown/70">최대 전환율</div>
                    <div className="text-xl font-bold text-cookie-brown">
                      {Math.max(...(marketingUserStatus.shops?.map(c => c.cvr || 0) || [0])).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-white border border-purple-100">
                    <div className="text-xs text-cookie-brown/70">총 매출</div>
                    <div className="text-xl font-bold text-cookie-brown">{marketingUserStatus.total_revenue?.toLocaleString() || '계산중'}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 최적화 버튼 */}
          <div className="flex justify-center">
            <button
              onClick={runMarketingOptimization}
              disabled={marketingOptimizing || !marketingUserStatus}
              className="px-8 py-4 bg-gradient-to-r from-cookie-orange to-cookie-yellow text-white font-bold text-lg rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3"
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
            <div className="bg-gradient-to-br from-cookie-yellow/10 via-white to-cookie-orange/10 rounded-2xl p-6 border-2 border-cookie-orange/20 shadow-lg">
              <h3 className="text-lg font-bold text-cookie-brown mb-4 flex items-center gap-2">
                <TrendingUp size={20} className="text-cookie-orange" />
                최적화 결과 - 개인화된 마케팅 추천
              </h3>

              {/* 예상 효과 */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">예상 전환율 증가</div>
                  <div className="text-xl font-bold text-green-600 flex items-center gap-1">
                    +{Number(marketingResult.total_cvr_gain || 0).toFixed(1)}%
                    <ArrowUpRight size={18} />
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">추천 개수</div>
                  <div className="text-xl font-bold text-blue-600">{marketingResult.recommendations?.length || 0}개</div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">평균 효율</div>
                  <div className="text-xl font-bold text-pink-600">{marketingResult.recommendations?.length > 0
                    ? (marketingResult.recommendations.reduce((sum, r) => sum + Number(r.efficiency || 0), 0) / marketingResult.recommendations.length * 100).toFixed(1)
                    : 0}%</div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">최적화 방식</div>
                  <div className="text-xl font-bold text-purple-600">AI</div>
                </div>
              </div>

              {/* 추천 리스트 */}
              <div className="bg-white rounded-xl p-4 border border-cookie-orange/10">
                <h4 className="font-bold text-cookie-brown mb-4">우선순위별 마케팅 추천</h4>
                <div className="space-y-2">
                  {marketingResult.recommendations?.slice(0, 8).map((rec, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 rounded-xl bg-gradient-to-r from-gray-50 to-white border border-gray-100 hover:border-cookie-orange/30 transition-colors">
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm ${
                        idx === 0 ? 'bg-gradient-to-br from-amber-500 to-yellow-500' :
                        idx === 1 ? 'bg-gradient-to-br from-gray-400 to-gray-500' :
                        idx === 2 ? 'bg-gradient-to-br from-orange-400 to-orange-500' :
                        'bg-gradient-to-br from-cookie-orange to-cookie-yellow'
                      }`}>
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-cookie-brown">{rec.channel_name}</div>
                        <div className="text-xs text-cookie-brown/60">{rec.campaign_type} {rec.from_budget} → {rec.to_budget}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-600">+{Number(rec.cvr_gain || 0).toFixed(1)}%</div>
                        <div className="text-xs text-cookie-brown/50">전환율 증가</div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-cookie-brown text-sm">
                          {(() => {
                            const cost = rec.cost;
                            const formatNum = (n) => n >= 1000 ? `${(n / 1000).toFixed(0)}K` : `${n}`;
                            if (!cost || typeof cost !== 'object') return '0';
                            if (rec.campaign_type === 'cpc') {
                              return `${formatNum(Number(cost.ad_spend || 0))}원`;
                            } else if (rec.campaign_type === 'display') {
                              return `${formatNum(Number(cost.ad_spend || 0))}원`;
                            } else if (rec.campaign_type === 'social') {
                              return `${formatNum(Number(cost.ad_spend || 0))}원`;
                            }
                            return formatNum(Number(cost.ad_spend || 0));
                          })()}
                        </div>
                        <div className="text-xs text-cookie-brown/50">
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
              <RefreshCw size={48} className="mx-auto mb-3 text-cookie-orange animate-spin" />
              <p className="text-sm font-semibold text-cookie-brown/50">셀러 정보를 불러오는 중...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
