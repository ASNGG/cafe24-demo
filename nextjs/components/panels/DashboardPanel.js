// components/panels/DashboardPanel.js
// CAFE24 AI Platform - 대시보드 패널 (Recharts 버전)

import { useCallback, useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import KpiCard from '@/components/KpiCard';
import EmptyState from '@/components/EmptyState';
import { SkeletonCard } from '@/components/Skeleton';
import {
  ShoppingBag, Users, Globe, BarChart3, TrendingUp, RefreshCw,
  AlertTriangle, Zap, ArrowUpRight, ArrowDownRight, Brain, Target
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import CustomTooltip from '@/components/common/CustomTooltip';
import { DASHBOARD_COLORS as COLORS } from '@/components/common/constants';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar
} from 'recharts';

// 파이 차트용 커스텀 툴팁
const PieTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0];
  return (
    <div className="rounded-xl border-2 border-cafe24-orange/20 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <p className="text-xs font-bold text-cafe24-brown">{data.name}</p>
      <p className="text-sm font-semibold" style={{ color: data.payload.fill }}>
        {data.value.toLocaleString()}명 ({((data.value / data.payload.total) * 100).toFixed(1)}%)
      </p>
    </div>
  );
};

// 커스텀 라벨
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  if (percent < 0.05) return null;
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="#1A1A2E"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      className="text-[10px] font-bold"
    >
      {`${name} ${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export default function DashboardPanel({ auth, selectedShop, apiCall }) {
  const [dashboard, setDashboard] = useState(null);
  const [insights, setInsights] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);

  // 드릴다운 상태
  const [drilldownSegment, setDrilldownSegment] = useState(null);
  const [drilldownData, setDrilldownData] = useState(null);
  const [drilldownLoading, setDrilldownLoading] = useState(false);

  // M44: loadData를 useCallback으로 안정화
  const loadData = useCallback(async () => {
    setLoading(true);

    const [summaryRes, insightsRes, alertsRes] = await Promise.all([
      apiCall({
        endpoint: '/api/dashboard/summary',
        auth,
        timeoutMs: 30000,
      }),
      apiCall({
        endpoint: '/api/dashboard/insights',
        auth,
        timeoutMs: 10000,
      }),
      apiCall({
        endpoint: '/api/dashboard/alerts?limit=5',
        auth,
        timeoutMs: 10000,
      }),
    ]);

    setLoading(false);

    if (summaryRes?.status === 'success') {
      setDashboard(summaryRes);
    } else {
      setDashboard(null);
      toast.error('대시보드 데이터를 불러올 수 없습니다');
    }

    if (insightsRes?.status === 'success' && insightsRes.insights) {
      setInsights(insightsRes.insights);
    }

    if (alertsRes?.status === 'success' && alertsRes.alerts) {
      setAlerts(alertsRes.alerts);
    }
  }, [auth, apiCall]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // L20: 대시보드 자동 폴링 (60초 간격, 탭 비활성 시 건너뛰기)
  useEffect(() => {
    const interval = setInterval(() => {
      if (document.hidden) return;
      loadData();
    }, 60000);
    return () => clearInterval(interval);
  }, [loadData]);

  // 세그먼트 드릴다운 핸들러
  const handleSegmentClick = async (data) => {
    if (!data || !data.name) return;

    setDrilldownSegment(data);
    setDrilldownLoading(true);

    try {
      // 세그먼트 상세 정보 API 호출
      const res = await apiCall({
        endpoint: `/api/users/segments/${encodeURIComponent(data.name)}/details`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'success') {
        setDrilldownData(res);
      } else {
        // 폴백: 기본 데이터 생성
        setDrilldownData({
          segment: data.name,
          count: data.value,
          percentage: ((data.value / data.total) * 100).toFixed(1),
          avg_monthly_revenue: 0,
          avg_product_count: 0,
          avg_order_count: 0,
          top_activities: [],
          retention_rate: '-',
        });
      }
    } catch (e) {
      // 폴백 데이터
      setDrilldownData({
        segment: data.name,
        count: data.value,
        percentage: ((data.value / data.total) * 100).toFixed(1),
        avg_monthly_revenue: 0,
        avg_product_count: 0,
        avg_order_count: 0,
        top_activities: [],
        retention_rate: '-',
      });
    } finally {
      setDrilldownLoading(false);
    }
  };

  // 드릴다운 닫기
  const closeDrilldown = useCallback(() => {
    setDrilldownSegment(null);
    setDrilldownData(null);
  }, []);

  // L25: Escape 키로 드릴다운 모달 닫기
  useEffect(() => {
    if (!drilldownSegment) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') closeDrilldown();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [drilldownSegment, closeDrilldown]);

  // 세그먼트 분포 차트 데이터 (의존성 세분화: dashboard 전체 → 하위 경로)
  const sellerSegments = dashboard?.seller_stats?.segments;
  const segmentData = useMemo(() => {
    if (!sellerSegments) return [];
    const total = Object.values(sellerSegments).reduce((a, b) => a + b, 0);
    return Object.entries(sellerSegments).map(([name, value], idx) => ({
      name,
      value,
      total,
      fill: COLORS.primary[idx % COLORS.primary.length],
    }));
  }, [sellerSegments]);

  // 운영 이벤트 통계 차트 데이터
  const orderByType = dashboard?.order_stats?.by_type;
  const eventData = useMemo(() => {
    if (!orderByType) return [];
    return Object.entries(orderByType).map(([name, value], idx) => ({
      name,
      value,
      fill: COLORS.primary[idx % COLORS.primary.length],
    }));
  }, [orderByType]);

  // 쇼핑몰 플랜별 데이터
  const shopByTier = dashboard?.shop_stats?.by_tier;
  const tierData = useMemo(() => {
    if (!shopByTier) return [];
    return Object.entries(shopByTier).map(([name, value]) => ({
      name,
      value,
      fill: COLORS.tiers[name] || '#1B6FF0',
    }));
  }, [shopByTier]);

  // GMV 데이터
  const dailyGmv = dashboard?.daily_gmv;
  const gmvData = useMemo(() => {
    return dailyGmv || [];
  }, [dailyGmv]);

  // CS 문의 카테고리별 데이터
  const csByCategory = dashboard?.cs_stats?.by_category;
  const categoryData = useMemo(() => {
    if (!csByCategory) return [];
    return Object.entries(csByCategory)
      .map(([name, value], idx) => ({
        name,
        value,
        fill: COLORS.primary[idx % COLORS.primary.length],
      }))
      .sort((a, b) => b.value - a.value);
  }, [csByCategory]);

  return (
    <div>
      <SectionHeader
        title="CAFE24 AI 대시보드"
        subtitle="플랫폼 현황 요약"
        right={
          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              disabled={loading}
              aria-label="데이터 새로고침"
              className="rounded-full border-2 border-cafe24-orange/20 bg-white/80 p-1.5 hover:bg-cafe24-beige transition disabled:opacity-50"
            >
              <RefreshCw size={14} className={`text-cafe24-brown ${loading ? 'animate-spin' : ''}`} />
            </button>
            {!loading && (
              <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
                dashboard
                  ? 'border-green-400/50 bg-green-50 text-green-700'
                  : 'border-red-400/50 bg-red-50 text-red-700'
              }`}>
                {dashboard ? 'LIVE' : 'NO DATA'}
              </span>
            )}
          </div>
        }
      />

      {loading && !dashboard ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : null}

      {dashboard ? (
        <>
          {/* KPI 카드 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <KpiCard
              title="쇼핑몰"
              value={`${dashboard.shop_stats?.total || 0}개`}
              subtitle={`Enterprise: ${dashboard.shop_stats?.by_tier?.Enterprise || 0}`}
              icon={<ShoppingBag size={18} className="text-cafe24-brown" />}
              tone="yellow"
            />
            <KpiCard
              title="전체 셀러"
              value={`${(dashboard.seller_stats?.total || 0).toLocaleString()}명`}
              subtitle={`이상 거래: ${dashboard.seller_stats?.anomaly_count || 0}`}
              icon={<Users size={18} className="text-cafe24-brown" />}
              tone="orange"
            />
            <KpiCard
              title="CS 문의"
              value={`${(dashboard.cs_stats?.total || 0).toLocaleString()}건`}
              subtitle={`만족도: ${dashboard.cs_stats?.avg_satisfaction || '-'}%`}
              icon={<Globe size={18} className="text-cafe24-brown" />}
              tone="cream"
            />
            <KpiCard
              title="운영 이벤트"
              value={`${(dashboard.order_stats?.total || 0).toLocaleString()}건`}
              subtitle="최근 30일"
              icon={<BarChart3 size={18} className="text-cafe24-brown" />}
              tone="green"
            />
          </div>

          {/* 일별 GMV 추이 차트 */}
          <div className="mb-6 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={18} className="text-cafe24-orange" />
              <span className="text-sm font-black text-cafe24-brown">일별 GMV 추이</span>
            </div>
            {gmvData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={gmvData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorGmv" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1B6FF0" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#1B6FF0" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#5C4A3D', fontSize: 11 }}
                  tickLine={{ stroke: '#FFD93D60' }}
                  axisLine={{ stroke: '#FFD93D60' }}
                />
                <YAxis
                  tick={{ fill: '#5C4A3D', fontSize: 11 }}
                  tickLine={{ stroke: '#FFD93D60' }}
                  axisLine={{ stroke: '#FFD93D60' }}
                  tickFormatter={(v) => `${(v/100000000).toFixed(1)}억`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="gmv"
                  name="GMV"
                  stroke="#1B6FF0"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorGmv)"
                  dot={{ fill: '#1B6FF0', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-sm text-cafe24-brown/60">
                GMV 데이터 없음
              </div>
            )}
          </div>

          {/* 메인 차트 그리드 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* 셀러 세그먼트 분포 - 파이 차트 (클릭 가능) */}
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 flex items-center justify-between">
                <span className="text-sm font-black text-cafe24-brown">셀러 세그먼트 분포</span>
                <span className="text-[10px] text-cafe24-brown/50 bg-cafe24-cream px-2 py-0.5 rounded-full">클릭하여 상세보기</span>
              </div>
              {segmentData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={segmentData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={3}
                      dataKey="value"
                      labelLine={false}
                      label={renderCustomLabel}
                      animationBegin={0}
                      animationDuration={800}
                      onClick={handleSegmentClick}
                      style={{ cursor: 'pointer' }}
                    >
                      {segmentData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.fill}
                          stroke="#fff"
                          strokeWidth={2}
                          className="hover:opacity-80 transition-opacity cursor-pointer"
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<PieTooltip />} />
                    <Legend
                      verticalAlign="bottom"
                      height={36}
                      formatter={(value) => <span className="text-xs font-semibold text-cafe24-brown">{value}</span>}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-sm text-cafe24-brown/60">
                  세그먼트 데이터 없음
                </div>
              )}
            </div>

            {/* 운영 이벤트 통계 - 바 차트 */}
            <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">운영 이벤트 통계</div>
              {eventData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={eventData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#5C4A3D', fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: '#FFD93D60' }}
                      angle={-20}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis
                      tick={{ fill: '#5C4A3D', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${(v/1000).toFixed(0)}K`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar
                      dataKey="value"
                      name="이벤트"
                      radius={[8, 8, 0, 0]}
                      animationDuration={800}
                    >
                      {eventData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-sm text-cafe24-brown/60">
                  이벤트 데이터 없음
                </div>
              )}
            </div>
          </div>

          {/* 쇼핑몰 플랜별 분포 - Radial Bar Chart */}
          {tierData.length > 0 && (
            <div className="mb-6 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">쇼핑몰 플랜별 분포</div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Radial 차트 */}
                <ResponsiveContainer width="100%" height={250}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="20%"
                    outerRadius="90%"
                    data={tierData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar
                      minAngle={15}
                      background
                      clockWise
                      dataKey="value"
                      cornerRadius={10}
                      animationDuration={800}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      iconSize={10}
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                      formatter={(value) => <span className="text-xs font-semibold text-cafe24-brown">{value}</span>}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>

                {/* 플랜 카드 그리드 */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 content-center">
                  {tierData.map(({ name, value, fill }) => (
                    <div
                      key={name}
                      className="rounded-2xl border-2 p-4 text-center transition-all hover:scale-105 hover:shadow-md"
                      style={{
                        borderColor: `${fill}50`,
                        background: `linear-gradient(135deg, ${fill}10 0%, ${fill}05 100%)`
                      }}
                    >
                      <div className="text-xs font-bold" style={{ color: fill }}>{name}</div>
                      <div className="text-2xl font-black text-cafe24-brown">{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* CS 문의 카테고리별 통계 */}
          {categoryData.length > 0 && (
            <div className="mb-6 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cafe24-brown">CS 문의 카테고리별 통계</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={categoryData} layout="vertical" margin={{ top: 5, right: 30, left: 50, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: '#5C4A3D', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: '#5C4A3D', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={50}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar
                    dataKey="value"
                    name="문의 건수"
                    radius={[0, 8, 8, 0]}
                    animationDuration={800}
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* AI 인사이트 & 빠른 액션 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* AI 인사이트 */}
            <div className="rounded-3xl border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <Brain size={18} className="text-purple-600" />
                <span className="text-sm font-black text-purple-900">AI 인사이트</span>
                <span className="ml-auto px-2 py-0.5 rounded-full bg-purple-500 text-white text-[10px] font-bold">
                  LIVE
                </span>
              </div>
              <div className="space-y-3">
                {insights.length > 0 ? insights.map((insight, idx) => {
                  const iconConfig = {
                    positive: { bg: 'bg-green-100', icon: <ArrowUpRight size={14} className="text-green-600" /> },
                    warning: { bg: 'bg-yellow-100', icon: <Target size={14} className="text-yellow-600" /> },
                    neutral: { bg: 'bg-blue-100', icon: <Zap size={14} className="text-blue-600" /> },
                  };
                  const config = iconConfig[insight.type] || iconConfig.neutral;

                  return (
                    <div key={idx} className="flex items-start gap-3 p-3 rounded-2xl bg-white/80 border border-purple-100">
                      <div className={`w-8 h-8 rounded-full ${config.bg} flex items-center justify-center flex-shrink-0`}>
                        {config.icon}
                      </div>
                      <div>
                        <div className="text-sm font-bold text-cafe24-brown">{insight.title}</div>
                        <div className="text-xs text-cafe24-brown/70">{insight.description}</div>
                      </div>
                    </div>
                  );
                }) : (
                  <div className="flex items-center justify-center p-4 text-sm text-cafe24-brown/50">
                    인사이트 로딩 중...
                  </div>
                )}
              </div>
            </div>

            {/* 실시간 알림 */}
            <div className="rounded-3xl border-2 border-red-200 bg-gradient-to-br from-red-50 to-white p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle size={18} className="text-red-600" />
                <span className="text-sm font-black text-red-900">실시간 알림</span>
                <span className="ml-auto px-2 py-0.5 rounded-full bg-red-500 text-white text-[10px] font-bold">
                  {alerts.length || dashboard?.seller_stats?.anomaly_count || 0}
                </span>
              </div>
              <div className="space-y-3">
                {alerts.length > 0 ? alerts.map((alert, idx) => {
                  const colorMap = {
                    red: { dot: 'bg-red-500', border: 'border-red-100', animate: idx === 0 },
                    orange: { dot: 'bg-orange-500', border: 'border-orange-100', animate: false },
                    yellow: { dot: 'bg-yellow-500', border: 'border-yellow-100', animate: false },
                  };
                  const colors = colorMap[alert.color] || colorMap.yellow;

                  return (
                    <div key={idx} className={`flex items-center gap-3 p-3 rounded-2xl bg-white/80 border ${colors.border}`}>
                      <div className={`w-2 h-2 rounded-full ${colors.dot} ${colors.animate ? 'animate-pulse' : ''}`} />
                      <div className="flex-1">
                        <div className="text-sm font-bold text-cafe24-brown">{alert.type}</div>
                        <div className="text-xs text-cafe24-brown/70">{alert.user_id} - {alert.detail}</div>
                      </div>
                      <span className="text-[10px] text-cafe24-brown/50">{alert.time_ago}</span>
                    </div>
                  );
                }) : (
                  <div className="flex items-center justify-center p-4 text-sm text-cafe24-brown/50">
                    알림이 없습니다
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      ) : (
        !loading && <EmptyState title="데이터가 없습니다" desc="백엔드 API 연결을 확인하세요." />
      )}

      {/* 세그먼트 드릴다운 모달 */}
      {drilldownSegment && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          onClick={closeDrilldown}
        >
          <div
            className="relative w-full max-w-md mx-4 rounded-3xl border-2 border-cafe24-orange/30 bg-white shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* 헤더 */}
            <div
              className="p-5 text-white"
              style={{ background: `linear-gradient(135deg, ${drilldownSegment.fill} 0%, ${drilldownSegment.fill}CC 100%)` }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs opacity-80 mb-1">세그먼트 상세</div>
                  <div className="text-xl font-black">{drilldownSegment.name}</div>
                </div>
                <button
                  onClick={closeDrilldown}
                  aria-label="닫기"
                  className="w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition"
                >
                  <span className="text-lg">&times;</span>
                </button>
              </div>
              <div className="mt-3 flex gap-4">
                <div>
                  <div className="text-2xl font-black">{drilldownSegment.value.toLocaleString()}</div>
                  <div className="text-xs opacity-80">총 셀러 수</div>
                </div>
                <div>
                  <div className="text-2xl font-black">
                    {((drilldownSegment.value / drilldownSegment.total) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs opacity-80">전체 비율</div>
                </div>
              </div>
            </div>

            {/* 상세 정보 */}
            <div className="p-5">
              {drilldownLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="w-8 h-8 border-4 border-cafe24-orange/30 border-t-cafe24-orange rounded-full animate-spin" />
                </div>
              ) : drilldownData ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-cafe24-cream/50">
                      <div className="text-xs text-cafe24-brown/60 mb-1">평균 월매출</div>
                      <div className="text-lg font-black text-cafe24-brown">{drilldownData.avg_monthly_revenue || '-'}만원</div>
                    </div>
                    <div className="p-3 rounded-xl bg-cafe24-cream/50">
                      <div className="text-xs text-cafe24-brown/60 mb-1">평균 상품수</div>
                      <div className="text-lg font-black text-cafe24-brown">{drilldownData.avg_product_count || '-'}개</div>
                    </div>
                    <div className="p-3 rounded-xl bg-cafe24-cream/50">
                      <div className="text-xs text-cafe24-brown/60 mb-1">평균 주문수</div>
                      <div className="text-lg font-black text-cafe24-brown">{drilldownData.avg_order_count || '-'}건</div>
                    </div>
                    <div className="p-3 rounded-xl bg-cafe24-cream/50">
                      <div className="text-xs text-cafe24-brown/60 mb-1">리텐션</div>
                      <div className="text-lg font-black text-cafe24-brown">{drilldownData.retention_rate || '-'}%</div>
                    </div>
                  </div>

                  {drilldownData.top_activities && (
                    <div>
                      <div className="text-xs text-cafe24-brown/60 mb-2">주요 활동</div>
                      <div className="flex flex-wrap gap-2">
                        {drilldownData.top_activities.map((activity, idx) => (
                          <span
                            key={idx}
                            className="px-3 py-1 rounded-full text-xs font-bold"
                            style={{ background: `${drilldownSegment.fill}20`, color: drilldownSegment.fill }}
                          >
                            {activity}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : null}
            </div>

            {/* 푸터 */}
            <div className="px-5 pb-5">
              <button
                onClick={closeDrilldown}
                className="w-full py-3 rounded-xl bg-cafe24-beige text-cafe24-brown font-bold text-sm hover:bg-cafe24-cream transition"
              >
                닫기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
