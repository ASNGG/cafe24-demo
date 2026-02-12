// components/panels/analysis/ShopTab.js
// 쇼핑몰 분석 탭

import { useMemo } from 'react';
import { ShoppingBag } from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import AnalysisEmptyState from './common/EmptyState';
import { ANALYSIS_COLORS as COLORS } from '@/components/common/constants';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

export default function ShopTab({ shopsData }) {
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
    <div className="space-y-6">
      {!shopsData ? (
        <AnalysisEmptyState
          icon={ShoppingBag}
          title="쇼핑몰 데이터를 불러올 수 없습니다"
          subtitle="백엔드 API 연결을 확인하세요"
        />
      ) : (
      <>
      {/* 쇼핑몰 운영 차트 */}
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <ShoppingBag size={18} className="text-cafe24-orange" />
          <span className="text-sm font-black text-cafe24-brown">인기 쇼핑몰 분석</span>
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
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="mb-4 text-sm font-black text-cafe24-brown">쇼핑몰별 상세 통계</div>
        <div className="space-y-3">
          {shopsData.map((shop, idx) => (
            <div key={shop.name} className="flex items-center gap-4 p-3 rounded-2xl bg-cafe24-beige/30 hover:bg-cafe24-beige/50 transition">
              <span
                className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm"
                style={{ backgroundColor: COLORS.tiers[shop.plan_tier] }}
              >
                {idx + 1}
              </span>
              <div className="flex-1">
                <div className="font-bold text-cafe24-brown">{shop.name}</div>
                <div className="text-xs text-cafe24-brown/60">{shop.plan_tier}</div>
              </div>
              <div className="flex gap-4 text-sm">
                <div className="text-center">
                  <div className="font-bold text-cafe24-brown">{shop.usage}%</div>
                  <div className="text-[10px] text-cafe24-brown/50">운영점수</div>
                </div>
                <div className="text-center">
                  <div className="font-bold text-cafe24-brown">{shop.cvr}%</div>
                  <div className="text-[10px] text-cafe24-brown/50">전환율</div>
                </div>
                <div className="text-center">
                  <div className="font-bold text-cafe24-brown">{shop.popularity}%</div>
                  <div className="text-[10px] text-cafe24-brown/50">인기도</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      </>
      )}
    </div>
  );
}
