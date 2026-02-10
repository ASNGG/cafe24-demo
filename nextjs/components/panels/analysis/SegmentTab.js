// components/panels/analysis/SegmentTab.js
// 세그먼트 분석 탭

import { useMemo } from 'react';
import { Users } from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

export default function SegmentTab({ selectedUser, segmentsData }) {
  const segmentCompareData = useMemo(() => {
    if (!segmentsData) return [];
    return Object.entries(segmentsData).map(([name, data]) => ({
      name: name.replace(' ', '\n'),
      셀러수: data.count,
      평균매출: data.avg_monthly_revenue,
      리텐션: data.retention,
    }));
  }, [segmentsData]);

  return (
    <div className="space-y-6">
      {/* 선택된 셀러 세그먼트 */}
      {selectedUser?.model_predictions?.segment && (
        <div className="rounded-3xl border-2 border-blue-300 bg-blue-50/80 p-5 shadow-sm backdrop-blur">
          <div className="flex items-center gap-2 mb-3">
            <Users size={18} className="text-blue-600" />
            <span className="text-sm font-black text-cafe24-brown">{selectedUser.id} 세그먼트 분류</span>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-xl font-black text-blue-600">{selectedUser.model_predictions.segment.segment_name}</div>
              <div className="text-xs text-cafe24-brown/60">세그먼트</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-black text-cafe24-brown">#{selectedUser.model_predictions.segment.cluster}</div>
              <div className="text-xs text-cafe24-brown/60">클러스터</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-black text-cafe24-brown">{selectedUser.plan_tier}</div>
              <div className="text-xs text-cafe24-brown/60">플랜</div>
            </div>
          </div>
          <div className="mt-2 text-[10px] text-cafe24-brown/40">{selectedUser.model_predictions.segment.model}</div>
        </div>
      )}
      {!segmentsData ? (
        <div className="text-center py-16 rounded-3xl border-2 border-cafe24-orange/20 bg-white/80">
          <Users size={48} className="mx-auto mb-3 text-cafe24-brown/30" />
          <p className="text-sm font-semibold text-cafe24-brown/50">세그먼트 데이터를 불러올 수 없습니다</p>
          <p className="text-xs text-cafe24-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
        </div>
      ) : (
      <>
      {/* 세그먼트 비교 */}
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <Users size={18} className="text-cafe24-orange" />
          <span className="text-sm font-black text-cafe24-brown">세그먼트 비교 분석</span>
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
      <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
        <div className="mb-4 text-sm font-black text-cafe24-brown">세그먼트별 상세 지표</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-cafe24-orange/10">
                <th className="text-left py-3 px-2 font-bold text-cafe24-brown">세그먼트</th>
                <th className="text-right py-3 px-2 font-bold text-cafe24-brown">셀러 수</th>
                <th className="text-right py-3 px-2 font-bold text-cafe24-brown">평균 월매출</th>
                <th className="text-right py-3 px-2 font-bold text-cafe24-brown">평균 상품 수</th>
                <th className="text-right py-3 px-2 font-bold text-cafe24-brown">평균 주문 수</th>
                <th className="text-right py-3 px-2 font-bold text-cafe24-brown">리텐션</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(segmentsData).map(([name, data]) => (
                <tr key={name} className="border-b border-cafe24-orange/5 hover:bg-cafe24-beige/30 transition">
                  <td className="py-3 px-2 font-semibold text-cafe24-brown">{name}</td>
                  <td className="py-3 px-2 text-right text-cafe24-brown/80">{data.count.toLocaleString()}명</td>
                  <td className="py-3 px-2 text-right text-cafe24-brown/80">{data.avg_monthly_revenue?.toLocaleString()}원</td>
                  <td className="py-3 px-2 text-right text-cafe24-brown/80">{data.avg_product_count}개</td>
                  <td className="py-3 px-2 text-right text-cafe24-brown/80">{data.avg_order_count}건</td>
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
  );
}
