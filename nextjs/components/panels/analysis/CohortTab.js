// components/panels/analysis/CohortTab.js
// 코호트 분석 탭

import { useState, useMemo } from 'react';
import { Target, Repeat, DollarSign } from 'lucide-react';
import CustomTooltip from '@/components/common/CustomTooltip';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

export default function CohortTab({ cohortData }) {
  const [cohortTab, setCohortTab] = useState('retention');

  const weekKeys = useMemo(() => {
    if (!cohortData?.retention?.length) return ['week0'];
    const allKeys = new Set();
    cohortData.retention.forEach(row => {
      Object.keys(row).forEach(k => {
        if (k.startsWith('week')) allKeys.add(k);
      });
    });
    return ['week0', ...Array.from(allKeys).filter(k => k !== 'week0').sort((a, b) => parseInt(a.replace('week', '')) - parseInt(b.replace('week', '')))];
  }, [cohortData]);

  return (
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
                  {weekKeys.map(week => (
                    <th key={week} className="text-center py-3 px-3 font-bold text-cookie-brown">
                      Week {week.replace('week', '')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(cohortData.retention || []).map((row, idx) => (
                  <tr key={idx} className="border-b border-cookie-orange/5">
                    <td className="py-3 px-3 font-semibold text-cookie-brown">{row.cohort}</td>
                    {weekKeys.map((week) => (
                      <td key={week} className="py-3 px-3 text-center">
                        {row[week] != null ? (
                          <span
                            className="inline-block px-3 py-1 rounded-lg text-xs font-bold"
                            style={{
                              backgroundColor: `rgba(255, 140, 66, ${Number(row[week]) / 100})`,
                              color: Number(row[week]) > 50 ? 'white' : '#5C4A3D'
                            }}
                          >
                            {typeof row[week] === 'number' ? row[week].toFixed(1) : row[week]}%
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
  );
}
