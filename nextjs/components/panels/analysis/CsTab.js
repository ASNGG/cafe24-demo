// components/panels/analysis/CsTab.js
// CS 분석 탭

import { Globe, MessageSquare } from 'lucide-react';

export default function CsTab({ csData }) {
  return (
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
  );
}
