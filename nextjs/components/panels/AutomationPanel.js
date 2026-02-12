// components/panels/AutomationPanel.js
// M68: 탭 컴포넌트 분리 완료 — RetentionTab, FaqTab, ReportTab
// 자동화 엔진 — 탐지 → 자동 실행 (이탈방지 · FAQ 생성 · 운영 리포트)

import { useState, useEffect } from 'react';
import {
  Zap, Shield, FileText, HelpCircle,
} from 'lucide-react';
import RetentionTab from '@/components/panels/automation/RetentionTab';
import FaqTab from '@/components/panels/automation/FaqTab';
import ReportTab from '@/components/panels/automation/ReportTab';

const TABS = [
  { key: 'retention', label: '이탈 방지', icon: Shield },
  { key: 'faq', label: 'FAQ 자동 생성', icon: HelpCircle },
  { key: 'report', label: '운영 리포트', icon: FileText },
];

export default function AutomationPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('retention');
  const [overviewStats, setOverviewStats] = useState(null);

  // M67: useEffect 의존성 배열 규칙 준수
  useEffect(() => {
    apiCall({ endpoint: '/api/automation/actions/stats', auth, timeoutMs: 10000 })
      .then(res => { if (res?.status === 'success') setOverviewStats(res); })
      .catch(() => {});
  }, [apiCall, auth]);

  return (
    <div className="space-y-4">
      {/* 개요 통계 */}
      {overviewStats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { icon: Zap, label: '총 자동화', value: overviewStats.total_actions || 0, gradient: 'from-amber-500 to-orange-500' },
            { icon: Shield, label: '이탈 방지', value: (overviewStats.by_type?.retention_coupon || 0) + (overviewStats.by_type?.retention_upgrade_offer || 0) + (overviewStats.by_type?.retention_manager_assign || 0) + (overviewStats.by_type?.retention_custom_message || 0), gradient: 'from-red-500 to-rose-500' },
            { icon: HelpCircle, label: 'FAQ 생성', value: overviewStats.by_type?.faq_generate || 0, gradient: 'from-blue-500 to-indigo-500' },
            { icon: FileText, label: '리포트', value: overviewStats.by_type?.report_generate || 0, gradient: 'from-emerald-500 to-green-500' },
          ].map((card, i) => {
            const CardIcon = card.icon;
            return (
              <div key={i} className="rounded-xl bg-white/80 border border-gray-200 p-3 backdrop-blur">
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-7 h-7 rounded-lg bg-gradient-to-br ${card.gradient} flex items-center justify-center`}>
                    <CardIcon size={14} className="text-white" />
                  </div>
                  <span className="text-[10px] text-gray-500">{card.label}</span>
                </div>
                <div className="text-lg font-bold text-gray-800">{(card.value || 0).toLocaleString()}</div>
              </div>
            );
          })}
        </div>
      )}

      {/* 서브탭 */}
      <div className="flex gap-2" role="tablist" aria-label="자동화 엔진 탭">
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={active}
              tabIndex={active ? 0 : -1}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all ${
                active
                  ? 'bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white shadow-lg shadow-orange-200'
                  : 'bg-white/80 text-gray-600 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === 'retention' && <RetentionTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'faq' && <FaqTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'report' && <ReportTab auth={auth} apiCall={apiCall} />}
    </div>
  );
}
