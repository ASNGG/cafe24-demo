// components/panels/ProcessMinerPanel.js
// M51: 1293줄 → 서브 컴포넌트로 분리

import { useState, useCallback, useRef } from 'react';
import { GitBranch, Timer, Sparkles } from 'lucide-react';
import { PROCESS_TYPES, CASE_OPTIONS } from './process-miner/utils';
import DiscoverTab from './process-miner/DiscoverTab';
import BottleneckTab from './process-miner/BottleneckTab';
import RecommendTab from './process-miner/RecommendTab';

const TABS = [
  { key: 'discover', label: '프로세스 발견', icon: GitBranch },
  { key: 'bottleneck', label: '병목 분석', icon: Timer },
  { key: 'recommend', label: 'AI 자동화 추천', icon: Sparkles },
];

export default function ProcessMinerPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('discover');
  const [processType, setProcessType] = useState('order');
  const [nCases, setNCases] = useState(100);
  const tabListRef = useRef(null);

  // L26: WAI-ARIA 탭 키보드 네비게이션
  const handleTabKeyDown = useCallback((e) => {
    const idx = TABS.findIndex(t => t.key === activeTab);
    let next = -1;
    if (e.key === 'ArrowRight') { e.preventDefault(); next = (idx + 1) % TABS.length; }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); next = (idx - 1 + TABS.length) % TABS.length; }
    else if (e.key === 'Home') { e.preventDefault(); next = 0; }
    else if (e.key === 'End') { e.preventDefault(); next = TABS.length - 1; }
    if (next >= 0) {
      setActiveTab(TABS[next].key);
      tabListRef.current?.querySelectorAll('[role="tab"]')?.[next]?.focus();
    }
  }, [activeTab]);

  return (
    <div className="space-y-4">
      {/* 서브탭 */}
      <div ref={tabListRef} className="flex gap-2" role="tablist" aria-label="Process Miner 탭">
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
              onKeyDown={handleTabKeyDown}
              className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all ${
                active
                  ? 'bg-gradient-to-r from-teal-500 to-teal-600 text-white shadow-lg shadow-teal-200'
                  : 'bg-white/80 text-gray-600 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* 공통 컨트롤 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[160px]">
            <label className="mb-1 block text-xs text-gray-500">프로세스 유형</label>
            <select
              value={processType}
              onChange={e => setProcessType(e.target.value)}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-teal-400 focus:outline-none"
            >
              {PROCESS_TYPES.map(pt => (
                <option key={pt.value} value={pt.value}>{pt.label}</option>
              ))}
            </select>
          </div>
          <div className="min-w-[120px]">
            <label className="mb-1 block text-xs text-gray-500">케이스 수</label>
            <select
              value={nCases}
              onChange={e => setNCases(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-teal-400 focus:outline-none"
            >
              {CASE_OPTIONS.map(n => (
                <option key={n} value={n}>{n}건</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {activeTab === 'discover' && (
        <DiscoverTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
      {activeTab === 'bottleneck' && (
        <BottleneckTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
      {activeTab === 'recommend' && (
        <RecommendTab auth={auth} apiCall={apiCall} processType={processType} nCases={nCases} />
      )}
    </div>
  );
}
