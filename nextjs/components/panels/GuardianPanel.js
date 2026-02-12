// components/panels/GuardianPanel.js
// M52: 884줄 → 서브 컴포넌트로 분리

import { useState, useCallback, useRef } from 'react';
import { Shield, RotateCcw, BarChart3 } from 'lucide-react';
import MonitorTab from './guardian/MonitorTab';
import RecoverTab from './guardian/RecoverTab';
import DashboardTab from './guardian/DashboardTab';

const TABS = [
  { key: 'monitor', label: '실시간 감시', icon: Shield },
  { key: 'recover', label: '복구 요청', icon: RotateCcw },
  { key: 'dashboard', label: '대시보드', icon: BarChart3 },
];

export default function GuardianPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('monitor');
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
      <div ref={tabListRef} className="flex gap-2" role="tablist" aria-label="Guardian 탭">
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
                  ? 'bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-lg shadow-indigo-200'
                  : 'bg-white/80 text-gray-600 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === 'monitor' && <MonitorTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'recover' && <RecoverTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'dashboard' && <DashboardTab auth={auth} apiCall={apiCall} />}
    </div>
  );
}
