// guardian/DashboardTab.js — 대시보드 탭

import { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import StatCard from '@/components/common/StatCard';
import {
  RotateCcw, Loader2, ChevronDown, ChevronRight,
  XCircle, AlertCircle, Database, Shield,
} from 'lucide-react';

function BlockedLogItem({ log }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-red-100 bg-red-50/30 overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left hover:bg-red-50/50 transition-colors">
        {open ? <ChevronDown size={14} className="text-red-400" /> : <ChevronRight size={14} className="text-red-400" />}
        <XCircle size={14} className="text-red-500" />
        <span className="text-xs text-gray-500">{(log.timestamp || '').slice(0, 16)}</span>
        <span className="text-xs font-bold text-gray-700">{log.user_id}: {log.action} {log.table_name} ({log.row_count}건)</span>
        {log.affected_amount > 0 && (
          <span className="ml-auto text-xs font-semibold text-red-600">₩{log.affected_amount?.toLocaleString()}</span>
        )}
      </button>
      {open && log.agent_reason && (
        <div className="border-t border-red-100 bg-white/60 px-3 py-2">
          <p className="text-xs font-medium text-gray-500 mb-1">Agent 사유:</p>
          <p className="text-xs text-gray-600 whitespace-pre-wrap">{log.agent_reason}</p>
        </div>
      )}
    </div>
  );
}

export default function DashboardTab({ auth, apiCall }) {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [blockedLogs, setBlockedLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, logsRes, blockedRes] = await Promise.all([
        apiCall({ endpoint: '/api/guardian/stats', auth, timeoutMs: 5000 }),
        apiCall({ endpoint: '/api/guardian/logs?limit=20', auth, timeoutMs: 5000 }),
        apiCall({ endpoint: '/api/guardian/logs?limit=10&status_filter=blocked', auth, timeoutMs: 5000 }),
      ]);
      if (statsRes?.status === 'success') setStats(statsRes);
      if (logsRes?.status === 'success') setLogs(logsRes.logs || []);
      if (blockedRes?.status === 'success') setBlockedLogs(blockedRes.logs || []);
    } catch (e) {
      toast.error('데이터 로드 실패');
    } finally {
      setLoading(false);
    }
  }, [auth, apiCall]);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 size={24} className="animate-spin text-indigo-400" />
        <span className="ml-2 text-sm text-gray-500">로딩 중...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard icon={Database} label="총 감사 로그" value={`${(stats?.total || 0).toLocaleString()}건`} color="indigo" />
        <StatCard icon={XCircle} label="차단된 쿼리" value={`${stats?.blocked || 0}건`} color="red" />
        <StatCard icon={AlertCircle} label="경고" value={`${stats?.warned || 0}건`} color="amber" />
        <StatCard icon={Shield} label="보호된 금액" value={`₩${((stats?.saved_amount || 0) / 10000).toFixed(0)}만`} color="emerald" />
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <p className="mb-3 text-sm font-bold text-gray-700">최근 차단 이력</p>
        {blockedLogs.length === 0 ? (
          <p className="text-center text-xs text-gray-400 py-8">
            아직 차단 이력이 없습니다. 실시간 감시 탭에서 시뮬레이션을 실행해보세요.
          </p>
        ) : (
          <div className="space-y-2">
            {blockedLogs.map(log => (
              <BlockedLogItem key={log.id} log={log} />
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-bold text-gray-700">전체 감사 로그 (최근 20건)</p>
          <button onClick={fetchData} className="rounded-lg border border-gray-200 p-1.5 text-gray-400 hover:bg-gray-50 hover:text-gray-600 transition-colors">
            <RotateCcw size={14} />
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100 text-left text-gray-400">
                <th className="pb-2 pr-3">시간</th>
                <th className="pb-2 pr-3">사용자</th>
                <th className="pb-2 pr-3">작업</th>
                <th className="pb-2 pr-3">테이블</th>
                <th className="pb-2 pr-3">행 수</th>
                <th className="pb-2 pr-3">상태</th>
                <th className="pb-2">위험도</th>
              </tr>
            </thead>
            <tbody>
              {logs.map(log => (
                <tr key={log.id} className="border-b border-gray-50 hover:bg-gray-50/50">
                  <td className="py-1.5 pr-3 text-gray-500">{(log.timestamp || '').slice(5, 16)}</td>
                  <td className="py-1.5 pr-3 font-medium text-gray-700">{log.user_id}</td>
                  <td className="py-1.5 pr-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      log.action === 'DELETE' ? 'bg-red-100 text-red-600'
                      : log.action === 'UPDATE' ? 'bg-amber-100 text-amber-600'
                      : log.action === 'DROP' ? 'bg-red-200 text-red-800'
                      : 'bg-gray-100 text-gray-600'
                    }`}>{log.action}</span>
                  </td>
                  <td className="py-1.5 pr-3 text-gray-600">{log.table_name}</td>
                  <td className="py-1.5 pr-3 text-gray-600">{log.row_count}</td>
                  <td className="py-1.5 pr-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      log.status === 'blocked' ? 'bg-red-100 text-red-600'
                      : log.status === 'warned' ? 'bg-amber-100 text-amber-600'
                      : log.status === 'restored' ? 'bg-blue-100 text-blue-600'
                      : 'bg-emerald-100 text-emerald-600'
                    }`}>{log.status}</span>
                  </td>
                  <td className="py-1.5">
                    <span className={`text-[10px] font-bold ${
                      log.risk_level === 'HIGH' ? 'text-red-500'
                      : log.risk_level === 'MEDIUM' ? 'text-amber-500'
                      : 'text-gray-400'
                    }`}>{log.risk_level}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
