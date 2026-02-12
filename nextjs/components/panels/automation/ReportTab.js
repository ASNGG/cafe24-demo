// components/panels/automation/ReportTab.js
// M68: AutomationPanel 분리 - 탭 3: 운영 리포트 자동 생성
import { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import {
  FileText, Play, Loader2, Clock, Eye,
} from 'lucide-react';
import PipelineFlow from '@/components/automation/PipelineFlow';
import { REPORT_STEPS } from '@/components/automation/constants';

const REPORT_TYPES = [
  { value: 'daily', label: '일간 리포트' },
  { value: 'weekly', label: '주간 리포트' },
  { value: 'monthly', label: '월간 리포트' },
];

const KPI_LABELS = {
  gmv_latest: '최근 GMV',
  gmv_7d_avg: 'GMV 7일 평균',
  gmv_30d_avg: 'GMV 30일 평균',
  active_sellers_latest: '활성 셀러',
  active_sellers_7d_avg: '셀러 7일 평균',
  orders_latest: '최근 주문수',
  orders_7d_avg: '주문 7일 평균',
  new_signups_latest: '신규 가입',
  total_shops: '총 쇼핑몰',
  total_sellers: '총 셀러',
  anomaly_sellers: '이상 셀러',
  fraud_total: '이상거래',
  total_tickets: 'CS 문의건',
  avg_satisfaction: '만족도',
  avg_resolution_hours: '평균 해결시간',
  gmv_wow_change_pct: 'GMV WoW(%)',
  orders_wow_change_pct: '주문 WoW(%)',
  active_sellers_wow_change_pct: '셀러 WoW(%)',
  latest_cohort: '최근 코호트',
};

const TREND_MAP = {
  gmv_latest: 'gmv_wow_change_pct',
  orders_latest: 'orders_wow_change_pct',
  active_sellers_latest: 'active_sellers_wow_change_pct',
  new_signups_latest: 'new_signups_wow_change_pct',
};

export default function ReportTab({ auth, apiCall }) {
  const [reportType, setReportType] = useState('daily');
  const [generating, setGenerating] = useState(false);
  const [report, setReport] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingReport, setViewingReport] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [currentStep, setCurrentStep] = useState(null);

  const generateReport = useCallback(async () => {
    setGenerating(true);
    setReport(null);
    setPipelineStatus({ collect: { status: 'processing' } });
    setCurrentStep('collect');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/report/generate',
        auth,
        method: 'POST',
        data: { report_type: reportType },
        timeoutMs: 120000,
      });
      if (res?.status === 'success') {
        setReport(res);
        toast.success(`${reportType} 리포트 생성 완료`);
        setPipelineStatus({
          collect: { status: 'complete', detail: '수집 완료' },
          aggregate: { status: 'complete', detail: 'KPI 집계' },
          write: { status: 'complete', detail: '작성 완료' },
          save: { status: 'complete', detail: '저장됨' },
        });
        setCurrentStep(null);
      } else {
        toast.error(res?.detail || '리포트 생성 실패');
        setPipelineStatus({ collect: { status: 'error' } });
      }
    } catch (e) {
      toast.error('리포트 생성 실패');
      setPipelineStatus({ collect: { status: 'error' } });
    } finally {
      setGenerating(false);
    }
  }, [apiCall, auth, reportType]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await apiCall({
        endpoint: '/api/automation/report/history?limit=20',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setHistory(res.reports || []);
        setShowHistory(true);
        setPipelineStatus(prev => ({
          ...prev,
          history: { status: 'complete', detail: `${(res.reports || []).length}건` },
        }));
      }
    } catch (e) {
      toast.error('리포트 이력 조회 실패');
    }
  }, [apiCall, auth]);

  const activeReport = report || viewingReport;

  const flattenSummary = (raw) => {
    const flat = {};
    Object.entries(raw).forEach(([k, v]) => {
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        Object.entries(v).forEach(([sk, sv]) => {
          if (typeof sv === 'number' || typeof sv === 'string') {
            flat[sk] = sv;
          }
        });
      } else if (typeof v === 'number' || typeof v === 'string') {
        flat[k] = v;
      }
    });
    return flat;
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="text-emerald-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">운영 리포트 자동 생성</h3>
            <span className="text-xs text-gray-500">플랫폼 KPI 집계 → LLM 리포트 작성</span>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={reportType}
              onChange={(e) => setReportType(e.target.value)}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              {REPORT_TYPES.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-600 disabled:opacity-50"
            >
              {generating ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              리포트 생성
            </button>
            <button
              onClick={fetchHistory}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50"
            >
              <Clock size={14} />
              생성 이력
            </button>
          </div>
        </div>
      </div>

      <PipelineFlow steps={REPORT_STEPS} stepStatuses={pipelineStatus} currentStep={currentStep} />

      {activeReport && (
        <div className="rounded-2xl border border-emerald-200 bg-gradient-to-r from-emerald-50 to-green-50 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <FileText className="text-emerald-600" size={18} />
              <h4 className="text-sm font-bold text-gray-800">
                {activeReport.report_type} 운영 리포트
              </h4>
              <span className="text-[10px] text-gray-500">
                {new Date((activeReport.timestamp || 0) * 1000).toLocaleString('ko-KR')}
              </span>
            </div>
            {viewingReport && (
              <button
                onClick={() => setViewingReport(null)}
                className="text-xs text-gray-400 hover:text-gray-600"
              >
                닫기
              </button>
            )}
          </div>
          {(() => {
            const reportContent = activeReport.content || '';
            const reportSections = reportContent.match(/^#{1,3}\s+.+$/gm)?.map((h, i) => ({
              text: h.replace(/^#+\s+/, ''),
              id: `rpt-${i}`,
            })) || [];
            return reportSections.length > 0 ? (
              <div className="flex gap-1.5 flex-wrap mb-3">
                {reportSections.map(s => (
                  <span key={s.id} className="text-[10px] px-2 py-1 rounded-full bg-emerald-100 text-emerald-700">
                    {s.text}
                  </span>
                ))}
              </div>
            ) : null;
          })()}
          <div className="rounded-xl bg-white/90 p-4 prose prose-sm max-w-none">
            <ReactMarkdown>{activeReport.content || ''}</ReactMarkdown>
          </div>
          {activeReport.data_summary && (() => {
            const flat = flattenSummary(activeReport.data_summary);
            return (
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(flat).slice(0, 12).map(([key, val]) => {
                  const trendKey = TREND_MAP[key];
                  return (
                    <div key={key} className="rounded-lg bg-white/80 p-2 text-center">
                      <div className="text-[10px] text-gray-500">{KPI_LABELS[key] || key}</div>
                      <div className="text-xs font-bold text-gray-800">
                        {typeof val === 'number'
                          ? (key.includes('pct') ? `${val > 0 ? '+' : ''}${val}%` : val.toLocaleString())
                          : String(val)}
                      </div>
                      {trendKey && flat[trendKey] !== undefined && (
                        <span className={`text-[10px] font-semibold flex items-center justify-center gap-0.5 ${
                          flat[trendKey] > 0 ? 'text-green-600' : flat[trendKey] < 0 ? 'text-red-600' : 'text-gray-400'
                        }`}>
                          {flat[trendKey] > 0 ? '↑' : flat[trendKey] < 0 ? '↓' : '→'}
                          {flat[trendKey] > 0 ? '+' : ''}{flat[trendKey]}%
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>
      )}

      {showHistory && history.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-gray-700">리포트 생성 이력</h4>
            <button onClick={() => setShowHistory(false)} className="text-xs text-gray-400 hover:text-gray-600">닫기</button>
          </div>
          <div className="space-y-2">
            {history.map((r, i) => (
              <div
                key={r.report_id || i}
                className="flex items-center justify-between rounded-xl border border-gray-200 p-2.5 hover:bg-gray-50 cursor-pointer"
                onClick={() => { setViewingReport(r); setShowHistory(false); }}
              >
                <div className="flex items-center gap-3">
                  <FileText size={14} className="text-emerald-500" />
                  <span className="text-xs font-semibold text-gray-700">{r.report_type} 리포트</span>
                  <span className="text-[10px] text-gray-500">
                    {new Date((r.timestamp || 0) * 1000).toLocaleString('ko-KR')}
                  </span>
                </div>
                <Eye size={14} className="text-gray-400" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
