// components/panels/GuardianPanel.js
// Data Guardian Agent — AI 기반 데이터 보호 시스템

import { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import {
  Shield, AlertTriangle, RotateCcw, BarChart3, Play, Loader2,
  CheckCircle2, XCircle, AlertCircle, ChevronDown, ChevronRight,
  Database, Clock, User, Zap, Search, FileText, Lock, Unlock, Bot
} from 'lucide-react';

// 서브탭
const TABS = [
  { key: 'monitor', label: '실시간 감시', icon: Shield },
  { key: 'recover', label: '복구 요청', icon: RotateCcw },
  { key: 'dashboard', label: '대시보드', icon: BarChart3 },
];

// 작업 유형별 메타데이터
const ACTION_META = {
  DELETE:   { danger: 'high',   rowRelevant: true,  label: 'DELETE',   desc: '행 삭제' },
  UPDATE:   { danger: 'high',   rowRelevant: true,  label: 'UPDATE',   desc: '행 수정' },
  DROP:     { danger: 'critical', rowRelevant: false, label: 'DROP',     desc: '테이블 삭제' },
  TRUNCATE: { danger: 'critical', rowRelevant: false, label: 'TRUNCATE', desc: '전체 행 삭제' },
  ALTER:    { danger: 'medium', rowRelevant: false, label: 'ALTER',    desc: '스키마 변경' },
  INSERT:   { danger: 'low',    rowRelevant: true,  label: 'INSERT',   desc: '행 추가' },
  SELECT:   { danger: 'none',   rowRelevant: true,  label: 'SELECT',   desc: '조회' },
};

// 테이블별 예상 행 수 (시뮬레이션용)
const TABLE_META = {
  orders:       { rows: 12847, label: 'orders',       core: true },
  payments:     { rows: 9532,  label: 'payments',     core: true },
  users:        { rows: 3210,  label: 'users',        core: true },
  products:     { rows: 18423, label: 'products',     core: true },
  shipments:    { rows: 7891,  label: 'shipments',    core: true },
  logs:         { rows: 54320, label: 'logs',         core: false },
  temp_reports: { rows: 1205,  label: 'temp_reports', core: false },
};

// 프리셋 시나리오
const PRESETS = [
  { label: '신입 대량 삭제', color: 'red', user_id: 'kim', action: 'DELETE', table: 'orders', row_count: 347, hour: 18 },
  { label: '야간 데이터 수정', color: 'yellow', user_id: 'park', action: 'UPDATE', table: 'payments', row_count: 150, hour: 23 },
  { label: '정상 로그 정리', color: 'green', user_id: 'lee', action: 'DELETE', table: 'logs', row_count: 5, hour: 14 },
  { label: 'DDL 명령어', color: 'red', user_id: 'choi', action: 'DROP', table: 'orders', row_count: 0, hour: 15 },
  { label: '전체 초기화', color: 'red', user_id: 'jung', action: 'TRUNCATE', table: 'payments', row_count: 0, hour: 2 },
  { label: '정상 조회', color: 'green', user_id: 'kim', action: 'SELECT', table: 'orders', row_count: 50, hour: 10 },
  { label: '반복 소량 삭제', color: 'yellow', user_id: 'lee', action: 'DELETE', table: 'orders', row_count: 8, hour: 15 },
  { label: '신규 테이블 접근', color: 'yellow', user_id: 'jung', action: 'UPDATE', table: 'payments', row_count: 50, hour: 10 },
];

export default function GuardianPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('monitor');

  return (
    <div className="space-y-4">
      {/* 서브탭 */}
      <div className="flex gap-2">
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
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

// ═══════════════════════════════════════
// 탭 1: 실시간 감시
// ═══════════════════════════════════════
const GUARD_MODES = [
  { key: 'rule+ml', label: '룰 + ML', desc: '룰엔진 + 이상탐지 병합' },
  { key: 'rule',    label: '룰엔진만', desc: '하드코딩 규칙 기반' },
  { key: 'ml',      label: 'ML만',    desc: 'Isolation Forest 이상탐지' },
];

function MonitorTab({ auth, apiCall }) {
  const [form, setForm] = useState({
    user_id: 'kim', action: 'DELETE', table: 'orders', row_count: 347, hour: 18,
  });
  const [guardMode, setGuardMode] = useState('rule+ml');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const meta = ACTION_META[form.action] || ACTION_META.DELETE;
  const tableMeta = TABLE_META[form.table];
  const isRowDisabled = !meta.rowRelevant;

  const applyPreset = (preset) => {
    setForm({ user_id: preset.user_id, action: preset.action, table: preset.table, row_count: preset.row_count, hour: preset.hour });
    setResult(null);
  };

  const handleActionChange = (action) => {
    const am = ACTION_META[action];
    if (!am.rowRelevant) {
      // DDL 계열: 행 수 자동으로 전체 행 수 or 0
      const tRows = TABLE_META[form.table]?.rows || 0;
      setForm(f => ({ ...f, action, row_count: action === 'TRUNCATE' ? tRows : 0 }));
    } else {
      setForm(f => ({ ...f, action }));
    }
  };

  const handleTableChange = (table) => {
    const am = ACTION_META[form.action];
    if (!am.rowRelevant) {
      const tRows = TABLE_META[table]?.rows || 0;
      setForm(f => ({ ...f, table, row_count: form.action === 'TRUNCATE' ? tRows : 0 }));
    } else {
      setForm(f => ({ ...f, table }));
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/guardian/analyze',
        auth,
        method: 'POST',
        data: { ...form, mode: guardMode },
        timeoutMs: 30000,
      });
      if (res?.status === 'SUCCESS') {
        setResult({ ...res, _mode: guardMode });
      } else {
        toast.error(res?.message || '분석 실패');
      }
    } catch (e) {
      toast.error('분석 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* 감시 모드 선택 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <p className="mb-3 text-sm font-bold text-gray-700">감시 모드</p>
        <div className="flex gap-2">
          {GUARD_MODES.map(m => (
            <button
              key={m.key}
              onClick={() => { setGuardMode(m.key); setResult(null); }}
              className={`flex-1 rounded-xl px-3 py-2 text-center text-xs font-semibold transition-all ${
                guardMode === m.key
                  ? 'bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-lg shadow-indigo-200'
                  : 'bg-gray-50 text-gray-500 border border-gray-200 hover:bg-gray-100'
              }`}
            >
              <div>{m.label}</div>
              <div className={`mt-0.5 text-[10px] font-normal ${guardMode === m.key ? 'text-indigo-100' : 'text-gray-400'}`}>{m.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 시나리오 프리셋 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <p className="mb-3 text-sm font-bold text-gray-700">시나리오 프리셋</p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p, i) => (
            <button
              key={i}
              onClick={() => applyPreset(p)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all hover:scale-105 ${
                p.color === 'red' ? 'bg-red-50 text-red-700 border border-red-200 hover:bg-red-100'
                : p.color === 'yellow' ? 'bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100'
                : 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* 쿼리 입력 폼 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-bold text-gray-700">쿼리 시뮬레이터</p>
          {/* 위험도 인디케이터 */}
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
            meta.danger === 'critical' ? 'bg-red-100 text-red-700'
            : meta.danger === 'high' ? 'bg-orange-100 text-orange-700'
            : meta.danger === 'medium' ? 'bg-amber-100 text-amber-700'
            : meta.danger === 'low' ? 'bg-blue-100 text-blue-700'
            : 'bg-gray-100 text-gray-500'
          }`}>
            {meta.danger === 'critical' ? 'CRITICAL' : meta.danger === 'high' ? 'HIGH RISK' : meta.danger === 'medium' ? 'MEDIUM' : meta.danger === 'low' ? 'LOW' : 'SAFE'}
          </span>
        </div>

        {/* SQL 미리보기 */}
        <div className="mb-3 rounded-lg bg-gray-900 px-4 py-2.5 font-mono text-sm text-gray-100">
          <span className={`font-bold ${
            meta.danger === 'critical' ? 'text-red-400'
            : meta.danger === 'high' ? 'text-orange-400'
            : meta.danger === 'medium' ? 'text-amber-400'
            : meta.danger === 'low' ? 'text-blue-400'
            : 'text-emerald-400'
          }`}>{form.action}</span>
          {form.action === 'DROP' ? (
            <> <span className="text-purple-300">TABLE</span> <span className="text-yellow-300">{form.table}</span></>
          ) : form.action === 'TRUNCATE' ? (
            <> <span className="text-purple-300">TABLE</span> <span className="text-yellow-300">{form.table}</span></>
          ) : form.action === 'ALTER' ? (
            <> <span className="text-purple-300">TABLE</span> <span className="text-yellow-300">{form.table}</span> <span className="text-gray-500">...</span></>
          ) : form.action === 'SELECT' ? (
            <> <span className="text-gray-400">*</span> <span className="text-purple-300">FROM</span> <span className="text-yellow-300">{form.table}</span> <span className="text-purple-300">LIMIT</span> <span className="text-cyan-300">{form.row_count}</span></>
          ) : form.action === 'INSERT' ? (
            <> <span className="text-purple-300">INTO</span> <span className="text-yellow-300">{form.table}</span> <span className="text-gray-400">({form.row_count}건)</span></>
          ) : (
            <> <span className="text-purple-300">FROM</span> <span className="text-yellow-300">{form.table}</span> <span className="text-purple-300">WHERE</span> <span className="text-gray-400">... </span><span className="text-gray-500">({form.row_count}건 영향)</span></>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          {/* 사용자 */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">사용자</label>
            <select value={form.user_id} onChange={e => setForm(f => ({...f, user_id: e.target.value}))}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none">
              {['kim','park','lee','choi','jung'].map(u => <option key={u} value={u}>{u}</option>)}
            </select>
          </div>
          {/* 작업 */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">작업</label>
            <select value={form.action} onChange={e => handleActionChange(e.target.value)}
              className={`w-full rounded-lg border px-3 py-2 text-sm font-semibold focus:outline-none ${
                meta.danger === 'critical' ? 'border-red-300 bg-red-50 text-red-700 focus:border-red-400'
                : meta.danger === 'high' ? 'border-orange-200 bg-orange-50 text-orange-700 focus:border-orange-400'
                : meta.danger === 'medium' ? 'border-amber-200 bg-amber-50 text-amber-700 focus:border-amber-400'
                : 'border-gray-200 bg-white text-gray-700 focus:border-indigo-400'
              }`}>
              {Object.entries(ACTION_META).map(([k, v]) => (
                <option key={k} value={k}>{k} — {v.desc}</option>
              ))}
            </select>
          </div>
          {/* 테이블 */}
          <div>
            <label className="mb-1 flex items-center gap-1 text-xs text-gray-500">
              테이블
              {tableMeta?.core && <span className="rounded bg-red-100 px-1 text-[10px] font-bold text-red-600">핵심</span>}
            </label>
            <select value={form.table} onChange={e => handleTableChange(e.target.value)}
              className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none">
              {Object.entries(TABLE_META).map(([k, v]) => (
                <option key={k} value={k}>{k} ({v.rows.toLocaleString()}행){v.core ? ' *' : ''}</option>
              ))}
            </select>
          </div>
          {/* 행 수 */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">
              {isRowDisabled ? (form.action === 'DROP' ? '대상' : '영향 행 수') : '대상 행 수'}
            </label>
            {isRowDisabled ? (
              <div className="flex w-full items-center rounded-lg border border-gray-200 bg-gray-100 px-3 py-2 text-sm text-gray-400">
                {form.action === 'DROP' ? 'TABLE 전체' : form.action === 'TRUNCATE' ? `전체 (${tableMeta?.rows?.toLocaleString() || 0}행)` : 'N/A (스키마)'}
              </div>
            ) : (
              <div className="relative">
                <input type="number" min={0} value={form.row_count} onChange={e => setForm(f => ({...f, row_count: parseInt(e.target.value) || 0}))}
                  className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none" />
                {tableMeta && form.row_count > 0 && (
                  <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-gray-400">
                    / {tableMeta.rows.toLocaleString()}
                  </span>
                )}
              </div>
            )}
          </div>
          {/* 시간대 */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">시간대</label>
            <div className="relative">
              <input type="number" min={0} max={23} value={form.hour} onChange={e => setForm(f => ({...f, hour: parseInt(e.target.value) || 0}))}
                className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-indigo-400 focus:outline-none" />
              {(form.hour >= 22 || form.hour < 6) && (
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-bold text-amber-500">야간</span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={runAnalysis}
          disabled={loading}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-200 transition-all hover:shadow-xl disabled:opacity-50"
        >
          {loading ? <><Loader2 size={16} className="animate-spin" /> Agent 분석 중...</> : <><Play size={16} /> 쿼리 실행</>}
        </button>
      </div>

      {/* 결과 */}
      {result && <AnalysisResult result={result} form={form} auth={auth} apiCall={apiCall} />}
    </div>
  );
}

// 분석 결과 표시
function AnalysisResult({ result, form, auth, apiCall }) {
  const { rule, ml, agent, _mode } = result;
  const showRule = _mode !== 'ml';
  const showMl = _mode !== 'rule';
  const level = rule?.level;
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [dbaEmail, setDbaEmail] = useState('');
  const [sending, setSending] = useState(false);

  const sendDbaAlert = async () => {
    if (!dbaEmail.trim()) { toast.error('이메일을 입력하세요'); return; }
    setSending(true);
    try {
      const res = await apiCall({
        endpoint: '/api/guardian/notify-dba',
        auth,
        method: 'POST',
        data: {
          email: dbaEmail,
          alert: {
            user_id: form?.user_id,
            action: form?.action,
            table: form?.table,
            row_count: form?.row_count,
            rule_reasons: rule?.reasons || [],
            agent_output: agent?.output || '',
          },
        },
        timeoutMs: 15000,
      });
      if (res?.status === 'SUCCESS') {
        toast.success(res.message || '발송 완료');
        setShowEmailInput(false);
      } else {
        toast.error(res?.message || '발송 실패');
      }
    } catch (e) {
      toast.error('발송 실패: ' + (e.message || ''));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* 룰엔진 결과 */}
      {rule && showRule && (
        <div className={`rounded-2xl border-2 p-4 ${
          level === 'pass' ? 'border-emerald-200 bg-emerald-50/50'
          : level === 'warn' ? 'border-amber-200 bg-amber-50/50'
          : 'border-red-200 bg-red-50/50'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <Zap size={16} className={level === 'pass' ? 'text-emerald-600' : level === 'warn' ? 'text-amber-600' : 'text-red-600'} />
            <span className="text-sm font-bold text-gray-800">룰엔진 판단</span>
            <span className="ml-auto text-xs text-gray-400">{rule?.elapsed_ms}ms</span>
          </div>
          <div className="flex items-center gap-2">
            {level === 'pass' && <CheckCircle2 size={20} className="text-emerald-500" />}
            {level === 'warn' && <AlertCircle size={20} className="text-amber-500" />}
            {level === 'block' && <XCircle size={20} className="text-red-500" />}
            <span className={`text-sm font-semibold ${
              level === 'pass' ? 'text-emerald-700' : level === 'warn' ? 'text-amber-700' : 'text-red-700'
            }`}>
              {level === 'pass' ? '통과' : level === 'warn' ? '경고' : '차단'}
            </span>
            <span className="text-sm text-gray-600">— {rule?.reasons?.join(', ')}</span>
          </div>
          {level === 'block' && (
            <p className="mt-2 text-xs text-indigo-600 font-medium">→ AI Agent를 호출하여 상세 분석합니다...</p>
          )}
        </div>
      )}

      {/* ML 모델 미학습 안내 */}
      {!ml && showMl && result.ml_not_ready && (
        <div className="rounded-2xl border-2 border-blue-200 bg-blue-50/50 p-4">
          <div className="flex items-center gap-2 mb-2">
            <Search size={16} className="text-blue-500" />
            <span className="text-sm font-bold text-gray-800">ML 이상탐지</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-blue-200 bg-white/80 px-3 py-3">
            <AlertCircle size={16} className="text-blue-500 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-gray-700">ML 모델을 사용할 수 없습니다</p>
              <p className="text-xs text-gray-500 mt-0.5">감사 로그가 20건 이상 쌓이면 IsolationForest 모델이 자동으로 학습됩니다.</p>
            </div>
          </div>
        </div>
      )}

      {/* ML 이상탐지 결과 */}
      {ml && showMl && (
        <div className={`rounded-2xl border-2 p-4 ${
          ml.combined_score > 0.7 ? 'border-red-200 bg-red-50/50'
          : ml.combined_score > 0.4 ? 'border-amber-200 bg-amber-50/50'
          : 'border-emerald-200 bg-emerald-50/50'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <Search size={16} className={ml.combined_score > 0.7 ? 'text-red-600' : ml.combined_score > 0.4 ? 'text-amber-600' : 'text-emerald-600'} />
            <span className="text-sm font-bold text-gray-800">ML 이상탐지</span>
            <span className="ml-auto rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-medium text-gray-500">{ml.model} · {ml.features_used} features</span>
          </div>
          {/* 점수 바 */}
          <div className="space-y-2">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-500">이상 점수 (Anomaly)</span>
                <span className={`text-xs font-bold ${ml.anomaly_score > 0.7 ? 'text-red-600' : ml.anomaly_score > 0.4 ? 'text-amber-600' : 'text-emerald-600'}`}>{(ml.anomaly_score * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200">
                <div className={`h-2 rounded-full transition-all ${ml.anomaly_score > 0.7 ? 'bg-red-500' : ml.anomaly_score > 0.4 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                  style={{ width: `${Math.min(ml.anomaly_score * 100, 100)}%` }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-500">사용자 이탈도 (Deviation)</span>
                <span className={`text-xs font-bold ${ml.user_deviation > 0.7 ? 'text-red-600' : ml.user_deviation > 0.4 ? 'text-amber-600' : 'text-emerald-600'}`}>{(ml.user_deviation * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200">
                <div className={`h-2 rounded-full transition-all ${ml.user_deviation > 0.7 ? 'bg-red-500' : ml.user_deviation > 0.4 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                  style={{ width: `${Math.min(ml.user_deviation * 100, 100)}%` }} />
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between rounded-lg bg-white/60 px-3 py-2">
              <span className="text-xs font-semibold text-gray-600">종합 점수</span>
              <span className={`text-lg font-black ${ml.combined_score > 0.7 ? 'text-red-600' : ml.combined_score > 0.4 ? 'text-amber-600' : 'text-emerald-600'}`}>
                {(ml.combined_score * 100).toFixed(1)}
                <span className="text-xs font-normal text-gray-400"> / 100</span>
              </span>
            </div>
          </div>
          {/* SHAP 위험 요인 분석 */}
          {ml.risk_factors?.length > 0 && (
            <div className="mt-3 rounded-xl border border-gray-200 bg-white/70 p-3">
              <p className="text-xs font-bold text-gray-600 mb-2">위험 요인 분석 (SHAP)</p>
              <div className="space-y-2">
                {ml.risk_factors.map((rf, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${rf.severity === 'high' ? 'bg-red-500' : 'bg-amber-500'}`} />
                    <span className="text-xs font-semibold text-gray-700 min-w-[72px]">{rf.factor}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-gray-100">
                      <div className={`h-1.5 rounded-full ${rf.severity === 'high' ? 'bg-red-400' : 'bg-amber-400'}`}
                        style={{ width: `${Math.min((rf.contribution || 0) * 500, 100)}%` }} />
                    </div>
                    <span className="text-[10px] text-gray-500 min-w-[100px] text-right">{rf.detail}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {ml.risk_factors?.length === 0 && ml.combined_score <= 0.3 && (
            <div className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50/50 px-3 py-2">
              <p className="text-xs text-emerald-600 font-medium">특이 위험 요인 없음 — 정상 범위 내 작업입니다.</p>
            </div>
          )}
          {/* LLM 해석 */}
          {ml.interpretation && (
            <div className="mt-3 rounded-xl border border-violet-200 bg-violet-50/50 px-3 py-2.5">
              <div className="flex items-start gap-2">
                <Bot size={14} className="text-violet-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-[10px] font-bold text-violet-500 mb-0.5">AI 해석</p>
                  <p className="text-xs text-gray-700 leading-relaxed">{ml.interpretation}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Agent 분석 */}
      {agent && (
        <div className="rounded-2xl border-2 border-indigo-200 bg-indigo-50/30 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Shield size={16} className="text-indigo-600" />
            <span className="text-sm font-bold text-gray-800">2단계: AI Agent 분석</span>
          </div>

          {/* Tool 호출 내역 */}
          {agent.steps?.length > 0 && (
            <div className="mb-4 space-y-2">
              {agent.steps.map((step, i) => (
                <ToolStep key={i} index={i + 1} step={step} />
              ))}
            </div>
          )}

          {/* Agent 최종 판단 */}
          <div className="rounded-xl border border-indigo-200 bg-white/80 p-4">
            <p className="mb-2 text-xs font-bold text-indigo-600">Agent 최종 판단</p>
            <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
              {agent.output}
            </div>
          </div>

          {/* 액션 버튼 */}
          <div className="mt-4 flex flex-wrap gap-3">
            <button className="flex items-center gap-2 rounded-xl bg-red-500 px-4 py-2.5 text-sm font-bold text-white shadow-lg transition-all hover:bg-red-600"
              onClick={() => toast.success('차단이 유지됩니다.')}>
              <Lock size={14} /> 차단 유지
            </button>
            <button className="flex items-center gap-2 rounded-xl border border-amber-300 bg-amber-50 px-4 py-2.5 text-sm font-bold text-amber-700 transition-all hover:bg-amber-100"
              onClick={() => setShowEmailInput(!showEmailInput)}>
              <Unlock size={14} /> 그래도 실행 (DBA 승인 필요)
            </button>
          </div>

          {/* DBA 이메일 알림 입력 */}
          {showEmailInput && (
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50/50 p-4">
              <p className="mb-2 text-sm font-bold text-amber-800">DBA 승인 요청 이메일 발송</p>
              <p className="mb-3 text-xs text-amber-600">입력한 이메일로 차단된 쿼리 상세 내역 + Agent 분석 결과가 전송됩니다.</p>
              <div className="flex gap-2">
                <input
                  type="email"
                  value={dbaEmail}
                  onChange={e => setDbaEmail(e.target.value)}
                  placeholder="DBA 이메일 주소 (예: dba@company.com)"
                  className="flex-1 rounded-lg border border-amber-200 bg-white px-3 py-2.5 text-sm focus:border-amber-400 focus:outline-none"
                  onKeyDown={e => e.key === 'Enter' && sendDbaAlert()}
                />
                <button
                  onClick={sendDbaAlert}
                  disabled={sending || !dbaEmail.trim()}
                  className="flex items-center gap-2 rounded-lg bg-amber-500 px-5 py-2.5 text-sm font-bold text-white transition-all hover:bg-amber-600 disabled:opacity-50"
                >
                  {sending ? <Loader2 size={14} className="animate-spin" /> : <AlertTriangle size={14} />}
                  {sending ? '발송 중...' : '알림 발송'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 결과 없음 폴백 */}
      {!(rule && showRule) && !(ml && showMl) && !agent && !result.ml_not_ready && (
        <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-gray-50/50 p-6 text-center">
          <AlertCircle size={24} className="mx-auto mb-2 text-gray-300" />
          <p className="text-sm font-semibold text-gray-500">분석 결과가 없습니다</p>
          <p className="mt-1 text-xs text-gray-400">분석 데이터가 부족합니다.</p>
        </div>
      )}
    </div>
  );
}

// Tool 호출 단계 표시
function ToolStep({ index, step }) {
  const [open, setOpen] = useState(true);
  const toolIcons = {
    analyze_impact: Database,
    get_user_pattern: User,
    search_similar: Search,
    execute_decision: Shield,
  };
  const Icon = toolIcons[step.tool] || FileText;
  const toolNames = {
    analyze_impact: '영향도 분석',
    get_user_pattern: '사용자 패턴',
    search_similar: '유사 사례 검색',
    execute_decision: '판단 실행',
  };

  return (
    <div className="rounded-xl border border-gray-200 bg-white/60 overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors">
        {open ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
        <Icon size={14} className="text-indigo-500" />
        <span className="text-xs font-semibold text-gray-700">Tool {index}: {toolNames[step.tool] || step.tool}</span>
      </button>
      {open && (
        <div className="border-t border-gray-100 bg-gray-50/50 px-3 py-2">
          <pre className="whitespace-pre-wrap text-xs text-gray-600 leading-relaxed">{step.output}</pre>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════
// 탭 2: 복구 요청
// ═══════════════════════════════════════
function RecoverTab({ auth, apiCall }) {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const EXAMPLES = [
    '아까 실수로 삭제한 12월 주문 데이터 347건 복구해주세요',
    'kim 사용자가 삭제한 payments 데이터 복구해줘',
    '오늘 차단된 orders 삭제 건 복구 SQL 만들어줘',
  ];

  const runRecovery = async () => {
    if (!message.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await apiCall({
        endpoint: '/api/guardian/recover',
        auth,
        method: 'POST',
        data: { message },
        timeoutMs: 30000,
      });
      if (res?.status === 'SUCCESS') {
        setResult(res);
      } else {
        toast.error(res?.message || '복구 요청 실패');
      }
    } catch (e) {
      toast.error('복구 요청 실패: ' + (e.message || ''));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <p className="mb-1 text-sm font-bold text-gray-700">자연어 복구 요청</p>
        <p className="mb-3 text-xs text-gray-400">삭제된 데이터를 자연어로 설명하면 Agent가 복구 SQL을 생성합니다. (DBA 승인 필요)</p>

        {/* 예시 */}
        <div className="mb-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex, i) => (
            <button key={i} onClick={() => setMessage(ex)}
              className="rounded-lg border border-gray-200 bg-gray-50 px-2.5 py-1 text-xs text-gray-600 hover:bg-gray-100 transition-colors">
              {ex.length > 35 ? ex.slice(0, 35) + '...' : ex}
            </button>
          ))}
        </div>

        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="예: 아까 실수로 삭제한 12월 주문 데이터 복구해주세요"
          className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm focus:border-indigo-400 focus:outline-none resize-none"
          rows={3}
        />

        <button
          onClick={runRecovery}
          disabled={loading || !message.trim()}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-emerald-200 transition-all hover:shadow-xl disabled:opacity-50"
        >
          {loading ? <><Loader2 size={16} className="animate-spin" /> 복구 계획 수립 중...</> : <><RotateCcw size={16} /> 복구 요청</>}
        </button>
      </div>

      {/* 결과 */}
      {result && (
        <div className="rounded-2xl border-2 border-emerald-200 bg-emerald-50/30 p-4">
          <div className="flex items-center gap-2 mb-3">
            <RotateCcw size={16} className="text-emerald-600" />
            <span className="text-sm font-bold text-gray-800">복구 계획</span>
          </div>

          {result.steps?.length > 0 && (
            <div className="mb-4 space-y-2">
              {result.steps.map((step, i) => (
                <ToolStep key={i} index={i + 1} step={step} />
              ))}
            </div>
          )}

          <div className="rounded-xl border border-emerald-200 bg-white/80 p-4">
            <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
              {result.output}
            </div>
          </div>

          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
            <p className="text-xs font-medium text-amber-700">
              ⚠️ 복구 SQL은 DBA 승인 후 실행됩니다. 승인 요청이 자동 발송됩니다.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════
// 탭 3: 대시보드
// ═══════════════════════════════════════
function DashboardTab({ auth, apiCall }) {
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
      if (statsRes?.status === 'SUCCESS') setStats(statsRes);
      if (logsRes?.status === 'SUCCESS') setLogs(logsRes.logs || []);
      if (blockedRes?.status === 'SUCCESS') setBlockedLogs(blockedRes.logs || []);
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
      {/* 통계 카드 */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard icon={Database} label="총 감사 로그" value={`${(stats?.total || 0).toLocaleString()}건`} color="indigo" />
        <StatCard icon={XCircle} label="차단된 쿼리" value={`${stats?.blocked || 0}건`} color="red" />
        <StatCard icon={AlertCircle} label="경고" value={`${stats?.warned || 0}건`} color="amber" />
        <StatCard icon={Shield} label="보호된 금액" value={`₩${((stats?.saved_amount || 0) / 10000).toFixed(0)}만`} color="emerald" />
      </div>

      {/* 차단 이력 */}
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

      {/* 전체 감사 로그 */}
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

// 통계 카드
function StatCard({ icon: Icon, label, value, color }) {
  const colors = {
    indigo: 'border-indigo-200 bg-indigo-50/50 text-indigo-600',
    red: 'border-red-200 bg-red-50/50 text-red-600',
    amber: 'border-amber-200 bg-amber-50/50 text-amber-600',
    emerald: 'border-emerald-200 bg-emerald-50/50 text-emerald-600',
  };
  return (
    <div className={`rounded-2xl border p-4 ${colors[color]}`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon size={16} />
        <span className="text-xs font-medium text-gray-500">{label}</span>
      </div>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}

// 차단 로그 아이템
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
