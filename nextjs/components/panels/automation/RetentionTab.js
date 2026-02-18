// components/panels/automation/RetentionTab.js
// M68: AutomationPanel 분리 - 탭 1: 셀러 이탈 방지 자동 조치
import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import {
  Shield, MessageSquare, Loader2, CheckCircle2,
  Users, TrendingDown, Send, Search, Clock, Tag,
} from 'lucide-react';
import PipelineFlow from '@/components/automation/PipelineFlow';
import { RETENTION_STEPS } from '@/components/automation/constants';

const ACTION_TYPES = [
  { key: 'coupon', label: '할인 쿠폰 발급', icon: Tag },
  { key: 'upgrade_offer', label: '플랜 업그레이드 제안', icon: TrendingDown },
  { key: 'manager_assign', label: '전담 매니저 배정', icon: Users },
  { key: 'custom_message', label: '맞춤 메시지 발송', icon: Send },
];

// React.memo로 셀러 카드 리렌더링 방지
const SellerCard = React.memo(function SellerCard({ s, i, isSelected, isChecked, msgLoading, onSelect, onToggleCheck, onGenerateMessage, riskColor }) {
  return (
    <div
      className={`rounded-xl border p-3 cursor-pointer transition-all hover:shadow-md ${
        isSelected ? 'ring-2 ring-cafe24-yellow border-cafe24-yellow' : 'border-gray-200'
      }`}
      onClick={() => onSelect(s.seller_id)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            checked={isChecked}
            onChange={(e) => {
              e.stopPropagation();
              onToggleCheck(s.seller_id);
            }}
            className="rounded border-gray-300 text-cafe24-yellow focus:ring-cafe24-yellow"
          />
          <span className="text-sm font-bold text-gray-800">{s.seller_id}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${riskColor(s.risk_level)}`}>
            {s.risk_level === 'high' ? '고위험' : '중위험'}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              이탈확률: <strong className="text-red-600">{Number(s.churn_probability).toFixed(1)}%</strong>
            </span>
            <div className="w-16 h-1.5 rounded-full bg-gray-200 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${
                  s.churn_probability > 80 ? 'bg-red-500' :
                  s.churn_probability > 60 ? 'bg-orange-500' : 'bg-yellow-500'
                }`}
                style={{ width: `${Math.min(Number(s.churn_probability), 100)}%` }}
              />
            </div>
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onGenerateMessage(s.seller_id); }}
          disabled={msgLoading}
          className="flex items-center gap-1 rounded-lg bg-cafe24-yellow px-2.5 py-1 text-xs font-semibold text-cafe24-brown hover:bg-cafe24-orange hover:text-white disabled:opacity-50"
        >
          {msgLoading && isSelected
            ? <Loader2 size={12} className="animate-spin" />
            : <MessageSquare size={12} />}
          메시지 생성
        </button>
      </div>
      {s.top_factors && s.top_factors.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {s.top_factors.slice(0, 5).map((f, fi) => (
            <span key={fi} className="text-[10px] bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
              {typeof f === 'string' ? f : f.factor || f.feature || f.name || JSON.stringify(f)}
            </span>
          ))}
        </div>
      )}
      {s.seller_info && (
        <div className="mt-2 grid grid-cols-5 gap-1.5">
          {[
            { label: '주문', value: s.seller_info.total_orders?.toLocaleString() },
            { label: '매출', value: `${Math.round((s.seller_info.total_revenue || 0) / 10000)}만` },
            { label: '접속', value: `${s.seller_info.days_since_last_login}일전` },
            { label: '환불률', value: `${s.seller_info.refund_rate}%` },
            { label: '상품', value: s.seller_info.product_count },
          ].map((stat, si) => (
            <div key={si} className="text-center p-1.5 rounded-lg bg-gray-50">
              <div className="text-[9px] text-gray-400">{stat.label}</div>
              <div className="text-[11px] font-bold text-gray-700">{stat.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default function RetentionTab({ auth, apiCall }) {
  const [sellers, setSellers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [message, setMessage] = useState(null);
  const [msgLoading, setMsgLoading] = useState(false);
  const [execLoading, setExecLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const [selectedSellers, setSelectedSellers] = useState(new Set());
  const [bulkLoading, setBulkLoading] = useState(false);

  const fetchAtRisk = useCallback(async () => {
    setLoading(true);
    setSellers([]);
    setSelectedSeller(null);
    setMessage(null);
    setPipelineStatus({ detect: { status: 'processing' } });
    setCurrentStep('detect');
    try {
      const res = await apiCall({
        endpoint: `/api/automation/retention/at-risk?threshold=${threshold}&limit=20`,
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setSellers(res.sellers || []);
        setPipelineStatus({
          detect: { status: 'complete', detail: `${(res.sellers || []).length}명 탐지` },
          analyze: { status: 'complete', detail: 'SHAP 분석' },
        });
        setCurrentStep(null);
        if ((res.sellers || []).length === 0) {
          toast('현재 이탈 위험 셀러가 없습니다', { icon: '✅' });
        }
      } else {
        toast.error(res?.detail || '조회 실패');
        setPipelineStatus({ detect: { status: 'error', detail: '탐지 실패' } });
      }
    } catch (e) {
      toast.error('이탈 위험 셀러 조회 실패');
      setPipelineStatus({ detect: { status: 'error', detail: '탐지 실패' } });
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth, threshold]);

  const generateMessage = useCallback(async (sellerId) => {
    setMsgLoading(true);
    setMessage(null);
    setSelectedSeller(sellerId);
    setPipelineStatus(prev => ({ ...prev, generate: { status: 'processing' } }));
    setCurrentStep('generate');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/message',
        auth,
        method: 'POST',
        data: { seller_id: sellerId },
        timeoutMs: 60000,
      });
      if (res?.status === 'success') {
        setMessage(res);
        setPipelineStatus(prev => ({ ...prev, generate: { status: 'complete', detail: '메시지 생성됨' } }));
        setCurrentStep(null);
      } else {
        toast.error(res?.detail || '메시지 생성 실패');
      }
    } catch (e) {
      toast.error('리텐션 메시지 생성 실패');
    } finally {
      setMsgLoading(false);
    }
  }, [apiCall, auth]);

  const executeAction = useCallback(async (sellerId, actionType) => {
    setExecLoading(true);
    setPipelineStatus(prev => ({ ...prev, execute: { status: 'processing' } }));
    setCurrentStep('execute');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/execute',
        auth,
        method: 'POST',
        data: { seller_id: sellerId, action_type: actionType },
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        toast.success(`조치 실행 완료: ${actionType}`);
        setPipelineStatus(prev => ({
          ...prev,
          execute: { status: 'complete', detail: actionType },
          log: { status: 'complete', detail: '기록 완료' },
        }));
        setCurrentStep(null);
      } else {
        toast.error(res?.detail || '조치 실행 실패');
      }
    } catch (e) {
      toast.error('조치 실행 실패');
    } finally {
      setExecLoading(false);
    }
  }, [apiCall, auth]);

  const executeBulk = useCallback(async (actionType) => {
    setBulkLoading(true);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/execute-bulk',
        auth,
        method: 'POST',
        data: { seller_ids: Array.from(selectedSellers), action_type: actionType },
        timeoutMs: 60000,
      });
      if (res?.status === 'success') {
        toast.success(`${res.success_count}명에게 ${actionType} 조치 완료`);
        setSelectedSellers(new Set());
      } else {
        toast.error(res?.detail || '벌크 조치 실패');
      }
    } catch (e) {
      toast.error('벌크 조치 실행 실패');
    } finally {
      setBulkLoading(false);
    }
  }, [apiCall, auth, selectedSellers]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/history?limit=20',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setHistory(res.history || []);
        setShowHistory(true);
      }
    } catch (e) {
      toast.error('이력 조회 실패');
    }
  }, [apiCall, auth]);

  const riskColor = (level) => {
    if (level === 'high') return 'text-red-600 bg-red-50 border-red-200';
    return 'text-yellow-600 bg-yellow-50 border-yellow-200';
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="text-red-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">셀러 이탈 방지 자동 조치</h3>
            <span className="text-xs text-gray-500">ML 예측 → LLM 메시지 → 자동 실행</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500">임계값:</label>
            <select
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              <option value={0.5}>50%</option>
              <option value={0.6}>60%</option>
              <option value={0.7}>70%</option>
              <option value={0.8}>80%</option>
            </select>
            <button
              onClick={fetchAtRisk}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-lg bg-red-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-red-600 disabled:opacity-50"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              위험 셀러 탐지
            </button>
            <button
              onClick={fetchHistory}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50"
            >
              <Clock size={14} />
              조치 이력
            </button>
          </div>
        </div>
      </div>

      <PipelineFlow steps={RETENTION_STEPS} stepStatuses={pipelineStatus} currentStep={currentStep} />

      {sellers.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <h4 className="text-sm font-bold text-gray-700 mb-3">
            이탈 위험 셀러 ({sellers.length}명)
          </h4>
          {selectedSellers.size > 0 && (
            <div className="flex items-center gap-3 rounded-xl bg-cafe24-yellow/10 border border-cafe24-yellow/30 p-3 mb-3">
              <span className="text-xs font-bold text-cafe24-brown">{selectedSellers.size}명 선택</span>
              <div className="flex gap-1.5">
                {ACTION_TYPES.map(act => {
                  const ActIcon = act.icon;
                  return (
                    <button
                      key={act.key}
                      onClick={() => executeBulk(act.key)}
                      disabled={bulkLoading}
                      className="flex items-center gap-1 rounded-lg border border-cafe24-yellow/50 bg-white px-2 py-1 text-[10px] font-semibold text-cafe24-brown hover:bg-cafe24-yellow/10 disabled:opacity-50"
                    >
                      <ActIcon size={10} />
                      {act.label}
                    </button>
                  );
                })}
              </div>
              <button onClick={() => setSelectedSellers(new Set())} className="ml-auto text-[10px] text-gray-400 hover:text-gray-600">선택 해제</button>
            </div>
          )}
          <div className="space-y-2">
            {sellers.map((s, i) => (
              <SellerCard
                key={s.seller_id || i}
                s={s}
                i={i}
                isSelected={selectedSeller === s.seller_id}
                isChecked={selectedSellers.has(s.seller_id)}
                msgLoading={msgLoading}
                onSelect={setSelectedSeller}
                onToggleCheck={(sellerId) => {
                  setSelectedSellers(prev => {
                    const next = new Set(prev);
                    if (next.has(sellerId)) next.delete(sellerId);
                    else next.add(sellerId);
                    return next;
                  });
                }}
                onGenerateMessage={generateMessage}
                riskColor={riskColor}
              />
            ))}
          </div>
        </div>
      )}

      {message && (
        <div className="rounded-2xl border border-cafe24-yellow/50 bg-gradient-to-r from-yellow-50 to-orange-50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare className="text-cafe24-orange" size={18} />
            <h4 className="text-sm font-bold text-gray-800">
              {message.seller_id} 리텐션 메시지
            </h4>
            {message.urgency && (
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                message.urgency === 'high' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
              }`}>
                {message.urgency === 'high' ? '긴급' : '보통'}
              </span>
            )}
          </div>
          <div className="rounded-xl bg-white/80 p-3 text-xs text-gray-700 leading-relaxed mb-3">
            <ReactMarkdown>{message.message || ''}</ReactMarkdown>
          </div>
          <div className="flex flex-wrap gap-2">
            {ACTION_TYPES.map(act => {
              const ActIcon = act.icon;
              return (
                <button
                  key={act.key}
                  onClick={() => executeAction(message.seller_id, act.key)}
                  disabled={execLoading}
                  className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 hover:border-cafe24-yellow hover:bg-cafe24-yellow/10 disabled:opacity-50 transition-all"
                >
                  {execLoading ? <Loader2 size={12} className="animate-spin" /> : <ActIcon size={12} />}
                  {act.label}
                </button>
              );
            })}
          </div>
          {message.recommended_actions && message.recommended_actions.length > 0 && (
            <div className="mt-3 text-[10px] text-gray-500">
              AI 추천: {message.recommended_actions.join(', ')}
            </div>
          )}
        </div>
      )}

      {showHistory && history.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-gray-700">조치 이력</h4>
            <button onClick={() => setShowHistory(false)} className="text-xs text-gray-400 hover:text-gray-600">닫기</button>
          </div>
          <div className="space-y-1.5 max-h-60 overflow-y-auto">
            {history.map((h, i) => (
              <div key={i} className="flex items-center gap-3 text-xs border-b border-gray-100 pb-1.5">
                <CheckCircle2 size={12} className="text-green-500" />
                <span className="text-gray-500 w-24">{new Date(h.timestamp * 1000).toLocaleString('ko-KR')}</span>
                <span className="font-semibold text-gray-700">{h.seller_id || h.target_id}</span>
                <span className="text-gray-500">{h.action_type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
