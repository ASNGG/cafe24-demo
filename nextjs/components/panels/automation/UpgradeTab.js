// components/panels/automation/UpgradeTab.js
// 셀러 플랜 업그레이드 자동 추천 탭
import React, { useState, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import {
  TrendingUp, MessageSquare, Loader2, CheckCircle2,
  Search, ArrowUpCircle, Headphones, Send,
} from 'lucide-react';
import PipelineFlow from '@/components/automation/PipelineFlow';
import { UPGRADE_STEPS } from '@/components/automation/constants';

const ACTION_TYPES = [
  { key: 'upgrade_recommend', label: '업그레이드 제안', icon: ArrowUpCircle },
  { key: 'benefit_info', label: '혜택 안내', icon: TrendingUp },
  { key: 'consultation_request', label: '상담 요청', icon: Headphones },
  { key: 'custom_message', label: '맞춤 메시지', icon: Send },
];

// React.memo로 셀러 카드 리렌더링 방지
const SellerCard = React.memo(function SellerCard({ s, isSelected, msgLoading, onSelect, onGenerateMessage, scoreColor }) {
  const score = Number(s.upgrade_score) || 0;

  return (
    <div
      className={`rounded-xl border p-3 cursor-pointer transition-all hover:shadow-md ${
        isSelected ? 'ring-2 ring-blue-400 border-blue-400' : 'border-gray-200'
      }`}
      onClick={() => onSelect(s.seller_id)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-gray-800">{s.seller_id}</span>
          <span className="text-xs px-2 py-0.5 rounded-full border font-semibold bg-blue-50 text-blue-600 border-blue-200">
            {s.current_plan || '기본'}
          </span>
          <span className="text-[10px] text-gray-400">→</span>
          <span className="text-xs px-2 py-0.5 rounded-full border font-semibold bg-emerald-50 text-emerald-600 border-emerald-200">
            {s.recommended_plan || '프로'}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              점수: <strong className={scoreColor(score)}>{score.toFixed(1)}</strong>
            </span>
            <div className="w-16 h-1.5 rounded-full bg-gray-200 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${
                  score >= 80 ? 'bg-emerald-500' :
                  score >= 60 ? 'bg-blue-500' : 'bg-sky-400'
                }`}
                style={{ width: `${Math.min(score, 100)}%` }}
              />
            </div>
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onGenerateMessage(s.seller_id); }}
          disabled={msgLoading}
          className="flex items-center gap-1 rounded-lg bg-blue-500 px-2.5 py-1 text-xs font-semibold text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {msgLoading && isSelected
            ? <Loader2 size={12} className="animate-spin" />
            : <MessageSquare size={12} />}
          메시지 생성
        </button>
      </div>
      {s.reasons && s.reasons.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {s.reasons.slice(0, 5).map((r, ri) => (
            <span key={ri} className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">
              {typeof r === 'string' ? r : r.reason || r.label || JSON.stringify(r)}
            </span>
          ))}
        </div>
      )}
      {s.seller_info && (
        <div className="mt-2 grid grid-cols-5 gap-1.5">
          {[
            { label: '주문', value: s.seller_info.total_orders?.toLocaleString() },
            { label: '매출', value: `${Math.round((s.seller_info.total_revenue || 0) / 10000)}만` },
            { label: '상품', value: s.seller_info.product_count },
            { label: '플랜', value: s.current_plan || '기본' },
            { label: '가입일', value: s.seller_info.join_days ? `${s.seller_info.join_days}일` : '-' },
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

export default function UpgradeTab({ auth, apiCall }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [message, setMessage] = useState(null);
  const [msgLoading, setMsgLoading] = useState(false);
  const [execLoading, setExecLoading] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const messageRef = useRef(null);

  const fetchCandidates = useCallback(async () => {
    setLoading(true);
    setCandidates([]);
    setSelectedSeller(null);
    setMessage(null);
    setPipelineStatus({ detect: { status: 'processing' } });
    setCurrentStep('detect');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/upgrade/candidates?limit=20',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setCandidates(res.candidates || []);
        setPipelineStatus({
          detect: { status: 'complete', detail: `${(res.candidates || []).length}명 탐지` },
          analyze: { status: 'complete', detail: '성과 분석' },
        });
        setCurrentStep(null);
        if ((res.candidates || []).length === 0) {
          toast('현재 업그레이드 후보 셀러가 없습니다', { icon: '✅' });
        }
      } else {
        toast.error(res?.detail || '조회 실패');
        setPipelineStatus({ detect: { status: 'error', detail: '탐지 실패' } });
      }
    } catch (e) {
      toast.error('업그레이드 후보 조회 실패');
      setPipelineStatus({ detect: { status: 'error', detail: '탐지 실패' } });
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth]);

  const generateMessage = useCallback(async (sellerId) => {
    setMsgLoading(true);
    setMessage(null);
    setSelectedSeller(sellerId);
    setPipelineStatus(prev => ({ ...prev, generate: { status: 'processing' } }));
    setCurrentStep('generate');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/upgrade/message',
        auth,
        method: 'POST',
        data: { seller_id: sellerId },
        timeoutMs: 60000,
      });
      if (res?.status === 'success') {
        setMessage(res);
        setPipelineStatus(prev => ({ ...prev, generate: { status: 'complete', detail: '메시지 생성됨' } }));
        setCurrentStep(null);
        setTimeout(() => messageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
      } else {
        toast.error(res?.detail || '메시지 생성 실패');
      }
    } catch (e) {
      toast.error('업그레이드 추천 메시지 생성 실패');
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
        endpoint: '/api/automation/upgrade/execute',
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

  const scoreColor = (score) => {
    if (score >= 80) return 'text-emerald-600';
    if (score >= 60) return 'text-blue-600';
    return 'text-sky-500';
  };

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TrendingUp className="text-blue-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">셀러 플랜 업그레이드 추천</h3>
            <span className="text-xs text-gray-500">규칙 기반 탐지 → LLM 메시지 → 업그레이드 제안</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchCandidates}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-600 disabled:opacity-50"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              후보 탐지
            </button>
          </div>
        </div>
      </div>

      <PipelineFlow steps={UPGRADE_STEPS} stepStatuses={pipelineStatus} currentStep={currentStep} />

      {candidates.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <h4 className="text-sm font-bold text-gray-700 mb-3">
            업그레이드 후보 셀러 ({candidates.length}명)
          </h4>
          <div className="space-y-2">
            {candidates.map((s, i) => (
              <React.Fragment key={s.seller_id || i}>
                <SellerCard
                  s={s}
                  isSelected={selectedSeller === s.seller_id}
                  msgLoading={msgLoading}
                  onSelect={setSelectedSeller}
                  onGenerateMessage={generateMessage}
                  scoreColor={scoreColor}
                />
                {message && selectedSeller === s.seller_id && (
                  <div ref={messageRef} className="rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50 to-emerald-50 p-4 ml-4">
                    <div className="flex items-center gap-2 mb-3">
                      <MessageSquare className="text-blue-500" size={18} />
                      <h4 className="text-sm font-bold text-gray-800">
                        {message.seller_id} 업그레이드 추천 메시지
                      </h4>
                      {message.recommended_plan && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-emerald-100 text-emerald-700">
                          {message.recommended_plan}
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
                            className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 hover:border-blue-400 hover:bg-blue-50 disabled:opacity-50 transition-all"
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
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
