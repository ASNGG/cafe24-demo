// components/panels/LabPanel.js - CS 자동화 파이프라인 실험실
import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Inbox, Search, MessageSquare, Send, TrendingUp,
  ChevronRight, ChevronLeft, RotateCcw, Zap, AlertTriangle,
  CheckCircle2, Clock, Mail, MessageCircle, Smartphone, Bell,
  Loader2, Sparkles, FileText, Edit3, User, Play,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import toast from 'react-hot-toast';
import { ReactFlow, Handle, Position } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// ─── 상수 ───
const STEPS = [
  { key: 'classify', label: '접수', icon: Inbox, desc: '일괄 분류 + 분기' },
  { key: 'review',   label: '검토', icon: Search, desc: '우선순위 + 검토' },
  { key: 'answer',   label: '답변', icon: MessageSquare, desc: 'RAG 답변 생성' },
  { key: 'reply',    label: '회신', icon: Send, desc: '채널별 자동 전송' },
  { key: 'improve',  label: '개선', icon: TrendingUp, desc: '피드백 & 대시보드' },
];

// 접수함에 들어온 셀러 문의 5건 (단순 + 복잡 혼합)
// preferredChannels: 고객이 선택한 희망 답변 채널 ('any' = 어느 방식이든 상관없음)
const INBOX_INQUIRIES = [
  { text: '배송비 조건부 무료 설정(5만원 이상 무료) 방법을 모르겠습니다.', tier: 'Basic', preferredChannels: ['email'] },
  { text: '세금계산서 자동 발행 설정은 어디서 하나요?', tier: 'Standard', preferredChannels: ['kakao'] },
  { text: '상품 대량 등록 엑셀 업로드에서 오류가 발생합니다. 양식을 확인하고 싶습니다.', tier: 'Standard', preferredChannels: ['email', 'kakao'] },
  { text: 'PG사 연동 중 이니시스 인증키 오류가 발생하고, 동시에 네이버페이 정산도 누락되고 있습니다. 두 건 다 긴급히 해결 부탁드립니다.', tier: 'Premium', preferredChannels: ['any'] },
  { text: '카페24 API 웹훅 콜백이 간헐적으로 실패합니다. 서버 로그를 확인해주시고 원인 분석 부탁드립니다.', tier: 'Enterprise', preferredChannels: ['email', 'sms'] },
];

const SELLER_TIERS = ['Basic', 'Standard', 'Premium', 'Enterprise'];

const PRIORITY_COLORS = {
  urgent: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-300' },
  high:   { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-300' },
  normal: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-300' },
  low:    { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-300' },
};

const CHANNELS = [
  { key: 'email', label: '이메일', icon: Mail, color: 'from-blue-500 to-blue-600', enabled: true },
  { key: 'kakao', label: '카카오톡', icon: MessageCircle, color: 'from-yellow-400 to-yellow-500', enabled: false },
  { key: 'sms',   label: 'SMS', icon: Smartphone, color: 'from-green-500 to-green-600', enabled: false },
  { key: 'inapp', label: '인앱 알림', icon: Bell, color: 'from-purple-500 to-purple-600', enabled: false },
];

const TIER_COLORS = {
  Basic: 'bg-gray-100 text-gray-600',
  Standard: 'bg-blue-100 text-blue-700',
  Premium: 'bg-purple-100 text-purple-700',
  Enterprise: 'bg-amber-100 text-amber-700',
};

// ─── 메인 컴포넌트 ───
export default function LabPanel({ auth, apiCall, settings }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(new Set());

  // Step 1 - 접수 (일괄 분류 + DnD)
  const [classifyResults, setClassifyResults] = useState([]); // [{text, tier, result}]
  const [classifyLoading, setClassifyLoading] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);

  // DnD 2열 관리 (분류 후 신뢰도로 초기 분기, 이후 드래그로 이동 가능)
  const [autoIdxs, setAutoIdxs] = useState([]);       // 자동 처리 컬럼 (idx 배열)
  const [manualIdxs, setManualIdxs] = useState([]);    // 담당자 검토 컬럼 (idx 배열)
  const [checkedAuto, setCheckedAuto] = useState(new Set()); // 자동 처리 체크된 항목
  const [dragOverZone, setDragOverZone] = useState(null);    // 드래그 하이라이트
  const [batchAnswers, setBatchAnswers] = useState({});       // { idx: 답변텍스트 }
  const [batchLoading, setBatchLoading] = useState(false);

  // 파이프라인 결과 (선택된 문의)
  const [pipelineResult, setPipelineResult] = useState(null);
  const [selectedInquiry, setSelectedInquiry] = useState(null); // {text, tier}

  // Step 3 - 답변
  const [draftAnswer, setDraftAnswer] = useState('');
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [ragContext, setRagContext] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const abortRef = useRef(null);

  // Step 4 - 회신 (다중 채널 선택)
  const [selectedChannels, setSelectedChannels] = useState(new Set());
  const [sent, setSent] = useState(false);

  // Step 5 - 개선
  const [pipelineHistory, setPipelineHistory] = useState([]);

  // ─── 일괄 분류 ───
  const runBatchClassify = useCallback(async () => {
    setClassifyLoading(true);
    setClassifyResults([]);
    setSelectedIdx(null);
    setPipelineResult(null);
    setSelectedInquiry(null);
    setAutoIdxs([]);
    setManualIdxs([]);
    setCheckedAuto(new Set());
    setBatchAnswers({});

    try {
      const promises = INBOX_INQUIRIES.map(async (inq) => {
        const res = await apiCall({
          endpoint: '/api/classify/inquiry',
          method: 'POST',
          auth,
          data: { text: inq.text },
        });
        return { text: inq.text, tier: inq.tier, preferredChannels: inq.preferredChannels, result: res };
      });
      const settled = await Promise.all(promises);
      setClassifyResults(settled);
      // 신뢰도 기준으로 초기 분기 (이후 드래그로 이동 가능)
      const autoList = [];
      const manualList = [];
      settled.forEach((item, i) => {
        const conf = item.result?.confidence || 0;
        if (conf >= confidenceThreshold) {
          autoList.push(i);
        } else {
          manualList.push(i);
        }
      });
      setAutoIdxs(autoList);
      setManualIdxs(manualList);
      setCompletedSteps(new Set([0]));
    } catch (e) {
      toast.error(`분류 오류: ${e.message || e}`);
    } finally {
      setClassifyLoading(false);
    }
  }, [apiCall, auth]);

  // ─── DnD 핸들러 ───
  const handleDragStart = useCallback((e, idx) => {
    e.dataTransfer.setData('text/plain', String(idx));
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleDropToAuto = useCallback((e) => {
    e.preventDefault();
    const idx = Number(e.dataTransfer.getData('text/plain'));
    setManualIdxs(prev => prev.filter(i => i !== idx));
    setAutoIdxs(prev => prev.includes(idx) ? prev : [...prev, idx]);
    setDragOverZone(null);
    // 담당자 검토에서 드래그해 온 경우 선택 해제
    if (selectedIdx === idx) {
      setSelectedIdx(null);
      setSelectedInquiry(null);
      setPipelineResult(null);
    }
  }, [selectedIdx]);

  const handleDropToManual = useCallback((e) => {
    e.preventDefault();
    const idx = Number(e.dataTransfer.getData('text/plain'));
    setAutoIdxs(prev => prev.filter(i => i !== idx));
    setCheckedAuto(prev => { const s = new Set(prev); s.delete(idx); return s; });
    setManualIdxs(prev => prev.includes(idx) ? prev : [...prev, idx]);
    setDragOverZone(null);
  }, []);

  // ─── 체크박스 (자동 처리 선택 시 담당자 검토 선택 해제) ───
  const toggleAutoCheck = useCallback((idx) => {
    setSelectedIdx(null);
    setSelectedInquiry(null);
    setPipelineResult(null);
    setCheckedAuto(prev => {
      const s = new Set(prev);
      s.has(idx) ? s.delete(idx) : s.add(idx);
      return s;
    });
  }, []);

  const toggleAllAuto = useCallback(() => {
    setSelectedIdx(null);
    setSelectedInquiry(null);
    setPipelineResult(null);
    setCheckedAuto(prev => prev.size === autoIdxs.length ? new Set() : new Set(autoIdxs));
  }, [autoIdxs]);

  // ─── 일괄 자동 답변 생성 (RAG 기반) ───
  const generateBatchAnswers = useCallback(async (targetIdxs) => {
    if (!targetIdxs || targetIdxs.length === 0) return;

    // 이미 답변 생성된 항목은 명시적 체크박스 선택 시에만 재생성
    const toGenerate = targetIdxs.filter(idx => !batchAnswers[idx] || checkedAuto.has(idx));
    if (toGenerate.length === 0) {
      toast('이미 모든 답변이 생성되었습니다. 재생성하려면 항목을 선택하세요.');
      return;
    }

    setBatchLoading(true);

    for (const idx of toGenerate) {
      const item = classifyResults[idx];
      if (!item) continue;

      const category = item.result?.predicted_category || '기타';
      try {
        const resp = await fetch('/api/cs/pipeline-answer', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(auth?.username && auth?.password
              ? { Authorization: 'Basic ' + btoa(`${auth.username}:${auth.password}`) }
              : {}),
          },
          body: JSON.stringify({
            inquiry_text: item.text,
            inquiry_category: category,
            seller_tier: item.tier,
            rag_mode: 'rag',  // 자동 처리는 항상 RAG 기반
            apiKey: settings?.apiKey || '',
          }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let fullText = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const parsed = JSON.parse(line.slice(6));
              if (parsed.type === 'token') fullText += parsed.data;
            } catch {}
          }
        }

        setBatchAnswers(prev => ({ ...prev, [idx]: fullText || '(답변 생성 실패)' }));

        // 이력 추가
        setPipelineHistory(prev => [...prev, {
          time: new Date().toLocaleTimeString('ko-KR'),
          text: item.text.slice(0, 40) + (item.text.length > 40 ? '...' : ''),
          category,
          routing: 'auto',
          priority: 'normal',
        }]);
      } catch (e) {
        setBatchAnswers(prev => ({ ...prev, [idx]: `(오류: ${e.message})` }));
      }
    }

    setBatchLoading(false);
    toast.success(`${toGenerate.length}건 자동 답변 생성 완료`);
  }, [classifyResults, auth, settings, batchAnswers, checkedAuto]);

  // ─── 문의 선택 → 풀 파이프라인 실행 ───
  const selectInquiry = useCallback(async (idx) => {
    const item = classifyResults[idx];
    if (!item) return;

    setSelectedIdx(idx);
    setSelectedInquiry({ text: item.text, tier: item.tier, preferredChannels: item.preferredChannels });
    setPipelineLoading(true);
    setPipelineResult(null);
    setDraftAnswer('');
    setStreamingAnswer('');
    setSelectedChannels(new Set());
    setSent(false);

    try {
      const res = await apiCall({
        endpoint: '/api/cs/pipeline',
        method: 'POST',
        auth,
        data: {
          inquiry_text: item.text,
          seller_tier: item.tier,
          order_value: 50000,
          is_repeat_issue: false,
          confidence_threshold: confidenceThreshold,
        },
      });

      if (res?.status === 'SUCCESS' || res?.steps) {
        setPipelineResult(res);
        setCompletedSteps(prev => new Set([...prev, 0, 1]));
        setCurrentStep(1);

        setPipelineHistory(prev => [...prev, {
          time: new Date().toLocaleTimeString('ko-KR'),
          text: item.text.slice(0, 40) + (item.text.length > 40 ? '...' : ''),
          category: res.steps?.classify?.predicted_category || '?',
          routing: res.steps?.review?.routing || '?',
          priority: res.steps?.review?.priority?.predicted_priority || '?',
        }]);
      } else {
        toast.error('파이프라인 실행 실패');
      }
    } catch (e) {
      toast.error(`오류: ${e.message || e}`);
    } finally {
      setPipelineLoading(false);
    }
  }, [classifyResults, confidenceThreshold, apiCall, auth]);

  // ─── 답변 생성 (SSE 스트리밍) ───
  const generateAnswer = useCallback(async () => {
    if (!pipelineResult || !selectedInquiry) return;

    setIsStreaming(true);
    setStreamingAnswer('');
    setDraftAnswer('');
    setRagContext(null);

    const category = pipelineResult.steps?.classify?.predicted_category || '기타';

    try {
      const controller = new AbortController();
      abortRef.current = controller;

      const backendBase = '/api/cs/pipeline-answer';
      const body = JSON.stringify({
        inquiry_text: selectedInquiry.text,
        inquiry_category: category,
        seller_tier: selectedInquiry.tier,
        order_id: null,
        rag_mode: settings?.ragMode || 'rag',
        apiKey: settings?.apiKey || '',
      });

      const headers = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      };
      if (auth?.username && auth?.password) {
        headers['Authorization'] = 'Basic ' + btoa(`${auth.username}:${auth.password}`);
      }

      const resp = await fetch(backendBase, {
        method: 'POST',
        headers,
        body,
        signal: controller.signal,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const parsed = JSON.parse(line.slice(6));
            if (parsed.type === 'rag_context') {
              setRagContext(parsed.data);
            } else if (parsed.type === 'token') {
              fullText += parsed.data;
              setStreamingAnswer(fullText);
            } else if (parsed.type === 'done') {
              setDraftAnswer(fullText);
              setIsStreaming(false);
            } else if (parsed.type === 'error') {
              toast.error(`답변 생성 오류: ${parsed.data}`);
              setIsStreaming(false);
            }
          } catch {}
        }
      }

      if (!fullText && !draftAnswer) {
        setIsStreaming(false);
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        toast.error(`스트리밍 오류: ${e.message}`);
      }
      setIsStreaming(false);
    }
  }, [pipelineResult, selectedInquiry, settings, auth, draftAnswer]);

  // cleanup abort on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  // ─── 자동 답변 개별 수정 ───
  const updateBatchAnswer = useCallback((idx, newText) => {
    setBatchAnswers(prev => ({ ...prev, [idx]: newText }));
  }, []);

  // ─── 초기화 ───
  const resetPipeline = useCallback(() => {
    setCurrentStep(0);
    setCompletedSteps(new Set());
    setClassifyResults([]);
    setSelectedIdx(null);
    setPipelineResult(null);
    setSelectedInquiry(null);
    setDraftAnswer('');
    setStreamingAnswer('');
    setRagContext(null);
    setSelectedChannels(new Set());
    setSent(false);
    setIsEditing(false);
    setAutoIdxs([]);
    setManualIdxs([]);
    setCheckedAuto(new Set());
    setDragOverZone(null);
    setBatchAnswers({});
    setBatchLoading(false);
    if (abortRef.current) abortRef.current.abort();
  }, []);

  // ─── 단계 이동 ───
  const goNext = () => {
    if (currentStep < 4) {
      // 자동 처리 모드: 접수(0) → 답변(2) 이동 (검토만 스킵)
      const isAutoMode = autoIdxs.length > 0 && !pipelineResult;
      if (currentStep === 0 && isAutoMode) {
        setCompletedSteps(prev => new Set([...prev, 0, 1]));
        setCurrentStep(2);
      } else {
        setCompletedSteps(prev => new Set([...prev, currentStep]));
        setCurrentStep(currentStep + 1);
      }
    }
  };
  const goPrev = () => {
    if (currentStep > 0) setCurrentStep(currentStep - 1);
  };

  // ─── 렌더링 ───
  return (
    <div className="space-y-6">
      <>
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-cookie-brown flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cookie-orange" />
            실험실 - CS 자동화 파이프라인
          </h2>
          <p className="text-sm text-cookie-brown/60 mt-1">
            단순/반복 문의는 자동 처리, 복잡한 문의만 담당자 검토
          </p>
        </div>
        <button
          onClick={resetPipeline}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border border-cookie-brown/20 hover:bg-cookie-yellow/10 text-cookie-brown/70 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          초기화
        </button>
      </div>

      {/* 스텝 인디케이터 */}
      <StepIndicator steps={STEPS} current={currentStep} completed={completedSteps} onStepClick={setCurrentStep} />

      {/* 스텝 콘텐츠 */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
        >
          {currentStep === 0 && (
            <StepClassify
              classifyResults={classifyResults}
              classifyLoading={classifyLoading}
              runBatchClassify={runBatchClassify}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              autoIdxs={autoIdxs}
              manualIdxs={manualIdxs}
              checkedAuto={checkedAuto}
              dragOverZone={dragOverZone}
              setDragOverZone={setDragOverZone}
              handleDragStart={handleDragStart}
              handleDropToAuto={handleDropToAuto}
              handleDropToManual={handleDropToManual}
              toggleAutoCheck={toggleAutoCheck}
              toggleAllAuto={toggleAllAuto}
              selectedIdx={selectedIdx}
              selectInquiry={selectInquiry}
              pipelineLoading={pipelineLoading}
            />
          )}
          {currentStep === 1 && (
            <StepReview
              result={pipelineResult?.steps?.review}
              classifyResult={pipelineResult?.steps?.classify}
              threshold={confidenceThreshold}
              selectedInquiry={selectedInquiry}
              classifyResults={classifyResults}
              autoIdxs={autoIdxs}
              manualIdxs={manualIdxs}
            />
          )}
          {currentStep === 2 && (
            <StepAnswer
              result={pipelineResult?.steps?.answer}
              draftAnswer={draftAnswer}
              setDraftAnswer={setDraftAnswer}
              streamingAnswer={streamingAnswer}
              isStreaming={isStreaming}
              generateAnswer={generateAnswer}
              ragContext={ragContext}
              isEditing={isEditing}
              setIsEditing={setIsEditing}
              settings={settings}
              classifyResults={classifyResults}
              autoIdxs={autoIdxs}
              batchAnswers={batchAnswers}
              batchLoading={batchLoading}
              generateBatchAnswers={generateBatchAnswers}
              checkedAuto={checkedAuto}
              toggleAutoCheck={toggleAutoCheck}
              toggleAllAuto={toggleAllAuto}
              updateBatchAnswer={updateBatchAnswer}
            />
          )}
          {currentStep === 3 && (
            <StepReply
              channels={CHANNELS}
              selectedChannels={selectedChannels}
              setSelectedChannels={setSelectedChannels}
              sent={sent}
              setSent={setSent}
              draftAnswer={draftAnswer}
              inquiry={selectedInquiry}
              classifyResults={classifyResults}
              autoIdxs={autoIdxs}
              batchAnswers={batchAnswers}
              updateBatchAnswer={updateBatchAnswer}
              auth={auth}
            />
          )}
          {currentStep === 4 && (
            <StepImprove
              result={pipelineResult?.steps?.improve}
              history={pipelineHistory}
              apiCall={apiCall}
              auth={auth}
            />
          )}
        </motion.div>
      </AnimatePresence>

      {/* 네비게이션 */}
      <div className="flex justify-between items-center pt-2">
        <button
          onClick={goPrev}
          disabled={currentStep === 0}
          className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg border border-cookie-brown/20 hover:bg-cookie-yellow/10 text-cookie-brown/70 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          이전 단계
        </button>
        <span className="text-xs text-cookie-brown/40">
          {currentStep + 1} / {STEPS.length}
        </span>
        <button
          onClick={goNext}
          disabled={currentStep === 4}
          className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg bg-cookie-orange text-white hover:bg-cookie-orange/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          다음 단계
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
      </>
    </div>
  );
}

// ─── 마크다운 Bold → React <strong> 변환 ───
const renderMd = (text) => {
  if (!text) return null;
  return text.split(/(\*\*.*?\*\*)/g).map((seg, i) =>
    seg.startsWith('**') && seg.endsWith('**')
      ? <strong key={i} className="font-semibold">{seg.slice(2, -2)}</strong>
      : seg
  );
};

// ─── 스텝 인디케이터 ───
function StepIndicator({ steps, current, completed, onStepClick }) {
  return (
    <div className="flex items-center justify-between bg-white rounded-xl p-4 shadow-sm border border-cookie-brown/10">
      {steps.map((step, i) => {
        const Icon = step.icon;
        const isActive = i === current;
        const isDone = completed.has(i);

        return (
          <div key={step.key} className="flex items-center flex-1">
            <button
              onClick={() => onStepClick(i)}
              className="flex flex-col items-center gap-1.5 flex-1 group"
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isActive
                    ? 'bg-cookie-orange text-white shadow-lg shadow-cookie-orange/30 scale-110'
                    : isDone
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-100 text-gray-400 group-hover:bg-gray-200'
                }`}
              >
                {isDone && !isActive ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>
              <span
                className={`text-xs font-medium ${
                  isActive ? 'text-cookie-orange' : isDone ? 'text-green-600' : 'text-gray-400'
                }`}
              >
                {step.label}
              </span>
              <span className="text-[10px] text-gray-400 hidden sm:block text-center whitespace-nowrap">{step.desc}</span>
            </button>
            {i < steps.length - 1 && (
              <div
                className={`h-0.5 flex-1 mx-2 rounded transition-colors ${
                  completed.has(i) ? 'bg-green-400' : 'bg-gray-200'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Step 1: 접수 - 일괄 분류 + DnD 자동/수동 분기 ───
function StepClassify({
  classifyResults, classifyLoading, runBatchClassify,
  confidenceThreshold, setConfidenceThreshold,
  autoIdxs, manualIdxs, checkedAuto, dragOverZone, setDragOverZone,
  handleDragStart, handleDropToAuto, handleDropToManual,
  toggleAutoCheck, toggleAllAuto,
  selectedIdx, selectInquiry, pipelineLoading,
}) {
  const hasResults = classifyResults.length > 0;
  const hasSplit = autoIdxs.length > 0 || manualIdxs.length > 0;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
        <Inbox className="w-5 h-5 text-cookie-orange" />
        Step 1. 접수 - 셀러 문의 일괄 분류
      </div>

      {/* 접수함 테이블 (분류 전) */}
      {!hasSplit && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-cookie-brown/70 font-medium">접수함 ({INBOX_INQUIRIES.length}건)</span>
            <button
              onClick={runBatchClassify}
              disabled={classifyLoading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cookie-orange text-white text-sm font-medium hover:bg-cookie-orange/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {classifyLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {classifyLoading ? '분류 중...' : '일괄 분류'}
            </button>
          </div>

          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-gray-500 text-xs">
                  <th className="text-left py-2.5 px-3 w-8">#</th>
                  <th className="text-left py-2.5 px-3">문의 내용</th>
                  <th className="text-left py-2.5 px-3 w-24">셀러 등급</th>
                  <th className="text-left py-2.5 px-3 w-28">희망 채널</th>
                </tr>
              </thead>
              <tbody>
                {INBOX_INQUIRIES.map((inq, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2.5 px-3 text-gray-400 text-xs">{i + 1}</td>
                    <td className="py-2.5 px-3 text-gray-700 text-xs leading-relaxed">{inq.text}</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[inq.tier] || 'bg-gray-100 text-gray-600'}`}>
                        {inq.tier}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex gap-1 flex-wrap">
                        {(inq.preferredChannels || []).map(ch => (
                          <span key={ch} className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            ch === 'any' ? 'bg-gray-100 text-gray-500' : 'bg-blue-50 text-blue-600'
                          }`}>
                            {ch === 'any' ? '무관' : ch === 'email' ? '이메일' : ch === 'kakao' ? '카카오' : ch === 'sms' ? 'SMS' : ch}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 분류 완료 → DnD 2열 */}
      {hasSplit && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          {/* 상단: 재분류 + 임계값 */}
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={runBatchClassify}
              disabled={classifyLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-cookie-brown/20 hover:bg-cookie-yellow/10 text-cookie-brown/60"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              재분류
            </button>
            <div className="flex items-center gap-2 text-xs text-cookie-brown/60">
              <span>임계값</span>
              <input
                type="range" min={0.5} max={0.95} step={0.05}
                value={confidenceThreshold}
                onChange={e => setConfidenceThreshold(Number(e.target.value))}
                className="w-20 accent-cookie-orange h-1"
              />
              <span className="font-bold text-cookie-orange w-8">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          <p className="text-xs text-gray-400">
            문의를 드래그하여 자동 처리 ↔ 담당자 검토 사이로 이동할 수 있습니다.
          </p>

          {/* 2열: 자동 처리 / 담당자 검토 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 자동 처리 드롭존 */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOverZone('auto'); }}
              onDragLeave={() => setDragOverZone(null)}
              onDrop={handleDropToAuto}
              className={`rounded-lg border-2 transition-all min-h-[200px] ${
                dragOverZone === 'auto'
                  ? 'border-dashed border-green-400 bg-green-50/80'
                  : 'border-green-200 bg-green-50/50'
              }`}
            >
              <div className="flex items-center gap-2 px-4 py-3 border-b border-green-200 bg-green-50">
                <Zap className="w-4 h-4 text-green-600" />
                <span className="text-sm font-bold text-green-700">자동 처리</span>
                <span className="ml-auto text-xs text-green-600 font-medium">{autoIdxs.length}건</span>
              </div>
              {/* 전체 선택 체크박스 */}
              {autoIdxs.length > 0 && (
                <div className="flex items-center gap-2 px-4 py-2 border-b border-green-100">
                  <input
                    type="checkbox"
                    checked={checkedAuto.size === autoIdxs.length && autoIdxs.length > 0}
                    onChange={toggleAllAuto}
                    className="w-3.5 h-3.5 accent-green-600 rounded"
                  />
                  <span className="text-[10px] text-gray-500">전체 선택</span>
                </div>
              )}
              <div className="p-2 space-y-1.5">
                {autoIdxs.length === 0 ? (
                  <p className="text-xs text-gray-400 text-center py-8">
                    {dragOverZone === 'auto' ? '여기에 놓으세요' : '문의를 드래그하세요'}
                  </p>
                ) : (
                  autoIdxs.map((idx) => (
                    <DraggableCard
                      key={idx}
                      idx={idx}
                      item={classifyResults[idx]}
                      variant="auto"
                      checked={checkedAuto.has(idx)}
                      onCheck={() => toggleAutoCheck(idx)}
                      onDragStart={handleDragStart}
                    />
                  ))
                )}
              </div>
              {autoIdxs.length > 0 && (
                <p className="text-[10px] text-green-600/60 text-center py-2 border-t border-green-100">
                  다음 단계(답변)에서 RAG 기반 답변이 생성됩니다
                </p>
              )}
            </div>

            {/* 담당자 검토 드롭존 */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOverZone('manual'); }}
              onDragLeave={() => setDragOverZone(null)}
              onDrop={handleDropToManual}
              className={`rounded-lg border-2 transition-all min-h-[200px] ${
                dragOverZone === 'manual'
                  ? 'border-dashed border-amber-400 bg-amber-50/80'
                  : 'border-amber-200 bg-amber-50/50'
              }`}
            >
              <div className="flex items-center gap-2 px-4 py-3 border-b border-amber-200 bg-amber-50">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span className="text-sm font-bold text-amber-700">담당자 검토</span>
                <span className="ml-auto text-xs text-amber-600 font-medium">{manualIdxs.length}건</span>
              </div>
              <div className="p-2 space-y-1.5">
                {manualIdxs.length === 0 ? (
                  <p className="text-xs text-gray-400 text-center py-8">
                    {dragOverZone === 'manual' ? '여기에 놓으세요' : '해당 없음'}
                  </p>
                ) : (
                  manualIdxs.map((idx) => (
                    <DraggableCard
                      key={idx}
                      idx={idx}
                      item={classifyResults[idx]}
                      variant="manual"
                      onDragStart={handleDragStart}
                      onClick={() => selectInquiry(idx)}
                      loading={pipelineLoading && selectedIdx === idx}
                      isSelected={selectedIdx === idx}
                    />
                  ))
                )}
              </div>
              {manualIdxs.length > 0 && (
                <p className="text-[10px] text-amber-600/60 text-center py-2 border-t border-amber-100">
                  클릭하여 상세 파이프라인 실행 (검토→답변→회신)
                </p>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

// ─── 드래그 가능한 문의 카드 ───
function DraggableCard({ idx, item, variant, checked, onCheck, onDragStart, onClick, loading, isSelected, answer, onUpdateAnswer }) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('');

  if (!item) return null;

  const conf = item.result?.confidence || 0;
  const isAuto = variant === 'auto';
  const borderColor = isAuto ? 'border-green-300' : 'border-amber-300';
  const hoverBorder = isAuto ? 'hover:border-green-400' : 'hover:border-amber-400';

  return (
    <div
      draggable={!editing}
      onDragStart={(e) => !editing && onDragStart(e, idx)}
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border cursor-grab active:cursor-grabbing transition-all ${
        isSelected ? 'border-cookie-orange bg-cookie-orange/5' : `${borderColor} bg-white ${hoverBorder} hover:shadow-sm`
      } ${loading ? 'opacity-60' : ''}`}
    >
      <div className="flex items-start gap-2">
        {/* 체크박스 (자동 처리만) */}
        {isAuto && onCheck && (
          <input
            type="checkbox"
            checked={!!checked}
            onChange={(e) => { e.stopPropagation(); onCheck(); }}
            onClick={(e) => e.stopPropagation()}
            className="mt-0.5 w-3.5 h-3.5 accent-green-600 rounded shrink-0"
          />
        )}
        {/* 아이콘 (담당자 검토) */}
        {!isAuto && (
          <div className="shrink-0 mt-0.5">
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin text-cookie-orange" />
            ) : isSelected ? (
              <CheckCircle2 className="w-4 h-4 text-cookie-orange" />
            ) : (
              <User className="w-4 h-4 text-gray-400" />
            )}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-700 leading-relaxed line-clamp-2">{item.text}</p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[item.tier] || ''}`}>
              {item.tier}
            </span>
            {item.result?.predicted_category && (
              <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange text-[10px] font-bold">
                {item.result.predicted_category}
              </span>
            )}
            <span className={`text-[10px] font-medium ${isAuto ? 'text-green-600' : 'text-amber-600'}`}>
              {(conf * 100).toFixed(0)}%
            </span>
            {/* 희망 채널 배지 */}
            {item.preferredChannels && (
              item.preferredChannels.includes('any')
                ? <span className="px-1 py-0.5 rounded text-[9px] bg-gray-100 text-gray-500">채널 무관</span>
                : item.preferredChannels.map(ch => (
                    <span key={ch} className="px-1 py-0.5 rounded text-[9px] bg-blue-50 text-blue-600">
                      {ch === 'email' ? '이메일' : ch === 'kakao' ? '카카오' : ch === 'sms' ? 'SMS' : ch === 'inapp' ? '인앱' : ch}
                    </span>
                  ))
            )}
          </div>

          {/* 자동 답변 결과 (접이식 + 수정 가능) */}
          {answer && (
            <details className="mt-2 group" onClick={(e) => e.stopPropagation()}>
              <summary className="text-[10px] text-green-600 font-medium cursor-pointer flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                답변 생성 완료 (클릭하여 보기)
              </summary>
              {editing ? (
                <div className="mt-1.5 space-y-1.5" onClick={(e) => e.stopPropagation()}>
                  <textarea
                    value={editText}
                    onChange={e => setEditText(e.target.value)}
                    rows={5}
                    className="w-full p-2 rounded border border-green-300 text-[11px] text-gray-700 resize-none focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-200"
                  />
                  <div className="flex gap-1.5 justify-end">
                    <button
                      onClick={(e) => { e.stopPropagation(); setEditing(false); }}
                      className="px-2 py-1 rounded text-[10px] text-gray-500 border border-gray-200 hover:bg-gray-50"
                    >
                      취소
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); onUpdateAnswer(idx, editText); setEditing(false); }}
                      className="px-2 py-1 rounded text-[10px] text-white bg-green-600 hover:bg-green-700"
                    >
                      저장
                    </button>
                  </div>
                </div>
              ) : (
                <div className="mt-1.5 relative group/answer">
                  <div className="p-2 rounded bg-green-50 border border-green-100 text-[11px] text-gray-600 max-h-24 overflow-y-auto whitespace-pre-wrap">
                    {answer}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setEditText(answer); setEditing(true); }}
                    className="absolute top-1 right-1 p-1 rounded bg-white/80 border border-green-200 opacity-0 group-hover/answer:opacity-100 transition-opacity hover:bg-green-50"
                    title="답변 수정"
                  >
                    <Edit3 className="w-3 h-3 text-green-600" />
                  </button>
                </div>
              )}
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Step 2: 검토 ───
function StepReview({ result, classifyResult, threshold, selectedInquiry, classifyResults, autoIdxs, manualIdxs }) {
  const hasAutoMode = !result && classifyResults?.length > 0;

  // 자동 처리 모드: 분류 결과 요약 표시
  if (hasAutoMode) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
        <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
          <Search className="w-5 h-5 text-cookie-orange" />
          Step 2. 검토 - 분류 결과 요약
        </div>

        {/* 분기 현황 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-green-50 border border-green-200 text-center">
            <Zap className="w-6 h-6 text-green-600 mx-auto mb-1" />
            <div className="text-2xl font-bold text-green-700">{autoIdxs?.length || 0}건</div>
            <div className="text-xs text-green-600">자동 처리</div>
          </div>
          <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 text-center">
            <AlertTriangle className="w-6 h-6 text-amber-600 mx-auto mb-1" />
            <div className="text-2xl font-bold text-amber-700">{manualIdxs?.length || 0}건</div>
            <div className="text-xs text-amber-600">담당자 검토</div>
          </div>
        </div>

        {/* 개별 분류 결과 */}
        <div className="space-y-2">
          <span className="text-sm font-medium text-cookie-brown/80">문의별 분류 상세</span>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                  <th className="text-left py-2 px-3 w-8">#</th>
                  <th className="text-left py-2 px-3">문의 내용</th>
                  <th className="text-left py-2 px-3 w-24 whitespace-nowrap">카테고리</th>
                  <th className="text-left py-2 px-3 w-16">신뢰도</th>
                  <th className="text-left py-2 px-3 w-16">분기</th>
                </tr>
              </thead>
              <tbody>
                {classifyResults.map((item, i) => {
                  const conf = item.result?.confidence || 0;
                  const isAutoItem = autoIdxs?.includes(i);
                  return (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-2 px-3 text-gray-400">{i + 1}</td>
                      <td className="py-2 px-3 text-gray-700 max-w-[200px] truncate">{item.text}</td>
                      <td className="py-2 px-3 whitespace-nowrap">
                        <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange font-bold">
                          {item.result?.predicted_category || '?'}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`font-medium ${conf >= threshold ? 'text-green-600' : 'text-amber-600'}`}>
                          {(conf * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${
                          isAutoItem ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}>
                          {isAutoItem ? '자동' : '수동'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-blue-50 border border-blue-100">
          <p className="text-xs text-blue-600">
            신뢰도 {(threshold * 100).toFixed(0)}% 이상은 자동 처리, 미만은 담당자 검토로 분기되었습니다.
            다음 단계에서 답변을 생성합니다.
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10">
        <EmptyStep message="먼저 Step 1에서 문의를 분류하세요." />
      </div>
    );
  }

  const confidence = result.confidence || classifyResult?.confidence || 0;
  const isAuto = confidence >= threshold;
  const priority = result.priority?.predicted_priority || 'normal';
  const pColors = PRIORITY_COLORS[priority] || PRIORITY_COLORS.normal;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
        <Search className="w-5 h-5 text-cookie-orange" />
        Step 2. 검토 - 상세 분석
      </div>

      {/* 선택된 문의 */}
      {selectedInquiry && (
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
            <FileText className="w-3.5 h-3.5" />
            선택된 문의
          </div>
          <p className="text-sm text-gray-700">{selectedInquiry.text}</p>
        </div>
      )}

      {/* 분기 결과 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* 분류 결과 */}
        <div className="p-4 rounded-lg bg-gray-50 border border-gray-200 space-y-2">
          <span className="text-xs text-gray-500 font-medium">분류 결과</span>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full bg-cookie-orange/10 text-cookie-orange text-sm font-bold">
              {classifyResult?.predicted_category || result.predicted_category}
            </span>
            <span className="text-sm text-gray-600">
              신뢰도 {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* 라우팅 결과 */}
        <div className={`p-4 rounded-lg border space-y-2 ${
          isAuto ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'
        }`}>
          <span className="text-xs text-gray-500 font-medium">라우팅 결정</span>
          <div className="flex items-center gap-2">
            {isAuto ? (
              <>
                <Zap className="w-5 h-5 text-green-600" />
                <span className="text-green-700 font-bold text-sm">자동 처리</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                <span className="text-amber-700 font-bold text-sm">담당자 검토 필요</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 우선순위 */}
      <div className={`p-4 rounded-lg border ${pColors.bg} ${pColors.border} space-y-2`}>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 font-medium">우선순위</span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${pColors.bg} ${pColors.text} border ${pColors.border}`}>
            {priority.toUpperCase()}
          </span>
        </div>
        {result.priority?.priority_description && (
          <p className={`text-sm ${pColors.text}`}>{result.priority.priority_description}</p>
        )}
      </div>

      {/* 추천 조치 */}
      {result.priority?.recommendations && result.priority.recommendations.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-cookie-brown/80">추천 조치</span>
          <ul className="space-y-1">
            {result.priority.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                <ChevronRight className="w-4 h-4 text-cookie-orange mt-0.5 shrink-0" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ─── Step 3: 답변 ───
function StepAnswer({
  result, draftAnswer, setDraftAnswer, streamingAnswer, isStreaming, generateAnswer, ragContext, isEditing, setIsEditing, settings,
  classifyResults, autoIdxs, batchAnswers, batchLoading, generateBatchAnswers, checkedAuto, toggleAutoCheck, toggleAllAuto, updateBatchAnswer,
}) {
  const [editingIdx, setEditingIdx] = useState(null);
  const [editText, setEditText] = useState('');
  const hasAutoMode = !result && classifyResults?.length > 0 && autoIdxs?.length > 0;
  const answeredCount = batchAnswers ? Object.keys(batchAnswers).filter(k => autoIdxs?.includes(Number(k))).length : 0;

  // 자동 처리 모드: 일괄 답변 생성
  if (hasAutoMode) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
            <MessageSquare className="w-5 h-5 text-cookie-orange" />
            Step 3. 답변 - 자동 처리 일괄 답변 생성
          </div>
          <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-500">
            RAG 모드: {settings?.ragMode || 'rag'}
          </span>
        </div>

        <p className="text-xs text-gray-500">
          자동 처리 대상 {autoIdxs.length}건에 대해 RAG + LLM 답변을 생성합니다.
          {answeredCount > 0 && ` (${answeredCount}건 생성 완료)`}
        </p>

        {/* 체크박스 선택 + 생성 버튼 */}
        <div className="flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-xs text-gray-600">
            <input
              type="checkbox"
              checked={checkedAuto?.size === autoIdxs.length && autoIdxs.length > 0}
              onChange={toggleAllAuto}
              className="w-3.5 h-3.5 accent-green-600 rounded"
            />
            전체 선택
          </label>
          <button
            onClick={() => generateBatchAnswers(checkedAuto?.size > 0 ? [...checkedAuto] : autoIdxs)}
            disabled={batchLoading}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-gradient-to-r from-cookie-orange to-cookie-yellow text-white text-sm font-medium hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {batchLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {batchLoading
              ? '생성 중...'
              : checkedAuto?.size > 0
              ? `선택 답변 생성 (${checkedAuto.size}건)`
              : `전체 답변 생성 (${autoIdxs.length}건)`}
          </button>
        </div>

        {/* 문의별 답변 카드 */}
        <div className="space-y-3">
          {autoIdxs.map((idx) => {
            const item = classifyResults[idx];
            if (!item) return null;
            const answer = batchAnswers?.[idx];
            const category = item.result?.predicted_category || '?';

            return (
              <div key={idx} className="rounded-lg border border-gray-200 overflow-hidden">
                {/* 문의 헤더 */}
                <div className="flex items-start gap-2 p-3 bg-gray-50 border-b border-gray-100">
                  <input
                    type="checkbox"
                    checked={!!checkedAuto?.has(idx)}
                    onChange={() => toggleAutoCheck(idx)}
                    className="mt-0.5 w-3.5 h-3.5 accent-green-600 rounded shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-gray-700 leading-relaxed">{item.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[item.tier] || ''}`}>
                        {item.tier}
                      </span>
                      <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange text-[10px] font-bold">
                        {category}
                      </span>
                    </div>
                  </div>
                </div>
                {/* 답변 영역 */}
                <div className="p-3">
                  {answer ? (
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1 text-xs text-green-600 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          답변 생성 완료
                        </div>
                        {editingIdx !== idx && (
                          <button
                            onClick={() => { setEditText(answer); setEditingIdx(idx); }}
                            className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border border-gray-200 hover:bg-gray-50 text-gray-500"
                          >
                            <Edit3 className="w-3 h-3" />
                            수정
                          </button>
                        )}
                      </div>
                      {editingIdx === idx ? (
                        <div className="space-y-1.5">
                          <textarea
                            value={editText}
                            onChange={e => setEditText(e.target.value)}
                            rows={6}
                            className="w-full p-2.5 rounded border border-green-300 text-xs text-gray-700 resize-none focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-200 leading-relaxed"
                          />
                          <div className="flex gap-1.5 justify-end">
                            <button
                              onClick={() => setEditingIdx(null)}
                              className="px-2.5 py-1 rounded text-[10px] text-gray-500 border border-gray-200 hover:bg-gray-50"
                            >
                              취소
                            </button>
                            <button
                              onClick={() => { updateBatchAnswer(idx, editText); setEditingIdx(null); }}
                              className="px-2.5 py-1 rounded text-[10px] text-white bg-green-600 hover:bg-green-700"
                            >
                              저장
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="p-2.5 rounded bg-green-50 border border-green-100 text-xs text-gray-600 max-h-32 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                          {renderMd(answer)}
                        </div>
                      )}
                    </div>
                  ) : batchLoading ? (
                    <div className="flex items-center gap-2 text-xs text-gray-400 py-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      대기 중...
                    </div>
                  ) : (
                    <p className="text-xs text-gray-400 py-2">답변 생성 버튼을 클릭하세요</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10">
        <EmptyStep message="먼저 Step 1에서 문의를 분류하세요." />
      </div>
    );
  }

  const displayText = draftAnswer || streamingAnswer;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
          <MessageSquare className="w-5 h-5 text-cookie-orange" />
          Step 3. 답변 - RAG + LLM 초안 생성
        </div>
        <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-500">
          RAG 모드: {settings?.ragMode || 'rag'}
        </span>
      </div>

      {/* 원본 문의 컨텍스트 */}
      {result.context && (
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
            <FileText className="w-3.5 h-3.5" />
            문의 컨텍스트
          </div>
          <p className="text-sm text-gray-700">
            카테고리: <strong>{result.context.inquiry_category}</strong> | 셀러 등급: <strong>{result.context.seller_tier}</strong>
          </p>
        </div>
      )}

      {/* 답변 생성 버튼 */}
      {!displayText && !isStreaming && (
        <button
          onClick={generateAnswer}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-cookie-orange to-cookie-yellow text-white font-medium hover:shadow-lg transition-all"
        >
          <Sparkles className="w-5 h-5" />
          RAG + LLM 답변 초안 생성
        </button>
      )}

      {/* RAG 소스 */}
      {ragContext && (
        <details className="group">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 flex items-center gap-1">
            <FileText className="w-3.5 h-3.5" />
            RAG 검색 결과 ({ragContext.source_count}건)
          </summary>
          <div className="mt-2 p-3 rounded-lg bg-blue-50 border border-blue-100 text-xs text-gray-600 max-h-32 overflow-y-auto">
            {ragContext.context_preview || '(검색 결과 없음)'}
          </div>
        </details>
      )}

      {/* 스트리밍 / 답변 표시 */}
      {(isStreaming || displayText) && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-cookie-brown/80 flex items-center gap-1.5">
              {isStreaming && <Loader2 className="w-4 h-4 animate-spin text-cookie-orange" />}
              {isStreaming ? '답변 생성 중...' : '답변 초안'}
            </span>
            {draftAnswer && !isStreaming && (
              <div className="flex gap-2">
                <button
                  onClick={() => setIsEditing(!isEditing)}
                  className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg border border-cookie-brown/20 hover:bg-cookie-yellow/10 text-cookie-brown/70"
                >
                  <Edit3 className="w-3.5 h-3.5" />
                  {isEditing ? '미리보기' : '편집'}
                </button>
                <button
                  onClick={generateAnswer}
                  className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg border border-cookie-orange/30 hover:bg-cookie-orange/10 text-cookie-orange"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  재생성
                </button>
              </div>
            )}
          </div>

          {isEditing && draftAnswer ? (
            <textarea
              value={draftAnswer}
              onChange={e => setDraftAnswer(e.target.value)}
              rows={10}
              className="w-full px-4 py-3 rounded-lg border border-cookie-orange/30 focus:border-cookie-orange focus:ring-1 focus:ring-cookie-orange/30 outline-none text-sm resize-none font-mono"
            />
          ) : (
            <div className="p-4 rounded-lg bg-gray-50 border border-gray-200 text-sm text-gray-700 whitespace-pre-wrap min-h-[120px] leading-relaxed">
              {renderMd(displayText)}
              {isStreaming && <span className="inline-block w-1.5 h-4 bg-cookie-orange animate-pulse ml-0.5 align-middle" />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Step 4: 회신 ───
// ─── React Flow 워크플로우 노드 정의 ───
const WORKFLOW_NODE_DEFS = [
  { id: 'webhook',  label: 'Webhook 트리거', x: 0,   y: 150, ntype: 'trigger' },
  { id: 'validate', label: '답변 검증',       x: 200, y: 150, ntype: 'process' },
  { id: 'router',   label: '채널 분기',       x: 400, y: 150, ntype: 'switch' },
  { id: 'channel_email', label: '이메일',     x: 620, y: 0,   ntype: 'channel' },
  { id: 'channel_kakao', label: '카카오톡',   x: 620, y: 100, ntype: 'channel' },
  { id: 'channel_sms',   label: 'SMS',       x: 620, y: 200, ntype: 'channel' },
  { id: 'channel_inapp', label: '인앱 알림',  x: 620, y: 300, ntype: 'channel' },
  { id: 'log',      label: '결과 기록',       x: 860, y: 150, ntype: 'output' },
];

const WORKFLOW_EDGE_DEFS = [
  { source: 'webhook',  target: 'validate' },
  { source: 'validate', target: 'router' },
  { source: 'router',   target: 'channel_email' },
  { source: 'router',   target: 'channel_kakao' },
  { source: 'router',   target: 'channel_sms' },
  { source: 'router',   target: 'channel_inapp' },
  { source: 'channel_email', target: 'log' },
  { source: 'channel_kakao', target: 'log' },
  { source: 'channel_sms',   target: 'log' },
  { source: 'channel_inapp', target: 'log' },
];

const NODE_STYLES = {
  trigger: { bg: 'bg-purple-100', border: 'border-purple-400', text: 'text-purple-700', icon: Zap },
  process: { bg: 'bg-blue-100', border: 'border-blue-400', text: 'text-blue-700', icon: CheckCircle2 },
  switch:  { bg: 'bg-amber-100', border: 'border-amber-400', text: 'text-amber-700', icon: Sparkles },
  channel: { bg: 'bg-sky-100', border: 'border-sky-400', text: 'text-sky-700', icon: Send },
  output:  { bg: 'bg-green-100', border: 'border-green-400', text: 'text-green-700', icon: FileText },
};

const CHANNEL_ICONS = {
  channel_email: Mail,
  channel_kakao: MessageCircle,
  channel_sms: Smartphone,
  channel_inapp: Bell,
};

const STATUS_STYLES = {
  idle:      'border-gray-300 bg-white',
  running:   'border-yellow-400 bg-yellow-50 shadow-md animate-pulse',
  completed: 'border-green-500 bg-green-50',
  disabled:  'border-dashed border-gray-200 bg-gray-50 opacity-40',
};

function WorkflowNode({ data }) {
  const { label, ntype, nodeStatus, detail } = data;
  const style = NODE_STYLES[ntype] || NODE_STYLES.process;
  const statusStyle = STATUS_STYLES[nodeStatus] || STATUS_STYLES.idle;
  const IconComp = CHANNEL_ICONS[data.nodeId] || style.icon;
  const isCompleted = nodeStatus === 'completed';
  const isRunning = nodeStatus === 'running';

  return (
    <div className={`relative px-3 py-2 rounded-lg border-2 ${statusStyle} min-w-[90px] text-center transition-all duration-300`}>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-gray-400" />
      <div className="flex flex-col items-center gap-1">
        <div className={`w-6 h-6 rounded-md flex items-center justify-center ${isCompleted ? 'bg-green-500 text-white' : isRunning ? 'bg-yellow-400 text-white' : style.bg + ' ' + style.text}`}>
          {isCompleted ? <CheckCircle2 className="w-3.5 h-3.5" /> : <IconComp className="w-3.5 h-3.5" />}
        </div>
        <span className={`text-[10px] font-semibold ${isCompleted ? 'text-green-700' : isRunning ? 'text-yellow-700' : 'text-gray-600'}`}>
          {label}
        </span>
        {detail && (
          <span className="text-[8px] text-gray-500 leading-tight">{detail}</span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-gray-400" />
    </div>
  );
}

const workflowNodeTypes = { workflowNode: WorkflowNode };

function StepReply({ channels, selectedChannels, setSelectedChannels, sent, setSent, draftAnswer, inquiry, classifyResults, autoIdxs, batchAnswers, updateBatchAnswer, auth }) {
  const [editingIdx, setEditingIdx] = useState(null);
  const [editText, setEditText] = useState('');
  const [autoReplyChannels, setAutoReplyChannels] = useState({});
  const [nodeStatuses, setNodeStatuses] = useState({});
  const [isSending, setIsSending] = useState(false);
  // 이메일 수신자 입력 (auto: idx→email, manual: 단일 문자열)
  const [autoEmails, setAutoEmails] = useState({});
  const [manualEmail, setManualEmail] = useState('');

  const batchAnswerCount = autoIdxs?.length > 0
    ? Object.keys(batchAnswers || {}).filter(k => autoIdxs.includes(Number(k))).length
    : 0;
  const isAutoMode = !draftAnswer && batchAnswerCount > 0;

  // 자동 처리 초기 채널 설정 (미구현 채널은 이메일로 대체)
  useEffect(() => {
    if (isAutoMode && Object.keys(autoReplyChannels).length === 0) {
      const enabledKeys = new Set(channels.filter(c => c.enabled).map(c => c.key));
      const initial = {};
      autoIdxs.forEach(idx => {
        const item = classifyResults?.[idx];
        if (!item) return;
        const pref = item.preferredChannels || ['any'];
        // 이메일만 구현된 상태 → 모든 문의에 이메일 자동 선택
        const validPref = pref.filter(k => enabledKeys.has(k));
        initial[idx] = new Set(validPref.length > 0 ? validPref : ['email']);
      });
      setAutoReplyChannels(initial);
    }
  }, [isAutoMode, autoIdxs, classifyResults]);

  // 현재 선택된 모든 채널 (auto / manual 공용)
  const activeChannelKeys = useMemo(() => {
    if (isAutoMode) {
      const all = new Set();
      Object.values(autoReplyChannels).forEach(s => s.forEach(c => all.add(c)));
      return all;
    }
    return selectedChannels;
  }, [isAutoMode, autoReplyChannels, selectedChannels]);

  // React Flow 노드/엣지 생성
  const { nodes: flowNodes, edges: flowEdges } = useMemo(() => {
    const nodes = WORKFLOW_NODE_DEFS.map(nd => {
      const isChannel = nd.id.startsWith('channel_');
      const chKey = isChannel ? nd.id.replace('channel_', '') : null;
      const isActive = !isChannel || activeChannelKeys.has(chKey);
      let nodeStatus = 'idle';
      if (nodeStatuses[nd.id]) {
        nodeStatus = nodeStatuses[nd.id].status;
      } else if (!isActive) {
        nodeStatus = 'disabled';
      }
      return {
        id: nd.id,
        type: 'workflowNode',
        position: { x: nd.x, y: nd.y },
        width: 110,
        height: 58,
        data: {
          label: nd.label,
          ntype: nd.ntype,
          nodeId: nd.id,
          nodeStatus,
          detail: nodeStatuses[nd.id]?.detail || '',
        },
        draggable: false,
        connectable: false,
      };
    });

    const edges = WORKFLOW_EDGE_DEFS.map(ed => {
      const isTargetChannel = ed.target.startsWith('channel_');
      const chKey = isTargetChannel ? ed.target.replace('channel_', '') : null;
      const isSourceChannel = ed.source.startsWith('channel_');
      const srcChKey = isSourceChannel ? ed.source.replace('channel_', '') : null;
      const isActive = (!isTargetChannel || activeChannelKeys.has(chKey)) && (!isSourceChannel || activeChannelKeys.has(srcChKey));
      return {
        id: `${ed.source}-${ed.target}`,
        source: ed.source,
        target: ed.target,
        animated: isActive && (nodeStatuses[ed.source]?.status === 'completed' || nodeStatuses[ed.target]?.status === 'running'),
        style: {
          stroke: isActive ? '#94a3b8' : '#e2e8f0',
          strokeWidth: isActive ? 1.5 : 1,
          strokeDasharray: isActive ? undefined : '4 4',
          opacity: isActive ? 1 : 0.4,
        },
      };
    });
    return { nodes, edges };
  }, [activeChannelKeys, nodeStatuses]);

  const toggleChannel = (key) => {
    if (sent || isSending) return;
    const ch = channels.find(c => c.key === key);
    if (ch && !ch.enabled) return;
    setSelectedChannels(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const toggleAutoChannel = (idx, key) => {
    if (sent || isSending) return;
    const ch = channels.find(c => c.key === key);
    if (ch && !ch.enabled) return;
    setAutoReplyChannels(prev => {
      const current = new Set(prev[idx] || []);
      current.has(key) ? current.delete(key) : current.add(key);
      return { ...prev, [idx]: current };
    });
  };

  const canSend = (() => {
    if (isSending || sent) return false;
    if (isAutoMode) {
      return autoIdxs.filter(idx => batchAnswers?.[idx]).every(idx => {
        const chs = autoReplyChannels[idx];
        if (!chs || chs.size === 0) return false;
        // 이메일 채널 선택 시 이메일 주소 필수
        if (chs.has('email') && !autoEmails[idx]?.includes('@')) return false;
        return true;
      });
    }
    if (selectedChannels.size === 0 || !draftAnswer) return false;
    if (selectedChannels.has('email') && !manualEmail.includes('@')) return false;
    return true;
  })();

  const handleSend = async () => {
    if (!canSend) {
      toast.error(isAutoMode ? '모든 문의에 회신 채널을 선택하세요.' : '회신 채널을 선택하세요.');
      return;
    }
    setIsSending(true);
    setNodeStatuses({ webhook: { status: 'completed', detail: '트리거 완료' } });

    // payload 구성 (recipient_email 포함)
    const inquiries = [];
    if (isAutoMode) {
      autoIdxs.filter(idx => batchAnswers?.[idx]).forEach(idx => {
        const item = classifyResults?.[idx];
        inquiries.push({
          inquiry_text: item?.text || '',
          answer_text: batchAnswers[idx] || '',
          channels: [...(autoReplyChannels[idx] || [])],
          seller_tier: item?.tier || 'Standard',
          category: item?.result?.predicted_category || '기타',
          recipient_email: autoEmails[idx] || '',
        });
      });
    } else {
      inquiries.push({
        inquiry_text: inquiry?.text || '',
        answer_text: draftAnswer || '',
        channels: [...selectedChannels],
        seller_tier: inquiry?.tier || 'Standard',
        category: '기타',
        recipient_email: manualEmail || '',
      });
    }

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (auth?.username && auth?.password) {
        headers['Authorization'] = `Basic ${btoa(`${auth.username}:${auth.password}`)}`;
      }

      // 1) job_id 발급 + n8n 트리거
      const triggerResp = await fetch('/api/cs/send-reply', {
        method: 'POST',
        headers,
        body: JSON.stringify({ inquiries }),
      });
      const triggerData = await triggerResp.json();
      if (!triggerData.job_id) throw new Error(triggerData.error || 'job_id 발급 실패');

      // 2) SSE 스트림 연결 (job_id 기반)
      const streamResp = await fetch(`/api/cs/stream?job_id=${triggerData.job_id}`, { headers });
      if (!streamResp.ok || !streamResp.body) throw new Error(`SSE stream HTTP ${streamResp.status}`);

      const reader = streamResp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === 'step') {
              setNodeStatuses(prev => ({
                ...prev,
                [evt.data.node]: { status: evt.data.status, detail: evt.data.detail || '' },
              }));
            } else if (evt.type === 'done') {
              setSent(true);
              const chLabels = (evt.data.channels || []).map(k => channels.find(c => c.key === k)?.label || k).join(', ');
              toast.success(`${evt.data.total}건이 ${chLabels}(으)로 전송 완료!`);
            } else if (evt.type === 'error') {
              toast.error(`전송 오류: ${evt.data}`);
            }
          } catch {}
        }
      }
    } catch (e) {
      toast.error(`전송 실패: ${e.message}`);
      setNodeStatuses(prev => ({ ...prev, validate: { status: 'completed', detail: 'fallback' }, router: { status: 'completed', detail: 'fallback' }, log: { status: 'completed', detail: 'fallback' } }));
      setSent(true);
    } finally {
      setIsSending(false);
    }
  };

  const ChannelBadge = ({ chKey, small }) => {
    const ch = channels.find(c => c.key === chKey);
    if (!ch) return null;
    const Icon = ch.icon;
    return (
      <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-gradient-to-r ${ch.color} text-white ${small ? 'text-[9px]' : 'text-[10px]'} font-medium`}>
        <Icon className={small ? 'w-2.5 h-2.5' : 'w-3 h-3'} />
        {ch.label}
      </span>
    );
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
        <Send className="w-5 h-5 text-cookie-orange" />
        Step 4. 회신 - 채널별 자동 발송
      </div>

      {/* ── 자동 처리 모드: per-inquiry 채널 + 답변 ── */}
      {isAutoMode && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 p-3 rounded-lg bg-green-50 border border-green-200">
            <Zap className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-green-700">
              자동 처리 {batchAnswerCount}건 - 고객 희망 채널별 회신
            </span>
          </div>

          <div className="space-y-3 max-h-[420px] overflow-y-auto">
            {autoIdxs.filter(idx => batchAnswers?.[idx]).map(idx => {
              const item = classifyResults?.[idx];
              if (!item) return null;
              const pref = item.preferredChannels || ['any'];
              const isAnyPref = pref.includes('any');
              const assignedChannels = autoReplyChannels[idx] || new Set();
              const isEditingThis = editingIdx === idx;

              return (
                <div key={idx} className="rounded-lg border border-gray-200 overflow-hidden">
                  <div className="p-3 bg-gray-50 border-b border-gray-100 space-y-2">
                    <div className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0 mt-0.5" />
                      <p className="text-xs text-gray-700 leading-relaxed flex-1 line-clamp-2">{item.text}</p>
                      <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange text-[10px] font-bold shrink-0">
                        {item.result?.predicted_category || '?'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 ml-5">
                      <span className="text-[10px] text-gray-500 shrink-0">고객 희망:</span>
                      {isAnyPref ? (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-200 text-gray-600 font-medium">
                          어느 방식이든 상관없음
                        </span>
                      ) : (
                        <div className="flex gap-1 flex-wrap">
                          {pref.map(k => <ChannelBadge key={k} chKey={k} small />)}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 ml-5 flex-wrap">
                      <span className="text-[10px] text-gray-500 shrink-0">회신 채널:</span>
                      {channels.map(ch => {
                        const Icon = ch.icon;
                        const selected = assignedChannels.has(ch.key);
                        const disabled = !ch.enabled || sent || isSending;
                        return (
                          <button
                            key={ch.key}
                            onClick={() => ch.enabled && toggleAutoChannel(idx, ch.key)}
                            disabled={disabled}
                            title={!ch.enabled ? '미구현' : ''}
                            className={`flex items-center gap-1 px-2 py-1 rounded-lg border text-[10px] font-medium transition-all ${
                              !ch.enabled
                                ? 'border-gray-100 bg-gray-50 text-gray-300 cursor-not-allowed line-through'
                                : selected
                                ? 'border-cookie-orange bg-cookie-orange/10 text-cookie-orange'
                                : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300'
                            } ${(sent || isSending) ? 'opacity-60 cursor-not-allowed' : ''}`}
                          >
                            <Icon className="w-3 h-3" />
                            {ch.label}
                            {!ch.enabled && <span className="text-[8px] text-gray-300 ml-0.5">(미구현)</span>}
                          </button>
                        );
                      })}
                    </div>
                    {/* 이메일 채널 선택 시 수신자 이메일 입력 */}
                    {assignedChannels.has('email') && (
                      <div className="ml-5 mt-2 space-y-1">
                        <label className="text-[11px] text-gray-500 font-medium ml-1">여기에 이메일을 입력해보세요</label>
                        <div className="flex items-center gap-2">
                          <Mail className="w-4 h-4 text-gray-400 shrink-0" />
                          <input
                            type="email"
                            placeholder="예: seller@shop.com"
                            value={autoEmails[idx] || ''}
                            onChange={e => setAutoEmails(prev => ({ ...prev, [idx]: e.target.value }))}
                            disabled={sent || isSending}
                            className="flex-1 px-3 py-2 rounded-lg border border-gray-200 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:border-cookie-orange focus:ring-1 focus:ring-cookie-orange/30 disabled:opacity-50"
                          />
                        </div>
                        <p className="text-[10px] text-blue-500 ml-6">전송 시 실제로 해당 이메일로 답변이 발송됩니다.</p>
                      </div>
                    )}
                  </div>
                  <div className="border-t border-gray-100 bg-white">
                    {isEditingThis ? (
                      <div className="p-2.5 space-y-2">
                        <textarea
                          value={editText}
                          onChange={e => setEditText(e.target.value)}
                          rows={5}
                          className="w-full p-2 rounded border border-green-300 text-xs text-gray-700 resize-none focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-200"
                        />
                        <div className="flex gap-1.5 justify-end">
                          <button
                            onClick={() => setEditingIdx(null)}
                            className="px-2.5 py-1 rounded text-[10px] text-gray-500 border border-gray-200 hover:bg-gray-50"
                          >
                            취소
                          </button>
                          <button
                            onClick={() => { updateBatchAnswer(idx, editText); setEditingIdx(null); }}
                            className="px-2.5 py-1 rounded text-[10px] text-white bg-green-600 hover:bg-green-700"
                          >
                            저장
                          </button>
                        </div>
                      </div>
                    ) : (
                      <details className="group">
                        <summary className="flex items-center gap-1 p-2.5 text-[10px] text-green-600 font-medium cursor-pointer hover:bg-gray-50">
                          <FileText className="w-3 h-3" />
                          답변 내용 보기
                        </summary>
                        <div className="p-2.5 pt-0 relative group/ans">
                          <div className="text-xs text-gray-600 whitespace-pre-wrap leading-relaxed max-h-24 overflow-y-auto">
                            {batchAnswers[idx]}
                          </div>
                          {!sent && !isSending && (
                            <button
                              onClick={() => { setEditText(batchAnswers[idx]); setEditingIdx(idx); }}
                              className="absolute top-0 right-2.5 p-1 rounded bg-white/80 border border-gray-200 opacity-0 group-hover/ans:opacity-100 transition-opacity hover:bg-gray-50"
                              title="답변 수정"
                            >
                              <Edit3 className="w-3 h-3 text-gray-500" />
                            </button>
                          )}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── 수동 처리 모드: 단일 문의 ── */}
      {!isAutoMode && inquiry && (
        <div className="space-y-3">
          {inquiry.preferredChannels && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-50 border border-blue-200">
              <User className="w-4 h-4 text-blue-600 shrink-0" />
              <span className="text-xs text-blue-700 font-medium shrink-0">고객 희망 채널:</span>
              {inquiry.preferredChannels.includes('any') ? (
                <span className="text-xs text-blue-600">어느 방식이든 상관없음 (아래에서 선택하세요)</span>
              ) : (
                <div className="flex gap-1.5 flex-wrap">
                  {inquiry.preferredChannels.map(k => {
                    const ch = channels.find(c => c.key === k);
                    if (!ch) return null;
                    const Icon = ch.icon;
                    return (
                      <span key={k} className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gradient-to-r ${ch.color} text-white text-[10px] font-medium`}>
                        <Icon className="w-3 h-3" />
                        {ch.label}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          <div>
            <span className="text-xs text-gray-500 font-medium mb-2 block">회신 채널 선택</span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {channels.map(ch => {
                const Icon = ch.icon;
                const selected = selectedChannels.has(ch.key);
                const isPreferred = inquiry?.preferredChannels?.includes(ch.key);
                const disabled = !ch.enabled || sent || isSending;
                return (
                  <button
                    key={ch.key}
                    onClick={() => ch.enabled && toggleChannel(ch.key)}
                    disabled={disabled}
                    className={`relative p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${
                      !ch.enabled
                        ? 'border-gray-100 bg-gray-50 cursor-not-allowed opacity-50'
                        : selected
                        ? 'border-cookie-orange bg-cookie-orange/5 shadow-md'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    } ${(sent || isSending) ? 'opacity-60 cursor-not-allowed' : ''}`}
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${ch.enabled ? ch.color : 'from-gray-300 to-gray-400'} flex items-center justify-center text-white`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-sm font-medium ${!ch.enabled ? 'text-gray-400' : selected ? 'text-cookie-orange' : 'text-gray-600'}`}>
                      {ch.label}
                    </span>
                    {!ch.enabled && (
                      <span className="text-[9px] text-gray-400 -mt-1">미구현</span>
                    )}
                    {selected && ch.enabled && (
                      <div className="absolute top-2 right-2 w-3 h-3 rounded-full bg-cookie-orange" />
                    )}
                    {isPreferred && (
                      <span className="absolute top-1 left-1 px-1 py-0.5 rounded text-[8px] font-bold bg-blue-100 text-blue-600">
                        희망
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            <p className="text-[11px] text-amber-600 bg-amber-50 rounded-lg px-3 py-2 mt-3">
              카카오톡, SMS, 인앱 알림은 현재 미구현 상태입니다. 고객 선호 채널과 관계없이 이메일로 자동 발송됩니다.
            </p>
            {/* 이메일 채널 선택 시 수신자 이메일 입력 */}
            {selectedChannels.has('email') && (
              <div className="mt-3 space-y-1">
                <label className="text-xs text-gray-500 font-medium ml-1">여기에 이메일을 입력해보세요</label>
                <div className="flex items-center gap-2">
                  <Mail className="w-4 h-4 text-gray-400 shrink-0" />
                  <input
                    type="email"
                    placeholder="예: seller@shop.com"
                    value={manualEmail}
                    onChange={e => setManualEmail(e.target.value)}
                    disabled={sent || isSending}
                    className="flex-1 px-3 py-2.5 rounded-lg border border-gray-200 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:border-cookie-orange focus:ring-1 focus:ring-cookie-orange/30 disabled:opacity-50"
                  />
                </div>
                <p className="text-[10px] text-blue-500 ml-6">전송 시 실제로 해당 이메일로 답변이 발송됩니다.</p>
                {manualEmail && !manualEmail.includes('@') && (
                  <p className="text-[10px] text-red-500 mt-1 ml-6">올바른 이메일 주소를 입력하세요.</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {!isAutoMode && !inquiry && (
        <EmptyStep message="먼저 답변을 생성하세요." />
      )}

      {/* ── React Flow 워크플로우 다이어그램 ── */}
      <div className="rounded-lg border border-gray-200 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
          <span className="text-xs text-gray-500 font-medium flex items-center gap-1.5">
            <Play className="w-3.5 h-3.5" />
            n8n 워크플로우 — 회신 자동화
          </span>
          {isSending && (
            <span className="flex items-center gap-1.5 text-[10px] text-yellow-600 font-medium">
              <Loader2 className="w-3 h-3 animate-spin" />
              실행 중...
            </span>
          )}
          {sent && !isSending && (
            <span className="flex items-center gap-1.5 text-[10px] text-green-600 font-medium">
              <CheckCircle2 className="w-3 h-3" />
              완료
            </span>
          )}
        </div>
        <div style={{ height: 340, width: '100%', position: 'relative' }}>
          <ReactFlow
            key={`wf-${activeChannelKeys.size}`}
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={workflowNodeTypes}
            defaultViewport={{ x: 40, y: 20, zoom: 0.95 }}
            fitView
            fitViewOptions={{ padding: 0.25, includeHiddenNodes: true }}
            onInit={(inst) => {
              const doFit = () => { try { inst.fitView({ padding: 0.25, includeHiddenNodes: true }); } catch {} };
              doFit();
              setTimeout(doFit, 120);
              setTimeout(doFit, 400);
            }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            panOnDrag={false}
            zoomOnScroll={false}
            zoomOnDoubleClick={false}
            preventScrolling={false}
            proOptions={{ hideAttribution: true }}
          />
        </div>
      </div>

      {/* 전송 버튼 */}
      <div className="flex justify-center">
        {sent ? (
          <div className="flex items-center gap-2 px-6 py-3 rounded-lg bg-green-50 border border-green-200 text-green-700 font-medium">
            <CheckCircle2 className="w-5 h-5" />
            {isAutoMode ? `${batchAnswerCount}건 일괄 전송 완료` : '회신 전송 완료'}
          </div>
        ) : (
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="flex items-center gap-2 px-6 py-3 rounded-lg bg-cookie-orange text-white font-medium hover:bg-cookie-orange/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isSending ? (
              <><Loader2 className="w-4 h-4 animate-spin" />전송 중...</>
            ) : (
              <><Send className="w-4 h-4" />{isAutoMode ? `${batchAnswerCount}건 일괄 전송` : `${selectedChannels.size > 0 ? [...selectedChannels].map(k => channels.find(c => c.key === k)?.label).join(' + ') + '으로 ' : ''}전송`}</>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Step 5: 개선 ───
function StepImprove({ result, history, apiCall, auth }) {
  const [stats, setStats] = useState(result?.statistics || null);
  const [loadingStats, setLoadingStats] = useState(false);

  const loadStats = useCallback(async () => {
    setLoadingStats(true);
    try {
      const res = await apiCall({ endpoint: '/api/cs/statistics', auth });
      if (res?.status === 'SUCCESS') {
        setStats(res);
      }
    } catch {}
    setLoadingStats(false);
  }, [apiCall, auth]);

  useEffect(() => {
    if (!stats) loadStats();
  }, [stats, loadStats]);

  const chartData = stats?.by_category
    ? Object.entries(stats.by_category).map(([name, val]) => ({
        name,
        tickets: val.total_tickets || 0,
        satisfaction: val.satisfaction_score || 0,
        hours: val.avg_resolution_hours || 0,
      }))
    : [];

  const BAR_COLORS = ['#D97B4A', '#EAC54F', '#5C4A3D', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ef4444', '#6b7280'];

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
          <TrendingUp className="w-5 h-5 text-cookie-orange" />
          Step 5. 개선 - 피드백 & 대시보드
        </div>
        <button
          onClick={loadStats}
          disabled={loadingStats}
          className="text-xs px-2.5 py-1 rounded-lg border border-cookie-brown/20 hover:bg-cookie-yellow/10 text-cookie-brown/60"
        >
          {loadingStats ? '로딩...' : '새로고침'}
        </button>
      </div>

      {/* KPI 카드 */}
      {stats && (
        <div className="grid grid-cols-3 gap-3">
          <KpiMini label="총 티켓 수" value={stats.total_tickets?.toLocaleString() || '-'} icon={Inbox} />
          <KpiMini label="평균 만족도" value={stats.avg_satisfaction_score ? `${stats.avg_satisfaction_score.toFixed(1)} / 5` : '-'} icon={TrendingUp} />
          <KpiMini label="평균 처리 시간" value={stats.avg_resolution_hours ? `${stats.avg_resolution_hours.toFixed(1)}h` : '-'} icon={Clock} />
        </div>
      )}

      {/* 카테고리별 차트 */}
      {chartData.length > 0 && (
        <div>
          <span className="text-sm font-medium text-cookie-brown/80 mb-2 block">카테고리별 티켓 수</span>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(val, name) => {
                    if (name === 'tickets') return [val, '티켓 수'];
                    return [val, name];
                  }}
                />
                <Bar dataKey="tickets" radius={[4, 4, 0, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 파이프라인 메타 */}
      {result?.pipeline_meta && (
        <div className="p-3 rounded-lg bg-blue-50 border border-blue-100 space-y-1">
          <span className="text-xs text-blue-600 font-medium">파이프라인 메타정보</span>
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
            <div>분류 모델 정확도: <strong>{(result.pipeline_meta.classification_model_accuracy * 100).toFixed(0)}%</strong></div>
            <div>자동 처리 기준: <strong>{result.pipeline_meta.auto_routing_rate}</strong></div>
          </div>
        </div>
      )}

      {/* 파이프라인 처리 이력 */}
      {history.length > 0 && (
        <div>
          <span className="text-sm font-medium text-cookie-brown/80 mb-2 block">파이프라인 처리 이력</span>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500">
                  <th className="text-left py-2 px-2">시간</th>
                  <th className="text-left py-2 px-2">문의 내용</th>
                  <th className="text-left py-2 px-2">카테고리</th>
                  <th className="text-left py-2 px-2">라우팅</th>
                  <th className="text-left py-2 px-2">우선순위</th>
                </tr>
              </thead>
              <tbody>
                {history.slice().reverse().map((row, i) => {
                  const pColor = PRIORITY_COLORS[row.priority] || PRIORITY_COLORS.normal;
                  return (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-1.5 px-2 text-gray-500">{row.time}</td>
                      <td className="py-1.5 px-2 text-gray-700">{row.text}</td>
                      <td className="py-1.5 px-2">
                        <span className="px-1.5 py-0.5 rounded bg-cookie-orange/10 text-cookie-orange font-medium">{row.category}</span>
                      </td>
                      <td className="py-1.5 px-2">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${
                          row.routing === 'auto' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}>{row.routing === 'auto' ? '자동' : '수동'}</span>
                      </td>
                      <td className="py-1.5 px-2">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${pColor.bg} ${pColor.text}`}>{row.priority}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── 유틸 컴포넌트 ───
function KpiMini({ label, value, icon: Icon }) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200 text-center">
      <Icon className="w-4 h-4 text-cookie-orange mx-auto mb-1" />
      <div className="text-lg font-bold text-cookie-brown">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function EmptyStep({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
      <AlertTriangle className="w-8 h-8 mb-3" />
      <p className="text-sm">{message}</p>
    </div>
  );
}
