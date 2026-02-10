// components/panels/lab/LabPanel.js - CS 자동화 파이프라인 실험실 (메인 컨테이너)
import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronRight, ChevronLeft, RotateCcw, Sparkles,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { createAuthHeaders, parseSSEStream } from '@/lib/sse';
import { STEPS, INBOX_INQUIRIES, CHANNELS } from './constants';
import StepIndicator from './StepIndicator';
import StepClassify from './StepClassify';
import StepReview from './StepReview';
import StepAnswer from './StepAnswer';
import StepReply from './StepReply';
import StepImprove from './StepImprove';

export default function LabPanel({ auth, apiCall, settings }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(new Set());

  // Step 1 - 접수 (일괄 분류 + DnD)
  const [classifyResults, setClassifyResults] = useState([]);
  const [classifyLoading, setClassifyLoading] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);

  // DnD 2열 관리
  const [autoIdxs, setAutoIdxs] = useState([]);
  const [manualIdxs, setManualIdxs] = useState([]);
  const [checkedAuto, setCheckedAuto] = useState(new Set());
  const [dragOverZone, setDragOverZone] = useState(null);
  const [batchAnswers, setBatchAnswers] = useState({});
  const [batchLoading, setBatchLoading] = useState(false);

  // 파이프라인 결과
  const [pipelineResult, setPipelineResult] = useState(null);
  const [selectedInquiry, setSelectedInquiry] = useState(null);

  // Step 3 - 답변
  const [draftAnswer, setDraftAnswer] = useState('');
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [ragContext, setRagContext] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const abortRef = useRef(null);

  // Step 4 - 회신
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

  // ─── 체크박스 ───
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

  // ─── 일괄 자동 답변 생성 (SSE 공통 유틸 사용) ───
  const generateBatchAnswers = useCallback(async (targetIdxs) => {
    if (!targetIdxs || targetIdxs.length === 0) return;

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
          headers: createAuthHeaders(auth),
          body: JSON.stringify({
            inquiry_text: item.text,
            inquiry_category: category,
            seller_tier: item.tier,
            rag_mode: 'rag',
            apiKey: settings?.apiKey || '',
          }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const fullText = await parseSSEStream(resp, {
          onToken: (_, accumulated) => {
            // 중간 업데이트는 하지 않음 (일괄 생성이라 최종 결과만 사용)
          },
        });

        setBatchAnswers(prev => ({ ...prev, [idx]: fullText || '(답변 생성 실패)' }));

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

      if (res?.status === 'success' || res?.steps) {
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

  // ─── 답변 생성 (SSE 공통 유틸 사용) ───
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

      const resp = await fetch('/api/cs/pipeline-answer', {
        method: 'POST',
        headers: createAuthHeaders(auth, { Accept: 'text/event-stream' }),
        body: JSON.stringify({
          inquiry_text: selectedInquiry.text,
          inquiry_category: category,
          seller_tier: selectedInquiry.tier,
          order_id: null,
          rag_mode: settings?.ragMode || 'rag',
          apiKey: settings?.apiKey || '',
        }),
        signal: controller.signal,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const fullText = await parseSSEStream(resp, {
        onToken: (_, accumulated) => {
          setStreamingAnswer(accumulated);
        },
        onDone: (_, accumulated) => {
          setDraftAnswer(accumulated);
          setIsStreaming(false);
        },
        onError: (data) => {
          toast.error(`답변 생성 오류: ${data}`);
          setIsStreaming(false);
        },
        onRagContext: (data) => {
          setRagContext(data);
        },
      });

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
          <h2 className="text-xl font-bold text-cafe24-brown flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cafe24-orange" />
            실험실 - CS 자동화 파이프라인
          </h2>
          <p className="text-sm text-cafe24-brown/60 mt-1">
            단순/반복 문의는 자동 처리, 복잡한 문의만 담당자 검토
          </p>
        </div>
        <button
          onClick={resetPipeline}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border border-cafe24-brown/20 hover:bg-cafe24-yellow/10 text-cafe24-brown/70 transition-colors"
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
          className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg border border-cafe24-brown/20 hover:bg-cafe24-yellow/10 text-cafe24-brown/70 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          이전 단계
        </button>
        <span className="text-xs text-cafe24-brown/40">
          {currentStep + 1} / {STEPS.length}
        </span>
        <button
          onClick={goNext}
          disabled={currentStep === 4}
          className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg bg-cafe24-orange text-white hover:bg-cafe24-orange/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          다음 단계
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
      </>
    </div>
  );
}
