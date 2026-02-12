// components/panels/lab/LabPanel.js - CS 자동화 파이프라인 실험실 (메인 컨테이너)
// H33: useState ~20개 → useReducer 전환
import { useReducer, useRef, useCallback, useEffect } from 'react';
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

// H33: 초기 상태
const initialState = {
  currentStep: 0,
  completedSteps: new Set(),
  // Step 1 - 접수
  classifyResults: [],
  classifyLoading: false,
  confidenceThreshold: 0.75,
  selectedIdx: null,
  pipelineLoading: false,
  // DnD
  autoIdxs: [],
  manualIdxs: [],
  checkedAuto: new Set(),
  dragOverZone: null,
  batchAnswers: {},
  batchLoading: false,
  // 파이프라인 결과
  pipelineResult: null,
  selectedInquiry: null,
  // Step 3 - 답변
  draftAnswer: '',
  streamingAnswer: '',
  isStreaming: false,
  ragContext: null,
  isEditing: false,
  // Step 4 - 회신
  selectedChannels: new Set(),
  sent: false,
  // Step 5 - 개선
  pipelineHistory: [],
};

// H33: 리듀서 (기존 state 명칭 유지)
function labReducer(state, action) {
  switch (action.type) {
    case 'SET':
      return { ...state, ...action.payload };
    case 'RESET':
      return { ...initialState, pipelineHistory: state.pipelineHistory };
    case 'FULL_RESET':
      return { ...initialState };
    case 'COMPLETE_STEP':
      return { ...state, completedSteps: new Set([...state.completedSteps, ...action.payload]) };
    case 'SET_CHECKED_AUTO': {
      const fn = action.payload;
      return { ...state, checkedAuto: fn(state.checkedAuto) };
    }
    case 'SET_BATCH_ANSWER':
      return { ...state, batchAnswers: { ...state.batchAnswers, [action.idx]: action.value } };
    case 'ADD_HISTORY':
      return { ...state, pipelineHistory: [...state.pipelineHistory, action.payload] };
    case 'SET_SELECTED_CHANNELS': {
      const fn = action.payload;
      return { ...state, selectedChannels: fn(state.selectedChannels) };
    }
    default:
      return state;
  }
}

export default function LabPanel({ auth, apiCall, settings }) {
  const [state, dispatch] = useReducer(labReducer, initialState);
  const abortRef = useRef(null);

  // 편의 destructure
  const {
    currentStep, completedSteps,
    classifyResults, classifyLoading, confidenceThreshold, selectedIdx, pipelineLoading,
    autoIdxs, manualIdxs, checkedAuto, dragOverZone, batchAnswers, batchLoading,
    pipelineResult, selectedInquiry,
    draftAnswer, streamingAnswer, isStreaming, ragContext, isEditing,
    selectedChannels, sent,
    pipelineHistory,
  } = state;

  // ─── setter 래퍼 (하위 컴포넌트 호환) ───
  const setCurrentStep = (v) => dispatch({ type: 'SET', payload: { currentStep: v } });
  const setConfidenceThreshold = (v) => dispatch({ type: 'SET', payload: { confidenceThreshold: v } });
  const setDragOverZone = (v) => dispatch({ type: 'SET', payload: { dragOverZone: v } });
  const setDraftAnswer = (v) => dispatch({ type: 'SET', payload: { draftAnswer: v } });
  const setIsEditing = (v) => dispatch({ type: 'SET', payload: { isEditing: v } });
  const setSelectedChannels = (fn) => dispatch({ type: 'SET_SELECTED_CHANNELS', payload: fn });
  const setSent = (v) => dispatch({ type: 'SET', payload: { sent: v } });

  // ─── 일괄 분류 ───
  const runBatchClassify = useCallback(async () => {
    dispatch({ type: 'SET', payload: {
      classifyLoading: true,
      classifyResults: [],
      selectedIdx: null,
      pipelineResult: null,
      selectedInquiry: null,
      autoIdxs: [],
      manualIdxs: [],
      checkedAuto: new Set(),
      batchAnswers: {},
    }});

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
      const autoList = [];
      const manualList = [];
      settled.forEach((item, i) => {
        const conf = item.result?.confidence || 0;
        if (conf >= state.confidenceThreshold) {
          autoList.push(i);
        } else {
          manualList.push(i);
        }
      });
      dispatch({ type: 'SET', payload: {
        classifyResults: settled,
        autoIdxs: autoList,
        manualIdxs: manualList,
        classifyLoading: false,
      }});
      dispatch({ type: 'COMPLETE_STEP', payload: [0] });
    } catch (e) {
      toast.error(`분류 오류: ${e.message || e}`);
      dispatch({ type: 'SET', payload: { classifyLoading: false } });
    }
  }, [apiCall, auth, state.confidenceThreshold]);

  // ─── DnD 핸들러 ───
  const handleDragStart = useCallback((e, idx) => {
    e.dataTransfer.setData('text/plain', String(idx));
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleDropToAuto = useCallback((e) => {
    e.preventDefault();
    const idx = Number(e.dataTransfer.getData('text/plain'));
    dispatch({ type: 'SET', payload: {
      manualIdxs: state.manualIdxs.filter(i => i !== idx),
      autoIdxs: state.autoIdxs.includes(idx) ? state.autoIdxs : [...state.autoIdxs, idx],
      dragOverZone: null,
      ...(state.selectedIdx === idx ? { selectedIdx: null, selectedInquiry: null, pipelineResult: null } : {}),
    }});
  }, [state.manualIdxs, state.autoIdxs, state.selectedIdx]);

  const handleDropToManual = useCallback((e) => {
    e.preventDefault();
    const idx = Number(e.dataTransfer.getData('text/plain'));
    const newChecked = new Set(state.checkedAuto);
    newChecked.delete(idx);
    dispatch({ type: 'SET', payload: {
      autoIdxs: state.autoIdxs.filter(i => i !== idx),
      checkedAuto: newChecked,
      manualIdxs: state.manualIdxs.includes(idx) ? state.manualIdxs : [...state.manualIdxs, idx],
      dragOverZone: null,
    }});
  }, [state.autoIdxs, state.manualIdxs, state.checkedAuto]);

  // ─── 체크박스 ───
  const toggleAutoCheck = useCallback((idx) => {
    dispatch({ type: 'SET', payload: { selectedIdx: null, selectedInquiry: null, pipelineResult: null } });
    dispatch({ type: 'SET_CHECKED_AUTO', payload: (prev) => {
      const s = new Set(prev);
      s.has(idx) ? s.delete(idx) : s.add(idx);
      return s;
    }});
  }, []);

  const toggleAllAuto = useCallback(() => {
    dispatch({ type: 'SET', payload: { selectedIdx: null, selectedInquiry: null, pipelineResult: null } });
    dispatch({ type: 'SET_CHECKED_AUTO', payload: (prev) =>
      prev.size === state.autoIdxs.length ? new Set() : new Set(state.autoIdxs)
    });
  }, [state.autoIdxs]);

  // ─── 일괄 자동 답변 생성 ───
  const generateBatchAnswers = useCallback(async (targetIdxs) => {
    if (!targetIdxs || targetIdxs.length === 0) return;

    const toGenerate = targetIdxs.filter(idx => !state.batchAnswers[idx] || state.checkedAuto.has(idx));
    if (toGenerate.length === 0) {
      toast('이미 모든 답변이 생성되었습니다. 재생성하려면 항목을 선택하세요.');
      return;
    }

    dispatch({ type: 'SET', payload: { batchLoading: true } });

    for (const idx of toGenerate) {
      const item = state.classifyResults[idx];
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
          onToken: () => {},
        });

        dispatch({ type: 'SET_BATCH_ANSWER', idx, value: fullText || '(답변 생성 실패)' });
        dispatch({ type: 'ADD_HISTORY', payload: {
          time: new Date().toLocaleTimeString('ko-KR'),
          text: item.text.slice(0, 40) + (item.text.length > 40 ? '...' : ''),
          category,
          routing: 'auto',
          priority: 'normal',
        }});
      } catch (e) {
        dispatch({ type: 'SET_BATCH_ANSWER', idx, value: `(오류: ${e.message})` });
      }
    }

    dispatch({ type: 'SET', payload: { batchLoading: false } });
    toast.success(`${toGenerate.length}건 자동 답변 생성 완료`);
  }, [state.classifyResults, auth, settings, state.batchAnswers, state.checkedAuto]);

  // ─── 문의 선택 → 풀 파이프라인 실행 ───
  const selectInquiry = useCallback(async (idx) => {
    const item = state.classifyResults[idx];
    if (!item) return;

    dispatch({ type: 'SET', payload: {
      selectedIdx: idx,
      selectedInquiry: { text: item.text, tier: item.tier, preferredChannels: item.preferredChannels },
      pipelineLoading: true,
      pipelineResult: null,
      draftAnswer: '',
      streamingAnswer: '',
      selectedChannels: new Set(),
      sent: false,
    }});

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
          confidence_threshold: state.confidenceThreshold,
        },
      });

      if (res?.status === 'success' || res?.steps) {
        dispatch({ type: 'SET', payload: { pipelineResult: res, currentStep: 1 } });
        dispatch({ type: 'COMPLETE_STEP', payload: [0, 1] });
        dispatch({ type: 'ADD_HISTORY', payload: {
          time: new Date().toLocaleTimeString('ko-KR'),
          text: item.text.slice(0, 40) + (item.text.length > 40 ? '...' : ''),
          category: res.steps?.classify?.predicted_category || '?',
          routing: res.steps?.review?.routing || '?',
          priority: res.steps?.review?.priority?.predicted_priority || '?',
        }});
      } else {
        toast.error('파이프라인 실행 실패');
      }
    } catch (e) {
      toast.error(`오류: ${e.message || e}`);
    } finally {
      dispatch({ type: 'SET', payload: { pipelineLoading: false } });
    }
  }, [state.classifyResults, state.confidenceThreshold, apiCall, auth]);

  // ─── 답변 생성 ───
  const generateAnswer = useCallback(async () => {
    if (!state.pipelineResult || !state.selectedInquiry) return;

    dispatch({ type: 'SET', payload: {
      isStreaming: true,
      streamingAnswer: '',
      draftAnswer: '',
      ragContext: null,
    }});

    const category = state.pipelineResult.steps?.classify?.predicted_category || '기타';

    try {
      const controller = new AbortController();
      abortRef.current = controller;

      const resp = await fetch('/api/cs/pipeline-answer', {
        method: 'POST',
        headers: createAuthHeaders(auth, { Accept: 'text/event-stream' }),
        body: JSON.stringify({
          inquiry_text: state.selectedInquiry.text,
          inquiry_category: category,
          seller_tier: state.selectedInquiry.tier,
          order_id: null,
          rag_mode: settings?.ragMode || 'rag',
          apiKey: settings?.apiKey || '',
        }),
        signal: controller.signal,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const fullText = await parseSSEStream(resp, {
        onToken: (_, accumulated) => {
          dispatch({ type: 'SET', payload: { streamingAnswer: accumulated } });
        },
        onDone: (_, accumulated) => {
          dispatch({ type: 'SET', payload: { draftAnswer: accumulated, isStreaming: false } });
        },
        onError: (data) => {
          toast.error(`답변 생성 오류: ${data}`);
          dispatch({ type: 'SET', payload: { isStreaming: false } });
        },
        onRagContext: (data) => {
          dispatch({ type: 'SET', payload: { ragContext: data } });
        },
      });

      if (!fullText && !state.draftAnswer) {
        dispatch({ type: 'SET', payload: { isStreaming: false } });
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        toast.error(`스트리밍 오류: ${e.message}`);
      }
      dispatch({ type: 'SET', payload: { isStreaming: false } });
    }
  }, [state.pipelineResult, state.selectedInquiry, settings, auth, state.draftAnswer]);

  // cleanup abort on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  // ─── 자동 답변 개별 수정 ───
  const updateBatchAnswer = useCallback((idx, newText) => {
    dispatch({ type: 'SET_BATCH_ANSWER', idx, value: newText });
  }, []);

  // ─── 초기화 ───
  const resetPipeline = useCallback(() => {
    dispatch({ type: 'FULL_RESET' });
    if (abortRef.current) abortRef.current.abort();
  }, []);

  // ─── 단계 이동 ───
  const goNext = () => {
    if (currentStep < 4) {
      const isAutoMode = autoIdxs.length > 0 && !pipelineResult;
      if (currentStep === 0 && isAutoMode) {
        dispatch({ type: 'COMPLETE_STEP', payload: [0, 1] });
        dispatch({ type: 'SET', payload: { currentStep: 2 } });
      } else {
        dispatch({ type: 'COMPLETE_STEP', payload: [currentStep] });
        dispatch({ type: 'SET', payload: { currentStep: currentStep + 1 } });
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
