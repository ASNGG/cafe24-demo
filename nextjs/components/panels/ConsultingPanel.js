// components/panels/ConsultingPanel.js
// 셀러 컨설팅 에이전트 — 4단계 대화형 워크플로우 (진단 → 전략 → 계획 → 실행)

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfmPlugin from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import {
  Loader2, Send, RotateCcw, Search, CheckCircle2, ChevronDown, ChevronUp,
  Bot, Briefcase, Target, ClipboardList, Rocket, Copy, ArrowLeft,
} from 'lucide-react';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import useConsultingStream, { CONSULTING_STEPS } from './hooks/useConsultingStream';

// 모듈 레벨 상수: remarkPlugins 배열 재생성 방지
const REMARK_PLUGINS = [remarkGfmPlugin];

// 단계별 아이콘 매핑
const STEP_ICONS = {
  diagnosis: Search,
  strategy: Target,
  plan: ClipboardList,
  execution: Rocket,
};

// 마크다운 컴포넌트 커스터마이징 (모듈 레벨 — 리렌더 방지)
const MD_COMPONENTS = {
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto -mx-1 my-2">
      <table className="w-full border-collapse" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-blue-50" {...props} />,
  th: ({ node, ...props }) => (
    <th className="border-2 border-blue-200/30 px-3 py-2 text-left text-xs font-extrabold text-gray-800" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border border-blue-100/30 px-3 py-2 align-top text-xs text-gray-700 whitespace-nowrap" {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre className="overflow-x-auto rounded-xl bg-gray-50 p-3 text-xs text-gray-800" {...props} />
  ),
  code: ({ node, inline, className, children, ...props }) => {
    if (inline) {
      return (
        <code className="rounded bg-blue-50 px-1 py-0.5 text-[11px] text-blue-800" {...props}>
          {children}
        </code>
      );
    }
    return <code className={className} {...props}>{children}</code>;
  },
  a: ({ node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer" className="font-extrabold text-blue-600 underline underline-offset-2 hover:text-blue-800" />
  ),
};

const MarkdownContent = React.memo(function MarkdownContent({ content }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={MD_COMPONENTS}>
      {content || ''}
    </ReactMarkdown>
  );
});

// 타이핑 인디케이터
function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.2s]" />
      <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.1s]" />
      <span className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" />
      <span className="ml-2 text-xs text-gray-500">컨설팅 에이전트 분석 중...</span>
    </div>
  );
}

// 단계 진행 바
const StepProgressBar = React.memo(function StepProgressBar({ stepHistory, currentStep, onRollback }) {
  return (
    <div className="flex items-center gap-1 w-full overflow-x-auto pb-1">
      {CONSULTING_STEPS.map((step, idx) => {
        const historyEntry = stepHistory.find((h) => h.step === step.key);
        const isCompleted = historyEntry?.status === 'completed';
        const isActive = historyEntry?.status === 'active' || currentStep === step.key;
        const isPending = !historyEntry;
        const Icon = STEP_ICONS[step.key] || Search;
        const canRollback = isCompleted;

        return (
          <React.Fragment key={step.key}>
            {idx > 0 && (
              <div className={`hidden sm:block h-0.5 flex-1 min-w-[16px] max-w-[40px] transition-colors duration-300 ${
                isCompleted ? 'bg-green-400' : isActive ? 'bg-blue-300' : 'bg-gray-200'
              }`} />
            )}
            <motion.button
              type="button"
              disabled={!canRollback}
              onClick={() => canRollback && onRollback(step.key)}
              className={`flex items-center gap-1.5 rounded-xl px-2.5 py-2 text-xs font-bold transition-all duration-200 shrink-0 ${
                isCompleted
                  ? 'bg-green-50 text-green-700 border border-green-200 hover:bg-green-100 cursor-pointer'
                  : isActive
                  ? 'bg-blue-500 text-white shadow-md shadow-blue-200'
                  : 'bg-gray-100 text-gray-400 border border-gray-200 cursor-default'
              }`}
              whileTap={canRollback ? { scale: 0.95 } : {}}
              title={canRollback ? `${step.label} 단계로 롤백` : ''}
            >
              {isCompleted ? (
                <CheckCircle2 size={14} />
              ) : isActive ? (
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <Icon size={14} />
                </motion.div>
              ) : (
                <Icon size={14} />
              )}
              <span className="hidden sm:inline">{step.number}. {step.label}</span>
              <span className="sm:hidden">{step.number}</span>
            </motion.button>
          </React.Fragment>
        );
      })}
    </div>
  );
});

// 도구 호출 표시
const ToolCallItem = React.memo(function ToolCallItem({ tool }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
      className="flex items-center gap-2 rounded-lg border border-blue-100 bg-blue-50/50 px-2.5 py-1.5 text-xs"
    >
      {tool.status === 'running' ? (
        <Loader2 size={13} className="animate-spin text-blue-500" />
      ) : (
        <CheckCircle2 size={13} className="text-green-500" />
      )}
      <span className="font-bold text-gray-800">{tool.tool}</span>
      {tool.args && Object.keys(tool.args).length > 0 && (
        <span className="text-gray-400 text-[10px]">
          ({Object.entries(tool.args).map(([k, v]) => `${k}=${v}`).join(', ')})
        </span>
      )}
      {tool.result_preview && (
        <span className="ml-auto text-gray-400 truncate max-w-[180px] text-[10px]">{tool.result_preview}</span>
      )}
    </motion.div>
  );
});

// 옵션 버튼 영역
function OptionButtons({ awaitingInput, onSelect, isLoading }) {
  if (!awaitingInput) return null;

  const { prompt, options, step } = awaitingInput;
  const isStrategy = step === 'strategy' || step === 'diagnosis';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border-2 border-blue-200 bg-blue-50/50 p-4 space-y-3"
    >
      {prompt && (
        <p className="text-sm font-bold text-gray-800">{prompt}</p>
      )}
      <div className="flex flex-wrap gap-2">
        {options.map((opt, idx) => {
          // 전략 선택 옵션은 파란색 강조
          const isPrimary = opt.includes('다음') || opt.includes('진행') || opt.includes('최적화') || opt.includes('통합');
          return (
            <button
              key={idx}
              type="button"
              disabled={isLoading}
              onClick={() => onSelect(opt)}
              className={`rounded-lg px-4 py-2.5 text-sm font-bold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed ${
                isPrimary
                  ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-md shadow-blue-200'
                  : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200 shadow-sm'
              }`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </motion.div>
  );
}

// 상태 뱃지
function StatusBadge({ isLoading, currentStep, sessionId, activeAgent }) {
  if (activeAgent) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-1 text-xs font-bold text-blue-700 border border-blue-200">
        <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
        {activeAgent}
      </span>
    );
  }
  if (isLoading) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-1 text-xs font-bold text-blue-600 border border-blue-200">
        <Loader2 size={12} className="animate-spin" />
        처리 중
      </span>
    );
  }
  if (currentStep) {
    const stepInfo = CONSULTING_STEPS.find((s) => s.key === currentStep);
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-50 px-2.5 py-1 text-xs font-bold text-green-700 border border-green-200">
        <CheckCircle2 size={12} />
        {stepInfo?.label || currentStep}
      </span>
    );
  }
  if (sessionId) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-1 text-xs font-bold text-gray-600">
        세션 활성
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-1 text-xs font-bold text-gray-500">
      대기
    </span>
  );
}

export default function ConsultingPanel({ auth, addLog, settings, apiCall }) {
  const [sellerId, setSellerId] = useState('SEL0001');
  const [input, setInput] = useState('');
  const [started, setStarted] = useState(false);
  const chatBoxRef = useRef(null);
  const prevLengthRef = useRef(0);

  const {
    messages,
    setMessages,
    isLoading,
    error,
    currentStep,
    stepHistory,
    awaitingInput,
    sessionId,
    toolCalls,
    activeAgent,
    sendMessage,
    stopStream,
    resetSession,
    rollbackTo,
  } = useConsultingStream({ auth, settings });

  const canSend = useMemo(() => !!input?.trim() && !isLoading, [input, isLoading]);

  // 컨설팅 시작
  const handleStart = useCallback(() => {
    if (!sellerId?.trim()) {
      toast.error('셀러 ID를 입력하세요');
      return;
    }
    setStarted(true);
    addLog?.('컨설팅', `${sellerId} 컨설팅 시작`);
    sendMessage('컨설팅 시작', { sellerId: sellerId.trim(), action: 'message' });
  }, [sellerId, sendMessage, addLog]);

  // 텍스트 메시지 전송
  const handleSend = useCallback(() => {
    const q = input.trim();
    if (!q) return;
    sendMessage(q, { sellerId, action: 'message' });
    setInput('');
    addLog?.('컨설팅', q.slice(0, 30));
  }, [input, sellerId, sendMessage, addLog]);

  // 옵션 선택
  const handleOptionSelect = useCallback((option) => {
    // 전략 방향 선택
    const strategyDirections = ['마케팅 강화', '리텐션(이탈방지)', '둘 다', '마케팅 최적화', '이탈 방지', '통합 전략'];
    const isStrategyChoice = strategyDirections.some((kw) => option.includes(kw));

    // "다음 단계로" 계열
    const isAdvance = option.includes('다음 단계') || option === '다음';

    // "승인" 계열
    const isConfirm = option === '승인' || option === '확인';

    // "이전 단계" / 롤백
    const isRollback = option.includes('이전 단계') || option.includes('다시') || option.includes('롤백');

    if (isAdvance) {
      sendMessage('다음', { sellerId, action: 'advance' });
    } else if (isConfirm) {
      sendMessage(option, { sellerId, action: 'message' });
    } else if (isStrategyChoice) {
      sendMessage(option, {
        sellerId,
        action: 'strategy_choice',
        strategyChoice: option,
      });
    } else if (isRollback) {
      const prevStep = stepHistory.length > 1
        ? stepHistory[stepHistory.length - 2]?.step
        : stepHistory[0]?.step;
      if (prevStep) {
        rollbackTo(prevStep);
      }
    } else {
      // 일반 옵션 (매출 분석 상세, 전략 수정 등) → 자유 대화로 전송
      sendMessage(option, { sellerId, action: 'message' });
    }
    addLog?.('컨설팅', `옵션 선택: ${option}`);
  }, [sellerId, sendMessage, rollbackTo, stepHistory, addLog]);

  // 새 컨설팅
  const handleReset = useCallback(() => {
    resetSession();
    setStarted(false);
    setSellerId('SEL0001');
    setInput('');
    addLog?.('컨설팅', '세션 초기화');
  }, [resetSession, addLog]);

  // 롤백 핸들러
  const handleRollback = useCallback((step) => {
    rollbackTo(step);
    addLog?.('컨설팅', `${step} 단계로 롤백`);
  }, [rollbackTo, addLog]);

  // 자동 스크롤
  const shouldAutoScrollRef = useRef(true);

  const updateAutoScrollFlag = useCallback(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom <= 80;
  }, []);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    el.addEventListener('scroll', updateAutoScrollFlag, { passive: true });
    return () => el.removeEventListener('scroll', updateAutoScrollFlag);
  }, [updateAutoScrollFlag]);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el || !shouldAutoScrollRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isLoading, toolCalls, awaitingInput]);

  useEffect(() => {
    prevLengthRef.current = messages?.length || 0;
  }, [messages]);

  // 활성 도구 호출 (현재 실행 중인 것만)
  const activeTools = useMemo(
    () => toolCalls.filter((t) => t.status === 'running'),
    [toolCalls]
  );

  return (
    <div className="space-y-4">
      <SectionHeader
        title="셀러 컨설팅 에이전트"
        subtitle="AI 기반 4단계 맞춤 컨설팅 (진단 - 전략 - 계획 - 실행)"
        right={
          <StatusBadge
            isLoading={isLoading}
            currentStep={currentStep}
            sessionId={sessionId}
            activeAgent={activeAgent}
          />
        }
      />

      <div className="grid grid-cols-12 gap-4">
        {/* 메인 영역 */}
        <div className="col-span-12 xl:col-span-9">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100">
            {/* 셀러 ID 입력 + 시작 */}
            {!started ? (
              <div className="p-6">
                <div className="max-w-lg mx-auto space-y-6">
                  <div className="text-center">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-50 mb-4">
                      <Briefcase size={28} className="text-blue-600" />
                    </div>
                    <h3 className="text-lg font-bold text-gray-800">셀러 컨설팅 시작</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      셀러 ID를 입력하고 AI 컨설팅을 시작하세요
                    </p>
                  </div>

                  <div className="space-y-3">
                    <label className="block text-sm font-bold text-gray-700">셀러 ID</label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input
                          type="text"
                          value={sellerId}
                          onChange={(e) => setSellerId(e.target.value.toUpperCase())}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleStart();
                          }}
                          placeholder="SEL0001"
                          className="w-full pl-10 pr-4 py-3 rounded-xl border-2 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 outline-none text-sm font-mono transition-all"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={handleStart}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-blue-200 active:scale-[0.98] flex items-center gap-2"
                      >
                        <Rocket size={16} />
                        컨설팅 시작
                      </button>
                    </div>
                  </div>

                  <div className="rounded-xl bg-gray-50 p-4 text-xs text-gray-500 space-y-2">
                    <div className="font-bold text-gray-600 mb-2">컨설팅 프로세스</div>
                    {CONSULTING_STEPS.map((s) => {
                      const Icon = STEP_ICONS[s.key];
                      return (
                        <div key={s.key} className="flex items-center gap-2">
                          <div className="w-6 h-6 rounded-lg bg-blue-50 flex items-center justify-center">
                            <Icon size={12} className="text-blue-500" />
                          </div>
                          <span className="font-bold text-gray-700">{s.number}단계</span>
                          <span>{s.label}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 space-y-3">
                {/* 셀러 정보 + 단계 진행 바 */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="inline-flex items-center gap-1 rounded-lg bg-blue-50 px-2.5 py-1.5 text-xs font-bold text-blue-700 border border-blue-200">
                      <Bot size={12} />
                      {sellerId}
                    </span>
                    {sessionId && (
                      <span className="text-[10px] text-gray-400 font-mono hidden sm:inline">
                        #{sessionId.slice(0, 8)}
                      </span>
                    )}
                  </div>
                  <div className="flex-1 w-full">
                    <StepProgressBar
                      stepHistory={stepHistory}
                      currentStep={currentStep}
                      onRollback={handleRollback}
                    />
                  </div>
                </div>

                {/* 대화 영역 */}
                <div
                  ref={chatBoxRef}
                  className="max-h-[50vh] md:max-h-[55vh] overflow-auto pr-1 space-y-3"
                >
                  {(messages || []).map((m, idx) => {
                    const isUser = m.role === 'user';
                    const isPending = !!m?._pending;
                    const msgKey = m?._id || idx;
                    const isNew = idx >= prevLengthRef.current;
                    const isAction = !!m?._isAction;

                    return (
                      <motion.div
                        key={msgKey}
                        initial={isNew ? { opacity: 0, x: isUser ? 20 : -20 } : false}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ type: 'spring', damping: 20, stiffness: 300 }}
                        className={`group relative ${isUser ? 'flex justify-end' : 'flex justify-start'}`}
                      >
                        <div
                          className={
                            isUser
                              ? isAction
                                ? 'max-w-[75%] rounded-xl bg-blue-100 border border-blue-200 px-3 py-2 text-sm text-blue-700 font-bold'
                                : 'max-w-[75%] rounded-xl bg-blue-600 text-white px-4 py-3 shadow-sm'
                              : 'w-full md:max-w-[85%] rounded-xl bg-gray-50 border border-gray-100 px-4 py-3 shadow-sm'
                          }
                        >
                          {/* 메시지 헤더 */}
                          <div className="text-[11px] font-extrabold mb-1.5 flex items-center justify-between">
                            <span className="flex items-center gap-1.5">
                              {isUser ? (
                                <span className={isAction ? 'text-blue-600' : 'text-blue-100'}>
                                  {auth?.username || 'USER'}
                                </span>
                              ) : (
                                <span className="text-gray-500 flex items-center gap-1">
                                  <Briefcase size={11} className="text-blue-500" />
                                  컨설팅 에이전트
                                  {isPending && activeAgent && (
                                    <span className="ml-1 rounded bg-blue-100 px-1.5 py-0.5 text-[9px] font-bold text-blue-600">
                                      {activeAgent}
                                    </span>
                                  )}
                                </span>
                              )}
                            </span>
                            {!isUser && isPending && (
                              <span className="inline-flex items-center gap-1.5 text-blue-500">
                                <span className="h-3 w-3 rounded-full border-2 border-blue-200 border-t-blue-500 animate-spin" />
                                <span className="text-[10px]">streaming</span>
                              </span>
                            )}
                          </div>

                          {/* 도구 호출 (AI 메시지 내부, pending일 때) */}
                          {!isUser && isPending && activeTools.length > 0 && (
                            <div className="mb-2 space-y-1">
                              {activeTools.map((te, ti) => (
                                <ToolCallItem key={ti} tool={te} />
                              ))}
                            </div>
                          )}

                          {/* 메시지 본문 */}
                          <div className={isUser ? 'text-sm' : 'prose prose-sm max-w-none'}>
                            {!isUser && isPending && !m.content?.trim() ? (
                              <TypingDots />
                            ) : isUser ? (
                              <span>{m.content}</span>
                            ) : (
                              <MarkdownContent content={m.content || ''} />
                            )}
                          </div>

                          {/* 호버 복사 버튼 */}
                          {!isPending && !isUser && (
                            <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(m.content || '');
                                  toast.success('복사되었습니다');
                                }}
                                className="p-1.5 rounded-lg bg-white border border-gray-200 text-gray-400 hover:text-gray-700 hover:bg-gray-50 transition shadow-sm"
                                title="복사"
                              >
                                <Copy size={13} />
                              </button>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    );
                  })}

                  {/* 빈 상태 */}
                  {!messages?.length && (
                    <div className="py-8 text-center text-sm text-gray-400">
                      컨설팅이 곧 시작됩니다...
                    </div>
                  )}
                </div>

                {/* 옵션 버튼 영역 */}
                <AnimatePresence>
                  {awaitingInput && (
                    <OptionButtons
                      awaitingInput={awaitingInput}
                      onSelect={handleOptionSelect}
                      isLoading={isLoading}
                    />
                  )}
                </AnimatePresence>

                {/* 에러 표시 */}
                {error && (
                  <div className="rounded-xl bg-red-50 border border-red-200 p-3 text-xs text-red-700">
                    {error}
                  </div>
                )}

                {/* 텍스트 입력 영역 */}
                <div className="flex gap-2">
                  <input
                    className="flex-1 px-4 py-2.5 rounded-xl border-2 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 outline-none text-sm transition-all"
                    placeholder="추가 요청이나 질문을 입력하세요 (Enter로 전송)"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && canSend) handleSend();
                    }}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={handleSend}
                    disabled={!canSend}
                    className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold text-sm transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shrink-0"
                  >
                    {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                    전송
                  </button>
                  {isLoading && (
                    <button
                      type="button"
                      onClick={() => {
                        stopStream();
                        toast('중단됨');
                      }}
                      className="px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-bold text-sm transition-all border border-gray-200 shrink-0"
                    >
                      중단
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 사이드바 */}
        <div className="col-span-12 xl:col-span-3 space-y-4">
          {/* 세션 정보 카드 */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
            <div className="text-sm font-bold text-gray-800 mb-3">세션 정보</div>
            <div className="space-y-2 text-xs text-gray-600">
              <div className="flex justify-between">
                <span className="text-gray-400">셀러 ID</span>
                <span className="font-mono font-bold">{started ? sellerId : '-'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">현재 단계</span>
                <span className="font-bold">
                  {currentStep
                    ? CONSULTING_STEPS.find((s) => s.key === currentStep)?.label || currentStep
                    : '-'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">세션 ID</span>
                <span className="font-mono text-[10px]">
                  {sessionId ? `#${sessionId.slice(0, 12)}` : '-'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">메시지</span>
                <span className="font-bold">{messages?.length || 0}개</span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleReset}
              className="mt-4 w-full rounded-xl border-2 border-blue-200/50 bg-blue-50 px-4 py-2.5 text-sm font-extrabold text-gray-700 transition hover:bg-blue-100 active:translate-y-[1px] flex items-center justify-center gap-2"
            >
              <RotateCcw size={14} />
              새 컨설팅
            </button>
          </div>

          {/* 컨설팅 안내 카드 */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
            <div className="text-sm font-bold text-gray-800 mb-3">컨설팅 안내</div>
            <div className="text-xs text-gray-600 space-y-3">
              <p>
                AI가 셀러 데이터를 분석하여 맞춤 컨설팅을 진행합니다. 각 단계에서 AI의 제안을 검토하고 방향을 선택할 수 있습니다.
              </p>

              <div className="rounded-xl bg-gray-50 p-3 space-y-2">
                <div className="font-bold text-gray-700">4단계 프로세스</div>
                {CONSULTING_STEPS.map((s) => {
                  const Icon = STEP_ICONS[s.key];
                  return (
                    <div key={s.key} className="flex items-start gap-2">
                      <div className="w-5 h-5 rounded-md bg-blue-50 flex items-center justify-center shrink-0 mt-0.5">
                        <Icon size={11} className="text-blue-500" />
                      </div>
                      <div>
                        <span className="font-bold text-gray-700">{s.label}</span>
                        <span className="text-gray-400 ml-1">
                          {s.key === 'diagnosis' && '- 셀러 현황 종합 분석'}
                          {s.key === 'strategy' && '- 맞춤 전략 수립'}
                          {s.key === 'plan' && '- 구체적 실행 계획'}
                          {s.key === 'execution' && '- 자동화 실행'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="rounded-xl bg-blue-50 border border-blue-100 p-3">
                <div className="font-bold text-blue-700 mb-1">롤백 기능</div>
                <p className="text-blue-600">
                  완료된 단계를 클릭하면 해당 단계로 되돌아갈 수 있습니다. 전략을 수정하고 싶을 때 활용하세요.
                </p>
              </div>
            </div>
          </div>

          {/* 도구 실행 이력 */}
          {toolCalls.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
              <div className="text-sm font-bold text-gray-800 mb-3">도구 실행 이력</div>
              <div className="space-y-1.5 max-h-[30vh] overflow-auto">
                {toolCalls.map((tc, idx) => (
                  <div
                    key={idx}
                    className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs ${
                      tc.status === 'running'
                        ? 'bg-blue-50 border border-blue-100'
                        : 'bg-gray-50 border border-gray-100'
                    }`}
                  >
                    {tc.status === 'running' ? (
                      <Loader2 size={11} className="animate-spin text-blue-500 shrink-0" />
                    ) : (
                      <CheckCircle2 size={11} className="text-green-500 shrink-0" />
                    )}
                    <span className="font-bold text-gray-700 truncate">{tc.tool}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* LLM 설정 요약 */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
            <div className="text-sm font-bold text-gray-800 mb-2">LLM 설정</div>
            <div className="text-xs text-gray-500 space-y-1">
              <div className="flex justify-between">
                <span>모델</span>
                <span className="font-mono font-bold text-gray-700">{settings?.selectedModel || 'gpt-4o-mini'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
