// components/panels/MultiAgentPanel.js
// CAFE24 AI 운영 플랫폼 - 멀티에이전트 패널 (Supervisor 패턴)

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfmPlugin from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import { Loader2, Zap, ChevronDown, ChevronUp, Copy, FlaskConical, RefreshCcw, RotateCcw, CheckCircle2, XCircle, Bot, Search, Target, ClipboardList, Rocket, Route } from 'lucide-react';
import useMultiAgentStream from './hooks/useMultiAgentStream';
import ToolExplorer from '@/components/common/ToolExplorer';
import { cafe24Btn, cafe24BtnInline, cafe24BtnSecondaryInline, cafe24BtnSecondary } from '@/components/common/buttonStyles';
import { MULTI_AGENT_WORKERS } from '@/components/automation/constants';

// 모듈 레벨 상수: remarkPlugins 배열 재생성 방지
const MULTI_REMARK_PLUGINS = [remarkGfmPlugin];

// 에이전트 뱃지 바: 활성 워커 에이전트들을 카드/뱃지로 표시
const AgentBadgeBar = React.memo(function AgentBadgeBar({ agentHistory, activeAgent, stepTimings }) {
  if (!agentHistory?.length) return null;
  return (
    <div className="rounded-2xl border border-cafe24-orange/15 bg-white/90 p-3 backdrop-blur">
      <div className="text-[10px] font-extrabold text-cafe24-brown/50 mb-2">워커 에이전트</div>
      <div className="flex flex-wrap gap-2">
        <AnimatePresence>
          {agentHistory.map((ah) => {
            const worker = MULTI_AGENT_WORKERS[ah.agent];
            const label = worker?.label || ah.agent;
            const Icon = worker?.icon || Bot;
            const isActive = ah.agent === activeAgent;
            const isDone = ah.status === 'done';
            const elapsed = stepTimings[ah.agent];

            return (
              <motion.div
                key={ah.agent}
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                transition={{ duration: 0.2 }}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-xs font-bold transition-all duration-200 hover:scale-105 ${
                  isActive
                    ? 'bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white shadow-md shadow-cafe24-orange/25 animate-pulse'
                    : isDone
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100'
                    : 'bg-gray-100 text-gray-500 border border-gray-200'
                }`}
              >
                <Icon size={13} />
                <span>{label}</span>
                {isDone && <CheckCircle2 size={11} className="text-emerald-500" />}
                {isDone && elapsed && (
                  <span className="text-[10px] font-normal opacity-70">
                    {(elapsed / 1000).toFixed(1)}s
                  </span>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
});

const StepResultCard = React.memo(function StepResultCard({ stepNum, result, agentName }) {
  const [open, setOpen] = useState(true);

  return (
    <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 shadow-sm backdrop-blur hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
      <button
        type="button"
        className="w-full flex items-center justify-between p-3 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <FlaskConical size={14} className="text-cafe24-orange" />
          <span className="text-xs font-extrabold text-cafe24-brown">
            Step {stepNum}{agentName ? ` - ${agentName}` : ''}
          </span>
        </div>
        {open ? (
          <ChevronUp size={14} className="text-cafe24-brown/50 transition-transform duration-300" />
        ) : (
          <ChevronDown size={14} className="text-cafe24-brown/50 transition-transform duration-300" />
        )}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3">
              <div className="prose prose-sm max-w-none text-cafe24-brown text-xs">
                <ReactMarkdown remarkPlugins={MULTI_REMARK_PLUGINS}>{result || ''}</ReactMarkdown>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.2s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.1s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce" />
      <span className="ml-2 text-xs text-cafe24-brown/60">AI 에이전트 분석 중...</span>
    </div>
  );
}

// 에이전트 전환 인라인 표시
const AgentTransitionMarker = React.memo(function AgentTransitionMarker({ agent }) {
  const worker = MULTI_AGENT_WORKERS[agent];
  const label = worker?.label || agent;
  return (
    <div className="flex items-center justify-center gap-2 py-1.5">
      <div className="h-px flex-1 bg-cafe24-orange/15" />
      <span className="inline-flex items-center gap-1 rounded-full bg-cafe24-yellow/15 px-2.5 py-0.5 text-[10px] font-bold text-cafe24-brown/60">
        <Bot size={10} className="text-cafe24-orange" />
        {label} Agent 활성
      </span>
      <div className="h-px flex-1 bg-cafe24-orange/15" />
    </div>
  );
});

// 모듈 레벨 상수: ReactMarkdown components 객체 리렌더 시 재생성 방지
const MULTI_MARKDOWN_COMPONENTS = {
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto -mx-1 my-2">
      <table className="w-full border-collapse" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-cafe24-yellow/20" {...props} />,
  th: ({ node, ...props }) => (
    <th className="border-2 border-cafe24-orange/20 px-3 py-2 text-left text-xs font-extrabold text-cafe24-brown" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border border-cafe24-orange/15 px-3 py-2 align-top text-xs text-cafe24-brown whitespace-nowrap" {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre className="overflow-x-auto rounded-xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown" {...props} />
  ),
  code: ({ node, inline, className, children, ...props }) => {
    if (inline) {
      return (
        <code className="rounded bg-cafe24-yellow/20 px-1 py-0.5 text-[11px] text-cafe24-brown" {...props}>
          {children}
        </code>
      );
    }
    return <code className={className} {...props}>{children}</code>;
  },
  a: ({ node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer" className="font-extrabold text-cafe24-orange underline underline-offset-2 hover:text-cafe24-brown" />
  ),
};

// Supervisor 워커 agent 이름 → 한글 라벨 (MULTI_AGENT_WORKERS에서 가져옴)

const MarkdownMessage = React.memo(function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={MULTI_REMARK_PLUGINS}
      components={MULTI_MARKDOWN_COMPONENTS}
    >
      {content || ''}
    </ReactMarkdown>
  );
});

const ToolCalls = React.memo(function ToolCalls({ toolCalls }) {
  const [open, setOpen] = useState(false);
  if (!toolCalls?.length) return null;
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-[10px] font-extrabold text-cafe24-brown/50 hover:text-cafe24-brown transition-colors"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        도구 실행 결과 ({toolCalls.length}개)
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-2">
              {toolCalls.map((tc, idx) => {
                const ok = tc?.result?.status === 'success';
                return (
                  <div key={idx} className="rounded-xl border border-cafe24-orange/15 bg-white/80 p-3 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        {ok ? <CheckCircle2 size={12} className="text-emerald-500" /> : <XCircle size={12} className="text-red-500" />}
                        <span className="font-bold text-xs text-cafe24-brown">{tc.tool}</span>
                      </div>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${ok ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-600'}`}>
                        {ok ? '성공' : '실패'}
                      </span>
                    </div>
                    <pre className="mt-2 overflow-auto rounded-lg bg-cafe24-yellow/5 p-2.5 text-[10px] text-cafe24-brown/80 font-mono max-h-[200px]">
                      {JSON.stringify(tc.result, null, 2)}
                    </pre>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});


function Chip({ label, onClick, disabled }) {
  return (
    <button
      className={`inline-flex items-center gap-2 rounded-full border-2 px-3 py-1.5 text-xs font-extrabold transition-all duration-200 whitespace-nowrap ${disabled ? 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed' : 'border-cafe24-orange/20 bg-white/80 text-cafe24-brown hover:bg-cafe24-yellow/20 hover:border-cafe24-orange/40 hover:shadow-md hover:scale-[1.02] hover:-translate-y-0.5 active:scale-[0.98]'}`}
      onClick={disabled ? undefined : onClick}
      data-tooltip={disabled ? '개발중 (비활성화)' : '클릭하면 질문이 바로 전송됩니다'}
      type="button"
      disabled={disabled}
    >
      <FlaskConical size={14} className={disabled ? 'text-gray-300' : 'text-cafe24-orange'} />
      <span>{label}</span>
    </button>
  );
}

function PipelineStatusBadge({ pipelineStatus, messageCount, activeAgent }) {
  if (activeAgent) {
    const worker = MULTI_AGENT_WORKERS[activeAgent];
    const label = worker?.label || activeAgent;
    return (
      <span className="badge inline-flex items-center gap-1">
        <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-pulse" />
        {label}
      </span>
    );
  }
  switch (pipelineStatus) {
    case 'running':
      return <span className="badge">실행 중</span>;
    case 'done':
      return <span className="badge">완료</span>;
    case 'error':
      return <span className="badge badge-danger">오류</span>;
    default:
      return <span className="badge">{messageCount > 0 ? `메시지 ${messageCount}` : '대기'}</span>;
  }
}

export default function MultiAgentPanel({ auth, selectedShop, addLog, settings, apiCall }) {
  const [input, setInput] = useState('');
  const [quickResult, setQuickResult] = useState(null);
  const chatBoxRef = useRef(null);
  const scrollRef = useRef(null);
  const prevLengthRef = useRef(0);
  const [collapsedMsgs, setCollapsedMsgs] = useState(new Set());
  const toggleCollapse = useCallback((idx) => {
    setCollapsedMsgs((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  const {
    messages,
    setMessages,
    isLoading,
    error,
    steps,
    currentStep,
    stepResults,
    pipelineStatus,
    sendMessage,
    stopStream,
    resetPipeline,
    toolExecutions,
    stepTimings,
    stepProgress,
    activeAgent,
    agentHistory,
    consultingMode,
    consultingStep,
    consultingStepHistory,
    consultingAwaitingInput,
    consultingSessionId,
  } = useMultiAgentStream({ auth, selectedShop, settings });

  const canSend = useMemo(() => !!input?.trim() && !isLoading, [input, isLoading]);

  const handleSend = useCallback(
    (q) => {
      const query = q || input;
      if (!query?.trim()) return;
      sendMessage(query);
      setInput('');
      addLog?.('AI에이전트', query);
    },
    [input, sendMessage, addLog]
  );

  const chips = useMemo(
    () => {
      const shopId = selectedShop || 'S0001';
      return [
        { label: 'SEL0001 이탈 위험 분석하고 리텐션 전략 실행해줘' },
        { label: 'SEL0001 셀러 종합 진단하고 이탈 위험도 분석해줘' },
        { label: 'SEL0001 이상거래 조사하고 CS 품질 점검해줘' },
        { label: 'CS 품질 통계 분석하고 전체 운영 현황 대시보드 요약해줘' },
        { label: '고위험 이탈 셀러 조회하고 세그먼트별 분포 분석해줘' },
        { label: `${shopId} 쇼핑몰 매출 성과 분석하고 코호트 리텐션 보여줘` },
        { label: 'SEL0001 셀러 활동 분석하고 마케팅 예산 최적화 돌려줘' },
        { label: 'SEL0001 컨설팅 해줘' },
      ];
    },
    [selectedShop]
  );

  // cafe24_send_question 이벤트 리스너
  const sendRef = useRef(handleSend);
  sendRef.current = handleSend;
  useEffect(() => {
    function handler(ev) {
      const q = ev?.detail?.q;
      if (q) sendRef.current(q);
    }
    window.addEventListener('cafe24_send_question', handler);
    return () => window.removeEventListener('cafe24_send_question', handler);
  }, []);

  // 빠른 분석 API 호출
  async function runQuick(endpoint) {
    setQuickResult(null);
    const res = await apiCall({
      endpoint,
      method: 'GET',
      auth,
      timeoutMs: 60000,
    });
    setQuickResult(res);
    addLog?.('빠른분석', endpoint);
  }

  // 에이전트 전환 마커 생성: steps 배열 기반으로 메시지 사이에 삽입할 전환 이벤트
  const agentTransitions = useMemo(() => {
    // steps 배열에서 각 에이전트가 시작된 시점의 step 번호를 기록
    return (steps || []).map((s) => s.agent);
  }, [steps]);

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
  }, [messages, isLoading, toolExecutions, agentHistory]);

  useEffect(() => {
    prevLengthRef.current = messages?.length || 0;
  }, [messages]);

  // 에이전트별 도구 호출 그룹핑
  const toolsByAgent = useMemo(() => {
    const map = {};
    (toolExecutions || []).forEach((te) => {
      const agent = te.agent || '_unknown';
      if (!map[agent]) map[agent] = [];
      map[agent].push(te);
    });
    return map;
  }, [toolExecutions]);

  // 단계별 결과 항목
  const stepResultEntries = useMemo(() => {
    if (!stepResults || typeof stepResults !== 'object') return [];
    return Object.entries(stepResults).map(([stepNum, result]) => {
      const stepInfo = (steps || []).find((s) => String(s.step) === String(stepNum));
      const agent = stepInfo?.agent || '';
      const workerInfo = MULTI_AGENT_WORKERS[agent] || {};
      return { stepNum, result, agentName: workerInfo.label || stepInfo?.description || agent };
    });
  }, [stepResults, steps]);

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-9">
        <SectionHeader
          title="AI 에이전트"
          subtitle="Supervisor 기반 AI 에이전트 분석"
          right={<PipelineStatusBadge pipelineStatus={pipelineStatus} messageCount={messages?.length || 0} activeAgent={activeAgent} />}
        />

        <div className="card space-y-4">
          {/* 채팅/결과 영역 */}
          <div ref={chatBoxRef} role="log" aria-live="polite" className="max-h-[55vh] md:max-h-[60vh] overflow-auto pr-1">
            {(messages || []).map((m, idx) => {
              const isUser = m.role === 'user';
              const isPending = !!m?._pending;
              const msgKey = m?._id || idx;
              const isNew = idx >= prevLengthRef.current;

              // 에이전트 전환 마커: 첫 assistant 메시지 직전에 표시
              // steps 배열에서 현재 메시지에 매핑되는 에이전트 전환 감지
              const transitionAgent = !isUser && idx > 0
                ? agentTransitions[Math.floor((idx - 1) / 2)] || null
                : null;

              return (
                <React.Fragment key={msgKey}>
                  {/* 에이전트 전환 인라인 마커 */}
                  {transitionAgent && (
                    <AgentTransitionMarker agent={transitionAgent} />
                  )}
                  <motion.div
                    initial={isNew ? { opacity: 0, x: isUser ? 20 : -20 } : false}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ type: "spring", damping: 20, stiffness: 300 }}
                    className={`group relative ${isUser ? 'flex justify-end mb-3' : 'flex justify-start mb-3'}`}
                  >
                    <div
                      className={
                        isUser
                          ? 'chat-bubble chat-bubble-user w-full md:max-w-[78%] rounded-[20px] shadow-sm hover:shadow-md transition-shadow duration-200'
                          : 'chat-bubble chat-bubble-ai w-full md:max-w-[78%] rounded-[20px] shadow-sm hover:shadow-md transition-shadow duration-200'
                      }
                    >
                      <div className="text-[11px] font-extrabold text-cafe24-brown/60 mb-2 flex items-center justify-between">
                        <span className="flex items-center gap-1.5">
                          {isUser ? (
                            auth?.username || 'USER'
                          ) : (
                            <>
                              <Bot size={11} className="text-cafe24-orange" />
                              CAFE24 AI
                              {/* 현재 활성 에이전트 태그 */}
                              {isPending && activeAgent && (
                                <span className="ml-1 rounded bg-cafe24-orange/10 px-1.5 py-0.5 text-[9px] font-bold text-cafe24-orange">
                                  {MULTI_AGENT_WORKERS[activeAgent]?.label || activeAgent}
                                </span>
                              )}
                            </>
                          )}
                        </span>
                        {!isUser && isPending ? (
                          <span className="inline-flex items-center gap-2 text-cafe24-orange">
                            <span className="h-3 w-3 rounded-full border-2 border-cafe24-yellow border-t-cafe24-orange animate-spin" />
                            <span className="text-[10px]">streaming</span>
                          </span>
                        ) : null}
                      </div>

                      <AnimatePresence initial={false}>
                        {!collapsedMsgs.has(idx) ? (
                          <motion.div
                            key="expanded"
                            initial={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                            className="overflow-hidden"
                          >
                            <div className="prose prose-sm max-w-none">
                              {!isUser && isPending && !m.content?.trim() ? (
                                <TypingDots />
                              ) : (
                                <MarkdownMessage content={m.content || ''} />
                              )}
                            </div>
                            <ToolCalls toolCalls={m.tool_calls} />
                          </motion.div>
                        ) : (
                          <motion.div
                            key="collapsed"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="text-xs text-cafe24-brown/40 truncate cursor-pointer py-1"
                            onClick={() => toggleCollapse(idx)}
                          >
                            {m.content?.slice(0, 80)}...
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {/* 접기/펼치기 버튼 — 메시지 하단 */}
                      {!isUser && !isPending && m.content?.trim() && (
                        <div className="flex justify-center mt-1">
                          <button
                            type="button"
                            onClick={() => toggleCollapse(idx)}
                            className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] text-cafe24-brown/40 hover:text-cafe24-brown hover:bg-cafe24-yellow/20 transition-colors"
                          >
                            {collapsedMsgs.has(idx) ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
                            {collapsedMsgs.has(idx) ? '펼치기' : '접기'}
                          </button>
                        </div>
                      )}

                      {/* 호버 액션 버튼 */}
                      {!isPending && (
                        <div className={`absolute ${isUser ? 'left-0' : 'right-0'} top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-1`}>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(m.content || '');
                              toast.success('복사되었습니다');
                            }}
                            className="p-1.5 rounded-lg bg-white/90 border border-cafe24-brown/20 text-cafe24-brown/60 hover:text-cafe24-brown hover:bg-cafe24-beige transition shadow-sm"
                            data-tooltip="복사"
                          >
                            <Copy size={14} />
                          </button>
                          {isUser && (
                            <button
                              onClick={() => handleSend(m.content || '')}
                              className="p-1.5 rounded-lg bg-white/90 border border-cafe24-brown/20 text-cafe24-brown/60 hover:text-cafe24-orange hover:bg-cafe24-beige transition shadow-sm"
                              data-tooltip="다시 질문"
                            >
                              <RefreshCcw size={14} />
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                </React.Fragment>
              );
            })}

            {!messages?.length && (
              <EmptyState
                title="AI 에이전트"
                desc="추천 질문을 클릭하거나 직접 입력하여 AI 에이전트와 대화를 시작하세요."
              />
            )}

            <div ref={scrollRef} />
          </div>

          {/* 실시간 실행 현황 카드 */}
          {agentHistory.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="rounded-2xl border-2 border-cafe24-orange/15 bg-gradient-to-r from-cafe24-yellow/5 via-white to-cafe24-orange/5 p-4 backdrop-blur"
            >
              <div className="flex items-center gap-2 mb-3">
                <Route size={16} className="text-cafe24-orange" />
                <span className="text-xs font-extrabold text-cafe24-brown">실행 현황</span>
                {pipelineStatus === 'running' && (
                  <span className="text-[10px] text-cafe24-orange font-bold animate-pulse">진행중</span>
                )}
                {pipelineStatus === 'done' && (
                  <span className="text-[10px] text-emerald-600 font-bold">완료</span>
                )}
              </div>
              <div className="space-y-1.5">
                <AnimatePresence>
                  {agentHistory.map((ah, idx) => {
                    const worker = MULTI_AGENT_WORKERS[ah.agent];
                    const Icon = worker?.icon || Bot;
                    const label = worker?.label || ah.agent;
                    const stepInfo = steps.find((s) => s.agent === ah.agent);
                    const desc = stepInfo?.description || ah.agent;
                    const status = ah.status;
                    const agentTools = toolsByAgent[ah.agent] || [];
                    return (
                      <motion.div
                        key={ah.agent}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.08, duration: 0.2 }}
                        className="space-y-0"
                      >
                        {/* 에이전트 행 */}
                        <div className={`flex items-center gap-3 rounded-xl px-3 py-2 text-xs transition-all duration-300 ${
                          status === 'active'
                            ? 'bg-cafe24-orange/10 border border-cafe24-orange/30 shadow-sm'
                            : status === 'done'
                            ? 'bg-emerald-50 border border-emerald-200'
                            : 'bg-gray-50 border border-gray-200'
                        }`}>
                          <span className={`flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold shrink-0 ${
                            status === 'done' ? 'bg-emerald-500 text-white'
                              : status === 'active' ? 'bg-cafe24-orange text-white animate-pulse'
                              : 'bg-gray-300 text-white'
                          }`}>
                            {status === 'done' ? <CheckCircle2 size={12} /> : idx + 1}
                          </span>
                          <Icon size={14} className={`shrink-0 ${
                            status === 'active' ? 'text-cafe24-orange'
                              : status === 'done' ? 'text-emerald-600'
                              : 'text-gray-400'
                          }`} />
                          <span className={`font-bold shrink-0 ${
                            status === 'active' ? 'text-cafe24-brown'
                              : status === 'done' ? 'text-emerald-700'
                              : 'text-gray-500'
                          }`}>{label}</span>
                          <span className={`truncate hidden sm:inline ${
                            status === 'active' ? 'text-cafe24-brown/60'
                              : status === 'done' ? 'text-emerald-600/60'
                              : 'text-gray-400'
                          }`}>{desc}</span>
                          <span className="ml-auto shrink-0">
                            {status === 'active' && <Loader2 size={12} className="animate-spin text-cafe24-orange" />}
                            {status === 'done' && ah.elapsed_ms && (
                              <span className="text-[10px] text-emerald-600 font-mono">{(ah.elapsed_ms / 1000).toFixed(1)}s</span>
                            )}
                          </span>
                        </div>
                        {/* 에이전트별 도구 호출 목록 */}
                        {agentTools.length > 0 && (
                          <div className="ml-7 border-l-2 border-cafe24-orange/10 pl-3 py-1 space-y-0.5">
                            {agentTools.map((te, ti) => {
                              const isRunning = te.status === 'running';
                              const isError = te.status === 'error';
                              return (
                                <motion.div
                                  key={ti}
                                  initial={{ opacity: 0, x: -6 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: ti * 0.04, duration: 0.15 }}
                                  className="flex items-center gap-1.5 py-1 text-[11px]"
                                >
                                  {isRunning ? (
                                    <span className="relative flex h-3.5 w-3.5 shrink-0 items-center justify-center">
                                      <span className="absolute inline-flex h-full w-full rounded-full bg-cafe24-orange/30 animate-ping" />
                                      <Loader2 size={11} className="relative animate-spin text-cafe24-orange" />
                                    </span>
                                  ) : isError ? (
                                    <XCircle size={11} className="text-red-500 shrink-0" />
                                  ) : (
                                    <CheckCircle2 size={11} className="text-emerald-500 shrink-0" />
                                  )}
                                  <span className={`font-bold ${isRunning ? 'text-cafe24-orange' : isError ? 'text-red-600' : 'text-cafe24-brown/70'}`}>
                                    {te.tool}
                                  </span>
                                  {te.args && Object.keys(te.args).length > 0 && (
                                    <span className="text-cafe24-brown/35 text-[10px] font-mono">
                                      ({Object.entries(te.args).map(([k, v]) => `${k}=${v}`).join(', ')})
                                    </span>
                                  )}
                                  {isRunning && (
                                    <span className="ml-auto text-[9px] text-cafe24-orange font-bold animate-pulse">호출중</span>
                                  )}
                                </motion.div>
                              );
                            })}
                          </div>
                        )}
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </motion.div>
          )}

          {/* 단계별 결과 접기/펼치기 */}
          {stepResultEntries.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-extrabold text-cafe24-brown/60">에이전트 결과</div>
              {stepResultEntries.map(({ stepNum, result, agentName }) => (
                <StepResultCard key={stepNum} stepNum={stepNum} result={result} agentName={agentName} />
              ))}
            </div>
          )}

          {/* 에러 표시 */}
          {error && (
            <div role="alert" className="rounded-xl bg-red-50 border border-red-200 p-3 text-xs text-red-700">
              {error}
            </div>
          )}

          {/* 컨설팅 모드: 스텝 진행바 + 옵션 버튼 */}
          {consultingMode && (
            <div className="space-y-3">
              {/* 스텝 진행바 */}
              <div className="flex items-center gap-1 w-full overflow-x-auto pb-1">
                {[
                  { key: 'diagnosis', label: '진단', Icon: Search },
                  { key: 'strategy', label: '전략 수립', Icon: Target },
                  { key: 'plan', label: '실행 계획', Icon: ClipboardList },
                  { key: 'execute', label: '실행', Icon: Rocket },
                ].map((step, idx) => {
                  const h = consultingStepHistory.find((s) => s.step === step.key);
                  const isCompleted = h?.status === 'completed';
                  const isActive = h?.status === 'active' || consultingStep === step.key;
                  return (
                    <React.Fragment key={step.key}>
                      {idx > 0 && (
                        <div className={`hidden sm:block h-0.5 flex-1 min-w-[16px] max-w-[40px] transition-colors ${
                          isCompleted ? 'bg-emerald-400' : isActive ? 'bg-cafe24-orange/50' : 'bg-gray-200'
                        }`} />
                      )}
                      <div className={`flex items-center gap-1.5 rounded-xl px-2.5 py-2 text-xs font-bold shrink-0 ${
                        isCompleted ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : isActive ? 'bg-cafe24-orange text-white shadow-md'
                          : 'bg-gray-100 text-gray-400 border border-gray-200'
                      }`}>
                        {isCompleted ? <CheckCircle2 size={14} /> : <step.Icon size={14} />}
                        <span className="hidden sm:inline">{idx + 1}. {step.label}</span>
                        <span className="sm:hidden">{idx + 1}</span>
                      </div>
                    </React.Fragment>
                  );
                })}
                {consultingSessionId && (
                  <span className="ml-auto text-[10px] text-gray-400 font-mono hidden sm:inline">
                    #{consultingSessionId.slice(0, 8)}
                  </span>
                )}
              </div>

              {/* 옵션 버튼 */}
              {consultingAwaitingInput && !isLoading && (
                <div className="rounded-xl border border-cafe24-orange/20 bg-cafe24-yellow/5 p-3 space-y-2">
                  {consultingAwaitingInput.prompt && (
                    <p className="text-xs font-bold text-cafe24-brown">{consultingAwaitingInput.prompt}</p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    {(consultingAwaitingInput.options || []).map((opt, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => handleSend(opt)}
                        className={`rounded-lg px-3 py-2 text-xs font-bold transition-all hover:scale-[1.02] active:scale-[0.98] ${
                          opt.includes('다음 단계') || opt === '승인'
                            ? 'bg-cafe24-orange hover:bg-cafe24-orange/90 text-white shadow-sm'
                            : 'bg-white hover:bg-gray-50 text-cafe24-brown border border-cafe24-brown/20'
                        }`}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 입력 영역 */}
          <div className="flex flex-col md:flex-row gap-2 hover:shadow-md transition-shadow duration-300 rounded-xl p-1">
            <input
              className="input focus:shadow-lg focus:-translate-y-0.5 transition-all duration-200"
              placeholder="AI 에이전트에게 질문 입력 (Enter로 전송)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && canSend) {
                  handleSend();
                }
              }}
            />

            <button
              className={`${cafe24BtnInline} w-[140px] hover:scale-105 active:scale-95 transition-transform duration-150`}
              onClick={() => handleSend()}
              disabled={!canSend}
              type="button"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              {isLoading ? '실행중...' : '실행'}
            </button>

            <button
              className={`${cafe24BtnSecondaryInline} w-[140px] hover:scale-105 active:scale-95 transition-transform duration-150`}
              onClick={() => {
                stopStream();
                toast('중단됨');
              }}
              disabled={!isLoading}
              data-tooltip="스트림 중단"
              type="button"
            >
              중단
            </button>
          </div>

          {/* 추천 질문 칩 */}
          <div className="flex flex-wrap gap-2">
            {chips.map((c) => (
              <Chip key={c.label} label={c.label} disabled={c.disabled} onClick={() => handleSend(c.label)} />
            ))}
          </div>
        </div>
      </div>

      <div className="col-span-12 xl:col-span-3">
        <div className="card">
          <div className="card-header">빠른 분석</div>
          <div className="text-sm text-cafe24-brown/70 mb-3">CAFE24 AI 도구 호출</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-2">
            <button className={cafe24Btn} onClick={() => runQuick('/api/shops')} type="button">쇼핑몰 목록</button>
            <button className={cafe24Btn} onClick={() => runQuick('/api/categories')} type="button">카테고리 목록</button>
            <button className={cafe24Btn} onClick={() => runQuick('/api/cs/glossary')} type="button">이커머스 용어집</button>
            <button className={cafe24Btn} onClick={() => runQuick('/api/sellers/segments/statistics')} type="button">세그먼트 통계</button>
          </div>
          {quickResult ? (
            <pre className="mt-3 max-h-[45vh] overflow-auto rounded-2xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown">
              {(() => {
                const str = JSON.stringify(quickResult, null, 2);
                if (str.length > 50000) return str.slice(0, 50000) + '\n\n... (결과가 너무 깁니다. 50KB 이후 생략)';
                return str;
              })()}
            </pre>
          ) : (
            <div className="mt-3 text-xs text-cafe24-brown/60">버튼을 클릭하면 API 호출 결과를 확인할 수 있어요.</div>
          )}
        </div>

        <div className="card mt-4">
          <div className="card-header">에이전트 정보</div>
          <div className="text-sm text-cafe24-brown/70 space-y-2">
            <p>
              Supervisor 멀티에이전트 구조입니다. Supervisor가 질문을 분석하여 전문 워커에게 위임하고 결과를 종합합니다. 멀티턴 대화로 후속 질문이 가능합니다.
            </p>
            <div className="rounded-xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown space-y-1">
              <div className="font-extrabold mb-1">전문 워커 에이전트 (7종)</div>
              <div>이탈 분석 - ML 이탈 예측 + SHAP + 위험 셀러 조회</div>
              <div>전략 수립 - 맞춤 리텐션 메시지 + 자동 조치 실행</div>
              <div>셀러 분석 - 종합 진단 + 세그먼트 + 이상거래 조사</div>
              <div>성과 분석 - 쇼핑몰 매출/코호트/마케팅 최적화</div>
              <div>CS 품질 - CS 통계 + 자동 분류 + 품질 평가</div>
              <div>리포트 - 대시보드 + KPI 종합 보고서</div>
              <div>플랫폼 검색 - RAG 지식 검색 + FAQ</div>
            </div>
            <div className="rounded-xl bg-blue-50 p-3 text-xs text-cafe24-brown space-y-1 mt-2">
              <div className="font-extrabold mb-1">셀러 컨설팅 (4단계 워크플로우)</div>
              <div>"SEL0001 컨설팅 해줘" 입력 시 자동 진입</div>
              <div>① 진단 → ② 전략 수립 → ③ 실행 계획 → ④ 실행</div>
              <div>각 단계에서 자유 대화 가능, '다음' 입력 시 진행</div>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            <button
              className={cafe24BtnSecondary}
              onClick={() => {
                resetPipeline();
                setMessages([]);
              }}
              type="button"
            >
              <RotateCcw size={14} className="inline mr-1" />
              대화 초기화
            </button>
          </div>
        </div>

        <div className="card mt-4">
          <div className="card-header">AI 도구 탐색기</div>
          <p className="text-sm text-cafe24-brown/70 mb-3">
            에이전트가 사용하는 AI 도구를 카테고리별로 탐색하세요.
          </p>
          <ToolExplorer />
        </div>

        <div className="card mt-4">
          <div className="card-header">LLM 설정 요약</div>
          <div className="text-sm text-cafe24-brown/70 space-y-1">
            <div>
              <span className="text-cafe24-brown/50">모델</span>:{' '}
              <span className="font-mono">{settings?.selectedModel || 'gpt-5-mini'}</span>
            </div>
            <div>
              <span className="text-cafe24-brown/50">Max Tokens</span>:{' '}
              <span className="font-mono">{settings?.maxTokens || 4000}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
