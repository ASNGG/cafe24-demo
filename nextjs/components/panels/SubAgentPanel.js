// components/panels/SubAgentPanel.js
// CAFE24 AI 운영 플랫폼 - 서브에이전트 패널

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfmPlugin from 'remark-gfm';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import { Loader2, Zap, ChevronDown, ChevronUp, Copy, FlaskConical, RefreshCcw, RotateCcw } from 'lucide-react';
import useSubAgentStream from './hooks/useSubAgentStream';
import { cafe24BtnInline, cafe24BtnSecondaryInline, cafe24BtnSecondary } from '@/components/common/buttonStyles';

const PipelineSteps = React.memo(function PipelineSteps({ steps }) {
  if (!steps?.length) return null;
  return (
    <div className="flex items-center gap-2 p-4 bg-white rounded-xl shadow-cafe24-sm overflow-x-auto">
      {steps.map((step, i) => (
        <div key={step.step ?? i} className="flex items-center shrink-0">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-extrabold transition-colors ${
              step.status === 'done'
                ? 'bg-green-500 text-white'
                : step.status === 'running'
                ? 'bg-cafe24-orange text-white animate-pulse'
                : 'bg-gray-200 text-gray-500'
            }`}
          >
            {step.status === 'done' ? '\u2713' : (step.step ?? i + 1)}
          </div>
          <span className="text-xs ml-1 font-bold text-cafe24-brown/70">
            {STEP_LABELS[step.agent] || step.description || step.agent || `Step ${step.step}`}
          </span>
          {i < steps.length - 1 && <div className="w-8 h-0.5 bg-gray-300 mx-1" />}
        </div>
      ))}
    </div>
  );
});

function StepResultCard({ stepNum, result, agentName }) {
  const [open, setOpen] = useState(true);
  const remarkPlugins = useMemo(() => [remarkGfmPlugin], []);

  return (
    <div className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 shadow-sm backdrop-blur">
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
          <ChevronUp size={14} className="text-cafe24-brown/50" />
        ) : (
          <ChevronDown size={14} className="text-cafe24-brown/50" />
        )}
      </button>
      {open && (
        <div className="px-3 pb-3">
          <div className="prose prose-sm max-w-none text-cafe24-brown text-xs">
            <ReactMarkdown remarkPlugins={remarkPlugins}>{result || ''}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.2s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.1s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce" />
      <span className="ml-2 text-xs text-cafe24-brown/60">서브에이전트 분석 중...</span>
    </div>
  );
}

// 모듈 레벨 상수: ReactMarkdown components 객체 리렌더 시 재생성 방지
const SUB_MARKDOWN_COMPONENTS = {
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
const SUB_REMARK_PLUGINS = [remarkGfmPlugin];

// 백엔드 스텝 이름 → 한글 라벨 매핑
const STEP_LABELS = {
  // 리텐션
  analyze_churn: '이탈 위험 분석',
  check_cs: 'CS 현황 확인',
  generate_strategy: '맞춤 전략 생성',
  execute_action: '리텐션 조치 실행',
  // 셀러 진단
  seller_analyze: '셀러 상세 분석',
  seller_risk: '리스크 평가',
  seller_optimize: '최적화 전략',
  // 쇼핑몰 성과
  shop_info: '쇼핑몰 정보 수집',
  shop_performance: '성과 분석',
  shop_marketing: '마케팅 최적화',
  // 딥 분석
  dashboard_overview: '대시보드 현황',
  trend_analysis: 'KPI 트렌드',
  gmv_forecast: 'GMV 예측',
  // 이상거래
  fraud_overview: '이상거래 현황',
  fraud_detect: '부정행위 탐지',
  fraud_report: '조사 보고서',
  // CS 품질
  cs_statistics: 'CS 통계 분석',
  cs_classify: '문의 분류',
  cs_auto_reply: '자동 응답 생성',
};

const MarkdownMessage = React.memo(function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={SUB_REMARK_PLUGINS}
      components={SUB_MARKDOWN_COMPONENTS}
    >
      {content || ''}
    </ReactMarkdown>
  );
});

const ToolCalls = React.memo(function ToolCalls({ toolCalls }) {
  if (!toolCalls?.length) return null;
  return (
    <details className="details mt-2">
      <summary>도구 실행 결과</summary>
      <div className="mt-2 space-y-3">
        {toolCalls.map((tc, idx) => {
          const ok = tc?.result?.status === 'success';
          return (
            <div key={idx} className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-3 shadow-sm backdrop-blur">
              <div className="flex items-center justify-between">
                <div className="font-extrabold text-cafe24-brown">{tc.tool}</div>
                <span className={ok ? 'badge badge-success' : 'badge badge-danger'}>
                  {ok ? '성공' : '실패'}
                </span>
              </div>
              <pre className="mt-2 overflow-auto rounded-xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown">
                {JSON.stringify(tc.result, null, 2)}
              </pre>
            </div>
          );
        })}
      </div>
    </details>
  );
});

function Chip({ label, onClick }) {
  return (
    <button
      className="inline-flex items-center gap-2 rounded-full border-2 border-cafe24-orange/20 bg-white/80 px-3 py-1.5 text-xs font-extrabold text-cafe24-brown hover:bg-cafe24-yellow/20 hover:border-cafe24-orange/40 hover:shadow-sm transition active:translate-y-[1px] whitespace-nowrap"
      onClick={onClick}
      title="클릭하면 질문이 바로 전송됩니다"
      type="button"
    >
      <FlaskConical size={14} className="text-cafe24-orange" />
      <span className="max-w-[220px] truncate">{label}</span>
    </button>
  );
}

function PipelineStatusBadge({ pipelineStatus, messageCount }) {
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

export default function SubAgentPanel({ auth, selectedShop, addLog, settings, apiCall }) {
  const [input, setInput] = useState('');
  const chatBoxRef = useRef(null);
  const scrollRef = useRef(null);
  const seenMsgIdsRef = useRef(new Set());

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
  } = useSubAgentStream({ auth, selectedShop, settings });

  const canSend = useMemo(() => !!input?.trim() && !isLoading, [input, isLoading]);

  const handleSend = useCallback(
    (q) => {
      const query = q || input;
      if (!query?.trim()) return;
      sendMessage(query);
      setInput('');
      addLog?.('서브에이전트', query);
    },
    [input, sendMessage, addLog]
  );

  const chips = useMemo(
    () => [
      // 리텐션
      '셀러 이탈 분석 후 리텐션 전략 자동 실행',
      '고위험 셀러 긴급 리텐션 캠페인',
      // 셀러 종합 진단
      '셀러 종합 진단 + 세그먼트 분석',
      // 쇼핑몰 성과
      '쇼핑몰 성과 리포트 + 마케팅 최적화',
      // 전체 현황 딥 분석
      '전체 현황 딥 분석 + KPI 종합',
      // 이상거래 조사
      '이상거래 조사 + 부정행위 분석 리포트',
      // CS 품질
      'CS 품질 분석 + 자동 응답 개선',
    ],
    []
  );

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
  }, [messages, isLoading]);

  // 단계별 결과 항목
  const stepResultEntries = useMemo(() => {
    if (!stepResults || typeof stepResults !== 'object') return [];
    return Object.entries(stepResults).map(([stepNum, result]) => {
      const stepInfo = (steps || []).find((s) => String(s.step) === String(stepNum));
      const agent = stepInfo?.agent || '';
      return { stepNum, result, agentName: STEP_LABELS[agent] || stepInfo?.description || agent };
    });
  }, [stepResults, steps]);

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-9">
        <SectionHeader
          title="서브에이전트 파이프라인"
          subtitle="다단계 AI 에이전트 오케스트레이션"
          right={<PipelineStatusBadge pipelineStatus={pipelineStatus} messageCount={messages?.length || 0} />}
        />

        <div className="card space-y-4">
          {/* 파이프라인 시각화 */}
          <PipelineSteps steps={steps} currentStep={currentStep} />

          {/* 채팅/결과 영역 */}
          <div ref={chatBoxRef} className="max-h-[55vh] md:max-h-[60vh] overflow-auto pr-1">
            {(messages || []).map((m, idx) => {
              const isUser = m.role === 'user';
              const isPending = !!m?._pending;
              const msgKey = m?._id || idx;
              const isNew = !seenMsgIdsRef.current.has(msgKey);
              if (isNew) seenMsgIdsRef.current.add(msgKey);

              return (
                <motion.div
                  key={msgKey}
                  initial={isNew ? { opacity: 0, y: 6 } : false}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                  className={`group relative ${isUser ? 'flex justify-end mb-3' : 'flex justify-start mb-3'}`}
                >
                  <div
                    className={
                      isUser
                        ? 'chat-bubble chat-bubble-user w-full md:max-w-[78%]'
                        : 'chat-bubble chat-bubble-ai w-full md:max-w-[78%]'
                    }
                  >
                    <div className="text-[11px] font-extrabold text-cafe24-brown/60 mb-2 flex items-center justify-between">
                      <span>{isUser ? auth?.username || 'USER' : 'SUB-AGENT'}</span>
                      {!isUser && isPending ? (
                        <span className="inline-flex items-center gap-2 text-cafe24-orange">
                          <span className="h-3 w-3 rounded-full border-2 border-cafe24-yellow border-t-cafe24-orange animate-spin" />
                          <span className="text-[10px]">streaming</span>
                        </span>
                      ) : null}
                    </div>

                    <div className="prose prose-sm max-w-none">
                      {!isUser && isPending && !m.content?.trim() ? (
                        <TypingDots />
                      ) : (
                        <MarkdownMessage content={m.content || ''} />
                      )}
                    </div>

                    <ToolCalls toolCalls={m.tool_calls} />

                    {/* 호버 액션 버튼 */}
                    {!isPending && (
                      <div className={`absolute ${isUser ? 'left-0' : 'right-0'} top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-1`}>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(m.content || '');
                            toast.success('복사되었습니다');
                          }}
                          className="p-1.5 rounded-lg bg-white/90 border border-cafe24-brown/20 text-cafe24-brown/60 hover:text-cafe24-brown hover:bg-cafe24-beige transition shadow-sm"
                          title="복사"
                        >
                          <Copy size={14} />
                        </button>
                        {isUser && (
                          <button
                            onClick={() => handleSend(m.content || '')}
                            className="p-1.5 rounded-lg bg-white/90 border border-cafe24-brown/20 text-cafe24-brown/60 hover:text-cafe24-orange hover:bg-cafe24-beige transition shadow-sm"
                            title="다시 질문"
                          >
                            <RefreshCcw size={14} />
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              );
            })}

            {!messages?.length && (
              <EmptyState
                title="서브에이전트 실험실"
                desc="아래 추천 질문을 클릭하거나 직접 입력하여 6가지 AI 파이프라인 분석을 시작하세요."
              />
            )}

            <div ref={scrollRef} />
          </div>

          {/* 단계별 결과 접기/펼치기 */}
          {stepResultEntries.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-extrabold text-cafe24-brown/60">단계별 결과</div>
              {stepResultEntries.map(({ stepNum, result, agentName }) => (
                <StepResultCard key={stepNum} stepNum={stepNum} result={result} agentName={agentName} />
              ))}
            </div>
          )}

          {/* 에러 표시 */}
          {error && (
            <div className="rounded-xl bg-red-50 border border-red-200 p-3 text-xs text-red-700">
              {error}
            </div>
          )}

          {/* 추천 질문 칩 */}
          <div className="flex flex-wrap gap-2">
            {chips.map((c) => (
              <Chip key={c} label={c} onClick={() => handleSend(c)} />
            ))}
          </div>

          {/* 입력 영역 */}
          <div className="flex flex-col md:flex-row gap-2">
            <input
              className="input"
              placeholder="서브에이전트에게 복합 질문 입력 (Enter로 전송)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && canSend) {
                  handleSend();
                }
              }}
            />

            <button
              className={`${cafe24BtnInline} w-[140px]`}
              onClick={() => handleSend()}
              disabled={!canSend}
              type="button"
            >
              {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              {isLoading ? '실행중...' : '실행'}
            </button>

            <button
              className={`${cafe24BtnSecondaryInline} w-[140px]`}
              onClick={() => {
                stopStream();
                toast('중단됨');
              }}
              disabled={!isLoading}
              title="파이프라인 중단"
              type="button"
            >
              중단
            </button>
          </div>
        </div>
      </div>

      <div className="col-span-12 xl:col-span-3">
        <div className="card">
          <div className="card-header">파이프라인 정보</div>
          <div className="text-sm text-cafe24-brown/70 space-y-2">
            <p>
              서브에이전트는 복합 분석 요청을 여러 단계로 분해하여 전문 에이전트들이 순차 처리합니다. 6가지 파이프라인을 지원합니다.
            </p>
            <div className="rounded-xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown space-y-1">
              <div className="font-extrabold mb-1">파이프라인 종류</div>
              <div>🔄 리텐션 전략 - 이탈 분석 → 전략 → 실행</div>
              <div>👤 셀러 종합 진단 - 분석 → 리스크 → 최적화</div>
              <div>🏪 쇼핑몰 성과 - 정보 → 성과 → 마케팅</div>
              <div>📊 전체 현황 딥 분석 - 대시보드 → 트렌드 → GMV</div>
              <div>🚨 이상거래 조사 - 현황 → 탐지 → 보고</div>
              <div>💬 CS 품질 분석 - 통계 → 분류 → 자동응답</div>
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
          <div className="card-header">LLM 설정 요약</div>
          <div className="text-sm text-cafe24-brown/70 space-y-1">
            <div>
              <span className="text-cafe24-brown/50">모델</span>:{' '}
              <span className="font-mono">{settings?.selectedModel || 'gpt-4o-mini'}</span>
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
