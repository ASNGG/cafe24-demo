// components/panels/AgentPanel.js
// CAFE24 AI 운영 플랫폼 - 에이전트 패널

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import { ArrowUpRight, Sparkles, Zap, Loader2, ShoppingBag, Copy, RefreshCcw, Check } from 'lucide-react';
import useAgentStream from './hooks/useAgentStream';
import { cafe24Btn, cafe24BtnSecondary, cafe24BtnInline, cafe24BtnSecondaryInline } from '@/components/common/buttonStyles';

const SEEN_KEY = 'cafe24_seen_example_hint';

// React.memo로 불필요한 리렌더링 방지
const ToolCalls = React.memo(function ToolCalls({ toolCalls }) {
  if (!toolCalls?.length) return null;
  return (
    <details className="details mt-2">
      <summary>도구 실행 결과</summary>
      <div className="mt-2 space-y-3">
        {toolCalls.map((tc, idx) => {
          const ok = tc?.result?.status === 'success';
          return (
            <div
              key={idx}
              className="rounded-2xl border-2 border-cafe24-orange/20 bg-white/80 p-3 shadow-sm backdrop-blur"
            >
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

// React.memo: 10개 칩이 매 렌더마다 리렌더되는 것 방지
// onClick에 label을 전달하여 부모에서 인라인 화살표 함수 생성 불필요
const Chip = React.memo(function Chip({ label, onClick }) {
  const handleClick = useCallback(() => onClick(label), [onClick, label]);
  return (
    <button
      className="inline-flex items-center gap-2 rounded-full border-2 border-cafe24-orange/20 bg-white/80 px-3 py-1.5 text-xs font-extrabold text-cafe24-brown hover:bg-cafe24-yellow/20 hover:border-cafe24-orange/40 hover:shadow-sm transition active:translate-y-[1px] whitespace-nowrap"
      onClick={handleClick}
      title="클릭하면 질문이 바로 전송됩니다"
      type="button"
    >
      <ShoppingBag size={14} className="text-cafe24-orange" />
      <span className="max-w-[220px] truncate">{label}</span>
      <ArrowUpRight size={14} className="text-cafe24-brown/50" />
    </button>
  );
});

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.2s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce [animation-delay:-0.1s]" />
      <span className="h-2 w-2 rounded-full bg-cafe24-orange animate-bounce" />
      <span className="ml-2 text-xs text-cafe24-brown/60">답변 생성 중…</span>
    </div>
  );
}

function TopProgressBar({ active }) {
  if (!active) return null;
  return (
    <div className="mb-3 h-1 w-full overflow-hidden rounded-full bg-cafe24-yellow/30">
      <div className="h-full w-1/3 animate-[cafe24_progress_1.2s_ease-in-out_infinite] bg-cafe24-orange" />
    </div>
  );
}

import remarkGfmPlugin from 'remark-gfm';

// 모듈 레벨 상수: 매 렌더마다 새 배열 생성 방지
const REMARK_PLUGINS = [remarkGfmPlugin];
const REHYPE_PLUGINS = [];

// 모듈 레벨 상수: ReactMarkdown components 객체 리렌더 시 재생성 방지
const MARKDOWN_COMPONENTS = {
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto -mx-1 my-2">
      <table className="w-full border-collapse" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-cafe24-yellow/20" {...props} />,
  th: ({ node, ...props }) => (
    <th
      className="border-2 border-cafe24-orange/20 px-3 py-2 text-left text-xs font-extrabold text-cafe24-brown"
      {...props}
    />
  ),
  td: ({ node, ...props }) => (
    <td
      className="border border-cafe24-orange/15 px-3 py-2 align-top text-xs text-cafe24-brown whitespace-nowrap"
      {...props}
    />
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
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  },
  a: ({ node, ...props }) => (
    <a
      {...props}
      target="_blank"
      rel="noopener noreferrer"
      className="font-extrabold text-cafe24-orange underline underline-offset-2 hover:text-cafe24-brown"
    />
  ),
};

// React.memo: 마크다운 파싱은 비용이 큰 작업이므로 content 변경 시에만 리렌더
const MarkdownMessage = React.memo(function MarkdownMessage({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={REMARK_PLUGINS}
      rehypePlugins={REHYPE_PLUGINS}
      components={MARKDOWN_COMPONENTS}
    >
      {content || ''}
    </ReactMarkdown>
  );
});

// ChatMessage: 개별 메시지 렌더링 컴포넌트 (React.memo로 불필요한 리렌더링 방지)
const ChatMessage = React.memo(function ChatMessage({ msg, isNew, username, onCopy, onResend }) {
  const isUser = msg.role === 'user';
  const isPending = !!msg._pending;

  return (
    <motion.div
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
          <span>{isUser ? username || 'USER' : 'CAFE24 AI'}</span>

          {!isUser && isPending ? (
            <span className="inline-flex items-center gap-2 text-cafe24-orange">
              <span className="h-3 w-3 rounded-full border-2 border-cafe24-yellow border-t-cafe24-orange animate-spin" />
              <span className="text-[10px]">streaming</span>
            </span>
          ) : null}
        </div>

        <div className="prose prose-sm max-w-none">
          {!isUser && isPending ? <TypingDots /> : <MarkdownMessage content={msg.content || ''} />}
        </div>

        <ToolCalls toolCalls={msg.tool_calls} />

        {/* 호버 시 나타나는 액션 버튼 */}
        {!isPending && (
          <div className={`absolute ${isUser ? 'left-0' : 'right-0'} top-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-1`}>
            <button
              onClick={() => onCopy(msg.content || '')}
              className="p-1.5 rounded-lg bg-white/90 border border-cafe24-brown/20 text-cafe24-brown/60 hover:text-cafe24-brown hover:bg-cafe24-beige transition shadow-sm"
              title="복사"
            >
              <Copy size={14} />
            </button>
            {isUser && (
              <button
                onClick={() => onResend(msg.content || '')}
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
}, (prev, next) => {
  return prev.msg.content === next.msg.content
    && prev.msg.role === next.msg.role
    && prev.msg._pending === next.msg._pending
    && prev.msg.tool_calls === next.msg.tool_calls;
});

export default function AgentPanel({
  auth,
  selectedShop,
  addLog,
  settings,
  setSettings,
  agentMessages,
  setAgentMessages,
  totalQueries,
  setTotalQueries,
  apiCall,
}) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [quickResult, setQuickResult] = useState(null);

  const chatBoxRef = useRef(null);
  const scrollRef = useRef(null);
  // M46: 이전 렌더 시 메시지 길이 추적 → 새 메시지만 애니메이션 (Set 메모리 누적 방지)
  const prevLengthRef = useRef(0);


  // M53: SSE 스트리밍 로직을 useAgentStream 훅으로 추출
  const { sendQuestion, stopStream } = useAgentStream({
    auth, selectedShop, settings,
    setAgentMessages, setTotalQueries, setLoading: setLoading,
    addLog,
  });

  const canSend = useMemo(() => !!input?.trim() && !loading, [input, loading]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seen = window.localStorage.getItem(SEEN_KEY);
    if (!seen) toast('왼쪽 예시 질문을 클릭하면 바로 분석이 시작됩니다', { icon: '🛒' });
  }, []);

  const handleSend = useCallback((q) => {
    if (typeof window !== 'undefined') window.localStorage.setItem(SEEN_KEY, '1');
    sendQuestion(q);
  }, [sendQuestion]);

  // CAFE24 관련 추천 질문 (이커머스 운영 분석)
  const chips = useMemo(() => {
    const shopId = selectedShop || 'S0001';
    return [
      `${shopId} 쇼핑몰 정보 알려줘`,
      `${shopId} 매출 성과 분석해줘`,
      'SEL0001 셀러 활동 현황',
      'Premium 등급 쇼핑몰 목록',
      '쇼핑몰 SEO 최적화 방법 알려줘',
      '카페24 결제 수단 안내해줘',
      '반품 처리 절차 알려줘',
    ];
  }, [selectedShop]);


  const shouldAutoScrollRef = useRef(true);

  const updateAutoScrollFlag = useCallback(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    const threshold = 80;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom <= threshold;
  }, []);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    el.addEventListener('scroll', updateAutoScrollFlag, { passive: true });
    return () => el.removeEventListener('scroll', updateAutoScrollFlag);
  }, [updateAutoScrollFlag]);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    if (!shouldAutoScrollRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [agentMessages, loading]);

  // M46: 렌더 후 prevLengthRef 갱신 (새 메시지 애니메이션 판별용)
  useEffect(() => {
    prevLengthRef.current = agentMessages?.length || 0;
  }, [agentMessages]);

  // Chip 클릭 시 안정적인 콜백 (인라인 화살표 함수 제거)
  const handleChipClick = useCallback((label) => {
    handleSend(label);
    setInput('');
  }, [handleSend]);

  // ChatMessage에서 사용할 안정적인 콜백
  const handleCopy = useCallback((content) => {
    navigator.clipboard.writeText(content || '');
    toast.success('복사되었습니다');
  }, []);

  const userKey = useMemo(() => String(auth?.username || '').trim(), [auth?.username]);
  const prevUserKeyRef = useRef(userKey);

  useEffect(() => {
    if (prevUserKeyRef.current === userKey) return;

    prevUserKeyRef.current = userKey;

    stopStream();
    setAgentMessages([]);
    setTotalQueries(0);
    setQuickResult(null);
    setInput('');
    setLoading(false);
  }, [userKey, stopStream, setAgentMessages, setTotalQueries]);

  // M70: sendQuestion ref로 이벤트 리스너 재등록 방지
  const sendQuestionRef = useRef(sendQuestion);
  sendQuestionRef.current = handleSend;

  useEffect(() => {
    function handler(ev) {
      const q = ev?.detail?.q;
      if (!q) return;
      sendQuestionRef.current(q);
    }
    window.addEventListener('cafe24_send_question', handler);
    return () => window.removeEventListener('cafe24_send_question', handler);
  }, []);

  async function runQuick(endpoint, method = 'GET', payload = null) {
    setQuickResult(null);

    const res = await apiCall({
      endpoint,
      method,
      auth,
      data: payload,
      timeoutMs: 60000,
    });

    setQuickResult(res);
    addLog('빠른분석', endpoint);
  }

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-9">
        <SectionHeader
          title="AI 에이전트"
          subtitle="GPT + ML 기반 이커머스 분석"
          right={<span className="badge">쿼리 {totalQueries || 0}</span>}
        />

        <div className="card">
          <div ref={chatBoxRef} className="max-h-[62vh] md:max-h-[70vh] overflow-auto pr-1">
            {(agentMessages || []).map((m, idx) => {
              // M46: 새 메시지만 애니메이션 (prevLengthRef로 메모리 누적 방지)
              const msgKey = m?._id || idx;
              const isNew = idx >= prevLengthRef.current;

              return (
                <ChatMessage
                  key={msgKey}
                  msg={m}
                  isNew={isNew}
                  username={auth?.username}
                  onCopy={handleCopy}
                  onResend={handleSend}
                />
              );
            })}

            {!agentMessages?.length ? (
              <EmptyState
                title="대화를 시작해보세요"
                desc="왼쪽 예시 질문을 누르거나 아래 추천 질문을 클릭하면 바로 시작됩니다."
              />
            ) : null}

            <div ref={scrollRef} />
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {chips.map((c) => (
              <Chip
                key={c}
                label={c}
                onClick={handleChipClick}
              />
            ))}
          </div>

          <div className="mt-3 flex flex-col md:flex-row gap-2">
            <input
              className="input"
              placeholder="질문 입력 (Enter로 전송)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && canSend) {
                  handleSend(input);
                  setInput('');
                }
              }}
            />

            <button
              className={`${cafe24BtnInline} w-[140px]`}
              onClick={() => {
                handleSend(input);
                setInput('');
              }}
              disabled={!canSend}
              type="button"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              {loading ? '분석중...' : '전송'}
            </button>

            <button
              className={`${cafe24BtnSecondaryInline} w-[140px]`}
              onClick={() => {
                stopStream();
                toast('중단됨');
              }}
              disabled={!loading}
              title="스트리밍 중단"
              type="button"
            >
              중단
            </button>
          </div>
        </div>
      </div>

      <div className="col-span-12 xl:col-span-3">
        <div className="card">
          <div className="card-header">빠른 분석</div>
          <div className="text-sm text-cafe24-brown/70 mb-3">
            CAFE24 AI 도구 호출
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-2">
            <button
              className={cafe24Btn}
              onClick={() => runQuick('/api/shops')}
              type="button"
            >
              쇼핑몰 목록
            </button>
            <button
              className={cafe24Btn}
              onClick={() => runQuick('/api/categories')}
              type="button"
            >
              카테고리 목록
            </button>
            <button
              className={cafe24Btn}
              onClick={() => runQuick('/api/cs/glossary')}
              type="button"
            >
              이커머스 용어집
            </button>
            <button
              className={cafe24Btn}
              onClick={() => runQuick('/api/sellers/segments/statistics')}
              type="button"
            >
              세그먼트 통계
            </button>
          </div>

          <div className="mt-3">
            <button className={cafe24BtnSecondary} onClick={() => setAgentMessages([])} type="button">
              대화 초기화
            </button>
          </div>

          {quickResult ? (
            <pre className="mt-3 max-h-[45vh] overflow-auto rounded-2xl bg-cafe24-yellow/10 p-3 text-xs text-cafe24-brown">
              {(() => {
                // L22: 대용량 JSON 크기 제한 (50KB)
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
          <div className="card-header">LLM 설정 요약</div>
          <div className="text-sm text-cafe24-brown/70 space-y-1">
            <div>
              <span className="text-cafe24-brown/50">모델</span>: <span className="font-mono">{settings?.selectedModel || 'gpt-4o-mini'}</span>
            </div>
            <div>
              <span className="text-cafe24-brown/50">Max Tokens</span>: <span className="font-mono">{settings?.maxTokens || 4000}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
