// components/panels/AgentPanel.js
// CAFE24 AI 운영 플랫폼 - 에이전트 패널

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import { ArrowUpRight, Sparkles, Zap, Loader2, ShoppingBag, Copy, RefreshCcw, Check } from 'lucide-react';
import useAgentStream from './hooks/useAgentStream';
import { cafe24Btn, cafe24BtnSecondary, cafe24BtnInline, cafe24BtnSecondaryInline } from '@/components/common/buttonStyles';

const SEEN_KEY = 'cafe24_seen_example_hint';

function ToolCalls({ toolCalls }) {
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
}

function Chip({ label, onClick }) {
  return (
    <button
      className="inline-flex items-center gap-2 rounded-full border-2 border-cafe24-orange/20 bg-white/80 px-3 py-1.5 text-xs font-extrabold text-cafe24-brown hover:bg-cafe24-yellow/20 hover:border-cafe24-orange/40 hover:shadow-sm transition active:translate-y-[1px] whitespace-nowrap"
      onClick={onClick}
      title="클릭하면 질문이 바로 전송됩니다"
      type="button"
    >
      <ShoppingBag size={14} className="text-cafe24-orange" />
      <span className="max-w-[220px] truncate">{label}</span>
      <ArrowUpRight size={14} className="text-cafe24-brown/50" />
    </button>
  );
}

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

function MarkdownMessage({ content }) {
  const remarkPlugins = useMemo(() => [remarkMath, remarkGfmPlugin], []);

  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      rehypePlugins={[rehypeKatex]}
      components={{
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
      }}
    >
      {content || ''}
    </ReactMarkdown>
  );
}

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
  // M46: 이미 렌더된 메시지 ID 추적 → 새 메시지만 애니메이션
  const seenMsgIdsRef = useRef(new Set());


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
      '셀러 이탈 예측 분석 보여줘',
      '셀러 활동 현황',
      '코호트 리텐션 분석',
      'KPI 트렌드 분석',
      '이상거래 탐지 현황',
      '셀러 세그먼트 통계',
      '대시보드 전체 현황',
      `${shopId} 쇼핑몰 정보 알려줘`,
      'Premium 등급 쇼핑몰 목록',
      'CS 문의 통계 보여줘',
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
              const isUser = m.role === 'user';
              const isPending = !!m?._pending;
              // M46: 새 메시지만 애니메이션
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
                      <span>{isUser ? auth?.username || 'USER' : 'CAFE24 AI'}</span>

                      {!isUser && isPending ? (
                        <span className="inline-flex items-center gap-2 text-cafe24-orange">
                          <span className="h-3 w-3 rounded-full border-2 border-cafe24-yellow border-t-cafe24-orange animate-spin" />
                          <span className="text-[10px]">streaming</span>
                        </span>
                      ) : null}
                    </div>

                    <div className="prose prose-sm max-w-none">
                      {!isUser && isPending ? <TypingDots /> : <MarkdownMessage content={m.content || ''} />}
                    </div>

                    <ToolCalls toolCalls={m.tool_calls} />

                    {/* 호버 시 나타나는 액션 버튼 */}
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
                            onClick={() => {
                              handleSend(m.content || '');
                            }}
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
                onClick={() => {
                  handleSend(c);
                  setInput('');
                }}
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
