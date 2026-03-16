// hooks/useConsultingStream.js — 셀러 컨설팅 전용 SSE 스트리밍 훅
// useBaseStream 공통 로직 기반, 4단계 워크플로우 + 롤백 지원

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import toast from 'react-hot-toast';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { makeBasicAuthHeader } from '@/lib/api';

const CONSULTING_STEPS = [
  { key: 'diagnosis', label: '진단', number: 1 },
  { key: 'strategy', label: '전략 수립', number: 2 },
  { key: 'plan', label: '실행 계획', number: 3 },
  { key: 'execution', label: '실행', number: 4 },
];

function newMsgId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export { CONSULTING_STEPS };

export default function useConsultingStream({ auth, settings }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStep, setCurrentStep] = useState(null);
  const [stepHistory, setStepHistory] = useState([]); // [{step, status:'active'|'completed', description}]
  const [awaitingInput, setAwaitingInput] = useState(null); // {step, prompt, options}
  const [sessionId, setSessionId] = useState(null);
  const [toolCalls, setToolCalls] = useState([]);
  const [activeAgent, setActiveAgent] = useState(null);

  const abortRef = useRef(null);
  const timeoutRef = useRef(null);
  const flushTimerRef = useRef(null);
  const stoppedRef = useRef(false);
  const runIdRef = useRef(0);
  const activeAssistantIdRef = useRef(null);
  const msgIndexRef = useRef(-1);

  // 클린업
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const resetSession = useCallback(() => {
    setMessages([]);
    setIsLoading(false);
    setError(null);
    setCurrentStep(null);
    setStepHistory([]);
    setAwaitingInput(null);
    setSessionId(null);
    setToolCalls([]);
    setActiveAgent(null);
    stoppedRef.current = false;
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const stopStream = useCallback(() => {
    setIsLoading(false);
    try {
      runIdRef.current += 1;
      stoppedRef.current = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
      const aid = activeAssistantIdRef.current;
      setMessages((prev) => {
        const arr = prev || [];
        let idx = msgIndexRef.current;
        let targetId = aid;
        if (idx >= 0 && idx < arr.length && arr[idx]?._id === targetId) {
          // 캐시 히트
        } else if (targetId) {
          idx = arr.findIndex((m) => m?._id === targetId);
        } else {
          const lastPending = [...arr].reverse().find((m) => m?.role === 'assistant' && m?._pending);
          targetId = lastPending?._id || null;
          if (!targetId) return arr;
          idx = arr.findIndex((m) => m?._id === targetId);
        }
        if (idx < 0) return arr;
        const msg = arr[idx] || {};
        const content = String(msg.content || '').trim();
        if (!content) return arr.filter((m) => m?._id !== targetId);
        const next = arr.slice();
        next[idx] = { ...arr[idx], content: content + '\n\n[중단됨]', _pending: false };
        return next;
      });
      activeAssistantIdRef.current = null;
    } catch (e) {
      activeAssistantIdRef.current = null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendMessage = useCallback(
    async (userInput, { sellerId, action = 'message', strategyChoice = null, rollbackTarget = null } = {}) => {
      const q = String(userInput || '').trim();
      if (!q && action === 'message') return;

      stopStream();
      stoppedRef.current = false;
      runIdRef.current += 1;
      const myRunId = runIdRef.current;

      setIsLoading(true);
      setError(null);
      setAwaitingInput(null);

      // 사용자 메시지 추가 (action이 message일 때만)
      const assistantId = newMsgId();
      activeAssistantIdRef.current = assistantId;

      if (action === 'message' && q) {
        const userMsg = { _id: newMsgId(), role: 'user', content: q };
        const assistantMsg = {
          _id: assistantId,
          role: 'assistant',
          content: '',
          _pending: true,
        };
        setMessages((prev) => {
          const arr = [...(prev || []), userMsg, assistantMsg];
          msgIndexRef.current = arr.length - 1;
          return arr;
        });
      } else {
        // 옵션 선택 등 비메시지 액션
        const label = strategyChoice || rollbackTarget || q || action;
        const userMsg = { _id: newMsgId(), role: 'user', content: `[${label}]`, _isAction: true };
        const assistantMsg = {
          _id: assistantId,
          role: 'assistant',
          content: '',
          _pending: true,
        };
        setMessages((prev) => {
          const arr = [...(prev || []), userMsg, assistantMsg];
          msgIndexRef.current = arr.length - 1;
          return arr;
        });
      }

      const username = auth?.username || '';
      const password = auth?.password_b64 || auth?.password || '';
      const isB64 = !!auth?.password_b64;

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      timeoutRef.current = setTimeout(() => {
        try {
          stoppedRef.current = true;
          ctrl.abort();
        } catch (e) {}
      }, 180000);

      let deltaBuf = '';
      const DELTA_FLUSH_THRESHOLD = 500;

      const flushDelta = () => {
        if (!deltaBuf) return;
        const chunk = deltaBuf;
        deltaBuf = '';
        setMessages((prev) => {
          const arr = prev || [];
          const idx = msgIndexRef.current;
          if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
          const m = arr[idx];
          const next = arr.slice();
          next[idx] = m?._pending
            ? { ...m, content: chunk, _pending: false }
            : { ...m, content: String(m.content || '') + chunk, _pending: false };
          return next;
        });
      };

      const isStale = () =>
        myRunId !== runIdRef.current ||
        stoppedRef.current ||
        ctrl.signal.aborted ||
        activeAssistantIdRef.current !== assistantId;

      try {
        await fetchEventSource('/api/agent/consulting/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            Authorization: makeBasicAuthHeader(username, password, isB64),
          },
          body: JSON.stringify({
            sellerId: sellerId || null,
            userInput: q || action,
            sessionId: sessionId,
            action: action,
            strategyChoice: strategyChoice,
            rollbackTarget: rollbackTarget,
            model: settings?.selectedModel || 'gpt-4o-mini',
          }),
          signal: ctrl.signal,
          openWhenHidden: true,

          async onopen(res) {
            if (isStale()) return;
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const ct = res.headers.get('content-type') || '';
            if (!ct.includes('text/event-stream')) throw new Error('Not an SSE response');
          },

          onmessage(ev) {
            if (isStale()) return;

            let data = {};
            try {
              data = ev.data ? JSON.parse(ev.data) : {};
            } catch (e) {
              return;
            }

            // step_change: 단계 전환
            if (ev.event === 'step_change') {
              const { step, step_number, total, description } = data;
              setCurrentStep(step);
              setStepHistory((prev) => {
                // 기존 단계를 completed로, 새 단계를 active로
                const updated = prev.map((s) =>
                  s.status === 'active' ? { ...s, status: 'completed' } : s
                );
                const exists = updated.find((s) => s.step === step);
                if (exists) {
                  return updated.map((s) =>
                    s.step === step ? { ...s, status: 'active', description } : s
                  );
                }
                return [...updated, { step, step_number, total, description, status: 'active' }];
              });

              // 단계 전환 구분선 메시지 삽입
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const stepLabel = description || step;
                const divider = `\n\n---\n**${step_number}단계: ${stepLabel}**\n\n`;
                const next = arr.slice();
                next[idx] = m?._pending
                  ? { ...m, content: divider, _pending: false }
                  : { ...m, content: String(m.content || '') + divider };
                return next;
              });
              return;
            }

            // awaiting_input: 사용자 입력 대기
            if (ev.event === 'awaiting_input') {
              setAwaitingInput({
                step: data.step,
                prompt: data.prompt,
                options: data.options || [],
              });
              return;
            }

            // agent_start
            if (ev.event === 'agent_start') {
              setActiveAgent(data.agent);
              return;
            }

            // agent_end
            if (ev.event === 'agent_end') {
              setActiveAgent(null);
              return;
            }

            // tool_start
            if (ev.event === 'tool_start') {
              const toolName = data.tool || '도구';
              setToolCalls((prev) => [
                ...prev,
                { tool: toolName, args: data.args || {}, status: 'running', timestamp: Date.now() },
              ]);
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const next = arr.slice();
                const statusMsg = `\n🔧 **${toolName}** 실행 중...`;
                next[idx] = m?._pending
                  ? { ...m, content: statusMsg, _pending: false }
                  : { ...m, content: String(m.content || '') + statusMsg };
                return next;
              });
              return;
            }

            // tool_end
            if (ev.event === 'tool_end') {
              const toolName = data.tool || '도구';
              setToolCalls((prev) =>
                prev.map((t) =>
                  t.tool === toolName && t.status === 'running'
                    ? { ...t, status: 'done', result_preview: data.result_preview }
                    : t
                )
              );
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                let content = String(m.content || '');
                content = content.replace(`🔧 **${toolName}** 실행 중...`, `✅ **${toolName}** 완료`);
                const next = arr.slice();
                next[idx] = { ...m, content };
                return next;
              });
              return;
            }

            // delta
            if (ev.event === 'delta') {
              const delta = String(data.delta || '');
              if (!delta) return;
              deltaBuf += delta;
              if (deltaBuf.length >= DELTA_FLUSH_THRESHOLD) {
                if (flushTimerRef.current) {
                  clearTimeout(flushTimerRef.current);
                  flushTimerRef.current = null;
                }
                flushDelta();
                return;
              }
              if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
              flushTimerRef.current = setTimeout(() => {
                flushTimerRef.current = null;
                if (isStale()) return;
                flushDelta();
              }, 50);
              return;
            }

            // done
            if (ev.event === 'done') {
              if (isStale()) return;
              if (flushTimerRef.current) {
                clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
              }
              flushDelta();

              const finalText = String(data.final || '');
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const streamed = String(m.content || '').trim();
                const next = arr.slice();
                next[idx] = {
                  ...m,
                  content: streamed || finalText || '',
                  _pending: false,
                };
                return next;
              });

              // 세션 ID 저장
              if (data.session_id) {
                setSessionId(data.session_id);
              }

              // 완료된 단계 업데이트
              if (data.step) {
                setStepHistory((prev) =>
                  prev.map((s) =>
                    s.step === data.step ? { ...s, status: 'completed' } : s
                  )
                );
              }

              setIsLoading(false);
              setActiveAgent(null);
              if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
              }
              abortRef.current = null;
              activeAssistantIdRef.current = null;

              if (data.ok) {
                toast.success('처리 완료');
              }
              return;
            }

            // error
            if (ev.event === 'error') {
              if (isStale()) return;
              if (flushTimerRef.current) {
                clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
              }
              flushDelta();
              const msg = data?.message ? String(data.message) : '스트리밍 오류';
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const next = arr.slice();
                next[idx] = { ...m, content: String(m.content || '') + `\n\n[오류]\n${msg}`, _pending: false };
                return next;
              });
              setError(msg);
              toast.error(msg);
              return;
            }
          },

          onerror(err) {
            throw err;
          },

          onclose() {
            if (isStale()) return;
            if (!activeAssistantIdRef.current) return;
            throw new Error('SSE closed');
          },
        });
      } catch (e) {
        if (isStale()) {
          setIsLoading(false);
          return;
        }
        if (flushTimerRef.current) {
          clearTimeout(flushTimerRef.current);
          flushTimerRef.current = null;
        }
        flushDelta();
        const msg = String(e || '요청 실패');
        setMessages((prev) => {
          const arr = prev || [];
          const idx = msgIndexRef.current;
          if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
          const m = arr[idx];
          const next = arr.slice();
          next[idx] = { ...m, content: String(m.content || '') + `\n\n[오류]\n${msg}`, _pending: false };
          return next;
        });
        setError(msg);
        setIsLoading(false);
        toast.error('요청 실패');
      } finally {
        if (flushTimerRef.current) {
          clearTimeout(flushTimerRef.current);
          flushTimerRef.current = null;
        }
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        abortRef.current = null;
        if (activeAssistantIdRef.current === assistantId) {
          activeAssistantIdRef.current = null;
        }
      }
    },
    [auth, settings, sessionId, stopStream]
  );

  const rollbackTo = useCallback(
    (step) => {
      // 롤백 시 해당 단계 이후 히스토리 제거
      setStepHistory((prev) => {
        const idx = prev.findIndex((s) => s.step === step);
        if (idx < 0) return prev;
        return prev.slice(0, idx + 1).map((s, i) =>
          i === idx ? { ...s, status: 'active' } : s
        );
      });
      setCurrentStep(step);
      setAwaitingInput(null);
      // 롤백 메시지 전송
      sendMessage(`${step} 단계로 롤백`, {
        action: 'rollback',
        rollbackTarget: step,
      });
    },
    [sendMessage]
  );

  return {
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
  };
}
