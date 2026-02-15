// hooks/useSubAgentStream.js — 서브에이전트 전용 SSE 스트리밍 훅
// useAgentStream.js 패턴 기반, agent_start/agent_end 이벤트 추가

import { useState, useCallback, useEffect, useRef } from 'react';
import toast from 'react-hot-toast';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { makeBasicAuthHeader } from '@/lib/api';

const WAITING_PLACEHOLDER = ['서브에이전트 분석 중입니다.', '잠시 기다려주세요.'].join('\n');

function newMsgId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export default function useSubAgentStream({ auth, selectedShop, settings }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [stepResults, setStepResults] = useState({});
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle|running|done|error

  const abortRef = useRef(null);
  const timeoutRef = useRef(null);
  const flushTimerRef = useRef(null);
  const stoppedRef = useRef(false);
  const runIdRef = useRef(0);
  const activeAssistantIdRef = useRef(null);

  // 컴포넌트 언마운트 시 타이머 클린업
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
    };
  }, []);

  const resetPipeline = useCallback(() => {
    setSteps([]);
    setCurrentStep(-1);
    setStepResults({});
    setPipelineStatus('idle');
    setError(null);
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

        let targetId = aid;
        if (!targetId) {
          const lastPending = [...arr].reverse().find((m) => m?.role === 'assistant' && m?._pending);
          targetId = lastPending?._id || null;
        }
        if (!targetId) return arr;

        const idx = arr.findIndex((m) => m?._id === targetId);
        if (idx < 0) return arr;

        const msg = arr[idx] || {};
        const content = String(msg.content || '').trim();
        const isPending = !!msg._pending;
        const isOnlyWaiting = content === String(WAITING_PLACEHOLDER).trim();

        if (!content || isPending || isOnlyWaiting) return arr.filter((m) => m?._id !== targetId);

        return arr.map((m) => {
          if (m?._id !== targetId) return m;
          const cur = String(m.content || '');
          return { ...m, content: cur + '\n\n[중단됨]', _pending: false };
        });
      });

      activeAssistantIdRef.current = null;
    } catch (e) {
      activeAssistantIdRef.current = null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendMessage = useCallback(
    async (input) => {
      const q = String(input || '').trim();
      if (!q) return;

      stopStream();

      stoppedRef.current = false;
      runIdRef.current += 1;
      const myRunId = runIdRef.current;

      // 파이프라인 상태 초기화
      resetPipeline();
      setIsLoading(true);
      setError(null);
      setPipelineStatus('running');

      const userMsg = { _id: newMsgId(), role: 'user', content: q };
      const assistantId = newMsgId();
      activeAssistantIdRef.current = assistantId;

      const assistantMsg = {
        _id: assistantId,
        role: 'assistant',
        content: WAITING_PLACEHOLDER,
        tool_calls: [],
        _pending: true,
      };

      setMessages((prev) => [...(prev || []), userMsg, assistantMsg]);

      const systemPromptToSend =
        settings?.systemPrompt && String(settings.systemPrompt).trim().length > 0
          ? String(settings.systemPrompt)
          : '';

      const username = auth?.username || '';
      const password = auth?.password || '';

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const timeoutMs = 120000; // 서브에이전트는 더 오래 걸릴 수 있으므로 120초
      timeoutRef.current = setTimeout(() => {
        try {
          stoppedRef.current = true;
          ctrl.abort();
        } catch (e) {}
      }, timeoutMs);

      let deltaBuf = '';

      const flushDelta = () => {
        if (!deltaBuf) return;
        const chunk = deltaBuf;
        deltaBuf = '';

        setMessages((prev) =>
          (prev || []).map((m) => {
            if (m?._id !== assistantId) return m;

            const isPending = !!m?._pending;
            if (isPending) return { ...m, content: chunk, _pending: false };
            return { ...m, content: String(m.content || '') + chunk, _pending: false };
          })
        );
      };

      const isStale = () =>
        myRunId !== runIdRef.current ||
        stoppedRef.current ||
        ctrl.signal.aborted ||
        activeAssistantIdRef.current !== assistantId;

      try {
        await fetchEventSource('/api/agent/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            Authorization: makeBasicAuthHeader(username, password),
          },
          body: JSON.stringify({
            user_input: q,
            sub_agent: true,
            shop_id: selectedShop || null,
            api_key: settings?.apiKey || '',
            model: settings?.selectedModel || 'gpt-4o-mini',
            max_tokens: Number(settings?.maxTokens ?? 4000),
            temperature: Number(settings?.temperature ?? 0.3),
            system_prompt: systemPromptToSend,
            rag_mode: settings?.ragMode || 'auto',
            debug: true,
          }),
          signal: ctrl.signal,

          async onopen(res) {
            if (isStale()) return;
            const ct = res.headers.get('content-type') || '';
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
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

            switch (ev.event) {
              case 'agent_start': {
                const { agent, step, total_steps, description } = data;
                setCurrentStep(step);
                setSteps((prev) => [
                  ...prev,
                  { agent, step, total_steps, description: description || agent, status: 'running' },
                ]);
                setPipelineStatus('running');
                break;
              }

              case 'agent_end': {
                const { agent, step, summary, result_summary } = data;
                const resultText = summary || result_summary || '';
                setStepResults((prev) => ({ ...prev, [step]: resultText }));
                setSteps((prev) =>
                  prev.map((s) => (s.step === step ? { ...s, status: 'done' } : s))
                );
                break;
              }

              case 'tool_start': {
                const toolName = data.tool || '도구';
                setMessages((prev) => {
                  const arr = prev || [];
                  const idx = arr.findIndex((m) => m?._id === assistantId);
                  if (idx < 0) return arr;
                  const m = arr[idx];
                  const statusMsg = `🔧 **${toolName}** 실행 중...`;
                  const updated = m?._pending
                    ? { ...m, content: statusMsg, _pending: true }
                    : { ...m, content: String(m.content || '') + '\n' + statusMsg, _pending: true };
                  const next = arr.slice();
                  next[idx] = updated;
                  return next;
                });
                break;
              }

              case 'tool_end': {
                const toolName = data.tool || '도구';
                setMessages((prev) => {
                  const arr = prev || [];
                  const idx = arr.findIndex((m) => m?._id === assistantId);
                  if (idx < 0) return arr;
                  const m = arr[idx];
                  let content = String(m.content || '');
                  content = content.replace(`🔧 **${toolName}** 실행 중...`, `✅ **${toolName}** 완료`);
                  const next = arr.slice();
                  next[idx] = { ...m, content, _pending: true };
                  return next;
                });
                break;
              }

              case 'delta': {
                const delta = String(data.delta || '');
                if (!delta) return;

                deltaBuf += delta;

                if (!flushTimerRef.current) {
                  flushTimerRef.current = setTimeout(() => {
                    flushTimerRef.current = null;
                    if (isStale()) return;
                    flushDelta();
                  }, 50);
                }
                break;
              }

              case 'done': {
                if (isStale()) return;

                if (flushTimerRef.current) {
                  clearTimeout(flushTimerRef.current);
                  flushTimerRef.current = null;
                }
                flushDelta();

                const finalText = String(data.final || '');
                const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];

                setMessages((prev) =>
                  (prev || []).map((m) => {
                    if (m?._id !== assistantId) return m;
                    return {
                      ...m,
                      content: finalText || String(m.content || ''),
                      tool_calls: toolCalls,
                      _pending: false,
                    };
                  })
                );

                // done 이벤트의 agent_results로 누락된 stepResults 보충
                const agentResults = Array.isArray(data.agent_results) ? data.agent_results : [];
                if (agentResults.length > 0) {
                  setStepResults((prev) => {
                    const merged = { ...prev };
                    agentResults.forEach((ar) => {
                      if (ar.step && ar.summary && !merged[ar.step]) {
                        merged[ar.step] = ar.summary;
                      }
                    });
                    return merged;
                  });
                }

                // 모든 단계를 완료 상태로
                setSteps((prev) => prev.map((s) => ({ ...s, status: 'done' })));
                setPipelineStatus('done');
                setIsLoading(false);

                if (timeoutRef.current) {
                  clearTimeout(timeoutRef.current);
                  timeoutRef.current = null;
                }
                abortRef.current = null;
                activeAssistantIdRef.current = null;

                toast.success('서브에이전트 분석 완료');
                break;
              }

              case 'error': {
                if (isStale()) return;

                if (flushTimerRef.current) {
                  clearTimeout(flushTimerRef.current);
                  flushTimerRef.current = null;
                }
                flushDelta();

                const msg = data?.message ? String(data.message) : '스트리밍 오류';

                setMessages((prev) =>
                  (prev || []).map((m) => {
                    if (m?._id !== assistantId) return m;
                    const cur = String(m.content || '');
                    return { ...m, content: cur + `\n\n[오류]\n${msg}`, _pending: false };
                  })
                );

                setError(msg);
                setPipelineStatus('error');
                toast.error(msg);
                break;
              }

              default:
                break;
            }
          },

          onerror(err) {
            throw err;
          },

          onclose() {
            if (isStale()) return;
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

        setMessages((prev) =>
          (prev || []).map((m) => {
            if (m?._id !== assistantId) return m;
            const cur = String(m.content || '');
            return { ...m, content: cur + `\n\n[오류]\n${msg}`, _pending: false };
          })
        );

        setError(msg);
        setPipelineStatus('error');
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
    [auth, settings, selectedShop, stopStream, resetPipeline]
  );

  return {
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
  };
}
