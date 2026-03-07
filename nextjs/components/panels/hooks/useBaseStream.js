// hooks/useBaseStream.js — SSE 스트리밍 공통 훅
// useAgentStream + useMultiAgentStream 70% 중복 로직 통합

import { useCallback, useEffect, useRef } from 'react';
import toast from 'react-hot-toast';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { makeBasicAuthHeader } from '@/lib/api';

export const WAITING_PLACEHOLDER = ['답변 생성 중입니다.', '잠시 기다려주세요.'].join('\n');

export function newMsgId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/**
 * SSE 스트리밍 공통 훅
 * @param {Object} opts
 * @param {Object} opts.auth - { username, password }
 * @param {string} opts.selectedShop
 * @param {Object} opts.settings - { apiKey, selectedModel, maxTokens, temperature, systemPrompt, ragMode }
 * @param {Function} opts.setMessages - 메시지 상태 setter
 * @param {Function} opts.setLoading - 로딩 상태 setter
 * @param {number} [opts.timeoutMs=60000] - SSE 타임아웃
 * @param {string} [opts.waitingText] - 대기 플레이스홀더 텍스트
 * @param {Object} [opts.bodyExtra={}] - fetchEventSource body에 추가할 필드
 * @param {Function} [opts.onExtraEvent] - 기본 이벤트 외 추가 이벤트 핸들러 (ev, data, assistantId) => boolean
 * @param {Function} [opts.onDone] - done 이벤트 추가 처리 (data) => void
 * @param {Function} [opts.onError] - error 이벤트 추가 처리 (msg) => void
 * @param {Function} [opts.onCatchError] - catch 블록 추가 처리 (msg) => void
 * @param {Function} [opts.onBeforeSend] - 메시지 전송 전 콜백 () => void
 * @param {Function} [opts.onSendDone] - 전송 완료 토스트 (ok) => void
 * @param {Function} [opts.addLog] - 로그 추가 함수
 */
export default function useBaseStream({
  auth,
  selectedShop,
  settings,
  setMessages,
  setLoading,
  timeoutMs = 60000,
  waitingText = WAITING_PLACEHOLDER,
  bodyExtra = {},
  onExtraEvent,
  onDone,
  onError,
  onCatchError,
  onBeforeSend,
  onSendDone,
  addLog,
}) {
  const abortRef = useRef(null);
  const timeoutRef = useRef(null);
  const flushTimerRef = useRef(null);
  const stoppedRef = useRef(false);
  const runIdRef = useRef(0);
  const activeAssistantIdRef = useRef(null);
  const msgIndexRef = useRef(-1);

  // 컴포넌트 언마운트 시 모든 타이머/abort 클린업
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
  }, []);

  const stopStream = useCallback(() => {
    setLoading(false);

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

        // msgIndexRef 활용하여 O(1) 접근 시도
        let idx = msgIndexRef.current;
        let targetId = aid;

        // 캐시된 인덱스 유효성 검증
        if (idx >= 0 && idx < arr.length && arr[idx]?._id === targetId) {
          // 캐시 히트
        } else if (targetId) {
          // 캐시 미스: fallback으로 findIndex
          idx = arr.findIndex((m) => m?._id === targetId);
        } else {
          // aid 없을 때: 마지막 pending assistant 찾기
          const lastPending = [...arr].reverse().find((m) => m?.role === 'assistant' && m?._pending);
          targetId = lastPending?._id || null;
          if (!targetId) return arr;
          idx = arr.findIndex((m) => m?._id === targetId);
        }
        if (idx < 0) return arr;

        const msg = arr[idx] || {};
        const content = String(msg.content || '').trim();
        // P0 버그 수정: isPending 조건 제거 → 실제 content가 있으면 [중단됨] 추가
        const isOnlyWaiting = content === String(waitingText).trim();

        if (!content || isOnlyWaiting) return arr.filter((m) => m?._id !== targetId);

        const next = arr.slice();
        const cur = String(arr[idx].content || '');
        next[idx] = { ...arr[idx], content: cur + '\n\n[중단됨]', _pending: false };
        return next;
      });

      activeAssistantIdRef.current = null;
    } catch (e) {
      activeAssistantIdRef.current = null;
    } finally {
      setLoading(false);
    }
  }, [setMessages, setLoading, waitingText]);

  const sendMessage = useCallback(
    async (question) => {
      const q = String(question || '').trim();
      if (!q) return;

      stopStream();

      stoppedRef.current = false;
      runIdRef.current += 1;
      const myRunId = runIdRef.current;

      // 전송 전 콜백
      onBeforeSend?.();
      setLoading(true);
      addLog?.('질문', q.slice(0, 30));

      const userMsg = { _id: newMsgId(), role: 'user', content: q };
      const assistantId = newMsgId();
      activeAssistantIdRef.current = assistantId;

      const assistantMsg = {
        _id: assistantId,
        role: 'assistant',
        content: waitingText,
        tool_calls: [],
        _pending: true,
      };

      setMessages((prev) => {
        const arr = [...(prev || []), userMsg, assistantMsg];
        msgIndexRef.current = arr.length - 1;
        return arr;
      });

      const systemPromptToSend =
        settings?.systemPrompt && String(settings.systemPrompt).trim().length > 0
          ? String(settings.systemPrompt)
          : '';

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
      }, timeoutMs);

      let deltaBuf = '';
      const DELTA_FLUSH_THRESHOLD = 500; // 버퍼가 이 크기 이상이면 즉시 flush

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
        await fetchEventSource('/api/agent/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            Authorization: makeBasicAuthHeader(username, password, isB64),
          },
          body: JSON.stringify({
            user_input: q,
            shop_id: selectedShop || null,
            api_key: settings?.apiKey || '',
            model: settings?.selectedModel || 'gpt-4o-mini',
            max_tokens: Number(settings?.maxTokens ?? 4000),
            temperature: Number(settings?.temperature ?? 0.3),
            system_prompt: systemPromptToSend,
            rag_mode: settings?.ragMode || 'auto',
            debug: process.env.NODE_ENV === 'development',
            ...bodyExtra,
          }),
          signal: ctrl.signal,
          openWhenHidden: true,

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

            // 추가 이벤트 핸들러 (true 반환 시 기본 처리 스킵)
            if (onExtraEvent?.(ev, data, assistantId)) return;

            if (ev.event === 'tool_start') {
              const toolName = data.tool || '도구';
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const statusMsg = `🔧 **${toolName}** 실행 중...`;
                // _pending 유지: 초기 상태면 true 유지, delta 수신 후면 false 유지
                // → 다음 delta가 기존 content를 덮어쓰지 않도록 방지
                const updated = m?._pending
                  ? { ...m, content: statusMsg, _pending: true }
                  : { ...m, content: String(m.content || '') + '\n' + statusMsg };
                const next = arr.slice();
                next[idx] = updated;
                return next;
              });
              return;
            }

            if (ev.event === 'tool_end') {
              const toolName = data.tool || '도구';
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

            if (ev.event === 'delta') {
              const delta = String(data.delta || '');
              if (!delta) return;

              deltaBuf += delta;

              // 버퍼가 임계값 이상이면 즉시 flush (메모리 누적 방지)
              if (deltaBuf.length >= DELTA_FLUSH_THRESHOLD) {
                if (flushTimerRef.current) {
                  clearTimeout(flushTimerRef.current);
                  flushTimerRef.current = null;
                }
                flushDelta();
                return;
              }

              // debounce: 마지막 delta 기준 50ms 후 flush
              if (flushTimerRef.current) {
                clearTimeout(flushTimerRef.current);
              }
              flushTimerRef.current = setTimeout(() => {
                flushTimerRef.current = null;
                if (isStale()) return;
                flushDelta();
              }, 50);
              return;
            }

            if (ev.event === 'done') {
              if (isStale()) return;

              if (flushTimerRef.current) {
                clearTimeout(flushTimerRef.current);
                flushTimerRef.current = null;
              }
              flushDelta();

              const finalText = String(data.final || '');
              const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];
              setMessages((prev) => {
                const arr = prev || [];
                const idx = msgIndexRef.current;
                if (idx < 0 || idx >= arr.length || arr[idx]?._id !== assistantId) return arr;
                const m = arr[idx];
                const next = arr.slice();
                next[idx] = {
                  ...m,
                  content: finalText || String(m.content || ''),
                  tool_calls: toolCalls,
                  _pending: false,
                };
                return next;
              });

              // 추가 done 처리
              onDone?.(data);

              setLoading(false);

              if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
              }
              abortRef.current = null;
              activeAssistantIdRef.current = null;

              // 완료 토스트
              if (onSendDone) {
                onSendDone(!!data.ok);
              } else {
                if (data.ok) toast.success('분석 완료');
                else toast.error('요청 실패: 백엔드/네트워크를 확인하세요');
              }
              return;
            }

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

              onError?.(msg);
              toast.error(msg);
              return;
            }
          },

          onerror(err) {
            throw err;
          },

          onclose() {
            if (isStale()) return;
            // done 이벤트 후 정상 종료 시에는 에러 불필요
            if (!activeAssistantIdRef.current) return;
            throw new Error('SSE closed');
          },
        });
      } catch (e) {
        if (isStale()) {
          setLoading(false);
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

        onCatchError?.(msg);
        setLoading(false);
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
    [addLog, auth, settings, setMessages, setLoading, stopStream, selectedShop, timeoutMs, waitingText, bodyExtra, onExtraEvent, onDone, onError, onCatchError, onBeforeSend, onSendDone]
  );

  return { sendMessage, stopStream };
}
