// hooks/useAutomationStream.js — 자동화 탭 SSE 스트리밍 공통 훅
// retention/report/upgrade 탭에서 공통 사용

import { useCallback, useRef, useState } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { makeBasicAuthHeader } from '@/lib/api';

export default function useAutomationStream({ auth }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [stepStatuses, setStepStatuses] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const startStream = useCallback(async ({ endpoint, data, onEvent, onDone, onError }) => {
    setIsStreaming(true);
    setError(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const username = auth?.username || '';
    const password = auth?.password_b64 || auth?.password || '';
    const isB64 = !!auth?.password_b64;

    try {
      await fetchEventSource(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: makeBasicAuthHeader(username, password, isB64),
        },
        body: JSON.stringify(data),
        signal: ctrl.signal,
        openWhenHidden: true,

        onmessage(ev) {
          if (!ev.data) return;
          let parsed;
          try {
            parsed = JSON.parse(ev.data);
          } catch {
            return;
          }

          if (ev.event === 'step_start') {
            setCurrentStep(parsed.step);
            setStepStatuses(prev => ({ ...prev, [parsed.step]: { status: 'processing' } }));
          } else if (ev.event === 'step_end') {
            setStepStatuses(prev => ({
              ...prev,
              [parsed.step]: {
                status: 'complete',
                detail: parsed.detail || (parsed.elapsed_ms ? `${parsed.elapsed_ms}ms` : undefined),
              },
            }));
            setCurrentStep(null);
          } else if (ev.event === 'done') {
            setIsStreaming(false);
            onDone?.(parsed);
          } else if (ev.event === 'error') {
            setError(parsed.message);
            setIsStreaming(false);
            onError?.(parsed.message);
          }

          // 커스텀 이벤트 핸들러 (step_start/step_end 포함 모든 이벤트 전달)
          onEvent?.(ev.event, parsed);
        },

        onerror(err) {
          setIsStreaming(false);
          setError('스트리밍 연결 오류');
          onError?.('스트리밍 연결 오류');
          throw err; // fetchEventSource 재시도 방지
        },
      });
    } catch (e) {
      if (e.name !== 'AbortError') {
        setIsStreaming(false);
        setError(e.message);
      }
    }
  }, [auth]);

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const resetStatuses = useCallback(() => {
    setStepStatuses({});
    setCurrentStep(null);
    setError(null);
  }, []);

  return { startStream, stopStream, isStreaming, stepStatuses, currentStep, error, resetStatuses, setStepStatuses };
}
