// hooks/useSubAgentStream.js — 서브에이전트 전용 SSE 스트리밍 훅
// useBaseStream 공통 로직 기반, agent_start/agent_end/pipeline 상태 추가

import { useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import useBaseStream from './useBaseStream';

const SUB_WAITING = ['서브에이전트 분석 중입니다.', '잠시 기다려주세요.'].join('\n');

export default function useSubAgentStream({ auth, selectedShop, settings }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [stepResults, setStepResults] = useState({});
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle|running|done|error

  const resetPipeline = useCallback(() => {
    setSteps([]);
    setCurrentStep(-1);
    setStepResults({});
    setPipelineStatus('idle');
    setError(null);
  }, []);

  // 서브에이전트 전용 이벤트: agent_start, agent_end
  const onExtraEvent = useCallback((ev, data) => {
    if (ev.event === 'agent_start') {
      const { agent, step, total_steps, description } = data;
      setCurrentStep(step);
      setSteps((prev) => [
        ...prev,
        { agent, step, total_steps, description: description || agent, status: 'running' },
      ]);
      setPipelineStatus('running');
      return true;
    }

    if (ev.event === 'agent_end') {
      const { agent, step, summary, result_summary } = data;
      const resultText = summary || result_summary || '';
      setStepResults((prev) => ({ ...prev, [step]: resultText }));
      setSteps((prev) =>
        prev.map((s) => (s.step === step ? { ...s, status: 'done' } : s))
      );
      return true;
    }

    return false;
  }, []);

  // done 이벤트 시 파이프라인 상태 업데이트
  const onDone = useCallback((data) => {
    const agentResults = Array.isArray(data.agent_results) ? data.agent_results : [];
    setStepResults((prev) => {
      if (agentResults.length === 0) return prev;
      const merged = { ...prev };
      agentResults.forEach((ar) => {
        if (ar.step && ar.summary && !merged[ar.step]) {
          merged[ar.step] = ar.summary;
        }
      });
      return merged;
    });
    setSteps((prev) => prev.map((s) => ({ ...s, status: 'done' })));
    setPipelineStatus('done');
  }, []);

  const onSendDone = useCallback(() => {
    toast.success('서브에이전트 분석 완료');
  }, []);

  const onError = useCallback((msg) => {
    setError(msg);
    setPipelineStatus('error');
  }, []);

  const onCatchError = useCallback((msg) => {
    setError(msg);
    setPipelineStatus('error');
  }, []);

  const onBeforeSend = useCallback(() => {
    resetPipeline();
    setError(null);
    setPipelineStatus('running');
  }, [resetPipeline]);

  const bodyExtra = useMemo(() => ({ sub_agent: true }), []);

  const { sendMessage, stopStream } = useBaseStream({
    auth,
    selectedShop,
    settings,
    setMessages,
    setLoading: setIsLoading,
    timeoutMs: 120000,
    waitingText: SUB_WAITING,
    bodyExtra,
    onExtraEvent,
    onDone,
    onSendDone,
    onError,
    onCatchError,
    onBeforeSend,
  });

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
