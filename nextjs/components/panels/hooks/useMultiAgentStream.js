// hooks/useMultiAgentStream.js — 멀티에이전트 전용 SSE 스트리밍 훅
// useBaseStream 공통 로직 기반, Supervisor 패턴: 동적 에이전트 활동 추적

import { useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import useBaseStream from './useBaseStream';

const MULTI_WAITING = '';

export default function useMultiAgentStream({ auth, selectedShop, settings }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [steps, setSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(-1);
  const [stepResults, setStepResults] = useState({});
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle|running|done|error
  const [toolExecutions, setToolExecutions] = useState([]);
  const [stepTimings, setStepTimings] = useState({});
  const [stepProgress, setStepProgress] = useState(null);
  // Supervisor 패턴: 현재 활성 에이전트 + 에이전트 이력
  const [activeAgent, setActiveAgent] = useState(null);
  const [agentHistory, setAgentHistory] = useState([]); // [{agent, status:'active'|'done', elapsed_ms?}]

  const resetPipeline = useCallback(() => {
    setSteps([]);
    setCurrentStep(-1);
    setStepResults({});
    setPipelineStatus('idle');
    setError(null);
    setToolExecutions([]);
    setStepTimings({});
    setStepProgress(null);
    setActiveAgent(null);
    setAgentHistory([]);
  }, []);

  // 멀티에이전트 전용 이벤트: agent_start, agent_end, tool_start, tool_end, step_progress
  const onExtraEvent = useCallback((ev, data) => {
    if (ev.event === 'agent_start') {
      const { agent, step, total_steps, description } = data;
      setCurrentStep(step);
      setActiveAgent(agent);
      setSteps((prev) => [
        ...prev,
        { agent, step, total_steps, description: description || agent, status: 'running' },
      ]);
      // 에이전트 이력에 추가 (이미 있으면 active로 변경)
      setAgentHistory((prev) => {
        const existing = prev.find((a) => a.agent === agent);
        if (existing) {
          return prev.map((a) => a.agent === agent ? { ...a, status: 'active' } : a);
        }
        return [...prev, { agent, status: 'active' }];
      });
      setPipelineStatus('running');
      return true;
    }

    if (ev.event === 'agent_end') {
      const { agent, step, summary, result_summary, elapsed_ms } = data;
      const resultText = summary || result_summary || '';
      setStepResults((prev) => ({ ...prev, [step]: resultText }));
      setSteps((prev) =>
        prev.map((s) => (s.step === step ? { ...s, status: 'done' } : s))
      );
      if (elapsed_ms) {
        setStepTimings((prev) => ({ ...prev, [agent]: elapsed_ms }));
      }
      // 에이전트 이력 업데이트
      setAgentHistory((prev) =>
        prev.map((a) => a.agent === agent ? { ...a, status: 'done', elapsed_ms } : a)
      );
      setActiveAgent(null);
      return true;
    }

    if (ev.event === 'tool_start') {
      setToolExecutions((prev) => [
        ...prev,
        { tool: data.tool, args: data.args || {}, status: 'running', timestamp: Date.now() },
      ]);
      return true;
    }

    if (ev.event === 'tool_end') {
      const endStatus = data.status === 'error' ? 'error' : 'done';
      setToolExecutions((prev) =>
        prev.map((t) =>
          t.tool === data.tool && t.status === 'running'
            ? { ...t, status: endStatus, result_preview: data.result_preview }
            : t
        )
      );
      return true;
    }

    if (ev.event === 'step_progress') {
      setStepProgress({ step: data.step, progress: data.progress, detail: data.detail });
      return true;
    }

    return false;
  }, []);

  // done 이벤트 시 상태 업데이트
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
    setAgentHistory((prev) => prev.map((a) => ({ ...a, status: 'done' })));
    setActiveAgent(null);
    setPipelineStatus('done');
    setStepProgress(null);
  }, []);

  const onSendDone = useCallback(() => {
    toast.success('멀티에이전트 분석 완료');
  }, []);

  const onError = useCallback((msg) => {
    setError(msg);
    setPipelineStatus('error');
    setActiveAgent(null);
  }, []);

  const onCatchError = useCallback((msg) => {
    setError(msg);
    setPipelineStatus('error');
    setActiveAgent(null);
  }, []);

  // 멀티턴: 새 메시지 전송 시 에이전트 활동 상태만 초기화 (대화 이력 유지)
  const onBeforeSend = useCallback(() => {
    setSteps([]);
    setCurrentStep(-1);
    setStepResults({});
    setError(null);
    setToolExecutions([]);
    setStepTimings({});
    setStepProgress(null);
    setActiveAgent(null);
    setAgentHistory([]);
    setPipelineStatus('running');
  }, []);

  const bodyExtra = useMemo(() => ({ multi_agent: true }), []);

  const { sendMessage, stopStream } = useBaseStream({
    auth,
    selectedShop,
    settings,
    setMessages,
    setLoading: setIsLoading,
    timeoutMs: 120000,
    waitingText: MULTI_WAITING,
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
    toolExecutions,
    stepTimings,
    stepProgress,
    activeAgent,
    agentHistory,
  };
}
