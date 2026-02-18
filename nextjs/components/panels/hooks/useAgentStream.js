// hooks/useAgentStream.js — AgentPanel SSE 스트리밍 훅
// useBaseStream 공통 로직 기반 리팩토링

import { useCallback } from 'react';
import toast from 'react-hot-toast';
import useBaseStream from './useBaseStream';

export default function useAgentStream({
  auth,
  selectedShop,
  settings,
  setAgentMessages,
  setTotalQueries,
  setLoading,
  addLog,
}) {
  const onDone = useCallback(
    () => {
      setTotalQueries((prev) => (prev || 0) + 1);
    },
    [setTotalQueries]
  );

  const onSendDone = useCallback((ok) => {
    if (ok) toast.success('분석 완료');
    else toast.error('요청 실패: 백엔드/네트워크를 확인하세요');
  }, []);

  const { sendMessage, stopStream } = useBaseStream({
    auth,
    selectedShop,
    settings,
    setMessages: setAgentMessages,
    setLoading,
    addLog,
    timeoutMs: 60000,
    onDone,
    onSendDone,
  });

  return { sendQuestion: sendMessage, stopStream };
}
