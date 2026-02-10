// lib/sse.js - SSE 스트림 파싱 및 인증 헤더 공통 유틸

/**
 * Basic 인증 헤더를 포함한 fetch 헤더 객체 생성
 * @param {object} auth - { username, password }
 * @param {object} [extra] - 추가 헤더
 * @returns {object} headers
 */
export function createAuthHeaders(auth, extra = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...extra,
  };
  if (auth?.username && auth?.password) {
    headers['Authorization'] = 'Basic ' + btoa(`${auth.username}:${auth.password}`);
  }
  return headers;
}

/**
 * SSE(Server-Sent Events) 스트림을 파싱하여 핸들러에 전달
 *
 * @param {Response} response - fetch Response 객체 (body가 ReadableStream)
 * @param {object} handlers
 * @param {function} [handlers.onMessage] - 모든 파싱된 SSE 메시지 수신 (parsed JSON)
 * @param {function} [handlers.onToken]   - type === 'token' 이벤트
 * @param {function} [handlers.onDone]    - type === 'done' 이벤트
 * @param {function} [handlers.onError]   - type === 'error' 이벤트
 * @param {function} [handlers.onRagContext] - type === 'rag_context' 이벤트
 * @param {function} [handlers.onStep]    - type === 'step' 이벤트
 * @returns {Promise<string>} fullText - 누적된 토큰 텍스트
 */
export async function parseSSEStream(response, handlers = {}) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const parsed = JSON.parse(line.slice(6));

        if (handlers.onMessage) {
          handlers.onMessage(parsed);
        }

        switch (parsed.type) {
          case 'token':
            fullText += parsed.data;
            if (handlers.onToken) handlers.onToken(parsed.data, fullText);
            break;
          case 'done':
            if (handlers.onDone) handlers.onDone(parsed.data, fullText);
            break;
          case 'error':
            if (handlers.onError) handlers.onError(parsed.data);
            break;
          case 'rag_context':
            if (handlers.onRagContext) handlers.onRagContext(parsed.data);
            break;
          case 'step':
            if (handlers.onStep) handlers.onStep(parsed.data);
            break;
          default:
            break;
        }
      } catch {
        // JSON 파싱 실패 시 무시
      }
    }
  }

  return fullText;
}
