// SSE 프록시: 셀러 컨설팅 에이전트 스트리밍
import { createSSEProxyHandler } from '@/lib/sseProxy';

export const config = {
  api: { bodyParser: false, responseLimit: false, externalResolver: true },
};

export default createSSEProxyHandler({
  target: '/api/consulting/stream',
  logPrefix: 'consulting stream proxy',
});
