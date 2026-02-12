// H31: SSE 프록시 공통 유틸 사용
import { createSSEProxyHandler } from '@/lib/sseProxy';

export const config = {
  api: { bodyParser: false, responseLimit: false, externalResolver: true },
};

export default createSSEProxyHandler({
  target: '/api/cs/pipeline/answer',
  logPrefix: 'cs pipeline-answer proxy',
});
