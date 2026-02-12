// H31: SSE 프록시 공통 유틸 사용
import { createSSEProxyHandler } from '@/lib/sseProxy';

export const config = {
  api: { bodyParser: false, responseLimit: false, externalResolver: true },
};

const sseHandler = createSSEProxyHandler({
  allowedMethods: 'GET,OPTIONS',
  buildTarget: (req, backendBase) => {
    return `${backendBase}/api/cs/stream?job_id=${encodeURIComponent(req.query.job_id)}`;
  },
  logPrefix: 'cs stream proxy',
});

// job_id 필수 검증 래퍼
export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    return sseHandler(req, res);
  }

  if (!req.query.job_id) {
    res.statusCode = 400;
    res.end(JSON.stringify({ status: 'error', message: 'job_id required' }));
    return;
  }

  return sseHandler(req, res);
}
