// M64: JSON 프록시 공통 유틸 사용
import { createJSONProxyHandler } from '@/lib/sseProxy';

export const config = {
  api: { bodyParser: false, responseLimit: false, externalResolver: true },
};

export default createJSONProxyHandler({
  target: '/api/cs/send-reply',
  allowedMethods: 'GET,POST,OPTIONS',
  forwardAuth: true,
  logPrefix: 'cs send-reply proxy',
});
