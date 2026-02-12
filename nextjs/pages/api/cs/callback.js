// M64: JSON 프록시 공통 유틸 사용
import { createJSONProxyHandler } from '@/lib/sseProxy';

export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

export default createJSONProxyHandler({
  target: '/api/cs/callback',
  logPrefix: 'cs callback proxy',
});
