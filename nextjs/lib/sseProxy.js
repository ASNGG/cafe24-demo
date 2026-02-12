// lib/sseProxy.js - SSE/JSON 프록시 공통 유틸 (H31: 3개 SSE 핸들러 통합)

export const sseProxyConfig = {
  api: {
    bodyParser: false,
    responseLimit: false,
    externalResolver: true,
  },
};

/**
 * SSE 프록시 핸들러 생성
 * @param {object} opts
 * @param {string} opts.target - 백엔드 경로 (예: '/api/agent/stream')
 * @param {string} [opts.allowedMethods] - 허용 메서드 (기본: 'GET,POST,OPTIONS')
 * @param {function} [opts.buildTarget] - (req, backendBase) => targetUrl 커스텀 함수
 * @param {string} [opts.logPrefix] - 에러 로그 접두어
 */
export function createSSEProxyHandler({ target, allowedMethods = 'GET,POST,OPTIONS', buildTarget, logPrefix = 'sse proxy' }) {
  return async function handler(req, res) {
    if (req.method === 'OPTIONS') {
      res.statusCode = 204;
      res.setHeader('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
      res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, accept');
      res.setHeader('Access-Control-Allow-Methods', allowedMethods);
      res.end();
      return;
    }

    const backendBase = process.env.BACKEND_INTERNAL_URL || 'http://127.0.0.1:8001';
    const targetUrl = buildTarget ? buildTarget(req, backendBase) : `${backendBase}${target}`;

    try {
      const headers = {
        'content-type': req.headers['content-type'] || 'application/json',
        'authorization': req.headers['authorization'] || '',
        'accept': 'text/event-stream',
        'cache-control': 'no-cache',
        'connection': 'keep-alive',
      };

      const init = { method: req.method, headers };

      if (req.method !== 'GET' && req.method !== 'HEAD') {
        init.body = req;
        init.duplex = 'half';
      }

      const upstream = await fetch(targetUrl, init);

      res.statusCode = upstream.status;
      res.setHeader('Content-Type', upstream.headers.get('content-type') || 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.setHeader('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
      res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, accept');
      res.setHeader('Access-Control-Allow-Methods', allowedMethods);

      if (typeof res.flushHeaders === 'function') res.flushHeaders();

      if (!upstream.body) {
        res.end();
        return;
      }

      const reader = upstream.body.getReader();

      req.on('close', () => {
        try { reader.cancel(); } catch (e) {}
        try { res.end(); } catch (e) {}
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) res.write(Buffer.from(value));
      }

      res.end();
    } catch (e) {
      console.error(`[${logPrefix} error]`, e);
      res.statusCode = 500;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.end(JSON.stringify({ status: 'error', message: String(e?.message || e) }));
    }
  };
}

/**
 * JSON 프록시 핸들러 생성 (M64: callback/send-reply 통합)
 * @param {object} opts
 * @param {string} opts.target - 백엔드 경로 (예: '/api/cs/callback')
 * @param {string} [opts.allowedMethods] - 허용 메서드
 * @param {boolean} [opts.forwardAuth] - authorization 헤더 포워딩 여부
 * @param {string} [opts.logPrefix] - 에러 로그 접두어
 */
export function createJSONProxyHandler({ target, allowedMethods = 'POST,OPTIONS', forwardAuth = false, logPrefix = 'json proxy' }) {
  return async function handler(req, res) {
    if (req.method === 'OPTIONS') {
      res.statusCode = 204;
      res.setHeader('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
      res.setHeader('Access-Control-Allow-Headers', forwardAuth ? 'authorization, content-type, accept' : 'content-type');
      res.setHeader('Access-Control-Allow-Methods', allowedMethods);
      res.end();
      return;
    }

    const backendBase = process.env.BACKEND_INTERNAL_URL || 'http://127.0.0.1:8001';
    const targetUrl = `${backendBase}${target}`;

    try {
      const chunks = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);

      const headers = {
        'content-type': req.headers['content-type'] || 'application/json',
      };
      if (forwardAuth) {
        headers['authorization'] = req.headers['authorization'] || '';
      }

      const upstream = await fetch(targetUrl, {
        method: 'POST',
        headers,
        body,
      });

      const data = await upstream.text();
      res.statusCode = upstream.status;
      res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json; charset=utf-8');
      if (forwardAuth) {
        res.setHeader('Access-Control-Allow-Origin', process.env.CORS_ORIGIN || '*');
        res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, accept');
        res.setHeader('Access-Control-Allow-Methods', allowedMethods);
      }
      res.end(data);
    } catch (e) {
      console.error(`[${logPrefix} error]`, e);
      res.statusCode = 500;
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.end(JSON.stringify({ status: 'error', message: String(e?.message || e) }));
    }
  };
}
