export const config = {
  api: {
    bodyParser: false,
    responseLimit: false,
    externalResolver: true,
  },
};

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, accept');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.end();
    return;
  }

  const backendBase = process.env.BACKEND_INTERNAL_URL || 'http://127.0.0.1:8001';
  const target = `${backendBase}/api/cs/pipeline/answer`;

  try {
    const headers = {
      'content-type': req.headers['content-type'] || 'application/json',
      'authorization': req.headers['authorization'] || '',
      'accept': 'text/event-stream',
      'cache-control': 'no-cache',
      'connection': 'keep-alive',
    };

    const init = {
      method: req.method,
      headers,
    };

    if (req.method !== 'GET' && req.method !== 'HEAD') {
      init.body = req;
      init.duplex = 'half';
    }

    const upstream = await fetch(target, init);

    res.statusCode = upstream.status;

    res.setHeader(
      'Content-Type',
      upstream.headers.get('content-type') || 'text/event-stream; charset=utf-8'
    );
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, accept');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');

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
    console.error('[cs pipeline-answer proxy error]', e);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ status: 'error', message: String(e?.message || e) }));
  }
}
