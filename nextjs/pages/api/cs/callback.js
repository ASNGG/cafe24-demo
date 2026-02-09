export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'content-type');
    res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
    res.end();
    return;
  }

  const backendBase = process.env.BACKEND_INTERNAL_URL || 'http://127.0.0.1:8001';
  const target = `${backendBase}/api/cs/callback`;

  try {
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const body = Buffer.concat(chunks);

    const upstream = await fetch(target, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
    });

    const data = await upstream.text();
    res.statusCode = upstream.status;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(data);
  } catch (e) {
    console.error('[cs callback proxy error]', e);
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ status: 'FAILED', error: String(e?.message || e) }));
  }
}
