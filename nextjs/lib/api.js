// GET 요청용 인메모리 캐시 (TTL 60초, 정적 데이터 전용)
const _apiCache = new Map();
const CACHE_TTL = 60_000;

// 캐시 대상 엔드포인트 (정적 데이터만)
const CACHEABLE_ENDPOINTS = ['/api/shops', '/api/categories'];

function _getCacheKey(endpoint, auth) {
  return `${endpoint}::${auth?.username || ''}`;
}

function _getCached(key) {
  const entry = _apiCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL) {
    _apiCache.delete(key);
    return null;
  }
  return entry.data;
}

function _setCache(key, data) {
  _apiCache.set(key, { data, ts: Date.now() });
}

export function getApiBase() {
  // ✅ 중요: 외부 접속에서도 동작하게 기본값을 '같은 오리진'으로 둠
  // - 로컬 개발에서 백엔드가 다른 호스트/포트면 NEXT_PUBLIC_API_BASE를 지정
  //   예) NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
  const base = process.env.NEXT_PUBLIC_API_BASE || '';
  return String(base).replace(/\/$/, '');
}

/**
 * Basic Auth 헤더 생성
 * @param {string} username
 * @param {string} passwordOrB64 - 평문 비밀번호 또는 btoa 인코딩된 비밀번호
 * @param {boolean} [isB64=false] - true면 passwordOrB64를 atob으로 디코딩 후 사용
 */
export function makeBasicAuthHeader(username, passwordOrB64, isB64 = false) {
  if (typeof window === 'undefined') return '';
  const password = isB64 ? window.atob(passwordOrB64) : passwordOrB64;
  const token = window.btoa(`${username}:${password}`);
  return `Basic ${token}`;
}

export async function apiCall({
  endpoint,
  method = 'GET',
  data = null,
  auth = null,
  timeoutMs = 60000,
  headers = {},
  responseType = 'json',
  cache = 'no-store',
}) {
  // GET 캐시 히트 확인 (정적 데이터 엔드포인트만)
  if (method === 'GET' && CACHEABLE_ENDPOINTS.includes(endpoint)) {
    const cacheKey = _getCacheKey(endpoint, auth);
    const cached = _getCached(cacheKey);
    if (cached) return cached;
  }

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  const base = getApiBase();
  const url = `${base}${endpoint}`;

  const init = {
    method,
    cache,
    signal: controller.signal,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  };

  if (auth?.username && (auth?.password || auth?.password_b64)) {
    const pw = auth.password_b64 || auth.password;
    const isB64 = !!auth.password_b64;
    init.headers['Authorization'] = makeBasicAuthHeader(auth.username, pw, isB64);
  }

  if (method !== 'GET' && method !== 'HEAD' && data !== null) {
    init.body = JSON.stringify(data);
  }

  try {
    const resp = await fetch(url, init);
    clearTimeout(t);

    if (responseType === 'blob') {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.blob();
    }

    const json = await resp.json().catch(() => ({}));

    // 성공한 GET 정적 데이터 캐시 저장
    if (method === 'GET' && CACHEABLE_ENDPOINTS.includes(endpoint) && json?.status === 'success') {
      const cacheKey = _getCacheKey(endpoint, auth);
      _setCache(cacheKey, json);
    }

    return json;
  } catch (e) {
    clearTimeout(t);
    return { status: 'error', message: String(e?.message || e) };
  }
}
