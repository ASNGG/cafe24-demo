import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { apiCall } from '@/lib/api';
import { saveToSession, loadFromSession, STORAGE_KEYS } from '@/lib/storage';
import { User, Lock, ChevronDown, ShoppingBag, BarChart3, Package, Truck, CreditCard } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [showAccounts, setShowAccounts] = useState(true);
  const [quickLoginUser, setQuickLoginUser] = useState('');

  useEffect(() => {
    const auth = loadFromSession(STORAGE_KEYS.AUTH, null);
    if (auth?.username && auth?.password_b64) router.replace('/app');
  }, [router]);

  async function onLogin() {
    setErr('');
    setLoading(true);

    const res = await apiCall({
      endpoint: '/api/login',
      method: 'POST',
      auth: { username, password },
      timeoutMs: 30000,
    });

    setLoading(false);

    if (res?.status === 'success') {
      const auth = {
        username,
        password_b64: window.btoa(password),
        user_name: res.user_name,
        user_role: res.user_role,
      };
      saveToSession(STORAGE_KEYS.AUTH, auth);
      router.replace('/app');
    } else {
      setErr('아이디 또는 비밀번호가 틀렸습니다');
    }
  }

  function fillAccount(user, pass) {
    setUsername(user);
    setPassword(pass);
  }

  async function quickLogin(acc) {
    setErr('');
    setQuickLoginUser(acc.user);
    setUsername(acc.user);
    setPassword(acc.pass);

    const res = await apiCall({
      endpoint: '/api/login',
      method: 'POST',
      auth: { username: acc.user, password: acc.pass },
      timeoutMs: 30000,
    });

    setQuickLoginUser('');

    if (res?.status === 'success') {
      const auth = {
        username: acc.user,
        password_b64: window.btoa(acc.pass),
        user_name: res.user_name,
        user_role: res.user_role,
      };
      saveToSession(STORAGE_KEYS.AUTH, auth);
      router.replace('/app');
    } else {
      setErr('로그인에 실패했습니다. 다시 시도해주세요.');
    }
  }

  const accounts = [
    { label: '관리자', user: 'admin', pass: 'admin123', role: 'Admin' },
    { label: '운영자', user: 'operator', pass: 'oper123', role: 'Operator' },
    { label: '분석가', user: 'analyst', pass: 'analyst123', role: 'Analyst' },
    { label: '사용자', user: 'user', pass: 'user123', role: 'User' },
  ];

  // 플로팅 아이콘 데이터
  const floatingIcons = [
    { Icon: ShoppingBag, top: '10%', left: '10%', size: 'w-16 h-16', delay: 0 },
    { Icon: BarChart3, top: '20%', right: '15%', size: 'w-14 h-14', delay: 0.5 },
    { Icon: Package, bottom: '15%', left: '20%', size: 'w-12 h-12', delay: 1.0 },
    { Icon: Truck, bottom: '30%', right: '10%', size: 'w-14 h-14', delay: 1.5 },
    { Icon: CreditCard, top: '60%', left: '8%', size: 'w-10 h-10', delay: 2.0 },
  ];

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[var(--bg)] relative overflow-hidden">
      {/* 배경 장식 */}
      <div className="pointer-events-none fixed inset-0">
        {/* 그라데이션 블러 */}
        <div className="absolute top-10 left-10 w-48 h-48 bg-cafe24-yellow/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-64 h-64 bg-cafe24-orange/15 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 right-1/4 w-32 h-32 bg-cafe24-yellow/15 rounded-full blur-2xl"></div>

        {/* 플로팅 이커머스 아이콘 */}
        {floatingIcons.map(({ Icon, size, delay, ...pos }, idx) => (
          <div
            key={idx}
            className={`absolute ${size} opacity-[0.08] cafe24-float`}
            style={{
              ...pos,
              animationDelay: `${delay}s`,
            }}
          >
            <Icon className="w-full h-full text-cafe24-yellow" />
          </div>
        ))}
      </div>

      <div className="w-full max-w-sm relative z-10 animate-login-card-in">
        {/* 헤더 */}
        <div className="text-center mb-8">
          <div className="mb-4 inline-block animate-login-logo-in hover:scale-110 hover:rotate-[10deg] transition-transform">
            <div className="w-20 h-20 mx-auto rounded-3xl bg-white shadow-lg flex items-center justify-center cafe24-float border-2 border-cafe24-orange/20" style={{ animationDuration: '2s' }}>
              <img src="https://img.echosting.cafe24.com/imgcafe24com/images/common/cafe24.svg" alt="CAFE24" className="w-14 h-14 object-contain" />
            </div>
          </div>
          <h1 className="text-xl font-semibold cafe24-text">CAFE24 AI Platform</h1>
          <p className="text-sm text-cafe24-brown/60 mt-1">이커머스 운영 · AI 에이전트 · 데이터 분석</p>
          <div className="mt-3 inline-flex items-center gap-1.5 bg-cafe24-beige px-3 py-1 rounded-full">
            <span className="text-xs font-medium text-cafe24-brown/70">CAFE24</span>
          </div>
        </div>

        {/* 서버 운영 안내 */}
        <div className="mb-4 text-center px-4 py-2.5 rounded-xl bg-cafe24-beige/60 border border-cafe24-yellow/15">
          <p className="text-xs text-cafe24-brown/60">
            서버 운영 시간: <span className="font-semibold text-cafe24-orange">AM 9:00 ~ PM 6:00</span> (KST)
          </p>
          <p className="text-[10px] text-cafe24-brown/40 mt-0.5">운영 시간 외에는 서버가 꺼져 있을 수 있습니다</p>
        </div>

        {/* 로그인 카드 */}
        <div className="bg-white rounded-2xl border border-[var(--border)] shadow-soft p-6">
          <div className="space-y-4">
            {/* 아이디 입력 */}
            <div>
              <label className="text-sm font-medium text-cafe24-brown mb-1.5 block">아이디</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-cafe24-brown/40" />
                <input
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-[var(--border2)] bg-white text-sm text-cafe24-brown placeholder:text-cafe24-brown/40 outline-none transition-all focus:border-cafe24-yellow focus:ring-2 focus:ring-cafe24-yellow/10"
                  placeholder="아이디를 입력하세요"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                />
              </div>
            </div>

            {/* 비밀번호 입력 */}
            <div>
              <label className="text-sm font-medium text-cafe24-brown mb-1.5 block">비밀번호</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-cafe24-brown/40" />
                <input
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-[var(--border2)] bg-white text-sm text-cafe24-brown placeholder:text-cafe24-brown/40 outline-none transition-all focus:border-cafe24-yellow focus:ring-2 focus:ring-cafe24-yellow/10"
                  placeholder="비밀번호를 입력하세요"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && username && password) onLogin();
                  }}
                />
              </div>
            </div>

            {/* 에러 메시지 */}
            {err && (
              <div className="rounded-lg bg-red-50 border border-red-100 px-3 py-2 text-sm text-red-600 animate-login-card-in">
                {err}
              </div>
            )}

            {/* 로그인 버튼 */}
            <button
              onClick={onLogin}
              disabled={loading || !username || !password}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white font-semibold text-base shadow-cafe24-sm transition-all hover:shadow-cafe24-lg hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            >
              {loading ? (
                <span className="inline-flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  로그인 중...
                </span>
              ) : (
                '로그인'
              )}
            </button>

            {/* 테스트 계정 - 클릭하면 바로 로그인 */}
            {accounts.length > 0 && (
            <div className="pt-2">
              <button
                onClick={() => setShowAccounts(!showAccounts)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-cafe24-beige transition-colors text-sm text-cafe24-brown"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">체험용 계정</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-cafe24-yellow/15 text-cafe24-orange font-medium">클릭 시 바로 로그인</span>
                </div>
                <ChevronDown className={`w-4 h-4 transition-transform ${showAccounts ? 'rotate-180' : ''}`} />
              </button>

              {showAccounts && (
                <div className="mt-2 space-y-1.5 animate-login-card-in">
                  {accounts.map((acc) => (
                    <button
                      key={acc.user}
                      onClick={() => quickLogin(acc)}
                      disabled={!!quickLoginUser}
                      className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl border border-cafe24-yellow/10 bg-gradient-to-r from-cafe24-light/50 to-transparent hover:from-cafe24-yellow/10 hover:border-cafe24-yellow/30 transition-all text-left group disabled:opacity-50"
                    >
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-cafe24-beige flex items-center justify-center group-hover:bg-cafe24-yellow/20 transition-colors">
                          <User className="w-4 h-4 text-cafe24-brown/50 group-hover:text-cafe24-orange transition-colors" />
                        </div>
                        <div>
                          <span className="text-sm font-medium text-cafe24-brown block">{acc.label}</span>
                          <span className="text-[11px] text-cafe24-brown/40">{acc.user} / {acc.pass}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-cafe24-beige text-cafe24-brown/60 group-hover:bg-cafe24-yellow/10 group-hover:text-cafe24-yellow transition-colors">
                          {acc.role}
                        </span>
                        {quickLoginUser === acc.user ? (
                          <span className="w-4 h-4 border-2 border-cafe24-orange/30 border-t-cafe24-orange rounded-full animate-spin" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5 -rotate-90 text-cafe24-brown/30 group-hover:text-cafe24-orange transition-colors" />
                        )}
                      </div>
                    </button>
                  ))}
                  <p className="text-[11px] text-cafe24-brown/40 text-center pt-1">
                    처음이시라면 <span className="text-cafe24-orange font-medium">관리자</span> 계정을 추천합니다
                  </p>
                </div>
              )}
            </div>
            )}
          </div>
        </div>

        {/* 푸터 */}
        <p className="mt-6 text-center text-xs text-cafe24-brown/40">
          &copy; 2026 CAFE24 &middot; AI 운영 플랫폼
        </p>
      </div>
    </div>
  );
}
