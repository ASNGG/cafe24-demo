// pages/app.js - CAFE24 AI 운영 플랫폼
// 카페24 이커머스 AI 기반 내부 시스템

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';

import dynamic from 'next/dynamic';

import Layout from '@/components/Layout';
import Tabs from '@/components/Tabs';

const PanelLoader = () => (
  <div className="animate-pulse p-6 space-y-4">
    <div className="h-6 bg-gray-200 rounded w-1/3"></div>
    <div className="h-4 bg-gray-200 rounded w-2/3"></div>
    <div className="h-4 bg-gray-200 rounded w-1/2"></div>
  </div>
);

const AgentPanel = dynamic(() => import('@/components/panels/AgentPanel'), { ssr: false, loading: PanelLoader });
const DashboardPanel = dynamic(() => import('@/components/panels/DashboardPanel'), { ssr: false, loading: PanelLoader });
const AnalysisPanel = dynamic(() => import('@/components/panels/AnalysisPanel'), { ssr: false, loading: PanelLoader });
const ModelsPanel = dynamic(() => import('@/components/panels/ModelsPanel'), { ssr: false, loading: PanelLoader });
const SettingsPanel = dynamic(() => import('@/components/panels/SettingsPanel'), { ssr: false, loading: PanelLoader });
const UsersPanel = dynamic(() => import('@/components/panels/UsersPanel'), { ssr: false, loading: PanelLoader });
const LogsPanel = dynamic(() => import('@/components/panels/LogsPanel'), { ssr: false, loading: PanelLoader });
const RagPanel = dynamic(() => import('@/components/panels/RagPanel'), { ssr: false, loading: PanelLoader });
const LabPanel = dynamic(() => import('@/components/panels/LabPanel'), { ssr: false, loading: PanelLoader });
const GuardianPanel = dynamic(() => import('@/components/panels/GuardianPanel'), { ssr: false, loading: PanelLoader });
const AutomationPanel = dynamic(() => import('@/components/panels/AutomationPanel'), { ssr: false, loading: PanelLoader });

import { apiCall as apiCallRaw } from '@/lib/api';
import {
  loadFromStorage,
  saveToStorage,
  loadFromSession,
  removeFromSession,
  STORAGE_KEYS,
} from '@/lib/storage';

// CAFE24 AI 운영 플랫폼 예시 질문 (agent/tools.py 31개 도구 기반, 도구당 1개)
const EXAMPLE_QUESTIONS = {
  '🛒 쇼핑몰 & 플랫폼': [
    'S0001 쇼핑몰 상세 정보 알려줘',          // get_shop_info
    'S0001 이용 중인 서비스 목록 보여줘',       // get_shop_services
    '패션 카테고리 쇼핑몰 목록 보여줘',         // list_shops
    '카테고리 전체 목록 보여줘',               // list_categories
    '뷰티 카테고리 상세 정보 알려줘',           // get_category_info
    '이커머스 용어 GMV 설명해줘',              // get_ecommerce_glossary
  ],
  '📦 CS & 운영': [
    'CS 문의 카테고리별 통계 보여줘',           // get_cs_statistics
    '"배송 늦어요 환불해주세요" 문의 분류해줘',   // classify_inquiry
    '"결제 오류 해결해주세요" 자동 답변 생성해줘', // auto_reply_cs
    '최근 30일 주문 이벤트 통계 보여줘',         // get_order_statistics
  ],
  '🔮 AI 예측': [
    'SEL0001 이탈 확률 예측해줘',              // predict_seller_churn
    '고위험 이탈 셀러 목록 보여줘',             // get_churn_prediction + get_at_risk_sellers
    'S0001 다음 달 매출 예측해줘',             // predict_shop_revenue
    'S0001 쇼핑몰 성과 분석해줘',              // get_shop_performance
    'SEL0001 마케팅 예산 최적화 추천해줘',       // optimize_marketing
    'SEL0001 이상거래 조사해줘',               // detect_fraud
    '이상거래 전체 통계 보여줘',               // get_fraud_statistics
  ],
  '📈 비즈니스 KPI': [
    '최근 7일 KPI 트렌드 분석해줘',            // get_trend_analysis
    '코호트 리텐션 분석 보여줘',               // get_cohort_analysis
    '이번 달 GMV 예측해줘',                   // get_gmv_prediction
    '대시보드 전체 현황 요약해줘',              // get_dashboard_summary
  ],
  '👤 셀러 분석': [
    'SEL0001 셀러 종합 분석해줘',              // analyze_seller
    'SEL0001 셀러 세그먼트 분류해줘',           // get_seller_segment
    '셀러 세그먼트별 통계 보여줘',              // get_segment_statistics
    'SEL0001 최근 30일 활동 리포트',           // get_seller_activity_report
  ],
  '❓ 카페24 FAQ': [
    '카페24 결제수단 설정 방법 알려줘',          // search_platform (RAG)
    '취소/교환/반품 처리 방법 알려줘',           // search_platform (RAG)
    '쿠폰 생성하고 관리하는 방법',              // search_platform_lightrag
    '디컬렉션이 뭐야?',                       // search_platform_lightrag
  ],
};

const DEFAULT_SETTINGS = {
  apiKey: '',
  selectedModel: 'gpt-4o-mini',
  maxTokens: 8000,
  temperature: 0.3,
  systemPrompt: '',
  ragMode: 'rag', // 'rag' | 'lightrag' | 'k2rag' | 'auto'
};

function formatTimestamp(d) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
    d.getMinutes()
  )}:${pad(d.getSeconds())}`;
}

export default function AppPage() {
  const router = useRouter();

  const [auth, setAuth] = useState(null);
  const [shops, setShops] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedShop, setSelectedShop] = useState(null);

  const [settings, setSettings] = useState(null);
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  const [activityLog, setActivityLog] = useState([]);

  const [activeTab, setActiveTab] = useState('agent');

  const isAdmin = auth?.user_role === '관리자';

  const tabs = useMemo(() => {
    if (isAdmin) {
      return [
        { key: 'agent', label: '🤖 AI 에이전트' },
        { key: 'dashboard', label: '📊 대시보드' },
        { key: 'analysis', label: '📈 분석' },
        { key: 'models', label: '🧠 ML 모델' },
        { key: 'rag', label: '📚 RAG 문서' },
        { key: 'lab', label: '🧪 실험실 - CS 자동화 파이프라인' },
        { key: 'guardian', label: '🔒 실험실 - DB 보안 감시' },
        { key: 'automation', label: '⚡ 자동화 엔진' },
        { key: 'settings', label: '⚙️ LLM 설정' },
        { key: 'users', label: '👥 셀러 관리' },
        { key: 'logs', label: '📋 로그' },
      ];
    }
    return [
      { key: 'agent', label: '🤖 AI 에이전트' },
      { key: 'dashboard', label: '📊 대시보드' },
      { key: 'analysis', label: '📈 분석' },
      { key: 'lab', label: '🧪 실험실 - CS 자동화 파이프라인' },
      { key: 'guardian', label: '🔒 실험실 - DB 보안 감시' },
      { key: 'automation', label: '⚡ 자동화 엔진' },
    ];
  }, [isAdmin]);

  // apiCallRaw는 모듈 스코프 함수이므로 안정 참조 유지
  const apiCall = useCallback((args) => apiCallRaw(args), []);

  const addLog = useCallback(
    (action, detail) => {
      const row = {
        시간: formatTimestamp(new Date()),
        사용자: auth?.username || '-',
        작업: action,
        상세: detail,
      };
      setActivityLog((prev) => [...prev, row]);
    },
    [auth?.username]
  );

  const safeReplace = useCallback(
    (path) => {
      if (!router.isReady) return;
      const cur = router.asPath || '';
      if (cur === path) return;
      router.replace(path);
    },
    [router]
  );

  const onLogout = useCallback(() => {
    removeFromSession(STORAGE_KEYS.AUTH);
    safeReplace('/login');
  }, [safeReplace]);

  const clearLog = useCallback(() => {
    setActivityLog([]);
  }, []);

  // 반응형 zoom: 작은 화면에서 축소, 큰 화면에서 기본
  useEffect(() => {
    function applyZoom() {
      document.documentElement.style.zoom = window.innerWidth < 1280 ? '0.85' : '0.9';
    }
    applyZoom();
    window.addEventListener('resize', applyZoom);
    return () => {
      window.removeEventListener('resize', applyZoom);
      document.documentElement.style.zoom = '1';
    };
  }, []);

  useEffect(() => {
    if (!router.isReady) return;

    const a = loadFromSession(STORAGE_KEYS.AUTH, null);
    if (!a?.username || !a?.password_b64) {
      safeReplace('/login');
      return;
    }
    setAuth(a);

    const savedSettings = loadFromStorage(STORAGE_KEYS.SETTINGS, null);
    const mergedSettings = { ...DEFAULT_SETTINGS, ...(savedSettings || {}) };
    if (!mergedSettings.apiKey || mergedSettings.apiKey.trim() === '') {
      mergedSettings.apiKey = DEFAULT_SETTINGS.apiKey;
    }
    setSettings(mergedSettings);
    setSettingsLoaded(true);

    setActivityLog(loadFromStorage(STORAGE_KEYS.ACTIVITY_LOG, []));
  }, [router.isReady, safeReplace]);

  const systemPromptLoadedRef = useRef(false);

  useEffect(() => {
    if (!auth?.username || !auth?.password_b64) return;
    if (systemPromptLoadedRef.current) return;

    const cur = settings?.systemPrompt ? String(settings.systemPrompt).trim() : '';
    if (cur.length > 0) {
      systemPromptLoadedRef.current = true;
      return;
    }

    systemPromptLoadedRef.current = true;
    let mounted = true;

    async function loadSystemPrompt() {
      try {
        const res = await apiCall({
          endpoint: '/api/settings/prompt',
          method: 'GET',
          auth,
          timeoutMs: 30000,
        });

        if (!mounted) return;

        const data = res?.data || res || {};
        const prompt = data?.systemPrompt || data?.system_prompt || '';
        const promptStr = String(prompt || '').trim();

        if (promptStr.length > 0) {
          setSettings((prev) => ({ ...prev, systemPrompt: promptStr }));
        }
      } catch (e) {
        try {
          const fallback = await apiCall({
            endpoint: '/api/settings/default',
            method: 'GET',
            auth,
            timeoutMs: 30000,
          });

          if (!mounted) return;

          const prompt = fallback?.data?.systemPrompt || fallback?.data?.system_prompt || '';
          const promptStr = String(prompt || '').trim();

          if (promptStr.length > 0) {
            setSettings((prev) => ({ ...prev, systemPrompt: promptStr }));
          }
        } catch (e2) {}
      }
    }

    loadSystemPrompt();

    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiCall, auth]);

  // 쇼핑몰/카테고리 데이터 로드
  useEffect(() => {
    if (!auth?.username || !auth?.password_b64) return;

    let mounted = true;

    async function loadShops() {
      try {
        const res = await apiCall({ endpoint: '/api/shops', auth, timeoutMs: 30000 });
        if (!mounted) return;

        if (res?.status === 'success' && Array.isArray(res.shops)) {
          setShops(res.shops);
          if (!selectedShop && res.shops.length > 0) {
            setSelectedShop(res.shops[0].id);
          }
        }
      } catch (e) {
        console.error('Failed to load shops:', e);
      }
    }

    async function loadCategories() {
      try {
        const res = await apiCall({ endpoint: '/api/categories', auth, timeoutMs: 30000 });
        if (!mounted) return;

        if (res?.status === 'success' && Array.isArray(res.categories)) {
          setCategories(res.categories);
        }
      } catch (e) {
        console.error('Failed to load categories:', e);
      }
    }

    loadShops();
    loadCategories();

    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiCall, auth]);

  // localStorage 저장 통합 debounce (300ms)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (settingsLoaded && settings) {
        saveToStorage(STORAGE_KEYS.SETTINGS, settings);
      }
      saveToStorage(STORAGE_KEYS.ACTIVITY_LOG, activityLog);
    }, 300);
    return () => clearTimeout(timer);
  }, [settings, settingsLoaded, activityLog]);

  const onExampleQuestion = useCallback((q) => {
    setActiveTab('agent');
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('cafe24_send_question', { detail: { q } }));
    }
  }, []);

  if (!auth) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-cafe24-yellow/20 via-white to-cafe24-orange/10 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cafe24-yellow to-cafe24-orange shadow-lg flex items-center justify-center animate-bounce">
              <span className="text-3xl font-black text-white">C24</span>
            </div>
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-14 h-3 bg-cafe24-orange/20 rounded-full blur-sm animate-pulse"></div>
          </div>
          <div className="mt-6 text-cafe24-brown font-bold text-lg">로딩 중...</div>
          <div className="mt-2 flex justify-center gap-1">
            <span className="w-2 h-2 bg-cafe24-yellow rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="w-2 h-2 bg-cafe24-yellow rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="w-2 h-2 bg-cafe24-yellow rounded-full animate-bounce"></span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Layout
      auth={auth}
      exampleQuestions={EXAMPLE_QUESTIONS}
      onExampleQuestion={onExampleQuestion}
      onLogout={onLogout}
    >
      <div className="mb-4 animate-slide-up">
        <div className="flex items-center gap-3">
          <span className="text-3xl font-black text-cafe24-yellow">C24</span>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-cafe24-brown">CAFE24 AI Platform</h1>
              {settings?.selectedModel?.includes("mini") && (
                <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1.5 rounded-full font-bold whitespace-nowrap">
                  GPT-4o mini 모드
                </span>
              )}
            </div>
            <p className="text-sm text-cafe24-brown/70">이커머스 운영 · AI 에이전트 · 데이터 분석</p>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-cafe24-yellow/20 text-cafe24-brown">
            GPT-4 기반
          </span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-cafe24-orange/15 text-cafe24-orange">
            CAFE24
          </span>
        </div>
      </div>

      <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

      <div key={activeTab} className="animate-fade-in">
      {activeTab === 'agent' ? (
          <AgentPanel
            auth={auth}
            selectedShop={selectedShop}
            addLog={addLog}
            settings={settings}
            apiCall={apiCall}
          />
      ) : null}

      {activeTab === 'dashboard' ? (
        <DashboardPanel auth={auth} selectedShop={selectedShop} apiCall={apiCall} />
      ) : null}

      {activeTab === 'analysis' ? <AnalysisPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'models' && isAdmin ? <ModelsPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'rag' && isAdmin ? <RagPanel auth={auth} apiCall={apiCall} addLog={addLog} settings={settings} setSettings={setSettings} /> : null}

      {activeTab === 'settings' && isAdmin ? (
        <SettingsPanel settings={settings} setSettings={setSettings} addLog={addLog} apiCall={apiCall} auth={auth} />
      ) : null}

      {activeTab === 'users' && isAdmin ? <UsersPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'logs' && isAdmin ? (
        <LogsPanel activityLog={activityLog} clearLog={clearLog} />
      ) : null}

      {activeTab === 'lab' ? (
        <LabPanel auth={auth} apiCall={apiCall} settings={settings} />
      ) : null}

      {activeTab === 'guardian' ? (
        <GuardianPanel auth={auth} apiCall={apiCall} />
      ) : null}



      {activeTab === 'automation' ? (
        <AutomationPanel auth={auth} apiCall={apiCall} />
      ) : null}
      </div>
    </Layout>
  );
}
