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
const ProcessMinerPanel = dynamic(() => import('@/components/panels/ProcessMinerPanel'), { ssr: false, loading: PanelLoader });
const AutomationPanel = dynamic(() => import('@/components/panels/AutomationPanel'), { ssr: false, loading: PanelLoader });
const SubAgentPanel = dynamic(() => import('@/components/panels/SubAgentPanel'), { ssr: false, loading: PanelLoader });

import { apiCall as apiCallRaw } from '@/lib/api';
import {
  loadFromStorage,
  saveToStorage,
  loadFromSession,
  removeFromSession,
  STORAGE_KEYS,
} from '@/lib/storage';

// CAFE24 AI 운영 플랫폼 예시 질문 (agent/tools.py AVAILABLE_TOOLS 기반)
const EXAMPLE_QUESTIONS = {
  '🛒 쇼핑몰 & 플랫폼': [
    'S0001 쇼핑몰 정보 알려줘',
    'S0010 쇼핑몰 서비스 구성 알려줘',
    'Premium 등급 쇼핑몰 목록 보여줘',
    '패션 카테고리 쇼핑몰 현황',
    '쇼핑몰 플랜별 분포 보여줘',
    '카테고리 정보 전체 목록',
    '뷰티 카테고리 상세 정보',
    '이커머스 용어 GMV 설명해줘',
    '이커머스 용어집 보여줘',
    '플랫폼 전체 쇼핑몰 수 알려줘',
  ],
  '📦 CS & 운영': [
    'CS 문의 통계 보여줘',
    '"배송이 너무 늦어요 환불해주세요" CS 자동 분류해줘',
    '"결제가 안 돼요 카드 오류 떠요" 카테고리 분류',
    'CS 문의 카테고리별 현황 알려줘',
    '최근 30일 주문 이벤트 통계 보여줘',
    '환불 관련 CS 현황',
  ],
  '🔮 AI 예측 분석': [
    'SEL0001 셀러 이탈 확률 예측해줘',
    'SEL0100 이탈 위험도 분석해줘',
    'SEL0050 셀러 이탈할 것 같아?',
    '전체 이탈 예측 분석 결과 보여줘',
    '고위험 이탈 셀러 5명 보여줘',
    '이탈 요인 상위 5개 뭐야?',
    'S0001 쇼핑몰 다음달 매출 예측해줘',
    'S0010 쇼핑몰 성과 분석',
    'SEL0001 마케팅 예산 최적화 추천해줘',
    'SEL0100 ROI 최대화 전략 알려줘',
    '이상거래 전체 통계 보여줘',
    '이상거래 탐지 현황 알려줘',
  ],
  '📈 비즈니스 KPI': [
    '최근 7일 KPI 트렌드 분석해줘',
    '최근 14일 GMV 변화율 알려줘',
    '최근 7일 활성 셀러 변화 분석해줘',
    '최근 7일 신규 가입 추이 알려줘',
    '최근 7일 주문 수 변화 분석해줘',
    '코호트 리텐션 분석 보여줘',
    '2024-11 코호트 리텐션 어때?',
    '전체 코호트 Week 4 평균 리텐션 얼마야?',
    '이번 달 GMV 예측해줘',
    '최근 30일 매출 분석해줘',
    'AOV랑 ARPU 알려줘',
    '대시보드 전체 현황 요약해줘',
  ],
  '👤 셀러 분석': [
    'SEL0001 셀러 분석해줘',
    'SEL0050 셀러 프로필 알려줘',
    'SEL0100 행동 패턴 분석해줘',
    '셀러 세그먼트별 통계 보여줘',
    '파워 셀러 세그먼트 몇 명이야?',
    '우수 셀러 세그먼트 통계 알려줘',
    '휴면 셀러 세그먼트 현황 알려줘',
    '이상 셀러 전체 통계 보여줘',
    'SEL0001 최근 30일 활동 리포트',
    'SEL0100 최근 7일 활동 보여줘',
    '최근 30일 운영 이벤트 통계 보여줘',
    '최근 30일 정산 이벤트 통계 보여줘',
  ],
  '❓ 카페24 FAQ': [
    '카페24 결제수단 설정 방법 알려줘',
    '배송 설정은 어떻게 하나요?',
    '상품 등록 방법 알려줘',
    '무통장입금 계좌 설정 방법은?',
    '취소/교환/반품/환불 처리 방법',
    '쿠폰 생성하고 관리하는 방법',
    '적립금 설정 방법 알려줘',
    '디컬렉션이 뭐야?',
    '마켓플러스 사용 방법 알려줘',
    '게시판 설정 방법은?',
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

  const [agentMessages, setAgentMessages] = useState([]);
  const [activityLog, setActivityLog] = useState([]);
  const [totalQueries, setTotalQueries] = useState(0);

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
        { key: 'process-miner', label: '⛏️ 실험실 - 프로세스 마이너' },
        { key: 'sub-agent', label: '🧬 개발중 - 서브에이전트' },
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
      { key: 'process-miner', label: '⛏️ 실험실 - 프로세스 마이너' },
      { key: 'sub-agent', label: '🧬 개발중 - 서브에이전트' },
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

    setAgentMessages(loadFromStorage(STORAGE_KEYS.AGENT_MESSAGES, []));
    setActivityLog(loadFromStorage(STORAGE_KEYS.ACTIVITY_LOG, []));
    setTotalQueries(loadFromStorage(STORAGE_KEYS.TOTAL_QUERIES, 0));
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
      saveToStorage(STORAGE_KEYS.AGENT_MESSAGES, agentMessages);
      saveToStorage(STORAGE_KEYS.ACTIVITY_LOG, activityLog);
      saveToStorage(STORAGE_KEYS.TOTAL_QUERIES, totalQueries);
    }, 300);
    return () => clearTimeout(timer);
  }, [settings, settingsLoaded, agentMessages, activityLog, totalQueries]);

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
      <div className="mb-4">
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

      {activeTab === 'agent' ? (
          <AgentPanel
            auth={auth}
            selectedShop={selectedShop}
            addLog={addLog}
            settings={settings}
            setSettings={setSettings}
            agentMessages={agentMessages}
            setAgentMessages={setAgentMessages}
            totalQueries={totalQueries}
            setTotalQueries={setTotalQueries}
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

      {activeTab === 'process-miner' ? (
        <ProcessMinerPanel auth={auth} apiCall={apiCall} />
      ) : null}

      {activeTab === 'sub-agent' ? (
        <SubAgentPanel auth={auth} selectedShop={selectedShop} addLog={addLog} settings={settings} apiCall={apiCall} />
      ) : null}

      {activeTab === 'automation' ? (
        <AutomationPanel auth={auth} apiCall={apiCall} />
      ) : null}
    </Layout>
  );
}
