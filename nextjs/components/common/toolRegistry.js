/**
 * 에이전트 도구 레지스트리
 * backend 리팩토링 시작/agent/tools.py 기반 31개 도구 정보
 */

import { Store, UserSearch, BrainCircuit, Headphones, LayoutDashboard, ShieldCheck, Search, BarChart3 } from 'lucide-react';

export const TOOL_CATEGORIES = [
  {
    id: 'shop',
    name: '쇼핑몰 정보',
    icon: Store,
    color: 'blue',
    tools: [
      {
        name: 'get_shop_info',
        label: '쇼핑몰 상세 조회',
        description: '쇼핑몰 ID 또는 이름으로 상세 정보(카테고리, 티어, 매출 현황 등) 조회',
        params: ['shop_id'],
        returns: '쇼핑몰명, 카테고리, 티어, 월매출, 주문수, 전환율, 리뷰수',
        agents: ['search_agent'],
        example: 'S0001 쇼핑몰 상세 정보 알려줘',
      },
      {
        name: 'list_shops',
        label: '쇼핑몰 목록 조회',
        description: '카테고리, 티어, 지역으로 필터링하여 쇼핑몰 목록 조회',
        params: [],
        returns: '쇼핑몰 목록 (ID, 이름, 카테고리, 티어, 지역)',
        agents: ['search_agent'],
        example: '패션 카테고리 쇼핑몰 목록 보여줘',
      },
      {
        name: 'get_shop_services',
        label: '이용 서비스 조회',
        description: '특정 쇼핑몰의 이용 중인 서비스/앱 목록 조회',
        params: ['shop_id'],
        returns: '서비스명, 상태, 시작일, 요금제',
        agents: ['search_agent'],
        example: 'S0001 이용 중인 서비스 목록 보여줘',
      },
      {
        name: 'get_category_info',
        label: '카테고리 상세 조회',
        description: '카테고리 ID 또는 이름으로 업종 상세 정보 조회',
        params: ['category_id'],
        returns: '카테고리명, 쇼핑몰 수, 평균 매출, 성장률',
        agents: ['search_agent'],
        example: '뷰티 카테고리 상세 정보 알려줘',
      },
      {
        name: 'list_categories',
        label: '카테고리 목록 조회',
        description: '모든 상품 카테고리(업종) 목록 조회',
        params: [],
        returns: '카테고리 목록 (ID, 이름, 쇼핑몰 수)',
        agents: ['search_agent'],
        example: '카테고리 전체 목록 보여줘',
      },
    ],
  },
  {
    id: 'seller',
    name: '셀러 분석',
    icon: UserSearch,
    color: 'purple',
    tools: [
      {
        name: 'analyze_seller',
        label: '셀러 운영 분석',
        description: '셀러의 주문, GMV, 반품률, CS 건수 등 운영 지표 분석',
        params: ['seller_id'],
        returns: '월매출, 주문수, 반품률, CS건수, 활동점수, 성장률',
        agents: ['analysis_agent'],
        example: 'SEL0001 셀러 종합 분석해줘',
      },
      {
        name: 'get_seller_segment',
        label: '셀러 세그먼트 분류',
        description: 'K-Means 클러스터링 기반 세그먼트 분류 (파워/성장/일반/신규/휴면)',
        params: ['seller_id'],
        returns: '세그먼트명, 클러스터 특성, 동일 세그먼트 셀러 수',
        agents: ['analysis_agent'],
        example: 'SEL0001 셀러 세그먼트 분류해줘',
      },
      {
        name: 'detect_fraud',
        label: '이상거래 탐지',
        description: '셀러의 이상/부정행위 여부 탐지 (허위 주문, 리뷰 조작, 비정상 환불 등)',
        params: ['seller_id'],
        returns: '이상 여부, 위험도, 이상 유형, 의심 근거',
        agents: ['analysis_agent'],
        example: 'SEL0001 이상거래 조사해줘',
      },
      {
        name: 'get_segment_statistics',
        label: '세그먼트별 통계',
        description: '셀러 세그먼트별 셀러 수, 평균 GMV, 평균 주문 수, 이상 비율 조회',
        params: [],
        returns: '세그먼트별 셀러수, 평균 GMV, 평균 주문수, 이상 비율',
        agents: ['analysis_agent'],
        example: '셀러 세그먼트별 통계 보여줘',
      },
      {
        name: 'get_fraud_statistics',
        label: '이상거래 통계',
        description: '전체 이상거래 통계 (이상 셀러 수, 유형별 분포, 위험 수준)',
        params: [],
        returns: '이상 셀러 수, 사기 비율, 유형별 건수, 고위험 수',
        agents: ['analysis_agent'],
        example: '이상거래 전체 통계 보여줘',
      },
      {
        name: 'get_seller_activity_report',
        label: '셀러 활동 리포트',
        description: '특정 셀러의 주문/배송/CS/정산 활동 요약 리포트 생성',
        params: ['seller_id'],
        returns: '일별 활동 로그, 이벤트 유형별 집계, 활동 요약',
        agents: ['analysis_agent'],
        example: 'SEL0001 최근 30일 활동 리포트',
      },
    ],
  },
  {
    id: 'ml',
    name: 'ML 예측',
    icon: BrainCircuit,
    color: 'emerald',
    tools: [
      {
        name: 'predict_seller_churn',
        label: '셀러 이탈 예측',
        description: 'ML 모델과 SHAP으로 특정 셀러의 이탈 확률, 위험 수준, 주요 이탈 요인 분석',
        params: ['seller_id'],
        returns: '이탈 확률(%), 위험 등급, SHAP 요인 Top5',
        agents: ['analysis_agent'],
        example: 'SEL0001 이탈 확률 예측해줘',
      },
      {
        name: 'predict_shop_revenue',
        label: '쇼핑몰 매출 예측',
        description: 'LightGBM으로 쇼핑몰의 예상 월매출, 성장률, 매출 기여 요인 예측',
        params: ['shop_id'],
        returns: '예상 월매출, 성장률, 기여 요인 Top5',
        agents: ['analysis_agent'],
        example: 'S0001 다음 달 매출 예측해줘',
      },
      {
        name: 'get_shop_performance',
        label: '쇼핑몰 성과 분석',
        description: '쇼핑몰의 현재 성과 KPI (매출, 주문, 전환율, 리뷰 등) 조회',
        params: ['shop_id'],
        returns: '월매출, 주문수, 전환율, 평균 객단가, 리뷰 점수',
        agents: ['analysis_agent'],
        example: 'S0001 쇼핑몰 성과 분석해줘',
      },
      {
        name: 'optimize_marketing',
        label: '마케팅 최적화',
        description: 'P-PSO 알고리즘으로 채널별 최적 마케팅 예산 배분 및 ROAS 예측',
        params: ['seller_id'],
        returns: '채널별 최적 예산, 예상 ROAS, 총 예상 매출',
        agents: ['analysis_agent'],
        example: 'SEL0001 마케팅 예산 최적화 추천해줘',
      },
      {
        name: 'get_churn_prediction',
        label: '이탈 예측 분석',
        description: '전체 셀러의 이탈 예측 (고/중/저위험 분류, 주요 이탈 요인 5개)',
        params: [],
        returns: '위험 등급별 셀러 수, 전체 이탈 요인 Top5',
        agents: ['analysis_agent'],
        example: '고위험 이탈 셀러 목록 보여줘',
      },
      {
        name: 'get_cohort_analysis',
        label: '코호트 리텐션 분석',
        description: '월별 코호트의 주차별(Week1~12) 리텐션율 및 평균 리텐션 조회',
        params: [],
        returns: '코호트별 주차 리텐션율, 평균 리텐션율',
        agents: ['analysis_agent'],
        example: '코호트 리텐션 분석 보여줘',
      },
      {
        name: 'get_trend_analysis',
        label: '트렌드 KPI 분석',
        description: 'GMV, 활성 셀러, 주문수, 신규 가입 등 주요 지표의 변화율과 상관관계 분석',
        params: [],
        returns: 'KPI별 일별 추이, 변화율, 지표 간 상관계수',
        agents: ['analysis_agent'],
        example: '최근 7일 KPI 트렌드 분석해줘',
      },
      {
        name: 'get_gmv_prediction',
        label: 'GMV 예측',
        description: '예상 월간 GMV, 성장률, 셀러 티어별 거래 분포 예측',
        params: [],
        returns: '예상 월간 GMV, 성장률, 티어별 거래 비율',
        agents: ['analysis_agent'],
        example: '이번 달 GMV 예측해줘',
      },
    ],
  },
  {
    id: 'cs',
    name: 'CS 상담',
    icon: Headphones,
    color: 'amber',
    tools: [
      {
        name: 'auto_reply_cs',
        label: 'CS 자동 응답',
        description: '카페24 정책 기반 CS 문의 자동 응답 초안 생성',
        params: ['inquiry_text'],
        returns: '자동 응답 초안, 참조 정책, 신뢰도',
        agents: ['cs_agent'],
        example: '"결제 오류 해결해주세요" 자동 답변 생성해줘',
      },
      {
        name: 'check_cs_quality',
        label: 'CS 품질 평가',
        description: 'CS 티켓의 우선순위/긴급도 예측 및 권장사항 제공',
        params: ['ticket_category', 'seller_tier', 'sentiment_score', 'order_value'],
        returns: '우선순위, 긴급도, 예상 처리시간, 권장사항',
        agents: ['cs_agent'],
        example: 'SEL0001 이상거래 조사하고 CS 품질 점검해줘',
      },
      {
        name: 'get_ecommerce_glossary',
        label: '이커머스 용어집',
        description: '카페24 이커머스 용어(GMV, ARPU, 정산, 환불 등) 검색 및 조회',
        params: [],
        returns: '용어명, 정의, 카테고리, 관련 용어',
        agents: ['cs_agent'],
        example: '이커머스 용어 GMV 설명해줘',
      },
      {
        name: 'get_cs_statistics',
        label: 'CS 통계 조회',
        description: '카테고리별 CS 통계, 평균 처리 시간, 만족도 점수 조회',
        params: [],
        returns: '총 티켓수, 카테고리별 건수, 평균 해결시간, 만족도',
        agents: ['cs_agent', 'retention_agent'],
        example: 'CS 문의 카테고리별 통계 보여줘',
      },
      {
        name: 'classify_inquiry',
        label: '문의 자동 분류',
        description: 'TF-IDF + RandomForest로 CS 문의 텍스트 카테고리 자동 분류',
        params: ['text'],
        returns: '예측 카테고리, 신뢰도, 상위3 후보',
        agents: ['cs_agent'],
        example: '"배송 늦어요 환불해주세요" 문의 분류해줘',
      },
    ],
  },
  {
    id: 'dashboard',
    name: '대시보드/통계',
    icon: LayoutDashboard,
    color: 'teal',
    tools: [
      {
        name: 'get_dashboard_summary',
        label: '대시보드 요약',
        description: '플랫폼 전체 운영 현황 (쇼핑몰/셀러/CS/주문/정산 통계) 요약',
        params: [],
        returns: '쇼핑몰 수, 셀러 수, CS 현황, 주문 통계, 이상거래 현황',
        agents: ['analysis_agent'],
        example: '대시보드 전체 현황 요약해줘',
      },
      {
        name: 'get_order_statistics',
        label: '주문/이벤트 통계',
        description: '운영 이벤트(주문, 정산, 환불, CS 등) 타입별 집계 및 일별 추이',
        params: [],
        returns: '이벤트 유형별 건수, 일별 추이',
        agents: ['analysis_agent'],
        example: '최근 30일 주문 이벤트 통계 보여줘',
      },
    ],
  },
  {
    id: 'retention',
    name: '리텐션/RAG',
    icon: ShieldCheck,
    color: 'orange',
    tools: [
      {
        name: 'get_at_risk_sellers',
        label: '이탈 위험 셀러 조회',
        description: 'ML 이탈 예측 모델과 SHAP으로 이탈 위험 높은 셀러 목록 탐지',
        params: [],
        returns: '고위험 셀러 목록, 이탈 확률, 위험 요인',
        agents: ['retention_agent'],
        example: '고위험 이탈 셀러 목록 보여줘',
      },
      {
        name: 'generate_retention_message',
        label: '리텐션 메시지 생성',
        description: 'LLM으로 셀러 맞춤 리텐션 메시지 생성 (쿠폰, 업그레이드, 매니저 배정 등)',
        params: ['seller_id'],
        returns: '맞춤 메시지, 추천 액션, 예상 효과',
        agents: ['retention_agent'],
        example: 'SEL0001 이탈 위험 분석하고 리텐션 전략 실행해줘',
      },
      {
        name: 'execute_retention_action',
        label: '리텐션 조치 실행',
        description: '리텐션 조치 실행 (coupon, upgrade_offer, manager_assign, custom_message)',
        params: ['seller_id', 'action_type'],
        returns: '실행 결과, 조치 유형, 대상 셀러',
        agents: ['retention_agent'],
        example: 'SEL0001 리텐션 조치 실행해줘',
      },
      {
        name: 'search_platform',
        label: 'RAG 지식 검색',
        description: '카페24 플랫폼/정책/기능 관련 지식을 임베딩 기반으로 검색',
        params: ['query'],
        returns: '관련 문서 Top5, 유사도 점수, 문서 출처',
        agents: ['search_agent'],
        example: '카페24 결제수단 설정 방법 알려줘',
      },
      {
        name: 'search_platform_lightrag',
        label: 'LightRAG 검색',
        description: 'LightRAG 그래프 기반 플랫폼 지식 검색 (local/global/hybrid 모드)',
        params: ['query'],
        returns: '관련 문서, 지식 그래프 엔티티, 관계',
        agents: ['search_agent'],
        example: '디컬렉션이 뭐야?',
      },
    ],
  },
];

/** 전체 도구 목록 (flat) */
export const ALL_TOOLS_FLAT = TOOL_CATEGORIES.flatMap((cat) =>
  cat.tools.map((t) => ({ ...t, category: cat.id, categoryName: cat.name }))
);

/** 도구 이름으로 빠르게 조회하는 맵 */
export const TOOL_BY_NAME = Object.fromEntries(
  ALL_TOOLS_FLAT.map((t) => [t.name, t])
);

/** 에이전트별 도구 목록 */
export const TOOLS_BY_AGENT = ALL_TOOLS_FLAT.reduce((acc, t) => {
  for (const agent of t.agents) {
    if (!acc[agent]) acc[agent] = [];
    acc[agent].push(t);
  }
  return acc;
}, {});

/** 에이전트 메타 정보 */
export const AGENT_META = {
  search_agent: { label: '검색 에이전트', icon: Search, color: 'blue' },
  analysis_agent: { label: '분석 에이전트', icon: BarChart3, color: 'purple' },
  cs_agent: { label: 'CS 에이전트', icon: Headphones, color: 'pink' },
  retention_agent: { label: '리텐션 에이전트', icon: ShieldCheck, color: 'orange' },
};
