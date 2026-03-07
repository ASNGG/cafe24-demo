import {
  Search, BarChart3, Sparkles, Play, FileText,
  CheckCircle2, ClipboardList, HelpCircle,
  Users, AlertTriangle, TrendingUp, ShieldAlert, MessageSquare, BookOpen
} from 'lucide-react';

export const RETENTION_STEPS = [
  { key: 'detect', label: '위험 탐지', desc: 'ML 이탈 예측', icon: Search },
  { key: 'analyze', label: '이탈 분석', desc: 'SHAP 요인 분석', icon: BarChart3 },
  { key: 'generate', label: '메시지 생성', desc: 'LLM 맞춤 메시지', icon: Sparkles },
  { key: 'execute', label: '조치 실행', desc: '자동 조치 실행', icon: Play },
  { key: 'log', label: '결과 기록', desc: '액션 로깅', icon: FileText },
];

export const FAQ_STEPS_KMEANS = [
  { key: 'analyze', label: '분석·클러스터링', desc: 'TF-IDF + K-Means', icon: Search },
  { key: 'generate', label: 'FAQ 생성', desc: 'LLM 자동 생성', icon: Sparkles },
  { key: 'review', label: '검토/편집', desc: '초안 검토', icon: HelpCircle },
  { key: 'approve', label: '승인/배포', desc: 'FAQ 승인', icon: CheckCircle2 },
];

export const FAQ_STEPS_LLM = [
  { key: 'analyze', label: '분석·클러스터링', desc: 'LLM 의미 분류', icon: Search },
  { key: 'generate', label: 'FAQ 생성', desc: 'LLM 자동 생성', icon: Sparkles },
  { key: 'review', label: '검토/편집', desc: '초안 검토', icon: HelpCircle },
  { key: 'approve', label: '승인/배포', desc: 'FAQ 승인', icon: CheckCircle2 },
];

export const FAQ_STEPS = FAQ_STEPS_KMEANS;

export const REPORT_STEPS = [
  { key: 'collect', label: '데이터 수집', desc: '전체 KPI 수집', icon: Search },
  { key: 'aggregate', label: 'KPI 집계', desc: '트렌드 분석', icon: BarChart3 },
  { key: 'write', label: '리포트 작성', desc: 'LLM 리포트 작성', icon: Sparkles },
  { key: 'save', label: '결과 저장', desc: '히스토리 저장', icon: FileText },
  { key: 'history', label: '이력 관리', desc: '리포트 조회', icon: ClipboardList },
];

export const UPGRADE_STEPS = [
  { key: 'detect', label: '후보 탐지', desc: '규칙 기반 탐지', icon: Search },
  { key: 'analyze', label: '성과 분석', desc: '매출/주문 분석', icon: BarChart3 },
  { key: 'generate', label: '메시지 생성', desc: 'LLM 추천 메시지', icon: Sparkles },
  { key: 'execute', label: '제안 실행', desc: '업그레이드 제안', icon: Play },
  { key: 'log', label: '결과 기록', desc: '액션 로깅', icon: FileText },
];

export const CS_CATEGORIES = ["배송", "환불", "결제", "상품", "계정", "정산", "기술지원", "마케팅", "기타"];

// 멀티에이전트 Supervisor 워커 레지스트리
// key: 백엔드 MULTI_AGENT_WORKERS의 agent 이름 (transfer_to_<key>로 handoff)
export const MULTI_AGENT_WORKERS = {
  churn_analyst:        { label: '이탈 분석가',     icon: AlertTriangle,  desc: 'ML 이탈 예측 + SHAP 분석' },
  retention_strategist: { label: '리텐션 전략가',   icon: Sparkles,       desc: '맞춤 메시지 + 조치 실행' },
  seller_analyst:       { label: '셀러 분석가',     icon: Users,          desc: '활동 + 세그먼트 + 이상거래' },
  performance_analyst:  { label: '성과 분석가',     icon: TrendingUp,     desc: '매출 트렌드 + GMV + 마케팅' },
  fraud_investigator:   { label: '이상거래 조사관', icon: ShieldAlert,    desc: '부정 거래 탐지 + 영향도' },
  cs_quality_analyst:   { label: 'CS 품질 분석가',  icon: MessageSquare,  desc: 'CS 통계 + 자동 응답 + 품질' },
  report_writer:        { label: '리포트 작성가',   icon: FileText,       desc: '대시보드 + KPI 종합 보고서' },
  platform_searcher:    { label: '플랫폼 검색가',   icon: BookOpen,       desc: 'RAG 지식 검색 + 쇼핑몰/카테고리 조회' },
};
