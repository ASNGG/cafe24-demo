import {
  Search, BarChart3, Sparkles, Play, FileText,
  CheckCircle2, ClipboardList, HelpCircle
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
