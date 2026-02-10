// components/panels/lab/constants.js - LabPanel 공통 상수
import {
  Inbox, Search, MessageSquare, Send, TrendingUp,
  Mail, MessageCircle, Smartphone, Bell,
} from 'lucide-react';

export const STEPS = [
  { key: 'classify', label: '접수', icon: Inbox, desc: '일괄 분류 + 분기' },
  { key: 'review',   label: '검토', icon: Search, desc: '우선순위 + 검토' },
  { key: 'answer',   label: '답변', icon: MessageSquare, desc: 'RAG 답변 생성' },
  { key: 'reply',    label: '회신', icon: Send, desc: '채널별 자동 전송' },
  { key: 'improve',  label: '개선', icon: TrendingUp, desc: '피드백 & 대시보드' },
];

export const INBOX_INQUIRIES = [
  { text: '배송비 조건부 무료 설정(5만원 이상 무료) 방법을 모르겠습니다.', tier: 'Basic', preferredChannels: ['email'] },
  { text: '세금계산서 자동 발행 설정은 어디서 하나요?', tier: 'Standard', preferredChannels: ['kakao'] },
  { text: '상품 대량 등록 엑셀 업로드에서 오류가 발생합니다. 양식을 확인하고 싶습니다.', tier: 'Standard', preferredChannels: ['email', 'kakao'] },
  { text: 'PG사 연동 중 이니시스 인증키 오류가 발생하고, 동시에 네이버페이 정산도 누락되고 있습니다. 두 건 다 긴급히 해결 부탁드립니다.', tier: 'Premium', preferredChannels: ['any'] },
  { text: '카페24 API 웹훅 콜백이 간헐적으로 실패합니다. 서버 로그를 확인해주시고 원인 분석 부탁드립니다.', tier: 'Enterprise', preferredChannels: ['email', 'sms'] },
];

export const SELLER_TIERS = ['Basic', 'Standard', 'Premium', 'Enterprise'];

export const PRIORITY_COLORS = {
  urgent: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-300' },
  high:   { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-300' },
  normal: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-300' },
  low:    { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-300' },
};

export const CHANNELS = [
  { key: 'email', label: '이메일', icon: Mail, color: 'from-blue-500 to-blue-600', enabled: true },
  { key: 'kakao', label: '카카오톡', icon: MessageCircle, color: 'from-yellow-400 to-yellow-500', enabled: false },
  { key: 'sms',   label: 'SMS', icon: Smartphone, color: 'from-green-500 to-green-600', enabled: false },
  { key: 'inapp', label: '인앱 알림', icon: Bell, color: 'from-purple-500 to-purple-600', enabled: false },
];

export const TIER_COLORS = {
  Basic: 'bg-gray-100 text-gray-600',
  Standard: 'bg-blue-100 text-blue-700',
  Premium: 'bg-purple-100 text-purple-700',
  Enterprise: 'bg-amber-100 text-amber-700',
};
