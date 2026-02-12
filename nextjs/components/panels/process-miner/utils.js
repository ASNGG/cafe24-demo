// process-miner/utils.js — 공통 유틸리티

export function isSuccess(res) {
  return res?.status === 'success' || res?.status === 'ok';
}

export function getErrorMsg(res) {
  return res?.message || res?.error || '분석 실패';
}

export function formatMinutes(min) {
  if (min == null || isNaN(min)) return '-';
  if (min < 1) return `${(min * 60).toFixed(0)}초`;
  if (min < 60) return `${min.toFixed(1)}분`;
  if (min < 1440) return `${(min / 60).toFixed(1)}시간`;
  return `${(min / 1440).toFixed(1)}일`;
}

export function formatNumber(n) {
  if (n == null || isNaN(n)) return '-';
  return Number(n).toLocaleString();
}

export function formatPercent(n) {
  if (n == null || isNaN(n)) return '-';
  return `${(n * 100).toFixed(1)}%`;
}

export const PROCESS_TYPES = [
  { value: 'order', label: '주문 프로세스' },
  { value: 'cs', label: 'CS 문의 프로세스' },
  { value: 'settlement', label: '정산 프로세스' },
];

export const CASE_OPTIONS = [100, 200, 500];
