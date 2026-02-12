// components/common/DataState.js — M55: 로딩/에러/빈 상태 래퍼
// 사용법: <DataState loading={loading} error={error} empty={!data} emptyText="데이터 없음">{children}</DataState>

import { Loader2, AlertCircle } from 'lucide-react';

export default function DataState({
  loading = false,
  error = null,
  empty = false,
  loadingText = '로딩 중...',
  emptyText = '데이터가 없습니다',
  emptyDesc = '',
  className = '',
  children,
}) {
  if (loading) {
    return (
      <div className={`flex items-center justify-center py-12 ${className}`}>
        <Loader2 size={22} className="animate-spin text-gray-400" />
        <span className="ml-2 text-sm text-gray-500">{loadingText}</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`rounded-2xl border-2 border-red-200 bg-red-50/50 p-6 text-center ${className}`}>
        <AlertCircle size={24} className="mx-auto mb-2 text-red-400" />
        <p className="text-sm font-semibold text-red-700">{typeof error === 'string' ? error : '오류가 발생했습니다'}</p>
      </div>
    );
  }

  if (empty) {
    return (
      <div className={`rounded-2xl border-2 border-dashed border-gray-200 bg-gray-50/50 p-6 text-center ${className}`}>
        <p className="text-sm font-semibold text-gray-500">{emptyText}</p>
        {emptyDesc && <p className="mt-1 text-xs text-gray-400">{emptyDesc}</p>}
      </div>
    );
  }

  return children;
}
