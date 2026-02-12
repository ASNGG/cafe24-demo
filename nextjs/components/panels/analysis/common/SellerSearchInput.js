// components/panels/analysis/common/SellerSearchInput.js
// 셀러 검색 + 빠른 선택 공통 컴포넌트

import { Search } from 'lucide-react';

const DEFAULT_QUICK_IDS = ['SEL0001', 'SEL0025', 'SEL0050', 'SEL0100'];

export default function SellerSearchInput({
  value,
  onChange,
  onSearch,
  loading,
  quickSelectIds = DEFAULT_QUICK_IDS,
  placeholder = '셀러 ID 입력 (예: SEL0001)',
  buttonLabel = '검색',
  loadingLabel = '조회중...',
  inputRef,
}) {
  return (
    <div>
      <div className="flex gap-2 mb-3">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') onSearch(); }}
          placeholder={placeholder}
          className="flex-1 rounded-xl border-2 border-cafe24-orange/20 bg-white px-4 py-2.5 text-sm font-semibold text-cafe24-brown placeholder:text-cafe24-brown/40 outline-none focus:border-cafe24-orange transition-all"
        />
        <button
          onClick={onSearch}
          disabled={loading}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white font-bold text-sm shadow-md hover:shadow-lg transition disabled:opacity-50"
        >
          {loading ? loadingLabel : buttonLabel}
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-cafe24-brown/60">빠른 선택:</span>
        {quickSelectIds.map(id => (
          <button
            key={id}
            onClick={() => { onChange(id); onSearch(id); }}
            className="px-2 py-1 rounded-lg bg-cafe24-beige text-xs font-semibold text-cafe24-brown hover:bg-cafe24-yellow/30 transition"
          >
            {id}
          </button>
        ))}
      </div>
    </div>
  );
}
