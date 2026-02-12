// components/panels/lab/useCheckboxSelection.js
// M65: 체크박스 + 전체선택 로직 중복 → 커스텀 훅
import { useCallback, useState } from 'react';

/**
 * useCheckboxSelection - 체크박스 선택 + 전체 선택 로직
 * @param {Array} items - 전체 아이템 목록 (인덱스 배열)
 * @returns {{ checked, toggle, toggleAll, setChecked, isAllChecked }}
 */
export default function useCheckboxSelection(items = []) {
  const [checked, setChecked] = useState(new Set());

  const toggle = useCallback((item) => {
    setChecked((prev) => {
      const next = new Set(prev);
      next.has(item) ? next.delete(item) : next.add(item);
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    setChecked((prev) => prev.size === items.length ? new Set() : new Set(items));
  }, [items]);

  const isAllChecked = checked.size === items.length && items.length > 0;

  return { checked, toggle, toggleAll, setChecked, isAllChecked };
}
