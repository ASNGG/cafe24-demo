// components/common/EditableAnswer.js - H32: 답변 편집 UI 공통 컴포넌트
// StepAnswer, StepReply, DraggableCard에서 반복되던 편집 UI를 통합
import { useState } from 'react';
import { Edit3 } from 'lucide-react';

/**
 * EditableAnswer - 답변 텍스트를 보여주고, 편집 모드를 제공
 *
 * @param {object} props
 * @param {string} props.answer - 답변 텍스트
 * @param {function} props.onSave - (newText) => void
 * @param {boolean} [props.disabled] - 편집 불가 상태
 * @param {number} [props.rows] - textarea 행 수 (기본 5)
 * @param {string} [props.className] - 래퍼 추가 클래스
 * @param {function} [props.renderContent] - 커스텀 렌더 (answer) => ReactNode
 * @param {string} [props.maxHeight] - 미리보기 최대 높이 (기본 'max-h-32')
 */
export default function EditableAnswer({
  answer,
  onSave,
  disabled = false,
  rows = 5,
  className = '',
  renderContent,
  maxHeight = 'max-h-32',
}) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('');

  const startEdit = () => {
    setEditText(answer || '');
    setEditing(true);
  };

  const handleSave = () => {
    onSave(editText);
    setEditing(false);
  };

  const handleCancel = () => {
    setEditing(false);
  };

  if (editing) {
    return (
      <div className={`space-y-1.5 ${className}`} onClick={(e) => e.stopPropagation()}>
        <textarea
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          rows={rows}
          className="w-full p-2.5 rounded border border-green-300 text-xs text-gray-700 resize-none focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-200 leading-relaxed"
        />
        <div className="flex gap-1.5 justify-end">
          <button
            onClick={handleCancel}
            className="px-2.5 py-1 rounded text-[10px] text-gray-500 border border-gray-200 hover:bg-gray-50"
          >
            취소
          </button>
          <button
            onClick={handleSave}
            className="px-2.5 py-1 rounded text-[10px] text-white bg-green-600 hover:bg-green-700"
          >
            저장
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative group/editable ${className}`} onClick={(e) => e.stopPropagation()}>
      <div className={`p-2.5 rounded bg-green-50 border border-green-100 text-xs text-gray-600 ${maxHeight} overflow-y-auto whitespace-pre-wrap leading-relaxed`}>
        {renderContent ? renderContent(answer) : answer}
      </div>
      {!disabled && (
        <button
          onClick={(e) => { e.stopPropagation(); startEdit(); }}
          className="absolute top-1 right-1 p-1 rounded bg-white/80 border border-gray-200 opacity-0 group-hover/editable:opacity-100 transition-opacity hover:bg-gray-50"
          title="답변 수정"
        >
          <Edit3 className="w-3 h-3 text-gray-500" />
        </button>
      )}
    </div>
  );
}
