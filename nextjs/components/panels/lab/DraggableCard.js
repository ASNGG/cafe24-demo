// components/panels/lab/DraggableCard.js - 드래그 가능한 문의 카드
import { useState } from 'react';
import {
  CheckCircle2, Loader2, User, Edit3,
} from 'lucide-react';
import { TIER_COLORS } from './constants';

export default function DraggableCard({ idx, item, variant, checked, onCheck, onDragStart, onClick, loading, isSelected, answer, onUpdateAnswer }) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('');

  if (!item) return null;

  const conf = item.result?.confidence || 0;
  const isAuto = variant === 'auto';
  const borderColor = isAuto ? 'border-green-300' : 'border-amber-300';
  const hoverBorder = isAuto ? 'hover:border-green-400' : 'hover:border-amber-400';

  return (
    <div
      draggable={!editing}
      onDragStart={(e) => !editing && onDragStart(e, idx)}
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border cursor-grab active:cursor-grabbing transition-all ${
        isSelected ? 'border-cookie-orange bg-cookie-orange/5' : `${borderColor} bg-white ${hoverBorder} hover:shadow-sm`
      } ${loading ? 'opacity-60' : ''}`}
    >
      <div className="flex items-start gap-2">
        {isAuto && onCheck && (
          <input
            type="checkbox"
            checked={!!checked}
            onChange={(e) => { e.stopPropagation(); onCheck(); }}
            onClick={(e) => e.stopPropagation()}
            className="mt-0.5 w-3.5 h-3.5 accent-green-600 rounded shrink-0"
          />
        )}
        {!isAuto && (
          <div className="shrink-0 mt-0.5">
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin text-cookie-orange" />
            ) : isSelected ? (
              <CheckCircle2 className="w-4 h-4 text-cookie-orange" />
            ) : (
              <User className="w-4 h-4 text-gray-400" />
            )}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-700 leading-relaxed line-clamp-2">{item.text}</p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[item.tier] || ''}`}>
              {item.tier}
            </span>
            {item.result?.predicted_category && (
              <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange text-[10px] font-bold">
                {item.result.predicted_category}
              </span>
            )}
            <span className={`text-[10px] font-medium ${isAuto ? 'text-green-600' : 'text-amber-600'}`}>
              {(conf * 100).toFixed(0)}%
            </span>
            {item.preferredChannels && (
              item.preferredChannels.includes('any')
                ? <span className="px-1 py-0.5 rounded text-[9px] bg-gray-100 text-gray-500">채널 무관</span>
                : item.preferredChannels.map(ch => (
                    <span key={ch} className="px-1 py-0.5 rounded text-[9px] bg-blue-50 text-blue-600">
                      {ch === 'email' ? '이메일' : ch === 'kakao' ? '카카오' : ch === 'sms' ? 'SMS' : ch === 'inapp' ? '인앱' : ch}
                    </span>
                  ))
            )}
          </div>

          {answer && (
            <details className="mt-2 group" onClick={(e) => e.stopPropagation()}>
              <summary className="text-[10px] text-green-600 font-medium cursor-pointer flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                답변 생성 완료 (클릭하여 보기)
              </summary>
              {editing ? (
                <div className="mt-1.5 space-y-1.5" onClick={(e) => e.stopPropagation()}>
                  <textarea
                    value={editText}
                    onChange={e => setEditText(e.target.value)}
                    rows={5}
                    className="w-full p-2 rounded border border-green-300 text-[11px] text-gray-700 resize-none focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-200"
                  />
                  <div className="flex gap-1.5 justify-end">
                    <button
                      onClick={(e) => { e.stopPropagation(); setEditing(false); }}
                      className="px-2 py-1 rounded text-[10px] text-gray-500 border border-gray-200 hover:bg-gray-50"
                    >
                      취소
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); onUpdateAnswer(idx, editText); setEditing(false); }}
                      className="px-2 py-1 rounded text-[10px] text-white bg-green-600 hover:bg-green-700"
                    >
                      저장
                    </button>
                  </div>
                </div>
              ) : (
                <div className="mt-1.5 relative group/answer">
                  <div className="p-2 rounded bg-green-50 border border-green-100 text-[11px] text-gray-600 max-h-24 overflow-y-auto whitespace-pre-wrap">
                    {answer}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setEditText(answer); setEditing(true); }}
                    className="absolute top-1 right-1 p-1 rounded bg-white/80 border border-green-200 opacity-0 group-hover/answer:opacity-100 transition-opacity hover:bg-green-50"
                    title="답변 수정"
                  >
                    <Edit3 className="w-3 h-3 text-green-600" />
                  </button>
                </div>
              )}
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
