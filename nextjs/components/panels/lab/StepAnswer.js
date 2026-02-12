// components/panels/lab/StepAnswer.js - Step 3: 답변
// H32: EditableAnswer 공통 컴포넌트 사용
import {
  MessageSquare, Sparkles, Loader2, Edit3, RotateCcw, FileText, CheckCircle2,
} from 'lucide-react';
import { TIER_COLORS } from './constants';
import { renderMd, EmptyStep } from './utils';
import EditableAnswer from '@/components/common/EditableAnswer';

export default function StepAnswer({
  result, draftAnswer, setDraftAnswer, streamingAnswer, isStreaming, generateAnswer, ragContext, isEditing, setIsEditing, settings,
  classifyResults, autoIdxs, batchAnswers, batchLoading, generateBatchAnswers, checkedAuto, toggleAutoCheck, toggleAllAuto, updateBatchAnswer,
}) {
  const hasAutoMode = !result && classifyResults?.length > 0 && autoIdxs?.length > 0;
  const answeredCount = batchAnswers ? Object.keys(batchAnswers).filter(k => autoIdxs?.includes(Number(k))).length : 0;

  // 자동 처리 모드: 일괄 답변 생성
  if (hasAutoMode) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10 space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-cafe24-brown font-semibold text-lg">
            <MessageSquare className="w-5 h-5 text-cafe24-orange" />
            Step 3. 답변 - 자동 처리 일괄 답변 생성
          </div>
          <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-500">
            RAG 모드: {settings?.ragMode || 'rag'}
          </span>
        </div>

        <p className="text-xs text-gray-500">
          자동 처리 대상 {autoIdxs.length}건에 대해 RAG + LLM 답변을 생성합니다.
          {answeredCount > 0 && ` (${answeredCount}건 생성 완료)`}
        </p>

        {/* 체크박스 선택 + 생성 버튼 */}
        <div className="flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-xs text-gray-600">
            <input
              type="checkbox"
              checked={checkedAuto?.size === autoIdxs.length && autoIdxs.length > 0}
              onChange={toggleAllAuto}
              className="w-3.5 h-3.5 accent-green-600 rounded"
            />
            전체 선택
          </label>
          <button
            onClick={() => generateBatchAnswers(checkedAuto?.size > 0 ? [...checkedAuto] : autoIdxs)}
            disabled={batchLoading}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-gradient-to-r from-cafe24-orange to-cafe24-yellow text-white text-sm font-medium hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {batchLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {batchLoading
              ? '생성 중...'
              : checkedAuto?.size > 0
              ? `선택 답변 생성 (${checkedAuto.size}건)`
              : `전체 답변 생성 (${autoIdxs.length}건)`}
          </button>
        </div>

        {/* 문의별 답변 카드 */}
        <div className="space-y-3">
          {autoIdxs.map((idx) => {
            const item = classifyResults[idx];
            if (!item) return null;
            const answer = batchAnswers?.[idx];
            const category = item.result?.predicted_category || '?';

            return (
              <div key={idx} className="rounded-lg border border-gray-200 overflow-hidden">
                {/* 문의 헤더 */}
                <div className="flex items-start gap-2 p-3 bg-gray-50 border-b border-gray-100">
                  <input
                    type="checkbox"
                    checked={!!checkedAuto?.has(idx)}
                    onChange={() => toggleAutoCheck(idx)}
                    className="mt-0.5 w-3.5 h-3.5 accent-green-600 rounded shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-gray-700 leading-relaxed">{item.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[item.tier] || ''}`}>
                        {item.tier}
                      </span>
                      <span className="px-1.5 py-0.5 rounded-full bg-cafe24-orange/10 text-cafe24-orange text-[10px] font-bold">
                        {category}
                      </span>
                    </div>
                  </div>
                </div>
                {/* 답변 영역 - H32: EditableAnswer 사용 */}
                <div className="p-3">
                  {answer ? (
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1 text-xs text-green-600 font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        답변 생성 완료
                      </div>
                      <EditableAnswer
                        answer={answer}
                        onSave={(newText) => updateBatchAnswer(idx, newText)}
                        rows={6}
                        renderContent={renderMd}
                      />
                    </div>
                  ) : batchLoading ? (
                    <div className="flex items-center gap-2 text-xs text-gray-400 py-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      대기 중...
                    </div>
                  ) : (
                    <p className="text-xs text-gray-400 py-2">답변 생성 버튼을 클릭하세요</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10">
        <EmptyStep message="먼저 Step 1에서 문의를 분류하세요." />
      </div>
    );
  }

  const displayText = draftAnswer || streamingAnswer;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-cafe24-brown font-semibold text-lg">
          <MessageSquare className="w-5 h-5 text-cafe24-orange" />
          Step 3. 답변 - RAG + LLM 초안 생성
        </div>
        <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-500">
          RAG 모드: {settings?.ragMode || 'rag'}
        </span>
      </div>

      {result.context && (
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
            <FileText className="w-3.5 h-3.5" />
            문의 컨텍스트
          </div>
          <p className="text-sm text-gray-700">
            카테고리: <strong>{result.context.inquiry_category}</strong> | 셀러 등급: <strong>{result.context.seller_tier}</strong>
          </p>
        </div>
      )}

      {!displayText && !isStreaming && (
        <button
          onClick={generateAnswer}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-cafe24-orange to-cafe24-yellow text-white font-medium hover:shadow-lg transition-all"
        >
          <Sparkles className="w-5 h-5" />
          RAG + LLM 답변 초안 생성
        </button>
      )}

      {ragContext && (
        <details className="group">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 flex items-center gap-1">
            <FileText className="w-3.5 h-3.5" />
            RAG 검색 결과 ({ragContext.source_count}건)
          </summary>
          <div className="mt-2 p-3 rounded-lg bg-blue-50 border border-blue-100 text-xs text-gray-600 max-h-32 overflow-y-auto">
            {ragContext.context_preview || '(검색 결과 없음)'}
          </div>
        </details>
      )}

      {(isStreaming || displayText) && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-cafe24-brown/80 flex items-center gap-1.5">
              {isStreaming && <Loader2 className="w-4 h-4 animate-spin text-cafe24-orange" />}
              {isStreaming ? '답변 생성 중...' : '답변 초안'}
            </span>
            {draftAnswer && !isStreaming && (
              <div className="flex gap-2">
                <button
                  onClick={() => setIsEditing(!isEditing)}
                  className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg border border-cafe24-brown/20 hover:bg-cafe24-yellow/10 text-cafe24-brown/70"
                >
                  <Edit3 className="w-3.5 h-3.5" />
                  {isEditing ? '미리보기' : '편집'}
                </button>
                <button
                  onClick={generateAnswer}
                  className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg border border-cafe24-orange/30 hover:bg-cafe24-orange/10 text-cafe24-orange"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  재생성
                </button>
              </div>
            )}
          </div>

          {isEditing && draftAnswer ? (
            <textarea
              value={draftAnswer}
              onChange={e => setDraftAnswer(e.target.value)}
              rows={10}
              className="w-full px-4 py-3 rounded-lg border border-cafe24-orange/30 focus:border-cafe24-orange focus:ring-1 focus:ring-cafe24-orange/30 outline-none text-sm resize-none font-mono"
            />
          ) : (
            <div className="p-4 rounded-lg bg-gray-50 border border-gray-200 text-sm text-gray-700 whitespace-pre-wrap min-h-[120px] leading-relaxed">
              {renderMd(displayText)}
              {isStreaming && <span className="inline-block w-1.5 h-4 bg-cafe24-orange animate-pulse ml-0.5 align-middle" />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
