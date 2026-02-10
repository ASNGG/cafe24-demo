// components/panels/lab/StepClassify.js - Step 1: 접수 - 일괄 분류 + DnD 자동/수동 분기
import { motion } from 'framer-motion';
import {
  Inbox, Zap, AlertTriangle, Loader2, RotateCcw,
} from 'lucide-react';
import { INBOX_INQUIRIES, TIER_COLORS } from './constants';
import DraggableCard from './DraggableCard';

export default function StepClassify({
  classifyResults, classifyLoading, runBatchClassify,
  confidenceThreshold, setConfidenceThreshold,
  autoIdxs, manualIdxs, checkedAuto, dragOverZone, setDragOverZone,
  handleDragStart, handleDropToAuto, handleDropToManual,
  toggleAutoCheck, toggleAllAuto,
  selectedIdx, selectInquiry, pipelineLoading,
}) {
  const hasResults = classifyResults.length > 0;
  const hasSplit = autoIdxs.length > 0 || manualIdxs.length > 0;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cafe24-brown font-semibold text-lg">
        <Inbox className="w-5 h-5 text-cafe24-orange" />
        Step 1. 접수 - 셀러 문의 일괄 분류
      </div>

      {/* 접수함 테이블 (분류 전) */}
      {!hasSplit && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm text-cafe24-brown/70 font-medium">접수함 ({INBOX_INQUIRIES.length}건)</span>
            <button
              onClick={runBatchClassify}
              disabled={classifyLoading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cafe24-orange text-white text-sm font-medium hover:bg-cafe24-orange/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {classifyLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
              {classifyLoading ? '분류 중...' : '일괄 분류'}
            </button>
          </div>

          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-gray-500 text-xs">
                  <th className="text-left py-2.5 px-3 w-8">#</th>
                  <th className="text-left py-2.5 px-3">문의 내용</th>
                  <th className="text-left py-2.5 px-3 w-24">셀러 등급</th>
                  <th className="text-left py-2.5 px-3 w-28">희망 채널</th>
                </tr>
              </thead>
              <tbody>
                {INBOX_INQUIRIES.map((inq, i) => (
                  <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2.5 px-3 text-gray-400 text-xs">{i + 1}</td>
                    <td className="py-2.5 px-3 text-gray-700 text-xs leading-relaxed">{inq.text}</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${TIER_COLORS[inq.tier] || 'bg-gray-100 text-gray-600'}`}>
                        {inq.tier}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex gap-1 flex-wrap">
                        {(inq.preferredChannels || []).map(ch => (
                          <span key={ch} className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            ch === 'any' ? 'bg-gray-100 text-gray-500' : 'bg-blue-50 text-blue-600'
                          }`}>
                            {ch === 'any' ? '무관' : ch === 'email' ? '이메일' : ch === 'kakao' ? '카카오' : ch === 'sms' ? 'SMS' : ch}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 분류 완료 → DnD 2열 */}
      {hasSplit && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          {/* 상단: 재분류 + 임계값 */}
          <div className="flex items-center justify-between gap-4">
            <button
              onClick={runBatchClassify}
              disabled={classifyLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-cafe24-brown/20 hover:bg-cafe24-yellow/10 text-cafe24-brown/60"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              재분류
            </button>
            <div className="flex items-center gap-2 text-xs text-cafe24-brown/60">
              <span>임계값</span>
              <input
                type="range" min={0.5} max={0.95} step={0.05}
                value={confidenceThreshold}
                onChange={e => setConfidenceThreshold(Number(e.target.value))}
                className="w-20 accent-cafe24-orange h-1"
              />
              <span className="font-bold text-cafe24-orange w-8">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          <p className="text-xs text-gray-400">
            문의를 드래그하여 자동 처리 ↔ 담당자 검토 사이로 이동할 수 있습니다.
          </p>

          {/* 2열: 자동 처리 / 담당자 검토 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 자동 처리 드롭존 */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOverZone('auto'); }}
              onDragLeave={() => setDragOverZone(null)}
              onDrop={handleDropToAuto}
              className={`rounded-lg border-2 transition-all min-h-[200px] ${
                dragOverZone === 'auto'
                  ? 'border-dashed border-green-400 bg-green-50/80'
                  : 'border-green-200 bg-green-50/50'
              }`}
            >
              <div className="flex items-center gap-2 px-4 py-3 border-b border-green-200 bg-green-50">
                <Zap className="w-4 h-4 text-green-600" />
                <span className="text-sm font-bold text-green-700">자동 처리</span>
                <span className="ml-auto text-xs text-green-600 font-medium">{autoIdxs.length}건</span>
              </div>
              {autoIdxs.length > 0 && (
                <div className="flex items-center gap-2 px-4 py-2 border-b border-green-100">
                  <input
                    type="checkbox"
                    checked={checkedAuto.size === autoIdxs.length && autoIdxs.length > 0}
                    onChange={toggleAllAuto}
                    className="w-3.5 h-3.5 accent-green-600 rounded"
                  />
                  <span className="text-[10px] text-gray-500">전체 선택</span>
                </div>
              )}
              <div className="p-2 space-y-1.5">
                {autoIdxs.length === 0 ? (
                  <p className="text-xs text-gray-400 text-center py-8">
                    {dragOverZone === 'auto' ? '여기에 놓으세요' : '문의를 드래그하세요'}
                  </p>
                ) : (
                  autoIdxs.map((idx) => (
                    <DraggableCard
                      key={idx}
                      idx={idx}
                      item={classifyResults[idx]}
                      variant="auto"
                      checked={checkedAuto.has(idx)}
                      onCheck={() => toggleAutoCheck(idx)}
                      onDragStart={handleDragStart}
                    />
                  ))
                )}
              </div>
              {autoIdxs.length > 0 && (
                <p className="text-[10px] text-green-600/60 text-center py-2 border-t border-green-100">
                  다음 단계(답변)에서 RAG 기반 답변이 생성됩니다
                </p>
              )}
            </div>

            {/* 담당자 검토 드롭존 */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOverZone('manual'); }}
              onDragLeave={() => setDragOverZone(null)}
              onDrop={handleDropToManual}
              className={`rounded-lg border-2 transition-all min-h-[200px] ${
                dragOverZone === 'manual'
                  ? 'border-dashed border-amber-400 bg-amber-50/80'
                  : 'border-amber-200 bg-amber-50/50'
              }`}
            >
              <div className="flex items-center gap-2 px-4 py-3 border-b border-amber-200 bg-amber-50">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span className="text-sm font-bold text-amber-700">담당자 검토</span>
                <span className="ml-auto text-xs text-amber-600 font-medium">{manualIdxs.length}건</span>
              </div>
              <div className="p-2 space-y-1.5">
                {manualIdxs.length === 0 ? (
                  <p className="text-xs text-gray-400 text-center py-8">
                    {dragOverZone === 'manual' ? '여기에 놓으세요' : '해당 없음'}
                  </p>
                ) : (
                  manualIdxs.map((idx) => (
                    <DraggableCard
                      key={idx}
                      idx={idx}
                      item={classifyResults[idx]}
                      variant="manual"
                      onDragStart={handleDragStart}
                      onClick={() => selectInquiry(idx)}
                      loading={pipelineLoading && selectedIdx === idx}
                      isSelected={selectedIdx === idx}
                    />
                  ))
                )}
              </div>
              {manualIdxs.length > 0 && (
                <p className="text-[10px] text-amber-600/60 text-center py-2 border-t border-amber-100">
                  클릭하여 상세 파이프라인 실행 (검토→답변→회신)
                </p>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
