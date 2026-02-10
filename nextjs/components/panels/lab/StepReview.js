// components/panels/lab/StepReview.js - Step 2: 검토
import {
  Search, Zap, AlertTriangle, ChevronRight, FileText,
} from 'lucide-react';
import { PRIORITY_COLORS } from './constants';
import { EmptyStep } from './utils';

export default function StepReview({ result, classifyResult, threshold, selectedInquiry, classifyResults, autoIdxs, manualIdxs }) {
  const hasAutoMode = !result && classifyResults?.length > 0;

  // 자동 처리 모드: 분류 결과 요약 표시
  if (hasAutoMode) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
        <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
          <Search className="w-5 h-5 text-cookie-orange" />
          Step 2. 검토 - 분류 결과 요약
        </div>

        {/* 분기 현황 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg bg-green-50 border border-green-200 text-center">
            <Zap className="w-6 h-6 text-green-600 mx-auto mb-1" />
            <div className="text-2xl font-bold text-green-700">{autoIdxs?.length || 0}건</div>
            <div className="text-xs text-green-600">자동 처리</div>
          </div>
          <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 text-center">
            <AlertTriangle className="w-6 h-6 text-amber-600 mx-auto mb-1" />
            <div className="text-2xl font-bold text-amber-700">{manualIdxs?.length || 0}건</div>
            <div className="text-xs text-amber-600">담당자 검토</div>
          </div>
        </div>

        {/* 개별 분류 결과 */}
        <div className="space-y-2">
          <span className="text-sm font-medium text-cookie-brown/80">문의별 분류 상세</span>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-gray-500">
                  <th className="text-left py-2 px-3 w-8">#</th>
                  <th className="text-left py-2 px-3">문의 내용</th>
                  <th className="text-left py-2 px-3 w-24 whitespace-nowrap">카테고리</th>
                  <th className="text-left py-2 px-3 w-16">신뢰도</th>
                  <th className="text-left py-2 px-3 w-16">분기</th>
                </tr>
              </thead>
              <tbody>
                {classifyResults.map((item, i) => {
                  const conf = item.result?.confidence || 0;
                  const isAutoItem = autoIdxs?.includes(i);
                  return (
                    <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-2 px-3 text-gray-400">{i + 1}</td>
                      <td className="py-2 px-3 text-gray-700 max-w-[200px] truncate">{item.text}</td>
                      <td className="py-2 px-3 whitespace-nowrap">
                        <span className="px-1.5 py-0.5 rounded-full bg-cookie-orange/10 text-cookie-orange font-bold">
                          {item.result?.predicted_category || '?'}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`font-medium ${conf >= threshold ? 'text-green-600' : 'text-amber-600'}`}>
                          {(conf * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.5 rounded font-medium ${
                          isAutoItem ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                        }`}>
                          {isAutoItem ? '자동' : '수동'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-blue-50 border border-blue-100">
          <p className="text-xs text-blue-600">
            신뢰도 {(threshold * 100).toFixed(0)}% 이상은 자동 처리, 미만은 담당자 검토로 분기되었습니다.
            다음 단계에서 답변을 생성합니다.
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10">
        <EmptyStep message="먼저 Step 1에서 문의를 분류하세요." />
      </div>
    );
  }

  const confidence = result.confidence || classifyResult?.confidence || 0;
  const isAuto = confidence >= threshold;
  const priority = result.priority?.predicted_priority || 'normal';
  const pColors = PRIORITY_COLORS[priority] || PRIORITY_COLORS.normal;

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cookie-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cookie-brown font-semibold text-lg">
        <Search className="w-5 h-5 text-cookie-orange" />
        Step 2. 검토 - 상세 분석
      </div>

      {selectedInquiry && (
        <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
          <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
            <FileText className="w-3.5 h-3.5" />
            선택된 문의
          </div>
          <p className="text-sm text-gray-700">{selectedInquiry.text}</p>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="p-4 rounded-lg bg-gray-50 border border-gray-200 space-y-2">
          <span className="text-xs text-gray-500 font-medium">분류 결과</span>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full bg-cookie-orange/10 text-cookie-orange text-sm font-bold">
              {classifyResult?.predicted_category || result.predicted_category}
            </span>
            <span className="text-sm text-gray-600">
              신뢰도 {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className={`p-4 rounded-lg border space-y-2 ${
          isAuto ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'
        }`}>
          <span className="text-xs text-gray-500 font-medium">라우팅 결정</span>
          <div className="flex items-center gap-2">
            {isAuto ? (
              <>
                <Zap className="w-5 h-5 text-green-600" />
                <span className="text-green-700 font-bold text-sm">자동 처리</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                <span className="text-amber-700 font-bold text-sm">담당자 검토 필요</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className={`p-4 rounded-lg border ${pColors.bg} ${pColors.border} space-y-2`}>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 font-medium">우선순위</span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${pColors.bg} ${pColors.text} border ${pColors.border}`}>
            {priority.toUpperCase()}
          </span>
        </div>
        {result.priority?.priority_description && (
          <p className={`text-sm ${pColors.text}`}>{result.priority.priority_description}</p>
        )}
      </div>

      {result.priority?.recommendations && result.priority.recommendations.length > 0 && (
        <div className="space-y-2">
          <span className="text-sm font-medium text-cookie-brown/80">추천 조치</span>
          <ul className="space-y-1">
            {result.priority.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                <ChevronRight className="w-4 h-4 text-cookie-orange mt-0.5 shrink-0" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
