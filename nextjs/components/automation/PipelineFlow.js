// M62: styled-jsx → CSS Module (매 렌더 CSS 파싱 방지)
import React from 'react';
import { CheckCircle2, XCircle } from 'lucide-react';
import styles from './PipelineFlow.module.css';

/**
 * PipelineFlow - Interactive pipeline step visualization
 *
 * @param {Object} props
 * @param {Array<{key:string, label:string, desc:string, icon:React.ComponentType}>} props.steps
 * @param {Object<string, {status:'pending'|'processing'|'complete'|'error', detail?:string}>} props.stepStatuses
 * @param {string|null} props.currentStep
 */
// L38: React.memo로 불필요한 리렌더 방지
export default React.memo(function PipelineFlow({ steps, stepStatuses = {}, currentStep }) {
  const getStatus = (key) => {
    if (stepStatuses[key]) return stepStatuses[key].status;
    return 'pending';
  };

  const getDetail = (key) => {
    return stepStatuses[key]?.detail || null;
  };

  const getLineClass = (leftKey, rightKey) => {
    const leftStatus = getStatus(leftKey);
    const rightStatus = getStatus(rightKey);

    if (leftStatus === 'complete' && rightStatus === 'complete') {
      return styles.lineComplete;
    }
    if (leftStatus === 'complete' && rightStatus === 'processing') {
      return styles.lineFlowing;
    }
    return styles.lineDefault;
  };

  const getNodeClasses = (status) => {
    switch (status) {
      case 'processing':
        return 'bg-gradient-to-br from-cafe24-yellow to-cafe24-orange text-white shadow-lg shadow-blue-200/50 animate-pulse';
      case 'complete':
        return 'bg-green-500 text-white border-2 border-green-400';
      case 'error':
        return 'bg-red-500 text-white border-2 border-red-400';
      default:
        return 'bg-gray-100 text-gray-400 border-2 border-gray-200';
    }
  };

  const getLabelClasses = (status) => {
    switch (status) {
      case 'processing':
        return 'text-cafe24-blue font-semibold';
      case 'complete':
        return 'text-green-600 font-medium';
      case 'error':
        return 'text-red-600 font-medium';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="rounded-2xl border border-gray-200 bg-white/90 p-4 backdrop-blur">
      <div className="overflow-x-auto">
        <div className="flex items-center min-w-max gap-0">
          {steps.map((step, idx) => {
            const status = getStatus(step.key);
            const detail = getDetail(step.key);
            const Icon = step.icon;
            const isLast = idx === steps.length - 1;

            return (
              <React.Fragment key={step.key}>
                {/* Node */}
                <div
                  className={`flex flex-col items-center gap-1.5 ${
                    status === 'processing' ? styles.nodeProcessing : ''
                  }`}
                  style={{ minWidth: 80 }}
                >
                  <div className="relative">
                    <div
                      className={`w-11 h-11 rounded-full flex items-center justify-center ${styles.nodeIcon} ${getNodeClasses(status)}`}
                    >
                      <Icon size={20} />
                    </div>

                    {status === 'complete' && (
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-white rounded-full flex items-center justify-center shadow-sm">
                        <CheckCircle2 size={14} className="text-green-500" />
                      </div>
                    )}

                    {status === 'error' && (
                      <div className="absolute -top-1 -right-1 w-4 h-4 bg-white rounded-full flex items-center justify-center shadow-sm">
                        <XCircle size={14} className="text-red-500" />
                      </div>
                    )}
                  </div>

                  <span className={`text-xs leading-tight text-center transition-all duration-500 ${getLabelClasses(status)}`}>
                    {step.label}
                  </span>

                  <span className="text-[10px] text-gray-400 leading-tight text-center">
                    {step.desc}
                  </span>

                  {status === 'complete' && detail && (
                    <span className="text-[10px] bg-green-50 text-green-600 px-1.5 py-0.5 rounded-full leading-tight">
                      {detail}
                    </span>
                  )}
                </div>

                {!isLast && (
                  <div className="flex items-center px-1 flex-1" style={{ minWidth: 32 }}>
                    <div className={getLineClass(step.key, steps[idx + 1].key)} />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
});
