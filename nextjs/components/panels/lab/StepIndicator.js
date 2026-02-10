// components/panels/lab/StepIndicator.js - 스텝 진행 표시기
import { CheckCircle2 } from 'lucide-react';

export default function StepIndicator({ steps, current, completed, onStepClick }) {
  return (
    <div className="flex items-center justify-between bg-white rounded-xl p-4 shadow-sm border border-cafe24-brown/10">
      {steps.map((step, i) => {
        const Icon = step.icon;
        const isActive = i === current;
        const isDone = completed.has(i);

        return (
          <div key={step.key} className="flex items-center flex-1">
            <button
              onClick={() => onStepClick(i)}
              className="flex flex-col items-center gap-1.5 flex-1 group"
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isActive
                    ? 'bg-cafe24-orange text-white shadow-lg shadow-cafe24-orange/30 scale-110'
                    : isDone
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-100 text-gray-400 group-hover:bg-gray-200'
                }`}
              >
                {isDone && !isActive ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>
              <span
                className={`text-xs font-medium ${
                  isActive ? 'text-cafe24-orange' : isDone ? 'text-green-600' : 'text-gray-400'
                }`}
              >
                {step.label}
              </span>
              <span className="text-[10px] text-gray-400 hidden sm:block text-center whitespace-nowrap">{step.desc}</span>
            </button>
            {i < steps.length - 1 && (
              <div
                className={`h-0.5 flex-1 mx-2 rounded transition-colors ${
                  completed.has(i) ? 'bg-green-400' : 'bg-gray-200'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
