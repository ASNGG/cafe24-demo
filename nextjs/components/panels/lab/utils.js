// components/panels/lab/utils.js - LabPanel 공통 유틸 컴포넌트
import { AlertTriangle } from 'lucide-react';

export const renderMd = (text) => {
  if (!text) return null;
  return text.split(/(\*\*.*?\*\*)/g).map((seg, i) =>
    seg.startsWith('**') && seg.endsWith('**')
      ? <strong key={i} className="font-semibold">{seg.slice(2, -2)}</strong>
      : seg
  );
};

export function KpiMini({ label, value, icon: Icon }) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200 text-center">
      <Icon className="w-4 h-4 text-cookie-orange mx-auto mb-1" />
      <div className="text-lg font-bold text-cookie-brown">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

export function EmptyStep({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
      <AlertTriangle className="w-8 h-8 mb-3" />
      <p className="text-sm">{message}</p>
    </div>
  );
}
