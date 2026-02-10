// components/panels/lab/WorkflowNode.js - ReactFlow 워크플로우 노드 컴포넌트 + 상수
import { Handle, Position } from '@xyflow/react';
import {
  Zap, CheckCircle2, Sparkles, Send, FileText,
  Mail, MessageCircle, Smartphone, Bell,
} from 'lucide-react';

export const WORKFLOW_NODE_DEFS = [
  { id: 'webhook',  label: 'Webhook 트리거', x: 0,   y: 150, ntype: 'trigger' },
  { id: 'validate', label: '답변 검증',       x: 200, y: 150, ntype: 'process' },
  { id: 'router',   label: '채널 분기',       x: 400, y: 150, ntype: 'switch' },
  { id: 'channel_email', label: '이메일',     x: 620, y: 0,   ntype: 'channel' },
  { id: 'channel_kakao', label: '카카오톡',   x: 620, y: 100, ntype: 'channel' },
  { id: 'channel_sms',   label: 'SMS',       x: 620, y: 200, ntype: 'channel' },
  { id: 'channel_inapp', label: '인앱 알림',  x: 620, y: 300, ntype: 'channel' },
  { id: 'log',      label: '결과 기록',       x: 860, y: 150, ntype: 'output' },
];

export const WORKFLOW_EDGE_DEFS = [
  { source: 'webhook',  target: 'validate' },
  { source: 'validate', target: 'router' },
  { source: 'router',   target: 'channel_email' },
  { source: 'router',   target: 'channel_kakao' },
  { source: 'router',   target: 'channel_sms' },
  { source: 'router',   target: 'channel_inapp' },
  { source: 'channel_email', target: 'log' },
  { source: 'channel_kakao', target: 'log' },
  { source: 'channel_sms',   target: 'log' },
  { source: 'channel_inapp', target: 'log' },
];

const NODE_STYLES = {
  trigger: { bg: 'bg-purple-100', border: 'border-purple-400', text: 'text-purple-700', icon: Zap },
  process: { bg: 'bg-blue-100', border: 'border-blue-400', text: 'text-blue-700', icon: CheckCircle2 },
  switch:  { bg: 'bg-amber-100', border: 'border-amber-400', text: 'text-amber-700', icon: Sparkles },
  channel: { bg: 'bg-sky-100', border: 'border-sky-400', text: 'text-sky-700', icon: Send },
  output:  { bg: 'bg-green-100', border: 'border-green-400', text: 'text-green-700', icon: FileText },
};

const CHANNEL_ICONS = {
  channel_email: Mail,
  channel_kakao: MessageCircle,
  channel_sms: Smartphone,
  channel_inapp: Bell,
};

const STATUS_STYLES = {
  idle:      'border-gray-300 bg-white',
  running:   'border-yellow-400 bg-yellow-50 shadow-md animate-pulse',
  completed: 'border-green-500 bg-green-50',
  disabled:  'border-dashed border-gray-200 bg-gray-50 opacity-40',
};

function WorkflowNode({ data }) {
  const { label, ntype, nodeStatus, detail } = data;
  const style = NODE_STYLES[ntype] || NODE_STYLES.process;
  const statusStyle = STATUS_STYLES[nodeStatus] || STATUS_STYLES.idle;
  const IconComp = CHANNEL_ICONS[data.nodeId] || style.icon;
  const isCompleted = nodeStatus === 'completed';
  const isRunning = nodeStatus === 'running';

  return (
    <div className={`relative px-3 py-2 rounded-lg border-2 ${statusStyle} min-w-[90px] text-center transition-all duration-300`}>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-gray-400" />
      <div className="flex flex-col items-center gap-1">
        <div className={`w-6 h-6 rounded-md flex items-center justify-center ${isCompleted ? 'bg-green-500 text-white' : isRunning ? 'bg-yellow-400 text-white' : style.bg + ' ' + style.text}`}>
          {isCompleted ? <CheckCircle2 className="w-3.5 h-3.5" /> : <IconComp className="w-3.5 h-3.5" />}
        </div>
        <span className={`text-[10px] font-semibold ${isCompleted ? 'text-green-700' : isRunning ? 'text-yellow-700' : 'text-gray-600'}`}>
          {label}
        </span>
        {detail && (
          <span className="text-[8px] text-gray-500 leading-tight">{detail}</span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-gray-400" />
    </div>
  );
}

export const workflowNodeTypes = { workflowNode: WorkflowNode };
