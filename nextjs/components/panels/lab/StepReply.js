// components/panels/lab/StepReply.js - Step 4: 회신 - 채널별 자동 발송
// H32: EditableAnswer 공통 컴포넌트 사용
// L35: onInit triple fitView → 1회
import { useState, useEffect, useMemo } from 'react';
import { ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  Send, Zap, CheckCircle2, Loader2, User, Edit3, FileText, Play, Mail,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { createAuthHeaders, parseSSEStream } from '@/lib/sse';
import { WORKFLOW_NODE_DEFS, WORKFLOW_EDGE_DEFS, workflowNodeTypes } from './WorkflowNode';
import { EmptyStep } from './utils';
import EditableAnswer from '@/components/common/EditableAnswer';

export default function StepReply({ channels, selectedChannels, setSelectedChannels, sent, setSent, draftAnswer, inquiry, classifyResults, autoIdxs, batchAnswers, updateBatchAnswer, auth }) {
  const [autoReplyChannels, setAutoReplyChannels] = useState({});
  const [nodeStatuses, setNodeStatuses] = useState({});
  const [isSending, setIsSending] = useState(false);
  const [autoEmails, setAutoEmails] = useState({});
  const [manualEmail, setManualEmail] = useState('');

  const batchAnswerCount = autoIdxs?.length > 0
    ? Object.keys(batchAnswers || {}).filter(k => autoIdxs.includes(Number(k))).length
    : 0;
  const isAutoMode = !draftAnswer && batchAnswerCount > 0;

  // 자동 처리 초기 채널 설정 (미구현 채널은 이메일로 대체)
  useEffect(() => {
    if (isAutoMode && Object.keys(autoReplyChannels).length === 0) {
      const enabledKeys = new Set(channels.filter(c => c.enabled).map(c => c.key));
      const initial = {};
      autoIdxs.forEach(idx => {
        const item = classifyResults?.[idx];
        if (!item) return;
        const pref = item.preferredChannels || ['any'];
        const validPref = pref.filter(k => enabledKeys.has(k));
        initial[idx] = new Set(validPref.length > 0 ? validPref : ['email']);
      });
      setAutoReplyChannels(initial);
    }
  }, [isAutoMode, autoIdxs, classifyResults]);

  const activeChannelKeys = useMemo(() => {
    if (isAutoMode) {
      const all = new Set();
      Object.values(autoReplyChannels).forEach(s => s.forEach(c => all.add(c)));
      return all;
    }
    return selectedChannels;
  }, [isAutoMode, autoReplyChannels, selectedChannels]);

  // React Flow 노드/엣지 생성
  const { nodes: flowNodes, edges: flowEdges } = useMemo(() => {
    const nodes = WORKFLOW_NODE_DEFS.map(nd => {
      const isChannel = nd.id.startsWith('channel_');
      const chKey = isChannel ? nd.id.replace('channel_', '') : null;
      const isActive = !isChannel || activeChannelKeys.has(chKey);
      let nodeStatus = 'idle';
      if (nodeStatuses[nd.id]) {
        nodeStatus = nodeStatuses[nd.id].status;
      } else if (!isActive) {
        nodeStatus = 'disabled';
      }
      return {
        id: nd.id,
        type: 'workflowNode',
        position: { x: nd.x, y: nd.y },
        width: 110,
        height: 58,
        data: {
          label: nd.label,
          ntype: nd.ntype,
          nodeId: nd.id,
          nodeStatus,
          detail: nodeStatuses[nd.id]?.detail || '',
        },
        draggable: false,
        connectable: false,
      };
    });

    const edges = WORKFLOW_EDGE_DEFS.map(ed => {
      const isTargetChannel = ed.target.startsWith('channel_');
      const chKey = isTargetChannel ? ed.target.replace('channel_', '') : null;
      const isSourceChannel = ed.source.startsWith('channel_');
      const srcChKey = isSourceChannel ? ed.source.replace('channel_', '') : null;
      const isActive = (!isTargetChannel || activeChannelKeys.has(chKey)) && (!isSourceChannel || activeChannelKeys.has(srcChKey));
      return {
        id: `${ed.source}-${ed.target}`,
        source: ed.source,
        target: ed.target,
        animated: isActive && (nodeStatuses[ed.source]?.status === 'completed' || nodeStatuses[ed.target]?.status === 'running'),
        style: {
          stroke: isActive ? '#94a3b8' : '#e2e8f0',
          strokeWidth: isActive ? 1.5 : 1,
          strokeDasharray: isActive ? undefined : '4 4',
          opacity: isActive ? 1 : 0.4,
        },
      };
    });
    return { nodes, edges };
  }, [activeChannelKeys, nodeStatuses]);

  const toggleChannel = (key) => {
    if (sent || isSending) return;
    const ch = channels.find(c => c.key === key);
    if (ch && !ch.enabled) return;
    setSelectedChannels(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const toggleAutoChannel = (idx, key) => {
    if (sent || isSending) return;
    const ch = channels.find(c => c.key === key);
    if (ch && !ch.enabled) return;
    setAutoReplyChannels(prev => {
      const current = new Set(prev[idx] || []);
      current.has(key) ? current.delete(key) : current.add(key);
      return { ...prev, [idx]: current };
    });
  };

  const canSend = (() => {
    if (isSending || sent) return false;
    if (isAutoMode) {
      return autoIdxs.filter(idx => batchAnswers?.[idx]).every(idx => {
        const chs = autoReplyChannels[idx];
        if (!chs || chs.size === 0) return false;
        if (chs.has('email') && !autoEmails[idx]?.includes('@')) return false;
        return true;
      });
    }
    if (selectedChannels.size === 0 || !draftAnswer) return false;
    if (selectedChannels.has('email') && !manualEmail.includes('@')) return false;
    return true;
  })();

  const handleSend = async () => {
    if (!canSend) {
      toast.error(isAutoMode ? '모든 문의에 회신 채널을 선택하세요.' : '회신 채널을 선택하세요.');
      return;
    }
    setIsSending(true);
    setNodeStatuses({ webhook: { status: 'completed', detail: '트리거 완료' } });

    const inquiries = [];
    if (isAutoMode) {
      autoIdxs.filter(idx => batchAnswers?.[idx]).forEach(idx => {
        const item = classifyResults?.[idx];
        inquiries.push({
          inquiry_text: item?.text || '',
          answer_text: batchAnswers[idx] || '',
          channels: [...(autoReplyChannels[idx] || [])],
          seller_tier: item?.tier || 'Standard',
          category: item?.result?.predicted_category || '기타',
          recipient_email: autoEmails[idx] || '',
        });
      });
    } else {
      inquiries.push({
        inquiry_text: inquiry?.text || '',
        answer_text: draftAnswer || '',
        channels: [...selectedChannels],
        seller_tier: inquiry?.tier || 'Standard',
        category: '기타',
        recipient_email: manualEmail || '',
      });
    }

    try {
      const headers = createAuthHeaders(auth);

      // 1) job_id 발급 + n8n 트리거
      const triggerResp = await fetch('/api/cs/send-reply', {
        method: 'POST',
        headers,
        body: JSON.stringify({ inquiries }),
      });
      const triggerData = await triggerResp.json();
      if (!triggerData.job_id) throw new Error(triggerData.message || 'job_id 발급 실패');

      // 2) SSE 스트림 연결 (job_id 기반)
      const streamResp = await fetch(`/api/cs/stream?job_id=${triggerData.job_id}`, { headers });
      if (!streamResp.ok || !streamResp.body) throw new Error(`SSE stream HTTP ${streamResp.status}`);

      await parseSSEStream(streamResp, {
        onStep: (data) => {
          setNodeStatuses(prev => ({
            ...prev,
            [data.node]: { status: data.status, detail: data.detail || '' },
          }));
        },
        onDone: (data) => {
          setSent(true);
          const chLabels = (data.channels || []).map(k => channels.find(c => c.key === k)?.label || k).join(', ');
          toast.success(`${data.total}건이 ${chLabels}(으)로 전송 완료!`);
        },
        onError: (data) => {
          toast.error(`전송 오류: ${data}`);
        },
      });
    } catch (e) {
      toast.error(`전송 실패: ${e.message}`);
      setNodeStatuses(prev => ({ ...prev, validate: { status: 'completed', detail: 'fallback' }, router: { status: 'completed', detail: 'fallback' }, log: { status: 'completed', detail: 'fallback' } }));
      setSent(true);
    } finally {
      setIsSending(false);
    }
  };

  const ChannelBadge = ({ chKey, small }) => {
    const ch = channels.find(c => c.key === chKey);
    if (!ch) return null;
    const Icon = ch.icon;
    return (
      <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-gradient-to-r ${ch.color} text-white ${small ? 'text-[9px]' : 'text-[10px]'} font-medium`}>
        <Icon className={small ? 'w-2.5 h-2.5' : 'w-3 h-3'} />
        {ch.label}
      </span>
    );
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-cafe24-brown/10 space-y-5">
      <div className="flex items-center gap-2 text-cafe24-brown font-semibold text-lg">
        <Send className="w-5 h-5 text-cafe24-orange" />
        Step 4. 회신 - 채널별 자동 발송
      </div>

      {/* 자동 처리 모드: per-inquiry 채널 + 답변 */}
      {isAutoMode && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 p-3 rounded-lg bg-green-50 border border-green-200">
            <Zap className="w-5 h-5 text-green-600" />
            <span className="text-sm font-medium text-green-700">
              자동 처리 {batchAnswerCount}건 - 고객 희망 채널별 회신
            </span>
          </div>

          <div className="space-y-3 max-h-[420px] overflow-y-auto">
            {autoIdxs.filter(idx => batchAnswers?.[idx]).map(idx => {
              const item = classifyResults?.[idx];
              if (!item) return null;
              const pref = item.preferredChannels || ['any'];
              const isAnyPref = pref.includes('any');
              const assignedChannels = autoReplyChannels[idx] || new Set();

              return (
                <div key={idx} className="rounded-lg border border-gray-200 overflow-hidden">
                  <div className="p-3 bg-gray-50 border-b border-gray-100 space-y-2">
                    <div className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0 mt-0.5" />
                      <p className="text-xs text-gray-700 leading-relaxed flex-1 line-clamp-2">{item.text}</p>
                      <span className="px-1.5 py-0.5 rounded-full bg-cafe24-orange/10 text-cafe24-orange text-[10px] font-bold shrink-0">
                        {item.result?.predicted_category || '?'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 ml-5">
                      <span className="text-[10px] text-gray-500 shrink-0">고객 희망:</span>
                      {isAnyPref ? (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-gray-200 text-gray-600 font-medium">
                          어느 방식이든 상관없음
                        </span>
                      ) : (
                        <div className="flex gap-1 flex-wrap">
                          {pref.map(k => <ChannelBadge key={k} chKey={k} small />)}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2 ml-5 flex-wrap">
                      <span className="text-[10px] text-gray-500 shrink-0">회신 채널:</span>
                      {channels.map(ch => {
                        const Icon = ch.icon;
                        const selected = assignedChannels.has(ch.key);
                        const disabled = !ch.enabled || sent || isSending;
                        return (
                          <button
                            key={ch.key}
                            onClick={() => ch.enabled && toggleAutoChannel(idx, ch.key)}
                            disabled={disabled}
                            title={!ch.enabled ? '미구현' : ''}
                            className={`flex items-center gap-1 px-2 py-1 rounded-lg border text-[10px] font-medium transition-all ${
                              !ch.enabled
                                ? 'border-gray-100 bg-gray-50 text-gray-300 cursor-not-allowed line-through'
                                : selected
                                ? 'border-cafe24-orange bg-cafe24-orange/10 text-cafe24-orange'
                                : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300'
                            } ${(sent || isSending) ? 'opacity-60 cursor-not-allowed' : ''}`}
                          >
                            <Icon className="w-3 h-3" />
                            {ch.label}
                            {!ch.enabled && <span className="text-[8px] text-gray-300 ml-0.5">(미구현)</span>}
                          </button>
                        );
                      })}
                    </div>
                    {assignedChannels.has('email') && (
                      <div className="ml-5 mt-2 space-y-1">
                        <label className="text-[11px] text-gray-500 font-medium ml-1">여기에 이메일을 입력해보세요</label>
                        <div className="flex items-center gap-2">
                          <Mail className="w-4 h-4 text-gray-400 shrink-0" />
                          <input
                            type="email"
                            placeholder="예: seller@shop.com"
                            value={autoEmails[idx] || ''}
                            onChange={e => setAutoEmails(prev => ({ ...prev, [idx]: e.target.value }))}
                            disabled={sent || isSending}
                            className="flex-1 px-3 py-2 rounded-lg border border-gray-200 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:border-cafe24-orange focus:ring-1 focus:ring-cafe24-orange/30 disabled:opacity-50"
                          />
                        </div>
                        <p className="text-[10px] text-blue-500 ml-6">전송 시 실제로 해당 이메일로 답변이 발송됩니다.</p>
                      </div>
                    )}
                  </div>
                  {/* H32: EditableAnswer 공통 컴포넌트 사용 */}
                  <div className="border-t border-gray-100 bg-white">
                    <details className="group">
                      <summary className="flex items-center gap-1 p-2.5 text-[10px] text-green-600 font-medium cursor-pointer hover:bg-gray-50">
                        <FileText className="w-3 h-3" />
                        답변 내용 보기
                      </summary>
                      <div className="p-2.5 pt-0">
                        <EditableAnswer
                          answer={batchAnswers[idx]}
                          onSave={(newText) => updateBatchAnswer(idx, newText)}
                          disabled={sent || isSending}
                          rows={5}
                          maxHeight="max-h-24"
                        />
                      </div>
                    </details>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 수동 처리 모드: 단일 문의 */}
      {!isAutoMode && inquiry && (
        <div className="space-y-3">
          {inquiry.preferredChannels && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-50 border border-blue-200">
              <User className="w-4 h-4 text-blue-600 shrink-0" />
              <span className="text-xs text-blue-700 font-medium shrink-0">고객 희망 채널:</span>
              {inquiry.preferredChannels.includes('any') ? (
                <span className="text-xs text-blue-600">어느 방식이든 상관없음 (아래에서 선택하세요)</span>
              ) : (
                <div className="flex gap-1.5 flex-wrap">
                  {inquiry.preferredChannels.map(k => {
                    const ch = channels.find(c => c.key === k);
                    if (!ch) return null;
                    const Icon = ch.icon;
                    return (
                      <span key={k} className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gradient-to-r ${ch.color} text-white text-[10px] font-medium`}>
                        <Icon className="w-3 h-3" />
                        {ch.label}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          <div>
            <span className="text-xs text-gray-500 font-medium mb-2 block">회신 채널 선택</span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {channels.map(ch => {
                const Icon = ch.icon;
                const selected = selectedChannels.has(ch.key);
                const isPreferred = inquiry?.preferredChannels?.includes(ch.key);
                const disabled = !ch.enabled || sent || isSending;
                return (
                  <button
                    key={ch.key}
                    onClick={() => ch.enabled && toggleChannel(ch.key)}
                    disabled={disabled}
                    className={`relative p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${
                      !ch.enabled
                        ? 'border-gray-100 bg-gray-50 cursor-not-allowed opacity-50'
                        : selected
                        ? 'border-cafe24-orange bg-cafe24-orange/5 shadow-md'
                        : 'border-gray-200 hover:border-gray-300 bg-white'
                    } ${(sent || isSending) ? 'opacity-60 cursor-not-allowed' : ''}`}
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${ch.enabled ? ch.color : 'from-gray-300 to-gray-400'} flex items-center justify-center text-white`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-sm font-medium ${!ch.enabled ? 'text-gray-400' : selected ? 'text-cafe24-orange' : 'text-gray-600'}`}>
                      {ch.label}
                    </span>
                    {!ch.enabled && (
                      <span className="text-[9px] text-gray-400 -mt-1">미구현</span>
                    )}
                    {selected && ch.enabled && (
                      <div className="absolute top-2 right-2 w-3 h-3 rounded-full bg-cafe24-orange" />
                    )}
                    {isPreferred && (
                      <span className="absolute top-1 left-1 px-1 py-0.5 rounded text-[8px] font-bold bg-blue-100 text-blue-600">
                        희망
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            <p className="text-[11px] text-amber-600 bg-amber-50 rounded-lg px-3 py-2 mt-3">
              카카오톡, SMS, 인앱 알림은 현재 미구현 상태입니다. 고객 선호 채널과 관계없이 이메일로 자동 발송됩니다.
            </p>
            {selectedChannels.has('email') && (
              <div className="mt-3 space-y-1">
                <label className="text-xs text-gray-500 font-medium ml-1">여기에 이메일을 입력해보세요</label>
                <div className="flex items-center gap-2">
                  <Mail className="w-4 h-4 text-gray-400 shrink-0" />
                  <input
                    type="email"
                    placeholder="예: seller@shop.com"
                    value={manualEmail}
                    onChange={e => setManualEmail(e.target.value)}
                    disabled={sent || isSending}
                    className="flex-1 px-3 py-2.5 rounded-lg border border-gray-200 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:border-cafe24-orange focus:ring-1 focus:ring-cafe24-orange/30 disabled:opacity-50"
                  />
                </div>
                <p className="text-[10px] text-blue-500 ml-6">전송 시 실제로 해당 이메일로 답변이 발송됩니다.</p>
                {manualEmail && !manualEmail.includes('@') && (
                  <p className="text-[10px] text-red-500 mt-1 ml-6">올바른 이메일 주소를 입력하세요.</p>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {!isAutoMode && !inquiry && (
        <EmptyStep message="먼저 답변을 생성하세요." />
      )}

      {/* React Flow 워크플로우 다이어그램 */}
      <div className="rounded-lg border border-gray-200 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
          <span className="text-xs text-gray-500 font-medium flex items-center gap-1.5">
            <Play className="w-3.5 h-3.5" />
            n8n 워크플로우 — 회신 자동화
          </span>
          {isSending && (
            <span className="flex items-center gap-1.5 text-[10px] text-yellow-600 font-medium">
              <Loader2 className="w-3 h-3 animate-spin" />
              실행 중...
            </span>
          )}
          {sent && !isSending && (
            <span className="flex items-center gap-1.5 text-[10px] text-green-600 font-medium">
              <CheckCircle2 className="w-3 h-3" />
              완료
            </span>
          )}
        </div>
        <div style={{ height: 340, width: '100%', position: 'relative' }}>
          {/* M63: key prop 제거 - 불필요한 재마운트 방지 */}
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={workflowNodeTypes}
            defaultViewport={{ x: 40, y: 20, zoom: 0.95 }}
            fitView
            fitViewOptions={{ padding: 0.25, includeHiddenNodes: true }}
            onInit={(inst) => {
              // L35: triple fitView → 1회 (레이아웃 안정 후 호출)
              setTimeout(() => { try { inst.fitView({ padding: 0.25, includeHiddenNodes: true }); } catch {} }, 150);
            }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            panOnDrag={false}
            zoomOnScroll={false}
            zoomOnDoubleClick={false}
            preventScrolling={false}
            proOptions={{ hideAttribution: true }}
          />
        </div>
      </div>

      {/* 전송 버튼 */}
      <div className="flex justify-center">
        {sent ? (
          <div className="flex items-center gap-2 px-6 py-3 rounded-lg bg-green-50 border border-green-200 text-green-700 font-medium">
            <CheckCircle2 className="w-5 h-5" />
            {isAutoMode ? `${batchAnswerCount}건 일괄 전송 완료` : '회신 전송 완료'}
          </div>
        ) : (
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="flex items-center gap-2 px-6 py-3 rounded-lg bg-cafe24-orange text-white font-medium hover:bg-cafe24-orange/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isSending ? (
              <><Loader2 className="w-4 h-4 animate-spin" />전송 중...</>
            ) : (
              <><Send className="w-4 h-4" />{isAutoMode ? `${batchAnswerCount}건 일괄 전송` : `${selectedChannels.size > 0 ? [...selectedChannels].map(k => channels.find(c => c.key === k)?.label).join(' + ') + '으로 ' : ''}전송`}</>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
