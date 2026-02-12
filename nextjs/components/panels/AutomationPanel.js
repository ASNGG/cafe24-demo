// components/panels/AutomationPanel.js
// 자동화 엔진 — 탐지 → 자동 실행 (이탈방지 · FAQ 생성 · 운영 리포트)

import { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import {
  Zap, Shield, MessageSquare, FileText, Play, Loader2,
  AlertTriangle, CheckCircle2, ChevronDown, ChevronRight,
  Users, TrendingDown, Send, RefreshCw, Trash2, Edit3,
  Clock, BarChart3, Download, Eye, Plus, Search,
  ThumbsUp, Tag, HelpCircle
} from 'lucide-react';

const TABS = [
  { key: 'retention', label: '이탈 방지', icon: Shield },
  { key: 'faq', label: 'FAQ 자동 생성', icon: HelpCircle },
  { key: 'report', label: '운영 리포트', icon: FileText },
];

export default function AutomationPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('retention');

  return (
    <div className="space-y-4">
      {/* 서브탭 */}
      <div className="flex gap-2" role="tablist" aria-label="자동화 엔진 탭">
        {TABS.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.key;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={active}
              tabIndex={active ? 0 : -1}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all ${
                active
                  ? 'bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white shadow-lg shadow-orange-200'
                  : 'bg-white/80 text-gray-600 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === 'retention' && <RetentionTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'faq' && <FaqTab auth={auth} apiCall={apiCall} />}
      {activeTab === 'report' && <ReportTab auth={auth} apiCall={apiCall} />}
    </div>
  );
}


// ═══════════════════════════════════════
// 탭 1: 셀러 이탈 방지 자동 조치
// ═══════════════════════════════════════
function RetentionTab({ auth, apiCall }) {
  const [sellers, setSellers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.6);
  const [selectedSeller, setSelectedSeller] = useState(null);
  const [message, setMessage] = useState(null);
  const [msgLoading, setMsgLoading] = useState(false);
  const [execLoading, setExecLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const fetchAtRisk = useCallback(async () => {
    setLoading(true);
    setSellers([]);
    setSelectedSeller(null);
    setMessage(null);
    try {
      const res = await apiCall({
        endpoint: `/api/automation/retention/at-risk?threshold=${threshold}&limit=20`,
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setSellers(res.sellers || []);
        if ((res.sellers || []).length === 0) {
          toast('현재 이탈 위험 셀러가 없습니다', { icon: '✅' });
        }
      } else {
        toast.error(res?.detail || '조회 실패');
      }
    } catch (e) {
      toast.error('이탈 위험 셀러 조회 실패');
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth, threshold]);

  const generateMessage = useCallback(async (sellerId) => {
    setMsgLoading(true);
    setMessage(null);
    setSelectedSeller(sellerId);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/message',
        auth,
        method: 'POST',
        data: { seller_id: sellerId },
        timeoutMs: 60000,
      });
      if (res?.status === 'success') {
        setMessage(res);
      } else {
        toast.error(res?.detail || '메시지 생성 실패');
      }
    } catch (e) {
      toast.error('리텐션 메시지 생성 실패');
    } finally {
      setMsgLoading(false);
    }
  }, [apiCall, auth]);

  const executeAction = useCallback(async (sellerId, actionType) => {
    setExecLoading(true);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/execute',
        auth,
        method: 'POST',
        data: { seller_id: sellerId, action_type: actionType },
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        toast.success(`조치 실행 완료: ${actionType}`);
      } else {
        toast.error(res?.detail || '조치 실행 실패');
      }
    } catch (e) {
      toast.error('조치 실행 실패');
    } finally {
      setExecLoading(false);
    }
  }, [apiCall, auth]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await apiCall({
        endpoint: '/api/automation/retention/history?limit=20',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setHistory(res.history || []);
        setShowHistory(true);
      }
    } catch (e) {
      toast.error('이력 조회 실패');
    }
  }, [apiCall, auth]);

  const riskColor = (level) => {
    if (level === 'high') return 'text-red-600 bg-red-50 border-red-200';
    return 'text-yellow-600 bg-yellow-50 border-yellow-200';
  };

  const ACTION_TYPES = [
    { key: 'coupon', label: '할인 쿠폰 발급', icon: Tag },
    { key: 'upgrade_offer', label: '플랜 업그레이드 제안', icon: TrendingDown },
    { key: 'manager_assign', label: '전담 매니저 배정', icon: Users },
    { key: 'custom_message', label: '맞춤 메시지 발송', icon: Send },
  ];

  return (
    <div className="space-y-4">
      {/* 컨트롤 패널 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="text-red-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">셀러 이탈 방지 자동 조치</h3>
            <span className="text-xs text-gray-500">ML 예측 → LLM 메시지 → 자동 실행</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500">임계값:</label>
            <select
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              <option value={0.5}>50%</option>
              <option value={0.6}>60%</option>
              <option value={0.7}>70%</option>
              <option value={0.8}>80%</option>
            </select>
            <button
              onClick={fetchAtRisk}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-lg bg-red-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-red-600 disabled:opacity-50"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              위험 셀러 탐지
            </button>
            <button
              onClick={fetchHistory}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50"
            >
              <Clock size={14} />
              조치 이력
            </button>
          </div>
        </div>
      </div>

      {/* 위험 셀러 목록 */}
      {sellers.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <h4 className="text-sm font-bold text-gray-700 mb-3">
            이탈 위험 셀러 ({sellers.length}명)
          </h4>
          <div className="space-y-2">
            {sellers.map((s, i) => (
              <div
                key={s.seller_id || i}
                className={`rounded-xl border p-3 cursor-pointer transition-all hover:shadow-md ${
                  selectedSeller === s.seller_id ? 'ring-2 ring-cafe24-yellow border-cafe24-yellow' : 'border-gray-200'
                }`}
                onClick={() => setSelectedSeller(s.seller_id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-bold text-gray-800">{s.seller_id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${riskColor(s.risk_level)}`}>
                      {s.risk_level === 'high' ? '고위험' : '중위험'}
                    </span>
                    <span className="text-xs text-gray-500">
                      이탈확률: <strong className="text-red-600">{Number(s.churn_probability).toFixed(1)}%</strong>
                    </span>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); generateMessage(s.seller_id); }}
                    disabled={msgLoading}
                    className="flex items-center gap-1 rounded-lg bg-cafe24-yellow px-2.5 py-1 text-xs font-semibold text-cafe24-brown hover:bg-cafe24-orange hover:text-white disabled:opacity-50"
                  >
                    {msgLoading && selectedSeller === s.seller_id
                      ? <Loader2 size={12} className="animate-spin" />
                      : <MessageSquare size={12} />}
                    메시지 생성
                  </button>
                </div>
                {s.top_factors && s.top_factors.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {s.top_factors.slice(0, 5).map((f, fi) => (
                      <span key={fi} className="text-[10px] bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
                        {typeof f === 'string' ? f : f.feature || f.name || JSON.stringify(f)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LLM 생성 메시지 + 조치 실행 */}
      {message && (
        <div className="rounded-2xl border border-cafe24-yellow/50 bg-gradient-to-r from-yellow-50 to-orange-50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare className="text-cafe24-orange" size={18} />
            <h4 className="text-sm font-bold text-gray-800">
              {message.seller_id} 리텐션 메시지
            </h4>
            {message.urgency && (
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                message.urgency === 'high' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
              }`}>
                {message.urgency === 'high' ? '긴급' : '보통'}
              </span>
            )}
          </div>
          <div className="rounded-xl bg-white/80 p-3 text-xs text-gray-700 leading-relaxed mb-3">
            <ReactMarkdown>{message.message || ''}</ReactMarkdown>
          </div>

          {/* 자동 조치 버튼 */}
          <div className="flex flex-wrap gap-2">
            {ACTION_TYPES.map(act => {
              const ActIcon = act.icon;
              return (
                <button
                  key={act.key}
                  onClick={() => executeAction(message.seller_id, act.key)}
                  disabled={execLoading}
                  className="flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 hover:border-cafe24-yellow hover:bg-cafe24-yellow/10 disabled:opacity-50 transition-all"
                >
                  {execLoading ? <Loader2 size={12} className="animate-spin" /> : <ActIcon size={12} />}
                  {act.label}
                </button>
              );
            })}
          </div>

          {message.recommended_actions && message.recommended_actions.length > 0 && (
            <div className="mt-3 text-[10px] text-gray-500">
              AI 추천: {message.recommended_actions.join(', ')}
            </div>
          )}
        </div>
      )}

      {/* 조치 이력 */}
      {showHistory && history.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-gray-700">조치 이력</h4>
            <button onClick={() => setShowHistory(false)} className="text-xs text-gray-400 hover:text-gray-600">닫기</button>
          </div>
          <div className="space-y-1.5 max-h-60 overflow-y-auto">
            {history.map((h, i) => (
              <div key={i} className="flex items-center gap-3 text-xs border-b border-gray-100 pb-1.5">
                <CheckCircle2 size={12} className="text-green-500" />
                <span className="text-gray-500 w-24">{new Date(h.timestamp * 1000).toLocaleString('ko-KR')}</span>
                <span className="font-semibold text-gray-700">{h.seller_id || h.target_id}</span>
                <span className="text-gray-500">{h.action_type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════
// 탭 2: CS FAQ 자동 생성
// ═══════════════════════════════════════
function FaqTab({ auth, apiCall }) {
  const [patterns, setPatterns] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [faqs, setFaqs] = useState([]);
  const [faqsLoading, setFaqsLoading] = useState(false);
  const [genCategory, setGenCategory] = useState('');
  const [genCount, setGenCount] = useState(5);
  const [editingFaq, setEditingFaq] = useState(null);
  const [editForm, setEditForm] = useState({ question: '', answer: '' });

  const analyzePatterns = useCallback(async () => {
    setAnalyzing(true);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/analyze',
        auth,
        method: 'POST',
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setPatterns(res);
        toast.success('CS 패턴 분석 완료');
      } else {
        toast.error(res?.detail || '분석 실패');
      }
    } catch (e) {
      toast.error('CS 패턴 분석 실패');
    } finally {
      setAnalyzing(false);
    }
  }, [apiCall, auth]);

  const generateFaqs = useCallback(async () => {
    setGenerating(true);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/generate',
        auth,
        method: 'POST',
        data: {
          category: genCategory || null,
          count: genCount,
        },
        timeoutMs: 120000,
      });
      if (res?.status === 'success') {
        toast.success(`FAQ ${res.generated_count || 0}개 생성 완료`);
        loadFaqs();
      } else {
        toast.error(res?.detail || 'FAQ 생성 실패');
      }
    } catch (e) {
      toast.error('FAQ 생성 실패');
    } finally {
      setGenerating(false);
    }
  }, [apiCall, auth, genCategory, genCount]);

  const loadFaqs = useCallback(async () => {
    setFaqsLoading(true);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/list',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setFaqs(res.faqs || []);
      }
    } catch (e) {
      toast.error('FAQ 목록 조회 실패');
    } finally {
      setFaqsLoading(false);
    }
  }, [apiCall, auth]);

  const approveFaq = useCallback(async (faqId) => {
    try {
      const res = await apiCall({
        endpoint: `/api/automation/faq/${faqId}/approve`,
        auth,
        method: 'PUT',
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        toast.success('FAQ 승인 완료');
        loadFaqs();
      } else {
        toast.error('승인 실패');
      }
    } catch (e) {
      toast.error('FAQ 승인 실패');
    }
  }, [apiCall, auth, loadFaqs]);

  const deleteFaq = useCallback(async (faqId) => {
    try {
      const res = await apiCall({
        endpoint: `/api/automation/faq/${faqId}`,
        auth,
        method: 'DELETE',
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        toast.success('FAQ 삭제 완료');
        loadFaqs();
      } else {
        toast.error('삭제 실패');
      }
    } catch (e) {
      toast.error('FAQ 삭제 실패');
    }
  }, [apiCall, auth, loadFaqs]);

  const startEdit = (faq) => {
    setEditingFaq(faq.id);
    setEditForm({ question: faq.question || '', answer: faq.answer || '' });
  };

  const saveEdit = useCallback(async (faqId) => {
    try {
      const res = await apiCall({
        endpoint: `/api/automation/faq/${faqId}`,
        auth,
        method: 'PUT',
        data: editForm,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        toast.success('FAQ 수정 완료');
        setEditingFaq(null);
        loadFaqs();
      } else {
        toast.error('수정 실패');
      }
    } catch (e) {
      toast.error('FAQ 수정 실패');
    }
  }, [apiCall, auth, editForm, loadFaqs]);

  return (
    <div className="space-y-4">
      {/* 컨트롤 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <HelpCircle className="text-blue-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">CS FAQ 자동 생성</h3>
            <span className="text-xs text-gray-500">문의 패턴 분석 → LLM FAQ 생성 → 승인 관리</span>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={analyzePatterns}
            disabled={analyzing}
            className="flex items-center gap-1.5 rounded-lg bg-blue-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-600 disabled:opacity-50"
          >
            {analyzing ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
            CS 패턴 분석
          </button>

          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="카테고리 (선택)"
              value={genCategory}
              onChange={(e) => setGenCategory(e.target.value)}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs w-28"
            />
            <select
              value={genCount}
              onChange={(e) => setGenCount(parseInt(e.target.value))}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              {[3, 5, 10].map(n => (
                <option key={n} value={n}>{n}개</option>
              ))}
            </select>
            <button
              onClick={generateFaqs}
              disabled={generating}
              className="flex items-center gap-1.5 rounded-lg bg-cafe24-yellow px-3 py-1.5 text-xs font-semibold text-cafe24-brown hover:bg-cafe24-orange hover:text-white disabled:opacity-50"
            >
              {generating ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
              FAQ 생성
            </button>
          </div>

          <button
            onClick={loadFaqs}
            disabled={faqsLoading}
            className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50"
          >
            <RefreshCw size={14} className={faqsLoading ? 'animate-spin' : ''} />
            목록 새로고침
          </button>
        </div>
      </div>

      {/* 패턴 분석 결과 */}
      {patterns && (
        <div className="rounded-2xl border border-blue-200 bg-blue-50/50 p-4">
          <h4 className="text-sm font-bold text-gray-700 mb-3">
            CS 문의 패턴 분석 (총 {patterns.total_inquiries?.toLocaleString() || 0}건)
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
            {(patterns.categories || []).slice(0, 10).map((cat, i) => (
              <div key={i} className="rounded-xl bg-white p-2.5 border border-blue-100">
                <div className="text-[10px] text-gray-500">{cat.category}</div>
                <div className="text-sm font-bold text-gray-800">{cat.count?.toLocaleString()}</div>
                <div className="text-[10px] text-blue-600">{cat.percentage}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* FAQ 목록 */}
      {faqs.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <h4 className="text-sm font-bold text-gray-700 mb-3">FAQ 목록 ({faqs.length}개)</h4>
          <div className="space-y-3">
            {faqs.map((faq, i) => (
              <div key={faq.id || i} className="rounded-xl border border-gray-200 p-3">
                {editingFaq === faq.id ? (
                  /* 수정 모드 */
                  <div className="space-y-2">
                    <input
                      type="text"
                      value={editForm.question}
                      onChange={(e) => setEditForm(f => ({ ...f, question: e.target.value }))}
                      className="w-full rounded-lg border border-gray-200 px-2 py-1 text-xs"
                      placeholder="질문"
                    />
                    <textarea
                      value={editForm.answer}
                      onChange={(e) => setEditForm(f => ({ ...f, answer: e.target.value }))}
                      className="w-full rounded-lg border border-gray-200 px-2 py-1 text-xs h-20"
                      placeholder="답변"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => saveEdit(faq.id)}
                        className="rounded-lg bg-green-500 px-2 py-1 text-xs text-white hover:bg-green-600"
                      >
                        저장
                      </button>
                      <button
                        onClick={() => setEditingFaq(null)}
                        className="rounded-lg border border-gray-200 px-2 py-1 text-xs text-gray-600"
                      >
                        취소
                      </button>
                    </div>
                  </div>
                ) : (
                  /* 표시 모드 */
                  <div>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
                            faq.status === 'approved'
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-600'
                          }`}>
                            {faq.status === 'approved' ? '승인됨' : '초안'}
                          </span>
                          {faq.category && (
                            <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded-full">
                              {faq.category}
                            </span>
                          )}
                        </div>
                        <div className="text-xs font-bold text-gray-800 mb-1">Q: {faq.question}</div>
                        <div className="text-xs text-gray-600 leading-relaxed">A: {faq.answer}</div>
                        {faq.tags && faq.tags.length > 0 && (
                          <div className="mt-1.5 flex flex-wrap gap-1">
                            {faq.tags.map((tag, ti) => (
                              <span key={ti} className="text-[10px] bg-gray-100 text-gray-500 px-1 py-0.5 rounded">
                                #{tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-1 ml-2">
                        {faq.status !== 'approved' && (
                          <button
                            onClick={() => approveFaq(faq.id)}
                            className="p-1 rounded hover:bg-green-100 text-green-600"
                            title="승인"
                          >
                            <ThumbsUp size={14} />
                          </button>
                        )}
                        <button
                          onClick={() => startEdit(faq)}
                          className="p-1 rounded hover:bg-blue-100 text-blue-600"
                          title="수정"
                        >
                          <Edit3 size={14} />
                        </button>
                        <button
                          onClick={() => deleteFaq(faq.id)}
                          className="p-1 rounded hover:bg-red-100 text-red-600"
                          title="삭제"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ═══════════════════════════════════════
// 탭 3: 운영 리포트 자동 생성
// ═══════════════════════════════════════
function ReportTab({ auth, apiCall }) {
  const [reportType, setReportType] = useState('daily');
  const [generating, setGenerating] = useState(false);
  const [report, setReport] = useState(null);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [viewingReport, setViewingReport] = useState(null);

  const generateReport = useCallback(async () => {
    setGenerating(true);
    setReport(null);
    try {
      const res = await apiCall({
        endpoint: '/api/automation/report/generate',
        auth,
        method: 'POST',
        data: { report_type: reportType },
        timeoutMs: 120000,
      });
      if (res?.status === 'success') {
        setReport(res);
        toast.success(`${reportType} 리포트 생성 완료`);
      } else {
        toast.error(res?.detail || '리포트 생성 실패');
      }
    } catch (e) {
      toast.error('리포트 생성 실패');
    } finally {
      setGenerating(false);
    }
  }, [apiCall, auth, reportType]);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await apiCall({
        endpoint: '/api/automation/report/history?limit=20',
        auth,
        timeoutMs: 30000,
      });
      if (res?.status === 'success') {
        setHistory(res.reports || []);
        setShowHistory(true);
      }
    } catch (e) {
      toast.error('리포트 이력 조회 실패');
    }
  }, [apiCall, auth]);

  const REPORT_TYPES = [
    { value: 'daily', label: '일간 리포트' },
    { value: 'weekly', label: '주간 리포트' },
    { value: 'monthly', label: '월간 리포트' },
  ];

  return (
    <div className="space-y-4">
      {/* 컨트롤 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="text-emerald-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">운영 리포트 자동 생성</h3>
            <span className="text-xs text-gray-500">플랫폼 KPI 집계 → LLM 리포트 작성</span>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={reportType}
              onChange={(e) => setReportType(e.target.value)}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              {REPORT_TYPES.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
            <button
              onClick={generateReport}
              disabled={generating}
              className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-600 disabled:opacity-50"
            >
              {generating ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              리포트 생성
            </button>
            <button
              onClick={fetchHistory}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50"
            >
              <Clock size={14} />
              생성 이력
            </button>
          </div>
        </div>
      </div>

      {/* 생성된 리포트 */}
      {(report || viewingReport) && (
        <div className="rounded-2xl border border-emerald-200 bg-gradient-to-r from-emerald-50 to-green-50 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <FileText className="text-emerald-600" size={18} />
              <h4 className="text-sm font-bold text-gray-800">
                {(report || viewingReport)?.report_type} 운영 리포트
              </h4>
              <span className="text-[10px] text-gray-500">
                {new Date(((report || viewingReport)?.timestamp || 0) * 1000).toLocaleString('ko-KR')}
              </span>
            </div>
            {viewingReport && (
              <button
                onClick={() => setViewingReport(null)}
                className="text-xs text-gray-400 hover:text-gray-600"
              >
                닫기
              </button>
            )}
          </div>
          <div className="rounded-xl bg-white/90 p-4 prose prose-sm max-w-none">
            <ReactMarkdown>{(report || viewingReport)?.content || ''}</ReactMarkdown>
          </div>
          {(report || viewingReport)?.data_summary && (
            <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries((report || viewingReport).data_summary).slice(0, 8).map(([key, val]) => (
                <div key={key} className="rounded-lg bg-white/80 p-2 text-center">
                  <div className="text-[10px] text-gray-500">{key}</div>
                  <div className="text-xs font-bold text-gray-800">
                    {typeof val === 'number' ? val.toLocaleString() : String(val)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 리포트 이력 */}
      {showHistory && history.length > 0 && (
        <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-bold text-gray-700">리포트 생성 이력</h4>
            <button onClick={() => setShowHistory(false)} className="text-xs text-gray-400 hover:text-gray-600">닫기</button>
          </div>
          <div className="space-y-2">
            {history.map((r, i) => (
              <div
                key={r.report_id || i}
                className="flex items-center justify-between rounded-xl border border-gray-200 p-2.5 hover:bg-gray-50 cursor-pointer"
                onClick={() => { setViewingReport(r); setShowHistory(false); }}
              >
                <div className="flex items-center gap-3">
                  <FileText size={14} className="text-emerald-500" />
                  <span className="text-xs font-semibold text-gray-700">{r.report_type} 리포트</span>
                  <span className="text-[10px] text-gray-500">
                    {new Date((r.timestamp || 0) * 1000).toLocaleString('ko-KR')}
                  </span>
                </div>
                <Eye size={14} className="text-gray-400" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
