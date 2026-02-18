// components/panels/automation/FaqTab.js
// M68: AutomationPanel 분리 - 탭 2: CS FAQ 자동 생성
import { useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import {
  Zap, Loader2, RefreshCw, Trash2, Edit3,
  Search, ThumbsUp, HelpCircle,
} from 'lucide-react';
import PipelineFlow from '@/components/automation/PipelineFlow';
import { FAQ_STEPS, CS_CATEGORIES } from '@/components/automation/constants';

// IIFE → useMemo: filteredFaqs 메모이제이션
function FaqList({ faqs, faqSearch, setFaqSearch, faqStatusFilter, setFaqStatusFilter, faqCategoryFilter, setFaqCategoryFilter, approveFaq, startEdit, deleteFaq }) {
  const filteredFaqs = useMemo(() => faqs.filter(f => {
    const matchSearch = !faqSearch || f.question?.toLowerCase().includes(faqSearch.toLowerCase()) || f.answer?.toLowerCase().includes(faqSearch.toLowerCase());
    const matchStatus = faqStatusFilter === 'all' || f.status === faqStatusFilter;
    const matchCat = !faqCategoryFilter || f.category === faqCategoryFilter;
    return matchSearch && matchStatus && matchCat;
  }), [faqs, faqSearch, faqStatusFilter, faqCategoryFilter]);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
      <h4 className="text-sm font-bold text-gray-700 mb-3">FAQ 목록 ({faqs.length}개)</h4>
      <div className="flex items-center gap-2 flex-wrap mb-3">
        <div className="relative flex-1 min-w-[160px]">
          <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="질문/답변 검색..."
            value={faqSearch}
            onChange={(e) => setFaqSearch(e.target.value)}
            className="w-full rounded-lg border border-gray-200 pl-7 pr-2 py-1.5 text-xs"
          />
        </div>
        <select
          value={faqStatusFilter}
          onChange={(e) => setFaqStatusFilter(e.target.value)}
          className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs"
        >
          <option value="all">전체 상태</option>
          <option value="draft">초안</option>
          <option value="approved">승인됨</option>
        </select>
        <select
          value={faqCategoryFilter}
          onChange={(e) => setFaqCategoryFilter(e.target.value)}
          className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs"
        >
          <option value="">전체 카테고리</option>
          {CS_CATEGORIES.map(cat => <option key={cat} value={cat}>{cat}</option>)}
        </select>
        {filteredFaqs.length !== faqs.length && (
          <span className="text-[10px] text-gray-400">{filteredFaqs.length}개 표시</span>
        )}
      </div>
      <div className="space-y-3">
        {filteredFaqs.map((faq, i) => (
          <div key={faq.id || i} className="rounded-xl border border-gray-200 p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
                    faq.status === 'approved' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
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
                  <button onClick={() => approveFaq(faq.id)} className="p-1 rounded hover:bg-green-100 text-green-600" title="승인">
                    <ThumbsUp size={14} />
                  </button>
                )}
                <button onClick={() => startEdit(faq)} className="p-1 rounded hover:bg-blue-100 text-blue-600" title="수정">
                  <Edit3 size={14} />
                </button>
                <button onClick={() => deleteFaq(faq.id)} className="p-1 rounded hover:bg-red-100 text-red-600" title="삭제">
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function FaqTab({ auth, apiCall }) {
  const [patterns, setPatterns] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [faqs, setFaqs] = useState([]);
  const [faqsLoading, setFaqsLoading] = useState(false);
  const [genCategory, setGenCategory] = useState('');
  const [genCount, setGenCount] = useState(5);
  const [editingFaq, setEditingFaq] = useState(null);
  const [editForm, setEditForm] = useState({ question: '', answer: '' });
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const [faqSearch, setFaqSearch] = useState('');
  const [faqStatusFilter, setFaqStatusFilter] = useState('all');
  const [faqCategoryFilter, setFaqCategoryFilter] = useState('');

  const analyzePatterns = useCallback(async () => {
    setAnalyzing(true);
    setPipelineStatus({ analyze: { status: 'processing' } });
    setCurrentStep('analyze');
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
        setPipelineStatus(prev => ({ ...prev, analyze: { status: 'complete', detail: '패턴 분석 완료' } }));
        setCurrentStep(null);
      } else {
        toast.error(res?.detail || '분석 실패');
        setPipelineStatus(prev => ({ ...prev, analyze: { status: 'error' } }));
      }
    } catch (e) {
      toast.error('CS 패턴 분석 실패');
      setPipelineStatus(prev => ({ ...prev, analyze: { status: 'error' } }));
    } finally {
      setAnalyzing(false);
    }
  }, [apiCall, auth]);

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

  const generateFaqs = useCallback(async () => {
    setGenerating(true);
    setPipelineStatus(prev => ({ ...prev, select: { status: 'complete', detail: genCategory || '전체' }, generate: { status: 'processing' } }));
    setCurrentStep('generate');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/generate',
        auth,
        method: 'POST',
        data: { category: genCategory || null, count: genCount },
        timeoutMs: 120000,
      });
      if (res?.status === 'success') {
        toast.success(`FAQ ${res.generated_count || 0}개 생성 완료`);
        setPipelineStatus(prev => ({
          ...prev,
          generate: { status: 'complete', detail: `${res.generated_count || 0}개` },
          review: { status: 'complete', detail: '검토 대기' },
        }));
        setCurrentStep(null);
        loadFaqs();
      } else {
        toast.error(res?.detail || 'FAQ 생성 실패');
      }
    } catch (e) {
      toast.error('FAQ 생성 실패');
    } finally {
      setGenerating(false);
    }
  }, [apiCall, auth, genCategory, genCount, loadFaqs]);

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
        setPipelineStatus(prev => ({ ...prev, approve: { status: 'complete', detail: '승인 완료' } }));
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
            <select
              value={genCategory}
              onChange={(e) => setGenCategory(e.target.value)}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs"
            >
              <option value="">전체 카테고리</option>
              {CS_CATEGORIES.map(cat => <option key={cat} value={cat}>{cat}</option>)}
            </select>
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

      <PipelineFlow steps={FAQ_STEPS} stepStatuses={pipelineStatus} currentStep={currentStep} />

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

      {faqs.length > 0 && <FaqList
        faqs={faqs}
        faqSearch={faqSearch}
        setFaqSearch={setFaqSearch}
        faqStatusFilter={faqStatusFilter}
        setFaqStatusFilter={setFaqStatusFilter}
        faqCategoryFilter={faqCategoryFilter}
        setFaqCategoryFilter={setFaqCategoryFilter}
        approveFaq={approveFaq}
        startEdit={startEdit}
        deleteFaq={deleteFaq}
      />}

      {editingFaq && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center"
             onClick={(e) => { if (e.target === e.currentTarget) setEditingFaq(null); }}>
          <div className="bg-white rounded-2xl p-6 w-full max-w-lg shadow-2xl">
            <h3 className="text-sm font-bold text-gray-800 mb-4">FAQ 수정</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500 mb-1 block">질문</label>
                <input value={editForm.question} onChange={(e) => setEditForm(f => ({...f, question: e.target.value}))}
                       className="w-full rounded-xl border border-gray-200 p-3 text-sm" />
              </div>
              <div>
                <label className="text-xs text-gray-500 mb-1 block">답변</label>
                <textarea value={editForm.answer} onChange={(e) => setEditForm(f => ({...f, answer: e.target.value}))}
                          className="w-full rounded-xl border border-gray-200 p-3 text-sm h-32 resize-y" />
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button onClick={() => setEditingFaq(null)} className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50">취소</button>
              <button onClick={() => saveEdit(editingFaq)} className="rounded-lg bg-green-500 text-white px-3 py-1.5 text-xs font-semibold hover:bg-green-600">저장</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
