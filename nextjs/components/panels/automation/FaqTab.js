// components/panels/automation/FaqTab.js
// CS FAQ 자동 생성 — TF-IDF + 실루엣 최적 K + K-Means / LLM 듀얼 클러스터링
import { useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import {
  Zap, Loader2, RefreshCw, Trash2, Edit3,
  Search, ThumbsUp, HelpCircle, ChevronDown, ChevronRight,
} from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, BarChart, Bar,
} from 'recharts';
import PipelineFlow from '@/components/automation/PipelineFlow';
import { FAQ_STEPS_KMEANS, FAQ_STEPS_LLM, CS_CATEGORIES } from '@/components/automation/constants';

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
          <input type="text" placeholder="질문/답변 검색..." value={faqSearch}
            onChange={(e) => setFaqSearch(e.target.value)}
            className="w-full rounded-lg border border-gray-200 pl-7 pr-2 py-1.5 text-xs" />
        </div>
        <select value={faqStatusFilter} onChange={(e) => setFaqStatusFilter(e.target.value)}
          className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs">
          <option value="all">전체 상태</option>
          <option value="draft">초안</option>
          <option value="approved">승인됨</option>
        </select>
        <select value={faqCategoryFilter} onChange={(e) => setFaqCategoryFilter(e.target.value)}
          className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs">
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
                    <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded-full">{faq.category}</span>
                  )}
                </div>
                <div className="text-xs font-bold text-gray-800 mb-1">Q: {faq.question}</div>
                <div className="text-xs text-gray-600 leading-relaxed">A: {faq.answer}</div>
                {faq.tags && faq.tags.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {faq.tags.map((tag, ti) => (
                      <span key={ti} className="text-[10px] bg-gray-100 text-gray-500 px-1 py-0.5 rounded">#{tag}</span>
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

// 클러스터 색상 팔레트
const CLUSTER_COLORS = [
  '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#6366f1', '#14b8a6',
];

// 실루엣 점수 색상
function silhouetteColor(score) {
  if (score >= 0.5) return 'text-emerald-600';
  if (score >= 0.3) return 'text-blue-600';
  return 'text-amber-600';
}

function silhouetteBg(score) {
  if (score >= 0.5) return 'bg-emerald-500';
  if (score >= 0.3) return 'bg-blue-500';
  return 'bg-amber-500';
}

export default function FaqTab({ auth, apiCall }) {
  const [patterns, setPatterns] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [faqs, setFaqs] = useState([]);
  const [faqsLoading, setFaqsLoading] = useState(false);
  const [genCategory, setGenCategory] = useState('');
  const [perCluster, setPerCluster] = useState(1);
  const [editingFaq, setEditingFaq] = useState(null);
  const [editForm, setEditForm] = useState({ question: '', answer: '' });
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [currentStep, setCurrentStep] = useState(null);
  const [faqSearch, setFaqSearch] = useState('');
  const [faqStatusFilter, setFaqStatusFilter] = useState('all');
  const [faqCategoryFilter, setFaqCategoryFilter] = useState('');
  const [expandedCats, setExpandedCats] = useState({});
  const [analyzeMode, setAnalyzeMode] = useState('kmeans'); // 'kmeans' | 'llm'

  const toggleCat = (cat) => setExpandedCats(prev => ({ ...prev, [cat]: !prev[cat] }));

  const analyzePatterns = useCallback(async () => {
    setAnalyzing(true);
    setPipelineStatus({ analyze: { status: 'processing' } });
    setCurrentStep('analyze');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/analyze',
        auth,
        method: 'POST',
        data: { mode: analyzeMode },
        timeoutMs: analyzeMode === 'llm' ? 120000 : 30000,
      });
      if (res?.status === 'success') {
        setPatterns(res);
        // 첫 번째 카테고리 자동 펼치기
        if (res.category_results?.length > 0) {
          setExpandedCats({ [res.category_results[0].category]: true });
        }
        toast.success(`CS 패턴 분석 완료 (${res.method === 'clustering' ? '클러스터링' : 'fallback'})`);
        setPipelineStatus(prev => ({
          ...prev,
          analyze: { status: 'complete', detail: `${res.total_inquiries}건 · ${res.category_results?.length || 0}개 카테고리` },
        }));
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
  }, [apiCall, auth, analyzeMode]);

  const loadFaqs = useCallback(async () => {
    setFaqsLoading(true);
    try {
      const res = await apiCall({ endpoint: '/api/automation/faq/list', auth, timeoutMs: 30000 });
      if (res?.status === 'success') setFaqs(res.faqs || []);
    } catch (e) {
      toast.error('FAQ 목록 조회 실패');
    } finally {
      setFaqsLoading(false);
    }
  }, [apiCall, auth]);

  const generateFaqs = useCallback(async () => {
    if (!patterns) {
      toast.error('CS 패턴 분석을 먼저 실행해주세요');
      return;
    }
    // 클러스터 수 × 클러스터당 FAQ 수 = 총 생성 개수
    const totalK = patterns.category_results?.reduce((sum, cr) => sum + (cr.optimal_k || 0), 0) || 5;
    const count = totalK * perCluster;
    setGenerating(true);
    setPipelineStatus(prev => ({ ...prev, generate: { status: 'processing' } }));
    setCurrentStep('generate');
    try {
      const res = await apiCall({
        endpoint: '/api/automation/faq/generate',
        auth, method: 'POST',
        data: { category: genCategory || null, count, mode: analyzeMode },
        timeoutMs: analyzeMode === 'llm' ? 120000 : 60000,
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
  }, [apiCall, auth, genCategory, perCluster, patterns, loadFaqs]);

  const approveFaq = useCallback(async (faqId) => {
    try {
      const res = await apiCall({ endpoint: `/api/automation/faq/${faqId}/approve`, auth, method: 'PUT', timeoutMs: 30000 });
      if (res?.status === 'success') { toast.success('FAQ 승인 완료'); setPipelineStatus(prev => ({ ...prev, approve: { status: 'complete', detail: '승인 완료' } })); loadFaqs(); }
      else toast.error('승인 실패');
    } catch (e) { toast.error('FAQ 승인 실패'); }
  }, [apiCall, auth, loadFaqs]);

  const deleteFaq = useCallback(async (faqId) => {
    try {
      const res = await apiCall({ endpoint: `/api/automation/faq/${faqId}`, auth, method: 'DELETE', timeoutMs: 30000 });
      if (res?.status === 'success') { toast.success('FAQ 삭제 완료'); loadFaqs(); }
      else toast.error('삭제 실패');
    } catch (e) { toast.error('FAQ 삭제 실패'); }
  }, [apiCall, auth, loadFaqs]);

  const startEdit = (faq) => { setEditingFaq(faq.id); setEditForm({ question: faq.question || '', answer: faq.answer || '' }); };

  const saveEdit = useCallback(async (faqId) => {
    try {
      const res = await apiCall({ endpoint: `/api/automation/faq/${faqId}`, auth, method: 'PUT', data: editForm, timeoutMs: 30000 });
      if (res?.status === 'success') { toast.success('FAQ 수정 완료'); setEditingFaq(null); loadFaqs(); }
      else toast.error('수정 실패');
    } catch (e) { toast.error('FAQ 수정 실패'); }
  }, [apiCall, auth, editForm, loadFaqs]);

  return (
    <div className="space-y-4">
      {/* 헤더 */}
      <div className="rounded-2xl border border-gray-200 bg-white/80 p-4 backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <HelpCircle className="text-blue-500" size={20} />
            <h3 className="text-base font-bold text-gray-800">CS FAQ 자동 생성</h3>
            <span className="text-xs text-gray-500">
              {analyzeMode === 'llm' ? 'LLM 의미 분류 → FAQ 자동 생성' : 'TF-IDF + 실루엣 최적 K + K-Means → LLM FAQ'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center rounded-lg border border-gray-200 overflow-hidden">
            <button onClick={() => setAnalyzeMode('kmeans')}
              className={`px-2.5 py-1.5 text-[11px] font-semibold transition-colors ${
                analyzeMode === 'kmeans' ? 'bg-blue-500 text-white' : 'bg-white text-gray-500 hover:bg-gray-50'
              }`}>
              K-Means
            </button>
            <button onClick={() => setAnalyzeMode('llm')}
              className={`px-2.5 py-1.5 text-[11px] font-semibold transition-colors ${
                analyzeMode === 'llm' ? 'bg-violet-500 text-white' : 'bg-white text-gray-500 hover:bg-gray-50'
              }`}>
              LLM
            </button>
          </div>
          <button onClick={analyzePatterns} disabled={analyzing}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50 ${
              analyzeMode === 'llm' ? 'bg-violet-500 hover:bg-violet-600' : 'bg-blue-500 hover:bg-blue-600'
            }`}>
            {analyzing ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
            CS 패턴 분석
          </button>
          <div className="flex items-center gap-2">
            <select value={genCategory} onChange={(e) => setGenCategory(e.target.value)}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs">
              <option value="">전체 카테고리</option>
              {CS_CATEGORIES.map(cat => <option key={cat} value={cat}>{cat}</option>)}
            </select>
            <label className="text-xs text-gray-500">클러스터당:</label>
            <select value={perCluster} onChange={(e) => setPerCluster(parseInt(e.target.value))}
              className="rounded-lg border border-gray-200 px-2 py-1 text-xs">
              {[1, 2, 3].map(n => <option key={n} value={n}>{n}개</option>)}
            </select>
            <button onClick={generateFaqs} disabled={generating}
              className="flex items-center gap-1.5 rounded-lg bg-cafe24-yellow px-3 py-1.5 text-xs font-semibold text-cafe24-brown hover:bg-cafe24-orange hover:text-white disabled:opacity-50">
              {generating ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
              FAQ 생성
            </button>
          </div>
          <button onClick={loadFaqs} disabled={faqsLoading}
            className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-semibold text-gray-600 hover:bg-gray-50">
            <RefreshCw size={14} className={faqsLoading ? 'animate-spin' : ''} />
            목록 새로고침
          </button>
        </div>
      </div>

      <PipelineFlow steps={analyzeMode === 'llm' ? FAQ_STEPS_LLM : FAQ_STEPS_KMEANS} stepStatuses={pipelineStatus} currentStep={currentStep} />

      {/* 클러스터링 결과: 카테고리별 아코디언 */}
      {patterns && (
        <div className="rounded-2xl border border-blue-200 bg-blue-50/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <h4 className="text-sm font-bold text-gray-700">
              CS 문의 클러스터링 (총 {patterns.total_inquiries?.toLocaleString() || 0}건)
            </h4>
            {patterns.method === 'clustering' && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">
                TF-IDF + Silhouette + K-Means
              </span>
            )}
            {patterns.method === 'llm' && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-violet-100 text-violet-700 font-semibold">
                LLM 의미 분류
              </span>
            )}
          </div>

          {/* 카테고리별 통계 그리드 */}
          {(patterns.categories || []).length > 0 && (
            <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-1.5 mb-4">
              {patterns.categories.map((cat, i) => {
                const catResult = (patterns.category_results || []).find(cr => cr.category === cat.category);
                return (
                  <button key={i} onClick={() => toggleCat(cat.category)}
                    className={`rounded-lg p-2 border text-left transition-all hover:shadow-sm ${
                      expandedCats[cat.category] ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-200' : 'border-blue-100 bg-white'
                    }`}>
                    <div className="text-[10px] text-gray-500 truncate">{cat.category}</div>
                    <div className="text-sm font-bold text-gray-800">{cat.count}</div>
                    {catResult && (
                      <div className="flex items-center gap-1 mt-0.5">
                        {patterns.method === 'llm' ? (
                          <span className="text-[9px] text-violet-500 font-bold">{catResult.optimal_k}개 그룹</span>
                        ) : (
                          <>
                            <span className="text-[9px] text-gray-400">K={catResult.optimal_k}</span>
                            <span className={`text-[9px] font-bold ${silhouetteColor(catResult.silhouette)}`}>
                              {catResult.silhouette?.toFixed(2)}
                            </span>
                          </>
                        )}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {/* 카테고리별 클러스터 상세 (아코디언) */}
          {(patterns.category_results || []).map((cr, ci) => (
            <div key={ci} className={`mb-2 rounded-xl border transition-all ${
              expandedCats[cr.category] ? 'border-blue-300 bg-white' : 'border-transparent'
            }`}>
              <button onClick={() => toggleCat(cr.category)}
                className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-blue-50/50 rounded-xl transition-colors">
                {expandedCats[cr.category]
                  ? <ChevronDown size={14} className="text-blue-500" />
                  : <ChevronRight size={14} className="text-gray-400" />}
                <span className="text-xs font-bold text-gray-800">{cr.category}</span>
                <span className="text-[10px] text-gray-500">{cr.count}건</span>
                {patterns.method === 'llm' ? (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-600 font-semibold">
                    {cr.optimal_k}개 그룹
                  </span>
                ) : (
                  <>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-600 font-semibold">
                      K={cr.optimal_k}
                    </span>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-gray-400">실루엣:</span>
                      <span className={`text-[10px] font-bold ${silhouetteColor(cr.silhouette)}`}>
                        {cr.silhouette?.toFixed(3)}
                      </span>
                      <div className="w-12 h-1.5 rounded-full bg-gray-200 overflow-hidden">
                        <div className={`h-full rounded-full ${silhouetteBg(cr.silhouette)}`}
                          style={{ width: `${Math.max(cr.silhouette * 100, 5)}%` }} />
                      </div>
                    </div>
                  </>
                )}
              </button>

              {expandedCats[cr.category] && (
                <div className="px-3 pb-3 space-y-3">
                  {/* K-Means 차트 영역 */}
                  {patterns.method !== 'llm' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {/* 실루엣 점수 바 차트 */}
                      {cr.scores && cr.scores.length > 1 && (
                        <div className="rounded-lg border border-gray-100 bg-white p-3">
                          <h5 className="text-[11px] font-bold text-gray-600 mb-2">실루엣 점수 (K별)</h5>
                          <ResponsiveContainer width="100%" height={140}>
                            <BarChart data={cr.scores} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                              <XAxis dataKey="k" tick={{ fontSize: 10 }} tickFormatter={(v) => `K=${v}`} />
                              <YAxis tick={{ fontSize: 10 }} domain={[0, 'auto']} />
                              <Tooltip formatter={(v) => v.toFixed(4)} labelFormatter={(v) => `K=${v}`} />
                              <Bar dataKey="silhouette" radius={[4, 4, 0, 0]}>
                                {cr.scores.map((s, i) => (
                                  <Cell key={i} fill={s.k === cr.optimal_k ? '#3b82f6' : '#d1d5db'} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      )}

                      {/* 군집 산점도 */}
                      {cr.points && cr.points.length > 0 && (
                        <div className="rounded-lg border border-gray-100 bg-white p-3">
                          <h5 className="text-[11px] font-bold text-gray-600 mb-2">군집 분포 (PCA 2D)</h5>
                          <ResponsiveContainer width="100%" height={140}>
                            <ScatterChart margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                              <XAxis dataKey="x" type="number" tick={{ fontSize: 9 }} name="PC1" />
                              <YAxis dataKey="y" type="number" tick={{ fontSize: 9 }} name="PC2" />
                              <Tooltip content={({ payload }) => {
                                if (!payload?.[0]) return null;
                                const d = payload[0].payload;
                                return (
                                  <div className="bg-white border border-gray-200 rounded-lg p-2 shadow-lg max-w-[200px]">
                                    <div className="text-[10px] text-gray-500">클러스터 {d.cluster}{d.isCentroid ? ' (중심)' : ''}</div>
                                    {d.text && <div className="text-[11px] text-gray-800 mt-0.5">{d.text}</div>}
                                  </div>
                                );
                              }} />
                              {/* 문의 포인트 */}
                              <Scatter data={cr.points} shape="circle">
                                {cr.points.map((p, i) => (
                                  <Cell key={i} fill={CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]} fillOpacity={0.6} r={3} />
                                ))}
                              </Scatter>
                              {/* 중심점 */}
                              {cr.centroids && (
                                <Scatter data={cr.centroids.map(c => ({ ...c, isCentroid: true, text: `중심점 ${c.cluster}` }))} shape="diamond">
                                  {cr.centroids.map((c, i) => (
                                    <Cell key={i} fill={CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length]} stroke="#000" strokeWidth={1} r={6} />
                                  ))}
                                </Scatter>
                              )}
                            </ScatterChart>
                          </ResponsiveContainer>
                          <div className="flex flex-wrap gap-2 mt-1.5">
                            {(cr.clusters || []).map((cl, i) => (
                              <span key={i} className="flex items-center gap-1 text-[9px] text-gray-500">
                                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: CLUSTER_COLORS[cl.cluster_id % CLUSTER_COLORS.length] }} />
                                C{cl.cluster_id} ({cl.size}건)
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 클러스터 목록 */}
                  {(cr.clusters || []).map((cl, cli) => (
                    <div key={cli} className="rounded-lg border border-gray-100 bg-gray-50/50 p-2.5">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className="text-[10px] w-5 h-5 rounded-full flex items-center justify-center text-white font-bold"
                          style={{ backgroundColor: CLUSTER_COLORS[cl.cluster_id % CLUSTER_COLORS.length] }}>
                          {cl.cluster_id}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700 font-bold">
                          {cl.size}건
                        </span>
                        {cl.label && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-50 text-violet-600 font-semibold">
                            {cl.label}
                          </span>
                        )}
                        <span className="text-xs font-semibold text-gray-800">{cl.representative}</span>
                      </div>
                      {cl.samples && cl.samples.length > 1 && (
                        <div className="flex flex-wrap gap-1 mt-1 ml-7">
                          {cl.samples.filter(s => s !== cl.representative).slice(0, 3).map((s, si) => (
                            <span key={si} className="text-[10px] bg-white text-gray-500 px-1.5 py-0.5 rounded border border-gray-100">
                              {s.length > 45 ? s.slice(0, 45) + '...' : s}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {faqs.length > 0 && <FaqList
        faqs={faqs} faqSearch={faqSearch} setFaqSearch={setFaqSearch}
        faqStatusFilter={faqStatusFilter} setFaqStatusFilter={setFaqStatusFilter}
        faqCategoryFilter={faqCategoryFilter} setFaqCategoryFilter={setFaqCategoryFilter}
        approveFaq={approveFaq} startEdit={startEdit} deleteFaq={deleteFaq}
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
