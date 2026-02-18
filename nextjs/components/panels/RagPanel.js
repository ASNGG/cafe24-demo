// components/panels/RagPanel.js
import { useCallback, useEffect, useState } from 'react';
import { Upload, FileText, Trash2, RefreshCw, CheckCircle, XCircle, AlertCircle, Image, ScanText, Zap, GitBranch, Search, Sparkles } from 'lucide-react';
import toast from 'react-hot-toast';
import SectionHeader from '../SectionHeader';

export default function RagPanel({ auth, apiCall, addLog, settings, setSettings }) {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lightragStatus, setLightragStatus] = useState(null);
  const [showOcrTooltip, setShowOcrTooltip] = useState(false);
  const [showDeleteTooltip, setShowDeleteTooltip] = useState(false);

  // M48: RAG 상태 + LightRAG 상태 병렬 로드
  const loadStatus = useCallback(async () => {
    if (!auth) return;
    setLoading(true);

    try {
      const [res, lightragRes] = await Promise.all([
        apiCall({ endpoint: '/api/rag/status', method: 'GET', auth }),
        apiCall({ endpoint: '/api/lightrag/status', method: 'GET', auth }),
      ]);

      if (res?.status === 'success') {
        setStatus(res);
      }

      if (lightragRes?.status === 'success') {
        setLightragStatus(lightragRes);
      }
    } catch (e) {
      console.error('RAG 상태 로드 실패:', e);
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth]);

  // 파일 목록 로드
  const loadFiles = useCallback(async () => {
    if (!auth) return;

    try {
      const res = await apiCall({
        endpoint: '/api/rag/files',
        method: 'GET',
        auth,
      });

      if (res?.status === 'success' && Array.isArray(res.files)) {
        setFiles(res.files);
      }
    } catch (e) {
      console.error('파일 목록 로드 실패:', e);
    }
  }, [apiCall, auth]);

  // 초기 로드
  useEffect(() => {
    loadStatus();
    loadFiles();
  }, [loadStatus, loadFiles]);

  // 인덱스 재빌드
  const handleReindex = useCallback(async () => {
    if (!auth) return;

    addLog?.('RAG 인덱스 재빌드', '');
    setLoading(true);

    try {
      const res = await apiCall({
        endpoint: '/api/rag/reload',
        method: 'POST',
        auth,
        data: { force: true },
        timeoutMs: 300000,
      });

      if (res?.status === 'success') {
        toast.success('인덱스가 재빌드되었습니다.');
        await loadStatus();
      } else {
        toast.error(`재빌드 실패: ${res.message || '알 수 없는 오류'}`);
      }
    } catch (e) {
      toast.error(`재빌드 실패: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [apiCall, auth, addLog, loadStatus]);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  const formatDate = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="space-y-4">
      <SectionHeader title="RAG 문서 관리" subtitle="PDF 및 문서 업로드/관리" />

      {/* RAG 상태 */}
      <div className="rounded-3xl border border-cafe24-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-black text-cafe24-brown">RAG 시스템 상태</h3>
          <button
            onClick={loadStatus}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-2xl border border-cafe24-brown/10 bg-white px-3 py-2 text-xs font-black text-cafe24-brown/80 hover:bg-cafe24-beige disabled:opacity-50"
            type="button"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            새로고침
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="rounded-2xl border border-cafe24-brown/10 bg-cafe24-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              {status?.rag_ready ? (
                <CheckCircle size={18} className="text-green-600" />
              ) : (
                <XCircle size={18} className="text-red-600" />
              )}
              <span className="text-xs font-black text-cafe24-brown/70">인덱스 상태</span>
            </div>
            <div className="text-lg font-black text-cafe24-brown">
              {status?.rag_ready ? '준비됨' : '비활성'}
            </div>
          </div>

          <div className="rounded-2xl border border-cafe24-brown/10 bg-cafe24-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={18} className="text-blue-600" />
              <span className="text-xs font-black text-cafe24-brown/70">문서 수</span>
            </div>
            <div className="text-lg font-black text-cafe24-brown">
              {status?.files_count || 0}
            </div>
          </div>

          <div className="rounded-2xl border border-cafe24-brown/10 bg-cafe24-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={18} className="text-purple-600" />
              <span className="text-xs font-black text-cafe24-brown/70">청크 수</span>
            </div>
            <div className="text-lg font-black text-cafe24-brown">
              {status?.chunks_count || 0}
            </div>
          </div>

          <div className="rounded-2xl border border-cafe24-brown/10 bg-cafe24-beige/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={18} className="text-amber-600" />
              <span className="text-xs font-black text-cafe24-brown/70">임베딩 모델</span>
            </div>
            <div className="text-sm font-bold text-cafe24-brown">
              {status?.embed_model || '-'}
            </div>
          </div>
        </div>

        {/* Advanced RAG Features */}
        <div className="mt-4 rounded-2xl border border-indigo-200 bg-indigo-50/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap size={16} className="text-indigo-600" />
            <span className="text-xs font-black text-indigo-800">RAG 기능 동작 여부 (인덱스 재빌드 시 활성화)</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {/* Contextual Retrieval - 미적용 */}
            <div className="flex items-center gap-2 opacity-50">
              <FileText size={14} className="text-cafe24-brown/50" />
              <div>
                <div className="text-xs font-black text-cafe24-brown/80">Contextual Retrieval</div>
                <div className="text-[10px] text-cafe24-brown/50">미적용</div>
              </div>
            </div>

            {/* Hybrid Search (BM25 + Vector) */}
            <div className="flex items-center gap-2">
              <Search size={14} className={status?.bm25_ready ? 'text-green-600' : 'text-cafe24-brown/50'} />
              <div>
                <div className="text-xs font-black text-cafe24-brown/80">Hybrid Search</div>
                <div className="text-[10px] text-cafe24-brown/60">
                  {status?.bm25_available ? (
                    status?.bm25_ready ? (
                      <span className="text-green-600">BM25 + Vector ✓</span>
                    ) : (
                      <span className="text-amber-600">BM25 대기중</span>
                    )
                  ) : (
                    <span className="text-cafe24-brown/50">미설치</span>
                  )}
                </div>
              </div>
            </div>

            {/* Reranking - 비활성화 */}
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-cafe24-brown/50" />
              <div>
                <div className="text-xs font-black text-cafe24-brown/80">Reranking</div>
                <div className="text-[10px] text-cafe24-brown/60">
                  <span className="text-cafe24-brown/50">비활성</span>
                </div>
              </div>
            </div>

            {/* Simple Knowledge Graph - 비활성화 */}
            <div className="flex items-center gap-2">
              <GitBranch size={14} className="text-cafe24-brown/50" />
              <div>
                <div className="text-xs font-black text-cafe24-brown/80">Simple KG</div>
                <div className="text-[10px] text-cafe24-brown/60">
                  <span className="text-cafe24-brown/50">비활성</span>
                </div>
              </div>
            </div>

          </div>
        </div>

        {status?.error && (
          <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-3 text-xs font-semibold text-red-800">
            <strong>오류:</strong> {status.error}
          </div>
        )}

        {/* 인덱스 재빌드 버튼 (비활성화 - LLM API 비용 방지) */}
        {auth?.user_role === '관리자' && (
          <div className="mt-4">
            <button
              disabled
              className="w-full rounded-2xl border border-cafe24-brown/20 bg-cafe24-brown/10 px-4 py-2.5 text-sm font-black text-cafe24-brown/40 cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
              title="프로덕션 환경에서 비활성화됨 (LLM API 비용 발생)"
            >
              <RefreshCw size={16} />
              인덱스 재빌드 (비활성화)
            </button>
          </div>
        )}
      </div>

      {/* RAG 모드 선택 */}
      <div className="rounded-3xl border border-cafe24-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center gap-2 mb-4">
          <Search size={18} className="text-indigo-600" />
          <h3 className="text-sm font-black text-cafe24-brown">AI 에이전트 RAG 검색 모드</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          {/* 기본 RAG */}
          <button
            onClick={() => setSettings?.({ ...settings, ragMode: 'rag' })}
            className={`rounded-2xl border-2 p-4 text-left transition ${
              settings?.ragMode === 'rag'
                ? 'border-blue-500 bg-blue-50'
                : 'border-cafe24-brown/10 bg-cafe24-beige/30 hover:bg-cafe24-beige/50'
            }`}
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Search size={16} className={settings?.ragMode === 'rag' ? 'text-blue-600' : 'text-cafe24-brown/60'} />
              <span className={`text-sm font-black ${settings?.ragMode === 'rag' ? 'text-blue-700' : 'text-cafe24-brown'}`}>
                RAG
              </span>
              {settings?.ragMode === 'rag' && (
                <CheckCircle size={14} className="text-blue-600 ml-auto" />
              )}
            </div>
            <div className="text-[11px] text-cafe24-brown/70 leading-relaxed">
              FAISS + BM25
              <br />
              <span className="text-cafe24-brown/50">싱글홉 질문에 최적</span>
            </div>
          </button>

          {/* LightRAG (시험용 - 비활성화) */}
          <button
            disabled
            className="rounded-2xl border-2 p-4 text-left transition border-cafe24-brown/10 bg-gray-100 opacity-50 cursor-not-allowed"
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm font-black text-gray-400">
                LightRAG
              </span>
              <span className="text-[9px] bg-yellow-200 text-yellow-700 px-1.5 py-0.5 rounded-full ml-auto font-bold">
                시험용
              </span>
            </div>
            <div className="text-[11px] text-gray-400 leading-relaxed">
              지식 그래프 기반
              <br />
              <span className="text-gray-300">멀티홉 질문에 최적 (준비중)</span>
            </div>
          </button>

          {/* K²RAG (시험중 - 비활성화) */}
          <button
            disabled
            className="rounded-2xl border-2 p-4 text-left transition border-cafe24-brown/10 bg-gray-100 opacity-50 cursor-not-allowed"
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={16} className="text-gray-400" />
              <span className="text-sm font-black text-gray-400">
                K²RAG
              </span>
              <span className="text-[9px] bg-yellow-200 text-yellow-700 px-1.5 py-0.5 rounded-full ml-auto font-bold">
                시험중
              </span>
            </div>
            <div className="text-[11px] text-gray-400 leading-relaxed">
              KG + Sub-Q + Hybrid
              <br />
              <span className="text-gray-300">고정밀 검색 (준비중)</span>
            </div>
          </button>

          {/* Auto (비활성화) */}
          <button
            disabled
            className="rounded-2xl border-2 p-4 text-left transition border-cafe24-brown/10 bg-gray-100 opacity-50 cursor-not-allowed"
            type="button"
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap size={16} className="text-gray-400" />
              <span className="text-sm font-black text-gray-400">
                자동 선택
              </span>
              <span className="text-[9px] bg-gray-200 text-gray-500 px-1.5 py-0.5 rounded-full ml-auto font-bold">
                비활성
              </span>
            </div>
            <div className="text-[11px] text-gray-400 leading-relaxed">
              AI가 질문에 맞게 선택
              <br />
              <span className="text-gray-300">두 방식 모두 사용 가능 (준비중)</span>
            </div>
          </button>
        </div>

        <div className="mt-3 text-[11px] text-cafe24-brown/60 text-center">
          선택한 모드는 AI 에이전트가 플랫폼 관련 질문에 답할 때 사용됩니다
        </div>
      </div>

      {/* 파일 업로드 - 비활성화 */}
      {auth?.user_role === '관리자' && (
        <div className="rounded-3xl border border-cafe24-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur opacity-60">
          <h3 className="text-sm font-black text-cafe24-brown mb-4">문서 업로드</h3>

          <div className="space-y-3">
            <div className="rounded-2xl border-2 border-dashed border-cafe24-brown/20 bg-gray-50 p-6 text-center">
              <Upload size={32} className="mx-auto mb-3 text-cafe24-brown/30" />
              <span className="inline-flex items-center gap-2 rounded-2xl border border-cafe24-brown/20 bg-white px-4 py-2 text-sm font-black text-cafe24-brown/40 cursor-not-allowed">
                <Upload size={16} />
                파일 선택 (비활성화)
              </span>
            </div>

            <button
              disabled
              className="w-full rounded-2xl border border-cafe24-brown/20 bg-gray-200 px-4 py-3 text-sm font-black text-cafe24-brown/40 cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <Upload size={16} />
              문서 업로드
            </button>

            <div className="text-xs font-semibold text-cafe24-brown/60">
              지원 형식: PDF, TXT, MD, JSON, CSV, LOG (파일당 최대 15MB)
            </div>
          </div>
        </div>
      )}

      {/* OCR 업로드 - 비활성화됨 */}
      <div className="rounded-3xl border border-cafe24-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur opacity-60">
        <div className="flex items-center gap-2 mb-4">
          <ScanText size={18} className="text-purple-400" />
          <h3 className="text-sm font-black text-cafe24-brown">OCR 이미지 업로드</h3>
        </div>

        <div className="space-y-3">
          <div className="rounded-2xl border-2 border-dashed border-purple-200 bg-purple-50/30 p-6 text-center">
            <Image size={32} className="mx-auto mb-3 text-purple-300" />
            <div className="inline-flex items-center gap-2 rounded-2xl border border-purple-200 bg-white/50 px-4 py-2 text-sm font-black text-purple-400 cursor-not-allowed">
              <Image size={16} />
              이미지 선택 (비활성화됨)
            </div>
          </div>

          <div
            className="relative"
            onMouseEnter={() => setShowOcrTooltip(true)}
            onMouseLeave={() => setShowOcrTooltip(false)}
          >
            <button
              disabled={true}
              className="w-full rounded-2xl border border-purple-300 bg-purple-100 px-4 py-3 text-sm font-black text-purple-400 cursor-not-allowed flex items-center justify-center gap-2"
              type="button"
            >
              <ScanText size={16} />
              OCR 추출 (비활성화됨)
            </button>

            {/* 커스텀 툴팁 */}
            <div
              className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
                showOcrTooltip
                  ? 'opacity-100 visible translate-y-0'
                  : 'opacity-0 invisible translate-y-1'
              }`}
            >
              <div className="bg-cafe24-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle size={14} className="text-amber-400" />
                  <span className="font-bold text-amber-400">기능 비활성화됨</span>
                </div>
                <p className="text-cafe24-beige leading-relaxed">
                  OCR 처리 후 RAG 저장 시 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                  현재 프로덕션 환경에서는 비활성화되어 있습니다.
                </p>
                <div className="mt-2 pt-2 border-t border-cafe24-brown/30 text-cafe24-brown/50 text-[10px]">
                  활성화가 필요하면 관리자에게 문의하세요
                </div>
              </div>
              {/* 툴팁 화살표 */}
              <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 bg-cafe24-brown rotate-45" />
            </div>
          </div>

          <div className="text-xs font-semibold text-cafe24-brown/60">
            지원 형식: JPG, PNG, BMP, TIFF, GIF, WEBP (최대 20MB) • 한국어/영어 지원
          </div>
        </div>
      </div>

      {/* 파일 목록 */}
      <div className="rounded-3xl border border-cafe24-brown/10 bg-white/70 p-5 shadow-sm backdrop-blur">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-black text-cafe24-brown">업로드된 문서</h3>
          <div className="flex items-center gap-2">
            {auth?.user_role === '관리자' && files.length > 0 && (
              <div
                className="relative"
                onMouseEnter={() => setShowDeleteTooltip(true)}
                onMouseLeave={() => setShowDeleteTooltip(false)}
              >
                <button
                  disabled={true}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-cafe24-brown/10 bg-white/50 px-2.5 py-1.5 text-xs font-black text-cafe24-brown/40 cursor-not-allowed"
                  type="button"
                >
                  <Trash2 size={14} />
                  삭제 (비활성화됨)
                </button>

                {/* 커스텀 툴팁 */}
                <div
                  className={`absolute bottom-full right-0 mb-2 w-64 z-50 transition-all duration-200 pointer-events-none ${
                    showDeleteTooltip
                      ? 'opacity-100 visible translate-y-0'
                      : 'opacity-0 invisible translate-y-1'
                  }`}
                >
                  <div className="bg-cafe24-brown text-white text-xs rounded-xl px-4 py-3 shadow-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle size={14} className="text-amber-400" />
                      <span className="font-bold text-amber-400">기능 비활성화됨</span>
                    </div>
                    <p className="text-cafe24-beige leading-relaxed">
                      문서 삭제 후 인덱스 재빌드 시 <span className="text-white font-semibold">LLM API 비용이 발생</span>합니다.
                      현재 프로덕션 환경에서는 비활성화되어 있습니다.
                    </p>
                    <div className="mt-2 pt-2 border-t border-cafe24-brown/30 text-cafe24-brown/50 text-[10px]">
                      활성화가 필요하면 관리자에게 문의하세요
                    </div>
                  </div>
                  {/* 툴팁 화살표 */}
                  <div className="absolute right-6 -bottom-1 w-2 h-2 bg-cafe24-brown rotate-45" />
                </div>
              </div>
            )}
            <span className="text-xs font-black text-cafe24-brown/60">{files.length}개</span>
          </div>
        </div>

        {files.length === 0 ? (
          <div className="rounded-2xl border border-cafe24-brown/10 bg-cafe24-beige/50 p-6 text-center text-sm font-semibold text-cafe24-brown/60">
            업로드된 문서가 없습니다
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((file) => (
              <div
                key={file.filename}
                className="flex items-center justify-between gap-3 rounded-2xl border border-cafe24-brown/10 bg-white p-3"
              >
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <FileText size={20} className="text-blue-600 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-bold text-cafe24-brown truncate">{file.filename}</div>
                    <div className="text-xs font-semibold text-cafe24-brown/60">
                      {formatBytes(file.size)} • {formatDate(file.modified)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
