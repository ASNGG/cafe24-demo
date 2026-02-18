// components/panels/ModelsPanel.js
// CAFE24 AI Platform - ML 모델 관리 패널

import { useEffect, useState, useCallback } from 'react';
import { Brain, Layers, FlaskConical, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';

export default function ModelsPanel({ auth, apiCall }) {
  const [mlflowData, setMlflowData] = useState([]);
  const [registeredModels, setRegisteredModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selecting, setSelecting] = useState(null);
  const [selectedModels, setSelectedModels] = useState({}); // 모델별 선택 상태 { modelName: version }
  const [message, setMessage] = useState(null);
  const [usingSample, setUsingSample] = useState(false);

  // H27: fetchMLflowData/handleRefresh 공통 함수 추출
  const CAFE24_KEYWORDS = ['cafe24', 'ops-ai', '이탈', '셀러', '매출', '이상', 'CS', '정산'];

  // useCallback으로 stale closure 방지 (auth, apiCall 의존성 명시)
  const fetchMLflowData = useCallback(async (reset = false) => {
    if (reset) {
      setMlflowData([]);
      setRegisteredModels([]);
    }
    setLoading(true);
    let gotRealData = false;

    try {
      const expRes = await apiCall({
        endpoint: '/api/mlflow/experiments',
        auth,
        timeoutMs: 10000,
      });

      if (expRes?.status === 'success' && expRes.data?.length > 0) {
        const cafe24Exps = expRes.data.filter(exp => {
          const name = exp.name.toLowerCase();
          return CAFE24_KEYWORDS.some(kw => name.includes(kw.toLowerCase()));
        });
        if (cafe24Exps.length > 0) {
          setMlflowData(cafe24Exps);
          gotRealData = true;
        }
      }
    } catch (e) {
      console.log('MLflow 실험 API fallback');
    }

    try {
      const modelsRes = await apiCall({
        endpoint: '/api/mlflow/models',
        auth,
        timeoutMs: 10000,
      });

      if (modelsRes?.status === 'success' && modelsRes.data?.length > 0) {
        setRegisteredModels(modelsRes.data);
        gotRealData = true;
      }
    } catch (e) {
      console.log('MLflow 모델 API fallback');
    }

    if (!gotRealData) {
      setMlflowData([]);
      setRegisteredModels([]);
      setUsingSample(true);
    } else {
      setUsingSample(false);
    }

    setLoading(false);
  }, [apiCall, auth]);

  // M47: 두 useEffect waterfall -> 단일 useEffect + Promise.all 병렬 로드
  useEffect(() => {
    if (!auth) return;
    async function initLoad() {
      const [selectedRes] = await Promise.all([
        apiCall({ endpoint: '/api/mlflow/models/selected', auth, timeoutMs: 5000 }).catch(() => null),
        fetchMLflowData(),
      ]);
      if (selectedRes?.status === 'success' && selectedRes.data) {
        setSelectedModels(selectedRes.data);
      }
    }
    initLoad();
  }, [auth, apiCall, fetchMLflowData]);

  const formatTimestamp = (ts) => {
    if (!ts) return '-';
    const date = new Date(ts);
    return date.toLocaleString('ko-KR');
  };

  const handleSelectModel = async (modelName, version) => {
    const modelKey = `${modelName}-${version}`;
    setSelecting(modelKey);
    setMessage(null);

    try {
      const res = await apiCall({
        endpoint: '/api/mlflow/models/select',
        auth,
        method: 'POST',
        data: { model_name: modelName, version: String(version) },
        timeoutMs: 30000,
      });

      setSelecting(null);

      if (res?.status === 'success') {
        setSelectedModels(prev => ({ ...prev, [modelName]: version }));
        setMessage({ type: 'success', text: res.message || `${modelName} v${version} 모델이 로드되었습니다` });
      } else {
        setMessage({ type: 'error', text: res?.message || '모델 로드 실패' });
      }
    } catch (e) {
      setSelecting(null);
      setMessage({ type: 'error', text: `${modelName} 모델 선택 실패` });
    }

    setTimeout(() => setMessage(null), 5000);
  };

  const handleRefresh = () => fetchMLflowData(true);

  return (
    <div>
      <SectionHeader
        title="CAFE24 ML 모델 관리"
        subtitle="이탈예측 · 이상탐지 · CS분류 · 매출예측 · 감성분석"
        right={
          <div className="flex items-center gap-2">
            <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
              usingSample
                ? 'border-amber-400/50 bg-amber-50 text-amber-700'
                : 'border-green-400/50 bg-green-50 text-green-700'
            }`}>
              {usingSample ? 'SAMPLE' : 'LIVE'}
            </span>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="rounded-full border-2 border-cafe24-orange/20 bg-white/80 p-1.5 hover:bg-cafe24-beige transition disabled:opacity-50"
            >
              <RefreshCw size={14} className={`text-cafe24-brown ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        }
      />

      {message && (
        <div className={`mb-4 p-3 rounded-2xl flex items-center gap-2 text-sm ${
          message.type === 'success'
            ? 'bg-green-50 border-2 border-green-200 text-green-700'
            : 'bg-red-50 border-2 border-red-200 text-red-700'
        }`}>
          {message.type === 'success' ? <CheckCircle size={16} /> : <XCircle size={16} />}
          {message.text}
        </div>
      )}

      {/* Model Registry Section */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Layers size={18} className="text-cafe24-orange" />
          <h3 className="text-sm font-black text-cafe24-brown">Model Registry</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {registeredModels.map((model) => (
            <div key={model.name} className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Brain size={16} className="text-cafe24-orange" />
                  <span className="font-bold text-cafe24-brown text-sm">{model.name}</span>
                </div>
                {model.model_type === 'artifact' && (
                  <span className="text-[10px] px-2 py-0.5 bg-purple-100 text-purple-700 rounded-full font-bold">
                    Artifact
                  </span>
                )}
              </div>
              <p className="text-xs text-cafe24-brown/60 mb-3">
                {model.description || '설명 없음'}
              </p>
              <div className="space-y-2">
                {model.versions.map((v) => {
                  const modelKey = `${model.name}-${v.version}`;
                  const isSelected = String(selectedModels[model.name]) === String(v.version);
                  const isSelecting = selecting === modelKey;

                  return (
                    <div key={v.version} className={`flex items-center justify-between p-2.5 rounded-xl transition ${
                      isSelected
                        ? 'bg-gradient-to-r from-cafe24-yellow/30 to-cafe24-orange/20 border-2 border-cafe24-orange'
                        : 'bg-cafe24-beige/50'
                    }`}>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-cafe24-brown">v{v.version}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                          v.stage === 'Production' ? 'bg-green-100 text-green-700' :
                          v.stage === 'Staging' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-500'
                        }`}>
                          {v.stage || 'None'}
                        </span>
                        {isSelected && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-cafe24-orange text-white">
                            ✓ 사용중
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => handleSelectModel(model.name, v.version)}
                        disabled={isSelecting || isSelected}
                        className={`text-xs px-3 py-1.5 rounded-lg font-bold shadow transition ${
                          isSelected
                            ? 'bg-gray-200 text-gray-500 cursor-default shadow-none'
                            : 'bg-gradient-to-r from-cafe24-yellow to-cafe24-orange text-white hover:shadow-md'
                        } disabled:opacity-50`}
                      >
                        {isSelecting ? '로딩...' : isSelected ? '선택됨' : '선택'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Experiments Section */}
      <div className="flex items-center gap-2 mb-4">
        <FlaskConical size={18} className="text-cafe24-orange" />
        <h3 className="text-sm font-black text-cafe24-brown">실험 기록</h3>
      </div>

      {mlflowData.length ? mlflowData.map((exp) => (
        <div key={exp.experiment_id} className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur mb-4">
          <div className="flex justify-between items-center mb-4">
            <span className="font-bold text-cafe24-brown">{exp.name}</span>
            <span className={`text-[10px] px-2 py-1 rounded-full font-bold ${
              exp.lifecycle_stage === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
            }`}>
              {exp.lifecycle_stage}
            </span>
          </div>

          {exp.runs && exp.runs.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cafe24-orange/10">
                    <th className="text-left py-2 px-2 text-cafe24-brown font-bold text-xs">Run Name</th>
                    <th className="text-left py-2 px-2 text-cafe24-brown font-bold text-xs">Status</th>
                    <th className="text-left py-2 px-2 text-cafe24-brown font-bold text-xs">시작 시간</th>
                    <th className="text-left py-2 px-2 text-cafe24-brown font-bold text-xs">Metrics</th>
                    <th className="text-left py-2 px-2 text-cafe24-brown font-bold text-xs">Params</th>
                  </tr>
                </thead>
                <tbody>
                  {exp.runs.map((run) => (
                    <tr key={run.run_id} className="border-b border-cafe24-orange/5 hover:bg-cafe24-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cafe24-brown">
                        {run.run_name || run.run_id.slice(0, 8)}
                      </td>
                      <td className="py-3 px-2">
                        <span className={`text-[10px] px-2 py-1 rounded-full font-bold ${
                          run.status === 'FINISHED' ? 'bg-green-100 text-green-700' :
                          run.status === 'RUNNING' ? 'bg-blue-100 text-blue-700 animate-pulse' :
                          run.status === 'error' ? 'bg-red-100 text-red-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {run.status}
                        </span>
                      </td>
                      <td className="py-3 px-2 text-cafe24-brown/60 text-xs">
                        {formatTimestamp(run.start_time)}
                      </td>
                      <td className="py-3 px-2">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(run.metrics || {}).map(([k, v]) => (
                            <span key={k} className="text-[10px] bg-cafe24-yellow/30 text-cafe24-brown px-2 py-0.5 rounded-full font-semibold">
                              {k}: {typeof v === 'number' ? v.toFixed(4) : v}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="py-3 px-2">
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(run.params || {}).slice(0, 3).map(([k, v]) => (
                            <span key={k} className="text-[10px] bg-cafe24-beige text-cafe24-brown/70 px-2 py-0.5 rounded-full">
                              {k}: {v}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-cafe24-brown/50 py-4 text-center">
              실험 기록이 없습니다.
            </div>
          )}
        </div>
      )) : (
        <div className="rounded-3xl border-2 border-cafe24-orange/20 bg-white/80 p-8 text-center">
          <FlaskConical size={32} className="mx-auto mb-3 text-cafe24-brown/30" />
          <p className="text-sm text-cafe24-brown/60">MLflow 실험이 없습니다.</p>
          <p className="text-xs text-cafe24-brown/40 mt-1">
            노트북을 실행하여 모델을 학습하세요.
          </p>
        </div>
      )}
    </div>
  );
}
