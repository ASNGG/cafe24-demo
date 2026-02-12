// components/panels/SettingsPanel.js
// LLM 설정 및 시스템 프롬프트는 백엔드에서 중앙 관리됩니다.
import { useEffect, useMemo, useState, useCallback } from "react";

const clampNumber = (v, min, max, fallback) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  if (typeof min === "number" && n < min) return min;
  if (typeof max === "number" && n > max) return max;
  return n;
};

// 프리셋 설정
const LLM_PRESETS = {
  precise: {
    name: "정확한 응답",
    description: "분석/데이터 작업에 적합",
    icon: "🎯",
    temperature: 0.1,
    topP: 0.9,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0,
  },
  balanced: {
    name: "균형잡힌",
    description: "일반적인 대화에 적합",
    icon: "⚖️",
    temperature: 0.5,
    topP: 1.0,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0,
  },
  creative: {
    name: "창의적",
    description: "아이디어/스토리텔링에 적합",
    icon: "✨",
    temperature: 0.9,
    topP: 1.0,
    presencePenalty: 0.3,
    frequencyPenalty: 0.2,
  },
};

// 슬라이더 컴포넌트
function Slider({ value, onChange, min, max, step, label, disabled, showValue = true }) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm text-cafe24-brown/70">{label}</label>
        {showValue && (
          <span className="text-sm font-mono text-cafe24-brown/80">{value.toFixed(step < 1 ? 2 : 0)}</span>
        )}
      </div>
      <div className="relative h-2 bg-cafe24-cream rounded-full overflow-hidden">
        <div
          className="absolute h-full bg-gradient-to-r from-cafe24-orange to-cafe24-pink transition-all duration-150"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        style={{ top: '20px', height: '8px' }}
      />
    </div>
  );
}

export default function SettingsPanel({ settings, setSettings, addLog, apiCall, auth }) {
  // ✅ GPT-4 계열 중심 + 필요시 확장
  const models = useMemo(
    () => [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4.1",
      "gpt-4.1-mini",
      "gpt-4-turbo",
    ],
    []
  );

  // 프롬프트 관련 상태
  const [loadingDefault, setLoadingDefault] = useState(false);
  const [draftPrompt, setDraftPrompt] = useState(settings?.systemPrompt || "");



  // LLM 설정 관련 상태
  const [llmSaved, setLlmSaved] = useState(true);
  const [isCustomLLM, setIsCustomLLM] = useState(false);

  // LLM 설정 임시 상태 (저장 버튼 누르기 전까지 여기에만 저장)
  const [draftLLM, setDraftLLM] = useState({
    selectedModel: settings?.selectedModel || "gpt-4o-mini",
    customModel: settings?.customModel || "",
    temperature: settings?.temperature ?? 0.3,
    topP: settings?.topP ?? 1.0,
    presencePenalty: settings?.presencePenalty ?? 0.0,
    frequencyPenalty: settings?.frequencyPenalty ?? 0.0,
    maxTokens: settings?.maxTokens ?? 8000,
    seed: settings?.seed ?? "",
    timeoutMs: settings?.timeoutMs ?? 30000,
    retries: settings?.retries ?? 2,
    stream: settings?.stream ?? true,
    apiKey: settings?.apiKey ?? "",
  });

  // draftLLM 기반 파생 값 (useState 이후에 위치해야 함)
  const selectedModel = (draftLLM?.selectedModel || "gpt-4o-mini").trim();
  const isGpt5 = selectedModel.toLowerCase().startsWith("gpt-5");
  const isMiniModel = selectedModel.toLowerCase().includes("mini");
  const maxTokensLimit = 16000;

  // settings.systemPrompt가 외부에서 변경되면 draftPrompt 동기화
  useEffect(() => {
    setDraftPrompt(settings?.systemPrompt || "");
  }, [settings?.systemPrompt]);

  // 백엔드에서 시스템 프롬프트 로드
  const loadPromptFromBackend = useCallback(async () => {
    if (typeof apiCall !== "function") return;

    setLoadingDefault(true);
    try {
      const res = await apiCall({
        endpoint: "/api/settings/prompt",
        method: "GET",
        auth,
        timeoutMs: 30000,
      });

      const data = res?.data || res || {};
      const prompt = String(data?.systemPrompt || data?.system_prompt || "").trim();

      if (prompt.length > 0) {
        setSettings((s) => ({ ...s, systemPrompt: prompt }));
        setDraftPrompt(prompt);
      }
    } catch (e) {
      console.error("프롬프트 로드 실패:", e);
    } finally {
      setLoadingDefault(false);
    }
  }, [apiCall, auth, setSettings]);

  // M49: 빈 의존성 useEffect 2개 통합 → 단일 초기화
  useEffect(() => {
    setDraftLLM({
      selectedModel: settings?.selectedModel || "gpt-4o-mini",
      customModel: settings?.customModel || "",
      temperature: settings?.temperature ?? 0.3,
      topP: settings?.topP ?? 1.0,
      presencePenalty: settings?.presencePenalty ?? 0.0,
      frequencyPenalty: settings?.frequencyPenalty ?? 0.0,
      maxTokens: settings?.maxTokens ?? 8000,
      seed: settings?.seed ?? "",
      timeoutMs: settings?.timeoutMs ?? 30000,
      retries: settings?.retries ?? 2,
      stream: settings?.stream ?? true,
      apiKey: settings?.apiKey ?? "",
    });
    setLlmSaved(true);
    loadPromptFromBackend();
  }, []);

  // LLM 설정 변경 감지 - draft에만 저장 (저장 버튼 누르기 전)
  const handleLLMSettingChange = useCallback((key, value) => {
    setDraftLLM((d) => {
      const updated = { ...d, [key]: value };

      // 모델 변경 시 maxTokens 자동 조정
      if (key === "selectedModel") {
        const isMini = value.toLowerCase().includes("mini");
        const newLimit = isMini ? 16000 : 4500;
        // 현재 maxTokens가 새 한도를 초과하면 기본값으로 조정
        if (d.maxTokens > newLimit) {
          updated.maxTokens = isMini ? 8000 : 4000;
        }
      }

      return updated;
    });
    setLlmSaved(false); // 변경사항 있음 표시
  }, []);

  // 저장 버튼 클릭 시 실제 settings에 반영
  const saveLLMSettings = useCallback(() => {
    setSettings((s) => ({ ...s, ...draftLLM }));
    setLlmSaved(true);
    if (addLog) addLog("LLM 설정 저장", `모델: ${draftLLM.selectedModel}`);
  }, [draftLLM, setSettings, addLog]);

  return (
    <div>
      <div className="flex items-end justify-between gap-3 mb-3">
        <div>
          <h2 className="text-lg md:text-xl font-semibold text-cafe24-brown">LLM 설정</h2>
          <p className="text-sm text-cafe24-brown/60">모델 파라미터 설정</p>
        </div>
        <span className="badge">Admin</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>모델 파라미터</span>
              {isCustomLLM ? (
                <span className="text-xs px-2 py-0.5 rounded-full bg-cafe24-orange/20 text-cafe24-orange">커스텀</span>
              ) : (
                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-200 text-gray-600">기본값</span>
              )}
            </div>
            {!llmSaved && (
              <span className="text-xs text-cafe24-orange font-medium">변경사항 있음</span>
            )}
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-sm text-cafe24-brown/70">모델</label>
              <select
                className="input mt-1 opacity-60 cursor-not-allowed"
                value={selectedModel}
                disabled
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <div className="mt-2">
                <label className="text-xs text-cafe24-brown/60">모델명 직접 입력(선택) <span className="text-cafe24-brown/40">(비활성)</span></label>
                <input
                  className="input mt-1 opacity-60 cursor-not-allowed"
                  type="text"
                  value={draftLLM?.customModel ?? ""}
                  placeholder="예: gpt-4o (비우면 위 선택값 사용)"
                  disabled
                />
              </div>
            </div>

            {/* 프리셋 버튼 */}
            <div>
              <label className="text-sm text-cafe24-brown/70 mb-2 block">빠른 프리셋</label>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(LLM_PRESETS).map(([key, preset]) => (
                  <button
                    key={key}
                    className="p-2 rounded-lg border-2 border-cafe24-cream hover:border-cafe24-orange/50 bg-white hover:bg-cafe24-cream/30 transition-all text-left group"
                    onClick={() => {
                      setDraftLLM((d) => ({
                        ...d,
                        temperature: preset.temperature,
                        topP: preset.topP,
                        presencePenalty: preset.presencePenalty,
                        frequencyPenalty: preset.frequencyPenalty,
                      }));
                      setLlmSaved(false);
                    }}
                  >
                    <div className="text-lg mb-1">{preset.icon}</div>
                    <div className="text-xs font-medium text-cafe24-brown group-hover:text-cafe24-orange transition-colors">{preset.name}</div>
                    <div className="text-[10px] text-cafe24-brown/50">{preset.description}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <Slider
                  label="Temperature"
                  value={draftLLM?.temperature ?? 0.3}
                  onChange={(v) => handleLLMSettingChange("temperature", v)}
                  min={0}
                  max={2}
                  step={0.1}
                  disabled={isGpt5}
                />
                {isGpt5 ? <div className="text-xs text-cafe24-brown/60 mt-1">gpt-5 계열은 temperature를 사용하지 않습니다.</div> : null}
                <div className="text-xs text-cafe24-brown/50 mt-1">낮을수록 정확, 높을수록 창의적</div>
              </div>

              <div>
                <Slider
                  label="Top P"
                  value={draftLLM?.topP ?? 1}
                  onChange={(v) => handleLLMSettingChange("topP", v)}
                  min={0}
                  max={1}
                  step={0.05}
                />
                <div className="text-xs text-cafe24-brown/50 mt-1">확률 분포 커트라인</div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <Slider
                  label="Presence Penalty"
                  value={draftLLM?.presencePenalty ?? 0}
                  onChange={(v) => handleLLMSettingChange("presencePenalty", v)}
                  min={-2}
                  max={2}
                  step={0.1}
                />
                <div className="text-xs text-cafe24-brown/50 mt-1">새 주제 언급 유도</div>
              </div>

              <div>
                <Slider
                  label="Frequency Penalty"
                  value={draftLLM?.frequencyPenalty ?? 0}
                  onChange={(v) => handleLLMSettingChange("frequencyPenalty", v)}
                  min={-2}
                  max={2}
                  step={0.1}
                />
                <div className="text-xs text-cafe24-brown/50 mt-1">반복 표현 억제</div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-sm text-cafe24-brown/70">Max Tokens (8000 고정)</label>
                <input
                  className="input mt-1 opacity-60 cursor-not-allowed"
                  type="number"
                  value={8000}
                  disabled
                />
              </div>

              <div>
                <label className="text-sm text-cafe24-brown/70">Seed (선택)</label>
                <input
                  className="input mt-1"
                  type="number"
                  step="1"
                  min="0"
                  value={draftLLM?.seed ?? ""}
                  placeholder="비우면 미사용"
                  onChange={(e) => handleLLMSettingChange("seed", e.target.value === "" ? "" : clampNumber(e.target.value, 0, 2147483647, 0))}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-sm text-cafe24-brown/70">요청 Timeout(ms)</label>
                <input
                  className="input mt-1"
                  type="number"
                  step="1000"
                  min="1000"
                  max="120000"
                  value={draftLLM?.timeoutMs ?? 30000}
                  onChange={(e) => handleLLMSettingChange("timeoutMs", clampNumber(e.target.value, 1000, 120000, 30000))}
                />
              </div>

              <div>
                <label className="text-sm text-cafe24-brown/70">Retry 횟수</label>
                <input
                  className="input mt-1"
                  type="number"
                  step="1"
                  min="0"
                  max="10"
                  value={draftLLM?.retries ?? 2}
                  onChange={(e) => handleLLMSettingChange("retries", clampNumber(e.target.value, 0, 10, 2))}
                />
              </div>
            </div>

            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm text-cafe24-brown/70">스트리밍 사용</div>
                <div className="text-xs text-cafe24-brown/60">UI에서 /api/agent/stream 사용 여부 플래그</div>
              </div>
              <input
                type="checkbox"
                className="toggle"
                checked={Boolean(draftLLM?.stream ?? true)}
                onChange={(e) => handleLLMSettingChange("stream", e.target.checked)}
              />
            </div>

            <div>
              <label className="text-sm text-cafe24-brown/70">OpenAI API Key (로컬 전용)</label>
              <input
                className="input mt-1"
                type="password"
                value={draftLLM?.apiKey ?? ""}
                onChange={(e) => handleLLMSettingChange("apiKey", e.target.value)}
              />
              <div className="text-xs text-cafe24-brown/50 mt-1">API Key는 보안상 로컬에만 저장됩니다.</div>
            </div>

            <div className="flex gap-2">
              <button
                className={`flex-1 ${llmSaved ? 'btn-secondary' : 'btn-primary'}`}
                onClick={saveLLMSettings}
                disabled={llmSaved}
              >
                {llmSaved ? '저장됨' : '설정 저장'}
              </button>
            </div>
            <p className="text-xs text-cafe24-brown/50">
              * 저장 버튼을 눌러야 설정이 브라우저에 저장됩니다.
              <br />* AI 에이전트 호출 시 저장된 설정값이 적용됩니다.
            </p>
          </div>
        </div>

        <div className="card">
          <div className="card-header flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>시스템 프롬프트</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700">백엔드 관리</span>
              {loadingDefault && (
                <span className="text-xs text-cafe24-orange">로딩 중...</span>
              )}
            </div>
          </div>
          <textarea
            className="input cursor-not-allowed opacity-80"
            style={{ height: 280 }}
            value={draftPrompt}
            placeholder="백엔드에서 시스템 프롬프트를 로드합니다..."
            disabled
          />
          <p className="text-xs text-cafe24-brown/50 mt-2">
            * 시스템 프롬프트는 백엔드에서 중앙 관리됩니다. 읽기 전용으로 표시됩니다.
          </p>
        </div>
      </div>
    </div>
  );
}
