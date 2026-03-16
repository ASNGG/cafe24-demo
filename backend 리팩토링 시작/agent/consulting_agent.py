"""
agent/consulting_agent.py - 셀러 컨설팅 에이전트
================================================
StateGraph 기반 4단계 인터랙티브 컨설팅 워크플로우
  diagnosis → strategy → plan → execute

각 단계 내에서 자유 대화 루프 지원:
  - 초기 진입: 도구 호출 + LLM 분석
  - 대화 모드: 사용자 질문/수정 요청에 응답
  - "다음" 키워드로만 단계 전환

롤백/리셋 지원, SSE 스트리밍, 세션 관리 포함.
"""

import asyncio
import json
import time
import uuid
import pathlib
from typing import TypedDict, Any, Dict, Optional

import yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from core.utils import safe_str, safe_int, safe_float, json_sanitize, format_openai_error
from agent.llm import get_llm, pick_api_key
from agent.tools import (
    tool_analyze_seller,
    tool_predict_seller_churn,
    tool_get_seller_segment,
    tool_optimize_marketing,
    tool_generate_retention_message,
    tool_execute_retention_action,
)
import state as st

# ============================================================
# 프롬프트 로드
# ============================================================
_PROMPTS_PATH = pathlib.Path(__file__).parent / "consulting_prompts.yaml"

def _load_prompts() -> dict:
    try:
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.logger.warning("CONSULTING_PROMPTS_LOAD_FAIL err=%s", e)
        return {}

_PROMPTS: dict = {}

def _get_prompts() -> dict:
    global _PROMPTS
    if not _PROMPTS:
        _PROMPTS = _load_prompts()
    return _PROMPTS


# ============================================================
# 상태 스키마
# ============================================================
class ConsultingState(TypedDict):
    current_step: str           # diagnosis / strategy / plan / execute / done
    seller_id: str
    user_input: str
    session_id: str

    # 단계별 결과
    diagnosis_result: dict
    strategy_result: dict
    plan_result: dict
    execute_result: dict

    # 컨텍스트 요약 (다음 단계에 전달)
    diagnosis_summary: str
    strategy_summary: str
    plan_summary: str

    # 사용자 선택
    strategy_direction: str     # marketing / retention / both
    plan_confirmed: bool

    # 단계 초기화 여부 (도구 호출 + 첫 분석 완료 여부)
    step_initialized: dict      # {"diagnosis": True, ...}

    # 대화 이력 (단계별 필터)
    chat_history: list          # [{"step": str, "role": str, "content": str}, ...]

    # SSE 스트리밍
    sse_callback: Any
    api_key: str
    model: str


# ============================================================
# 세션 관리
# ============================================================
_sessions: Dict[str, Dict[str, Any]] = {}
_SESSION_TTL_SEC = 30 * 60
_MAX_SESSIONS = 100


def _cleanup_expired_sessions():
    now = time.time()
    expired = [
        uname for uname, s in _sessions.items()
        if now - s.get("last_access", 0) > _SESSION_TTL_SEC
    ]
    for uname in expired:
        del _sessions[uname]


def _get_session(username: str, session_id: str | None) -> tuple[str, ConsultingState]:
    _cleanup_expired_sessions()

    if username in _sessions:
        sess = _sessions[username]
        if session_id and sess["session_id"] == session_id:
            sess["last_access"] = time.time()
            return sess["session_id"], sess["state"]

    if len(_sessions) >= _MAX_SESSIONS:
        _cleanup_expired_sessions()
        if len(_sessions) >= _MAX_SESSIONS:
            oldest = min(_sessions, key=lambda u: _sessions[u]["last_access"])
            del _sessions[oldest]

    new_id = session_id or str(uuid.uuid4())[:8]
    state: ConsultingState = {
        "current_step": "diagnosis",
        "seller_id": "",
        "user_input": "",
        "session_id": new_id,
        "diagnosis_result": {},
        "strategy_result": {},
        "plan_result": {},
        "execute_result": {},
        "diagnosis_summary": "",
        "strategy_summary": "",
        "plan_summary": "",
        "strategy_direction": "",
        "plan_confirmed": False,
        "step_initialized": {},
        "chat_history": [],
        "sse_callback": None,
        "api_key": "",
        "model": "gpt-4o-mini",
    }
    _sessions[username] = {
        "session_id": new_id,
        "state": state,
        "last_access": time.time(),
    }
    return new_id, state


def _save_session(username: str, state: ConsultingState):
    if username in _sessions:
        _sessions[username]["state"] = state
        _sessions[username]["last_access"] = time.time()


def get_user_sessions(username: str) -> list[dict]:
    _cleanup_expired_sessions()
    if username not in _sessions:
        return []
    sess = _sessions[username]
    return [{
        "session_id": sess["session_id"],
        "current_step": sess["state"]["current_step"],
        "seller_id": sess["state"]["seller_id"],
        "last_access": sess["last_access"],
    }]


def delete_session(username: str, session_id: str) -> bool:
    if username in _sessions and _sessions[username]["session_id"] == session_id:
        del _sessions[username]
        return True
    return False


# ============================================================
# 단계 정의 및 순서
# ============================================================
STEP_ORDER = ["diagnosis", "strategy", "plan", "execute", "done"]
STEP_LABELS = {
    "diagnosis": "🔍 진단",
    "strategy": "🎯 전략 수립",
    "plan": "📋 실행 계획",
    "execute": "🚀 실행",
    "done": "✅ 완료",
}

# 단계 전환 키워드
ADVANCE_KEYWORDS = ["다음", "다음 단계", "넘어가", "진행해", "넘어가자", "다음으로", "next"]
# 리셋 키워드
RESET_KEYWORDS = ["처음부터", "리셋", "reset", "다시 시작"]
# 롤백 키워드
ROLLBACK_KEYWORDS = ["돌아가", "이전", "rollback", "back", "이전 단계"]
# 실행 승인 키워드
CONFIRM_KEYWORDS = ["확인", "좋아", "네", "오케이", "ok", "승인", "실행해", "confirm", "ㅇㅋ", "고"]
# 전략 방향 키워드
DIRECTION_KEYWORDS = {
    "marketing": ["마케팅", "광고", "홍보", "marketing"],
    "retention": ["리텐션", "이탈", "유지", "retention", "이탈방지"],
    "both": ["둘 다", "모두", "양쪽", "both", "전체", "둘다"],
}

MAX_HISTORY_PER_STEP = 10  # 단계당 최대 대화 이력 수


def _next_step(current: str) -> str:
    idx = STEP_ORDER.index(current) if current in STEP_ORDER else 0
    return STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]


def _prev_step(current: str) -> str:
    idx = STEP_ORDER.index(current) if current in STEP_ORDER else 0
    return STEP_ORDER[max(idx - 1, 0)]


def _detect_intent(text: str) -> dict:
    """사용자 입력에서 의도 파싱"""
    t = text.strip().lower()
    result = {"advance": False, "reset": False, "rollback": False,
              "rollback_target": None, "confirm": False, "direction": None}

    if any(kw in t for kw in RESET_KEYWORDS):
        result["reset"] = True
        return result

    if any(kw in t for kw in ROLLBACK_KEYWORDS):
        result["rollback"] = True
        if "진단" in t or "diagnosis" in t:
            result["rollback_target"] = "diagnosis"
        elif "전략" in t or "strategy" in t:
            result["rollback_target"] = "strategy"
        elif "계획" in t or "plan" in t:
            result["rollback_target"] = "plan"
        return result

    if any(kw in t for kw in ADVANCE_KEYWORDS):
        result["advance"] = True

    if any(kw in t for kw in CONFIRM_KEYWORDS):
        result["confirm"] = True

    # 전략 방향
    for direction, keywords in DIRECTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            result["direction"] = direction
    # "둘 다" 우선
    if any(kw in t for kw in DIRECTION_KEYWORDS["both"]):
        result["direction"] = "both"

    return result


# ============================================================
# 롤백/리셋
# ============================================================
def _clear_downstream(state: ConsultingState, from_step: str):
    idx = STEP_ORDER.index(from_step) if from_step in STEP_ORDER else 0
    step_fields = {
        "diagnosis": ("diagnosis_result", "diagnosis_summary"),
        "strategy": ("strategy_result", "strategy_summary", "strategy_direction"),
        "plan": ("plan_result", "plan_summary", "plan_confirmed"),
        "execute": ("execute_result",),
    }
    for i in range(idx, len(STEP_ORDER) - 1):
        step = STEP_ORDER[i]
        if step in step_fields:
            for field in step_fields[step]:
                if field in ("plan_confirmed",):
                    state[field] = False
                elif field.endswith("_result"):
                    state[field] = {}
                else:
                    state[field] = ""
        # 초기화 상태 리셋
        state["step_initialized"][step] = False
    # 해당 단계 이후 대화 이력 제거
    state["chat_history"] = [
        m for m in state["chat_history"]
        if STEP_ORDER.index(m.get("step", "diagnosis")) < idx
    ]


# ============================================================
# 대화 이력 헬퍼
# ============================================================
def _get_step_history(state: ConsultingState, step: str) -> list[dict]:
    """특정 단계의 대화 이력 반환 (최근 MAX_HISTORY_PER_STEP개)"""
    history = [m for m in state["chat_history"] if m.get("step") == step]
    return history[-MAX_HISTORY_PER_STEP:]


def _add_to_history(state: ConsultingState, step: str, role: str, content: str):
    """대화 이력 추가"""
    state["chat_history"].append({"step": step, "role": role, "content": content})


# ============================================================
# 도구 호출 헬퍼
# ============================================================
async def _call_tool(sse_callback, tool_name: str, tool_fn, *args, **kwargs) -> dict:
    display_args = {}
    for k in ("seller_id", "threshold", "goal", "total_budget", "action_type"):
        if k in kwargs:
            display_args[k] = kwargs[k]
    if args:
        display_args["arg0"] = safe_str(args[0])[:50]

    await sse_callback("tool_start", {"tool": tool_name, "agent": "consulting", "args": display_args})

    try:
        result = tool_fn(*args, **kwargs)
        is_error = isinstance(result, dict) and result.get("status") == "error"

        result_preview = ""
        if isinstance(result, dict):
            result_preview = json.dumps(result, ensure_ascii=False, default=str)[:200]
        else:
            result_preview = safe_str(result)[:200]

        await sse_callback("tool_end", {
            "tool": tool_name,
            "agent": "consulting",
            "status": "error" if is_error else "success",
            "result_preview": result_preview,
        })
        return result
    except Exception as e:
        await sse_callback("tool_end", {
            "tool": tool_name,
            "agent": "consulting",
            "status": "error",
            "result_preview": f"오류: {e}",
        })
        return {"status": "error", "message": str(e)}


# ============================================================
# LLM 응답 스트리밍 (대화 이력 포함)
# ============================================================
async def _stream_llm_response(
    sse_callback,
    llm,
    system_prompt: str,
    user_content: str,
    tool_results: dict | None = None,
    chat_history: list | None = None,
) -> str:
    """LLM 응답을 SSE delta로 스트리밍 (대화 이력 포함)"""
    prompts = _get_prompts()
    common_rules = prompts.get("common_rules", "")
    full_system = f"{system_prompt}\n\n{common_rules}"

    user_text = user_content
    if tool_results:
        safe_data = json_sanitize(tool_results)
        tool_json = json.dumps(safe_data, ensure_ascii=False, indent=2)
        user_text += f"\n\n[도구 분석 결과]\n{tool_json}"

    messages = [SystemMessage(content=full_system)]

    # 대화 이력 추가
    if chat_history:
        for m in chat_history:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=user_text))

    buf = []
    async for chunk in llm.astream(messages):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            buf.append(content)
            await sse_callback("delta", {"delta": content})

    return "".join(buf)


# ============================================================
# 단계별 초기 실행 (도구 호출 + 첫 분석)
# ============================================================
async def _init_diagnosis(state: ConsultingState) -> str:
    """진단 단계 초기화: 도구 호출 + LLM 분석"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]

    await cb("agent_start", {"agent": "consulting_diagnosis", "description": "셀러 종합 진단 분석"})

    analysis = await _call_tool(cb, "analyze_seller", tool_analyze_seller, seller_id)
    churn = await _call_tool(cb, "predict_seller_churn", tool_predict_seller_churn, seller_id)
    segment = await _call_tool(cb, "get_seller_segment", tool_get_seller_segment, seller_id)

    tool_results = {
        "seller_analysis": analysis,
        "churn_prediction": churn,
        "segment_info": segment,
    }
    state["diagnosis_result"] = tool_results

    prompts = _get_prompts()
    system_prompt = prompts.get("steps", {}).get("diagnosis", {}).get("system", "셀러 진단을 수행하세요.")

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 ID '{seller_id}'에 대한 종합 진단을 수행해주세요.",
        tool_results,
    )

    # 요약 생성
    perf = analysis.get("performance", {}) if isinstance(analysis, dict) else {}
    churn_pct = churn.get("churn_probability_pct", "N/A") if isinstance(churn, dict) else "N/A"
    churn_risk = churn.get("risk_level", "N/A") if isinstance(churn, dict) else "N/A"
    seg_info = segment.get("segment", {}) if isinstance(segment, dict) else {}

    state["diagnosis_summary"] = (
        f"셀러 {seller_id} 진단 결과:\n"
        f"- 매출: ₩{safe_int(perf.get('total_revenue', 0)):,}, "
        f"주문수: {safe_int(perf.get('total_orders', 0))}건\n"
        f"- 객단가: ₩{safe_int(perf.get('avg_order_value', 0)):,}\n"
        f"- 전환율: {safe_float(perf.get('conversion_rate', 0))}%, "
        f"재구매율: {safe_float(perf.get('repeat_purchase_rate', 0))}%\n"
        f"- 이탈 위험: {churn_risk} ({churn_pct}%)\n"
        f"- 세그먼트: {seg_info.get('name', 'N/A')}\n"
        f"- 등급: {analysis.get('plan_tier', 'N/A') if isinstance(analysis, dict) else 'N/A'}\n"
    )

    state["step_initialized"]["diagnosis"] = True
    await cb("agent_end", {"agent": "consulting_diagnosis", "elapsed_ms": 0, "description": "셀러 종합 진단 완료"})

    return response


async def _init_strategy(state: ConsultingState) -> str:
    """전략 수립 초기화: 도구 호출 + LLM 전략 분석"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]
    direction = state["strategy_direction"] or "both"

    await cb("agent_start", {"agent": "consulting_strategy", "description": f"전략 수립 ({direction})"})

    tool_results = {"diagnosis": state["diagnosis_result"]}

    if direction in ("marketing", "both"):
        marketing = await _call_tool(
            cb, "optimize_marketing", tool_optimize_marketing,
            seller_id=seller_id, goal="maximize_roas",
        )
        tool_results["marketing_optimization"] = marketing

    if direction in ("retention", "both"):
        retention = await _call_tool(
            cb, "generate_retention_message", tool_generate_retention_message,
            seller_id=seller_id, api_key=api_key,
        )
        tool_results["retention_strategy"] = retention

    state["strategy_result"] = tool_results

    prompts = _get_prompts()
    system_template = prompts.get("steps", {}).get("strategy", {}).get("system", "전략을 수립하세요.")
    system_prompt = system_template.replace("{diagnosis_summary}", state["diagnosis_summary"])
    system_prompt = system_prompt.replace("{strategy_direction}", direction)

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 '{seller_id}'에 대한 {direction} 전략을 수립해주세요.",
        tool_results,
    )

    state["strategy_summary"] = (
        f"전략 방향: {direction}\n"
        f"셀러: {seller_id}\n"
        f"진단 기반 전략 수립 완료\n"
        f"LLM 전략 응답 요약: {response[:300]}..."
    )

    state["step_initialized"]["strategy"] = True
    await cb("agent_end", {"agent": "consulting_strategy", "elapsed_ms": 0, "description": "전략 수립 완료"})

    return response


async def _init_plan(state: ConsultingState) -> str:
    """실행 계획 초기화"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]

    await cb("agent_start", {"agent": "consulting_plan", "description": "실행 계획 수립"})

    tool_results = {
        "diagnosis": state["diagnosis_result"],
        "strategy": state["strategy_result"],
    }

    prompts = _get_prompts()
    system_template = prompts.get("steps", {}).get("plan", {}).get("system", "실행 계획을 수립하세요.")
    system_prompt = system_template.replace("{diagnosis_summary}", state["diagnosis_summary"])
    system_prompt = system_prompt.replace("{strategy_summary}", state["strategy_summary"])

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 '{seller_id}'에 대한 실행 계획을 수립해주세요.",
        tool_results,
    )

    state["plan_result"] = {"plan_response": response[:1000]}
    state["plan_summary"] = (
        f"실행 계획 수립 완료\n"
        f"셀러: {seller_id}\n"
        f"계획 요약: {response[:300]}..."
    )

    state["step_initialized"]["plan"] = True
    await cb("agent_end", {"agent": "consulting_plan", "elapsed_ms": 0, "description": "실행 계획 수립 완료"})

    return response


async def _init_execute(state: ConsultingState) -> str:
    """실행 단계 초기화"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]
    direction = state["strategy_direction"] or "both"

    await cb("agent_start", {"agent": "consulting_execute", "description": "자동 조치 실행"})

    tool_results = {}
    executed_actions = []

    if direction in ("retention", "both"):
        coupon_result = await _call_tool(
            cb, "execute_retention_action", tool_execute_retention_action,
            seller_id=seller_id, action_type="coupon", api_key=api_key,
        )
        tool_results["coupon_action"] = coupon_result
        executed_actions.append("쿠폰 발행")

        manager_result = await _call_tool(
            cb, "execute_retention_action", tool_execute_retention_action,
            seller_id=seller_id, action_type="manager_assign", api_key=api_key,
        )
        tool_results["manager_action"] = manager_result
        executed_actions.append("전담 매니저 배정")

    if direction in ("marketing", "both"):
        marketing = await _call_tool(
            cb, "optimize_marketing", tool_optimize_marketing,
            seller_id=seller_id, goal="balanced",
        )
        tool_results["marketing_final"] = marketing
        executed_actions.append("마케팅 예산 최적화 적용")

    state["execute_result"] = tool_results

    prompts = _get_prompts()
    system_template = prompts.get("steps", {}).get("execute", {}).get("system", "실행 결과를 보고하세요.")
    system_prompt = system_template.replace("{diagnosis_summary}", state["diagnosis_summary"])
    system_prompt = system_prompt.replace("{strategy_summary}", state["strategy_summary"])
    system_prompt = system_prompt.replace("{plan_summary}", state["plan_summary"])

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 '{seller_id}'에 대한 실행 결과를 보고해주세요.\n실행된 조치: {', '.join(executed_actions)}",
        tool_results,
    )

    state["step_initialized"]["execute"] = True
    state["current_step"] = "done"
    await cb("agent_end", {"agent": "consulting_execute", "elapsed_ms": 0, "description": "실행 완료"})

    return response


# ============================================================
# 단계 내 대화 처리 (자유 대화 루프)
# ============================================================
async def _chat_in_step(state: ConsultingState, user_input: str) -> str:
    """현재 단계 내에서 사용자 질문에 응답 (도구 결과 + 대화 이력 기반)"""
    cb = state["sse_callback"]
    current = state["current_step"]
    api_key = state["api_key"]

    # 현재 단계의 도구 결과 수집
    tool_results = {}
    if current == "diagnosis":
        tool_results = state["diagnosis_result"]
    elif current == "strategy":
        tool_results = state["strategy_result"]
    elif current == "plan":
        tool_results = {
            "diagnosis": state["diagnosis_result"],
            "strategy": state["strategy_result"],
            "plan": state["plan_result"],
        }
    elif current == "execute":
        tool_results = state["execute_result"]

    # 이전 단계 요약을 컨텍스트에 포함
    context_parts = []
    if state["diagnosis_summary"]:
        context_parts.append(f"[진단 요약]\n{state['diagnosis_summary']}")
    if current in ("strategy", "plan", "execute") and state["strategy_summary"]:
        context_parts.append(f"[전략 요약]\n{state['strategy_summary']}")
    if current in ("plan", "execute") and state["plan_summary"]:
        context_parts.append(f"[실행 계획 요약]\n{state['plan_summary']}")

    # 시스템 프롬프트: 대화 모드
    prompts = _get_prompts()
    step_prompts = prompts.get("steps", {}).get(current, {})
    base_system = step_prompts.get("system", "")
    chat_system = step_prompts.get("chat", "")

    # chat 프롬프트가 없으면 기본 생성
    if not chat_system:
        step_label = STEP_LABELS.get(current, current)
        chat_system = (
            f"당신은 카페24 셀러 컨설팅 에이전트입니다. 현재 **{step_label}** 단계에서 "
            f"사용자와 대화 중입니다.\n\n"
            f"아래 도구 분석 결과와 대화 이력을 참고하여 사용자의 질문에 답하세요.\n"
            f"사용자가 추가 분석, 수정, 설명을 요청하면 상세히 응답하세요.\n"
            f"사용자가 '다음'이라고 하면 다음 단계로 넘어갈 준비를 안내하세요."
        )

    # 이전 단계 요약을 시스템 프롬프트에 포함
    if context_parts:
        chat_system += "\n\n## 이전 단계 컨텍스트\n" + "\n\n".join(context_parts)

    # 템플릿 변수 치환
    chat_system = chat_system.replace("{diagnosis_summary}", state.get("diagnosis_summary", ""))
    chat_system = chat_system.replace("{strategy_summary}", state.get("strategy_summary", ""))
    chat_system = chat_system.replace("{plan_summary}", state.get("plan_summary", ""))
    chat_system = chat_system.replace("{strategy_direction}", state.get("strategy_direction", ""))

    # 대화 이력
    history = _get_step_history(state, current)

    llm = get_llm(state["model"], api_key, max_tokens=3000, streaming=True, temperature=0.4)
    response = await _stream_llm_response(
        cb, llm, chat_system, user_input, tool_results, history,
    )

    return response


# ============================================================
# 메인 스트리밍 엔트리 함수
# ============================================================
async def run_consulting_stream(
    seller_id: str,
    user_input: str,
    session_id: str | None,
    action: str,
    strategy_choice: str | None,
    username: str,
    sse_callback,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> None:
    """셀러 컨설팅 SSE 스트리밍 메인 함수 — 자유 대화 루프 지원"""
    api_key = pick_api_key(api_key)
    if not api_key:
        await sse_callback("done", {"ok": False, "final": "OpenAI API Key가 없습니다.", "tool_calls": []})
        return

    st.logger.info(
        "CONSULTING_STREAM_START user=%s seller=%s action=%s step_input=%s",
        username, seller_id, action, user_input[:50] if user_input else "",
    )

    try:
        sess_id, state = _get_session(username, session_id)

        state["seller_id"] = seller_id
        state["user_input"] = user_input
        state["session_id"] = sess_id
        state["sse_callback"] = sse_callback
        state["api_key"] = api_key
        state["model"] = model

        # 의도 파싱
        intent = _detect_intent(user_input or "")

        # 명시적 action 파라미터 처리
        if action == "rollback":
            intent["rollback"] = True
        elif action == "reset":
            intent["reset"] = True
        elif action == "strategy_choice" and strategy_choice:
            intent["direction"] = strategy_choice
        elif action == "advance":
            intent["advance"] = True

        # 전략 방향 적용
        if intent["direction"]:
            state["strategy_direction"] = intent["direction"]

        # 세션 정보 전송
        await sse_callback("session_info", {
            "session_id": sess_id,
            "current_step": state["current_step"],
            "seller_id": seller_id,
        })

        current = state["current_step"]

        # ── 리셋 ──
        if intent["reset"]:
            _clear_downstream(state, "diagnosis")
            state["current_step"] = "diagnosis"
            state["step_initialized"] = {}
            state["chat_history"] = []
            await sse_callback("step_change", {
                "step": "diagnosis", "step_number": 1, "total": 4,
                "description": "진단", "reset": True,
            })
            response = await _init_diagnosis(state)
            _add_to_history(state, "diagnosis", "assistant", response)
            _save_session(username, state)
            await _send_step_options(sse_callback, state)
            await sse_callback("done", {
                "ok": True, "session_id": sess_id,
                "current_step": state["current_step"], "tool_calls": [],
            })
            return

        # ── 롤백 ──
        if intent["rollback"]:
            target = intent.get("rollback_target") or _prev_step(current)
            if target != current:
                _clear_downstream(state, target)
                state["current_step"] = target
                step_num = STEP_ORDER.index(target) + 1
                await sse_callback("step_change", {
                    "step": target, "step_number": step_num, "total": 4,
                    "description": STEP_LABELS.get(target, target).replace("🔍 ", "").replace("🎯 ", "").replace("📋 ", "").replace("🚀 ", ""),
                    "rollback": True,
                })
                await sse_callback("delta", {
                    "delta": f"**{STEP_LABELS.get(target, target)}** 단계로 돌아왔습니다. 질문하거나 '다음'으로 진행하세요.\n",
                })
            else:
                await sse_callback("delta", {
                    "delta": f"이미 {STEP_LABELS.get(target, target)} 단계에 있습니다.\n",
                })
            _save_session(username, state)
            await _send_step_options(sse_callback, state)
            await sse_callback("done", {
                "ok": True, "session_id": sess_id,
                "current_step": state["current_step"], "tool_calls": [],
            })
            return

        # ── 단계 전환 (다음) ──
        if intent["advance"]:
            advanced = await _try_advance_step(state, username, sse_callback)
            if advanced:
                _save_session(username, state)
                await _send_step_options(sse_callback, state)
                await sse_callback("done", {
                    "ok": True, "session_id": sess_id,
                    "current_step": state["current_step"], "tool_calls": [],
                })
                return
            # 전환 불가능 시 (조건 미충족) 아래 대화 모드로 이동

        # ── 단계 초기 실행 또는 대화 ──
        if not state["step_initialized"].get(current):
            # 초기 실행 필요
            response = await _handle_step_init(state, intent, sse_callback)
            if response is None:
                # 조건 미충족 (전략 방향 미선택 등) — 이미 메시지 전송됨
                _save_session(username, state)
                await sse_callback("done", {
                    "ok": True, "session_id": sess_id,
                    "current_step": state["current_step"], "tool_calls": [],
                })
                return
            _add_to_history(state, current, "assistant", response)
        else:
            # 대화 모드: 사용자 질문/수정에 응답
            _add_to_history(state, current, "user", user_input)
            response = await _chat_in_step(state, user_input)
            _add_to_history(state, current, "assistant", response)

        _save_session(username, state)
        await _send_step_options(sse_callback, state)
        await sse_callback("done", {
            "ok": True, "session_id": sess_id,
            "current_step": state["current_step"], "tool_calls": [],
        })

        st.logger.info(
            "CONSULTING_STREAM_COMPLETE user=%s seller=%s step=%s",
            username, seller_id, state["current_step"],
        )

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("CONSULTING_STREAM_FAIL err=%s", err)
        msg = f"컨설팅 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        await sse_callback("done", {
            "ok": False, "final": msg, "tool_calls": [],
        })


# ============================================================
# 단계별 초기화 분기
# ============================================================
async def _handle_step_init(state: ConsultingState, intent: dict, sse_callback) -> str | None:
    """단계 초기화 처리. 조건 미충족 시 None 반환."""
    current = state["current_step"]
    step_num = STEP_ORDER.index(current) + 1

    await sse_callback("step_change", {
        "step": current, "step_number": step_num, "total": 4,
        "description": STEP_LABELS.get(current, current).replace("🔍 ", "").replace("🎯 ", "").replace("📋 ", "").replace("🚀 ", ""),
    })

    if current == "diagnosis":
        return await _init_diagnosis(state)

    elif current == "strategy":
        if not state["strategy_direction"]:
            # 전략 방향 미선택 → 안내
            await sse_callback("delta", {
                "delta": "전략 방향을 선택해주세요: **마케팅 강화**, **리텐션(이탈방지)**, 또는 **둘 다**\n\n"
                         "선택 후 자동으로 전략 분석이 시작됩니다.\n",
            })
            await sse_callback("awaiting_input", {
                "step": "strategy",
                "prompt": "전략 방향을 선택해주세요",
                "options": ["마케팅 강화", "리텐션(이탈방지)", "둘 다"],
            })
            return None
        return await _init_strategy(state)

    elif current == "plan":
        return await _init_plan(state)

    elif current == "execute":
        if not intent.get("confirm"):
            # 실행 전 승인 필요
            await sse_callback("delta", {
                "delta": "실행 계획이 수립되었습니다. **승인**을 눌러 자동 조치를 시작하거나, "
                         "수정할 부분을 말씀해주세요.\n",
            })
            await sse_callback("awaiting_input", {
                "step": "execute",
                "prompt": "실행을 승인하시겠습니까?",
                "options": ["승인", "계획 수정", "이전 단계로"],
            })
            return None
        return await _init_execute(state)

    elif current == "done":
        await sse_callback("delta", {
            "delta": "컨설팅이 완료되었습니다. 새로운 세션을 시작하려면 **처음부터**를 입력하세요.\n",
        })
        return "컨설팅 완료"

    return None


# ============================================================
# 단계 전환 시도
# ============================================================
async def _try_advance_step(state: ConsultingState, username: str, sse_callback) -> bool:
    """다음 단계로 전환. 성공 시 True, 조건 미충족 시 False."""
    current = state["current_step"]

    if current == "done":
        await sse_callback("delta", {"delta": "이미 컨설팅이 완료되었습니다.\n"})
        return True

    if not state["step_initialized"].get(current):
        # 현재 단계 초기화가 안 됐으면 먼저 초기화
        return False

    next_step = _next_step(current)

    # 전략 단계 → plan: 방향이 선택되어 있어야 함
    if current == "strategy" and not state["strategy_direction"]:
        await sse_callback("delta", {"delta": "전략 방향을 먼저 선택해주세요.\n"})
        return False

    state["current_step"] = next_step
    step_num = STEP_ORDER.index(next_step) + 1

    if next_step == "done":
        await sse_callback("delta", {"delta": "모든 단계가 완료되었습니다.\n"})
        return True

    await sse_callback("step_change", {
        "step": next_step, "step_number": step_num, "total": 4,
        "description": STEP_LABELS.get(next_step, next_step).replace("🔍 ", "").replace("🎯 ", "").replace("📋 ", "").replace("🚀 ", ""),
    })

    # 다음 단계 초기화
    intent = _detect_intent(state["user_input"] or "")
    response = await _handle_step_init(state, intent, sse_callback)
    if response:
        _add_to_history(state, next_step, "assistant", response)

    return True


# ============================================================
# 옵션 버튼 전송 (단계별 동적)
# ============================================================
async def _send_step_options(sse_callback, state: ConsultingState):
    """현재 단계에 맞는 옵션 버튼 전송"""
    current = state["current_step"]
    initialized = state["step_initialized"].get(current, False)

    if current == "diagnosis" and initialized:
        await sse_callback("awaiting_input", {
            "step": "diagnosis",
            "prompt": "진단 결과에 대해 질문하거나, '다음'으로 전략 수립을 시작하세요.",
            "options": ["매출 분석 상세", "이탈 위험 상세", "다음 단계로"],
        })
    elif current == "strategy" and not state["strategy_direction"]:
        await sse_callback("awaiting_input", {
            "step": "strategy",
            "prompt": "전략 방향을 선택해주세요",
            "options": ["마케팅 강화", "리텐션(이탈방지)", "둘 다"],
        })
    elif current == "strategy" and initialized:
        await sse_callback("awaiting_input", {
            "step": "strategy",
            "prompt": "전략에 대해 질문/수정하거나, '다음'으로 실행 계획을 수립하세요.",
            "options": ["전략 수정", "예산 조정", "다음 단계로"],
        })
    elif current == "plan" and initialized:
        await sse_callback("awaiting_input", {
            "step": "plan",
            "prompt": "실행 계획에 대해 질문/수정하거나, '다음'으로 실행을 시작하세요.",
            "options": ["계획 수정", "일정 조정", "다음 단계로 (실행)"],
        })
    elif current == "execute" and not initialized:
        await sse_callback("awaiting_input", {
            "step": "execute",
            "prompt": "실행을 승인하시겠습니까?",
            "options": ["승인", "계획 수정", "이전 단계로"],
        })
    elif current == "done":
        await sse_callback("awaiting_input", {
            "step": "done",
            "prompt": "컨설팅이 완료되었습니다.",
            "options": ["처음부터 다시", "결과 요약"],
        })
