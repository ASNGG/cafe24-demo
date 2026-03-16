"""
agent/consulting_agent.py - 셀러 컨설팅 에이전트
================================================
StateGraph 기반 4단계 인터랙티브 컨설팅 워크플로우
  diagnosis → strategy → plan → execute

롤백/리셋 지원, SSE 스트리밍, 세션 관리 포함.
기존 멀티에이전트 시스템과 완전히 분리된 독립 모듈.
"""

import asyncio
import json
import time
import uuid
import pathlib
from typing import TypedDict, Any, Dict, Optional

import yaml
from langchain_core.messages import SystemMessage, HumanMessage

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
    """YAML 프롬프트 파일 로드"""
    try:
        with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.logger.warning("CONSULTING_PROMPTS_LOAD_FAIL err=%s", e)
        return {}

_PROMPTS: dict = {}

def _get_prompts() -> dict:
    """프롬프트 싱글톤 (최초 호출 시 로드)"""
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

    # SSE 스트리밍
    sse_callback: Any           # async callback
    api_key: str
    model: str


# ============================================================
# 세션 관리
# ============================================================
_sessions: Dict[str, Dict[str, Any]] = {}   # username -> {session_id, state, last_access}
_SESSION_TTL_SEC = 30 * 60                  # 30분 TTL
_MAX_SESSIONS = 100                         # 최대 동시 세션 수


def _cleanup_expired_sessions():
    """만료 세션 정리"""
    now = time.time()
    expired = [
        uname for uname, s in _sessions.items()
        if now - s.get("last_access", 0) > _SESSION_TTL_SEC
    ]
    for uname in expired:
        del _sessions[uname]


def _get_session(username: str, session_id: str | None) -> tuple[str, ConsultingState]:
    """세션 조회 또는 생성"""
    _cleanup_expired_sessions()

    # 기존 세션 조회
    if username in _sessions:
        sess = _sessions[username]
        if session_id and sess["session_id"] == session_id:
            sess["last_access"] = time.time()
            return sess["session_id"], sess["state"]

    # 새 세션 생성
    if len(_sessions) >= _MAX_SESSIONS:
        _cleanup_expired_sessions()
        if len(_sessions) >= _MAX_SESSIONS:
            # 가장 오래된 세션 제거
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
    """세션 상태 저장"""
    if username in _sessions:
        _sessions[username]["state"] = state
        _sessions[username]["last_access"] = time.time()


def get_user_sessions(username: str) -> list[dict]:
    """사용자의 활성 세션 목록 반환"""
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
    """세션 삭제"""
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


def _next_step(current: str) -> str:
    """다음 단계 반환"""
    idx = STEP_ORDER.index(current) if current in STEP_ORDER else 0
    return STEP_ORDER[min(idx + 1, len(STEP_ORDER) - 1)]


def _prev_step(current: str) -> str:
    """이전 단계 반환"""
    idx = STEP_ORDER.index(current) if current in STEP_ORDER else 0
    return STEP_ORDER[max(idx - 1, 0)]


# ============================================================
# 롤백/리셋 처리
# ============================================================
def _clear_downstream(state: ConsultingState, from_step: str):
    """특정 단계 이후의 모든 데이터 초기화"""
    idx = STEP_ORDER.index(from_step) if from_step in STEP_ORDER else 0
    step_fields = {
        "diagnosis": ("diagnosis_result", "diagnosis_summary"),
        "strategy": ("strategy_result", "strategy_summary", "strategy_direction"),
        "plan": ("plan_result", "plan_summary", "plan_confirmed"),
        "execute": ("execute_result",),
    }
    for i in range(idx, len(STEP_ORDER) - 1):  # done 제외
        step = STEP_ORDER[i]
        if step in step_fields:
            for field in step_fields[step]:
                if field in ("plan_confirmed",):
                    state[field] = False
                elif field.endswith("_result"):
                    state[field] = {}
                else:
                    state[field] = ""


def _parse_user_action(user_input: str, current_step: str) -> dict:
    """사용자 입력에서 액션, 롤백 대상, 전략 선택 파싱"""
    text = user_input.strip().lower()
    result = {"action": "forward", "rollback_target": None, "strategy_choice": None}

    # 리셋 감지
    reset_kw = ["처음부터", "리셋", "reset", "다시 시작"]
    if any(kw in text for kw in reset_kw):
        result["action"] = "reset"
        return result

    # 롤백 감지
    rollback_kw = ["돌아가", "이전", "rollback", "back", "이전 단계"]
    if any(kw in text for kw in rollback_kw):
        result["action"] = "rollback"
        # 특정 단계 롤백 대상 파싱
        if "진단" in text or "diagnosis" in text:
            result["rollback_target"] = "diagnosis"
        elif "전략" in text or "strategy" in text:
            result["rollback_target"] = "strategy"
        elif "계획" in text or "plan" in text:
            result["rollback_target"] = "plan"
        return result

    # 확인/승인 감지
    confirm_kw = ["확인", "진행", "좋아", "네", "오케이", "ok", "승인", "실행해", "confirm", "ㅇㅋ"]
    if any(kw in text for kw in confirm_kw):
        result["action"] = "confirm"

    # 전략 방향 감지
    if "마케팅" in text or "광고" in text or "홍보" in text or "marketing" in text:
        result["strategy_choice"] = "marketing"
    if "리텐션" in text or "이탈" in text or "유지" in text or "retention" in text:
        if result["strategy_choice"] == "marketing":
            result["strategy_choice"] = "both"
        else:
            result["strategy_choice"] = "retention"
    if "둘 다" in text or "모두" in text or "both" in text or "전체" in text:
        result["strategy_choice"] = "both"

    return result


# ============================================================
# 도구 호출 헬퍼 (SSE 이벤트 발행 포함)
# ============================================================
async def _call_tool(sse_callback, tool_name: str, tool_fn, *args, **kwargs) -> dict:
    """도구 호출 + SSE tool_start/tool_end 이벤트 발행"""
    # 핵심 파라미터 추출 (표시용)
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
# LLM 응답 스트리밍 헬퍼
# ============================================================
async def _stream_llm_response(
    sse_callback,
    llm,
    system_prompt: str,
    user_content: str,
    tool_results: dict | None = None,
) -> str:
    """LLM 응답을 SSE delta로 스트리밍하고 전체 텍스트 반환"""
    prompts = _get_prompts()
    common_rules = prompts.get("common_rules", "")
    full_system = f"{system_prompt}\n\n{common_rules}"

    user_text = user_content
    if tool_results:
        safe_data = json_sanitize(tool_results)
        tool_json = json.dumps(safe_data, ensure_ascii=False, indent=2)
        user_text += f"\n\n[도구 분석 결과]\n{tool_json}"

    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_text),
    ]

    buf = []
    async for chunk in llm.astream(messages):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            buf.append(content)
            await sse_callback("delta", {"delta": content})

    return "".join(buf)


# ============================================================
# 단계별 노드 함수
# ============================================================
async def _run_diagnosis(state: ConsultingState) -> ConsultingState:
    """1단계: 셀러 진단"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]

    await cb("step_change", {
        "step": "diagnosis",
        "label": STEP_LABELS["diagnosis"],
        "step_index": 0,
        "total_steps": 4,
    })
    await cb("agent_start", {"agent": "consulting_diagnosis", "description": "셀러 종합 진단 분석"})

    # 도구 호출: 셀러 분석, 이탈 예측, 세그먼트
    analysis = await _call_tool(cb, "analyze_seller", tool_analyze_seller, seller_id)
    churn = await _call_tool(cb, "predict_seller_churn", tool_predict_seller_churn, seller_id)
    segment = await _call_tool(cb, "get_seller_segment", tool_get_seller_segment, seller_id)

    # 결과 통합
    tool_results = {
        "seller_analysis": analysis,
        "churn_prediction": churn,
        "segment_info": segment,
    }
    state["diagnosis_result"] = tool_results

    # LLM 응답 생성
    prompts = _get_prompts()
    system_prompt = prompts.get("steps", {}).get("diagnosis", {}).get("system", "셀러 진단을 수행하세요.")

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 ID '{seller_id}'에 대한 종합 진단을 수행해주세요.",
        tool_results,
    )

    # 요약 생성 (다음 단계 컨텍스트용)
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

    state["current_step"] = "strategy"

    elapsed_ms = 0  # 단순화
    await cb("agent_end", {"agent": "consulting_diagnosis", "elapsed_ms": elapsed_ms, "description": "셀러 종합 진단 완료"})
    await cb("awaiting_input", {
        "step": "strategy",
        "prompt": "전략 방향을 선택해주세요: 마케팅 강화, 리텐션(이탈방지), 또는 둘 다",
        "options": ["마케팅 강화", "리텐션(이탈방지)", "둘 다"],
    })

    return state


async def _run_strategy(state: ConsultingState) -> ConsultingState:
    """2단계: 전략 수립"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]
    direction = state["strategy_direction"] or "both"

    await cb("step_change", {
        "step": "strategy",
        "label": STEP_LABELS["strategy"],
        "step_index": 1,
        "total_steps": 4,
    })
    await cb("agent_start", {"agent": "consulting_strategy", "description": f"전략 수립 ({direction})"})

    tool_results = {"diagnosis": state["diagnosis_result"]}

    # 마케팅 전략: 마케팅 최적화 도구 호출
    if direction in ("marketing", "both"):
        marketing = await _call_tool(
            cb, "optimize_marketing", tool_optimize_marketing,
            seller_id=seller_id, goal="maximize_roas",
        )
        tool_results["marketing_optimization"] = marketing

    # 리텐션 전략: 리텐션 메시지 생성 도구 호출
    if direction in ("retention", "both"):
        retention = await _call_tool(
            cb, "generate_retention_message", tool_generate_retention_message,
            seller_id=seller_id, api_key=api_key,
        )
        tool_results["retention_strategy"] = retention

    state["strategy_result"] = tool_results

    # LLM 응답 생성
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

    # 요약 생성
    state["strategy_summary"] = (
        f"전략 방향: {direction}\n"
        f"셀러: {seller_id}\n"
        f"진단 기반 전략 수립 완료\n"
        f"LLM 전략 응답 요약: {response[:300]}..."
    )

    state["current_step"] = "plan"

    await cb("agent_end", {"agent": "consulting_strategy", "elapsed_ms": 0, "description": "전략 수립 완료"})
    await cb("awaiting_input", {
        "step": "plan",
        "prompt": "이 전략으로 실행 계획을 수립할까요? 확인 또는 수정 사항을 말씀해주세요.",
        "options": ["확인", "수정 요청"],
    })

    return state


async def _run_plan(state: ConsultingState) -> ConsultingState:
    """3단계: 실행 계획 수립"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]

    await cb("step_change", {
        "step": "plan",
        "label": STEP_LABELS["plan"],
        "step_index": 2,
        "total_steps": 4,
    })
    await cb("agent_start", {"agent": "consulting_plan", "description": "실행 계획 수립"})

    tool_results = {
        "diagnosis": state["diagnosis_result"],
        "strategy": state["strategy_result"],
    }

    # LLM 응답 생성
    prompts = _get_prompts()
    system_template = prompts.get("steps", {}).get("plan", {}).get("system", "실행 계획을 수립하세요.")
    system_prompt = system_template.replace("{diagnosis_summary}", state["diagnosis_summary"])
    system_prompt = system_prompt.replace("{strategy_summary}", state["strategy_summary"])

    llm = get_llm(state["model"], api_key, max_tokens=4000, streaming=True, temperature=0.3)
    response = await _stream_llm_response(
        cb, llm, system_prompt,
        f"셀러 '{seller_id}'에 대한 실행 계획을 수립해주세요.\n\n사용자 추가 요청: {state['user_input']}",
        tool_results,
    )

    state["plan_result"] = {"plan_response": response[:1000]}
    state["plan_summary"] = (
        f"실행 계획 수립 완료\n"
        f"셀러: {seller_id}\n"
        f"계획 요약: {response[:300]}..."
    )

    state["current_step"] = "execute"

    await cb("agent_end", {"agent": "consulting_plan", "elapsed_ms": 0, "description": "실행 계획 수립 완료"})
    await cb("awaiting_input", {
        "step": "execute",
        "prompt": "실행 계획을 승인하시겠습니까? 승인 시 자동 조치를 시작합니다.",
        "options": ["승인", "수정 요청"],
    })

    return state


async def _run_execute(state: ConsultingState) -> ConsultingState:
    """4단계: 실행"""
    cb = state["sse_callback"]
    seller_id = state["seller_id"]
    api_key = state["api_key"]
    direction = state["strategy_direction"] or "both"

    await cb("step_change", {
        "step": "execute",
        "label": STEP_LABELS["execute"],
        "step_index": 3,
        "total_steps": 4,
    })
    await cb("agent_start", {"agent": "consulting_execute", "description": "자동 조치 실행"})

    tool_results = {}
    executed_actions = []

    # 리텐션 조치 실행 (리텐션 또는 both)
    if direction in ("retention", "both"):
        # 쿠폰 발행
        coupon_result = await _call_tool(
            cb, "execute_retention_action", tool_execute_retention_action,
            seller_id=seller_id, action_type="coupon", api_key=api_key,
        )
        tool_results["coupon_action"] = coupon_result
        executed_actions.append("쿠폰 발행")

        # 매니저 배정
        manager_result = await _call_tool(
            cb, "execute_retention_action", tool_execute_retention_action,
            seller_id=seller_id, action_type="manager_assign", api_key=api_key,
        )
        tool_results["manager_action"] = manager_result
        executed_actions.append("전담 매니저 배정")

    # 마케팅 최적화 결과 재확인
    if direction in ("marketing", "both"):
        marketing = await _call_tool(
            cb, "optimize_marketing", tool_optimize_marketing,
            seller_id=seller_id, goal="balanced",
        )
        tool_results["marketing_final"] = marketing
        executed_actions.append("마케팅 예산 최적화 적용")

    state["execute_result"] = tool_results

    # LLM 응답 생성
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

    state["current_step"] = "done"

    await cb("agent_end", {"agent": "consulting_execute", "elapsed_ms": 0, "description": "실행 완료"})

    return state


# ============================================================
# 메인 스트리밍 엔트리 함수
# ============================================================
async def run_consulting_stream(
    seller_id: str,
    user_input: str,
    session_id: str | None,
    action: str,            # message / confirm / rollback / reset
    strategy_choice: str | None,
    username: str,
    sse_callback,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> None:
    """셀러 컨설팅 SSE 스트리밍 메인 함수

    Args:
        seller_id: 셀러 ID
        user_input: 사용자 입력 텍스트
        session_id: 세션 ID (없으면 새 세션 생성)
        action: 요청 액션 (message/confirm/rollback/reset)
        strategy_choice: 전략 방향 선택 (marketing/retention/both)
        username: 사용자명
        sse_callback: SSE 이벤트 콜백 (async callable)
        api_key: OpenAI API Key
        model: 모델명
    """
    api_key = pick_api_key(api_key)
    if not api_key:
        await sse_callback("done", {"ok": False, "final": "OpenAI API Key가 없습니다.", "tool_calls": []})
        return

    st.logger.info(
        "CONSULTING_STREAM_START user=%s seller=%s action=%s",
        username, seller_id, action,
    )

    try:
        # 세션 조회/생성
        sess_id, state = _get_session(username, session_id)

        # 상태 업데이트
        state["seller_id"] = seller_id
        state["user_input"] = user_input
        state["session_id"] = sess_id
        state["sse_callback"] = sse_callback
        state["api_key"] = api_key
        state["model"] = model

        # 사용자 입력에서 액션 파싱 (명시적 action이 없으면 텍스트 분석)
        if action == "message" and user_input:
            parsed = _parse_user_action(user_input, state["current_step"])
            if parsed["action"] != "forward":
                action = parsed["action"]
            if parsed["strategy_choice"] and not strategy_choice:
                strategy_choice = parsed["strategy_choice"]
            if parsed["action"] == "confirm":
                action = "confirm"

        # 전략 선택 적용
        if strategy_choice:
            state["strategy_direction"] = strategy_choice

        # 세션 정보 전송
        await sse_callback("session_info", {
            "session_id": sess_id,
            "current_step": state["current_step"],
            "seller_id": seller_id,
        })

        # ── 리셋 처리 ──
        if action == "reset":
            _clear_downstream(state, "diagnosis")
            state["current_step"] = "diagnosis"
            await sse_callback("step_change", {
                "step": "diagnosis",
                "label": STEP_LABELS["diagnosis"],
                "step_index": 0,
                "total_steps": 4,
                "reset": True,
            })
            # 진단부터 다시 시작
            state = await _run_diagnosis(state)
            _save_session(username, state)
            await sse_callback("done", {
                "ok": True,
                "final": "세션이 리셋되어 진단부터 다시 시작합니다.",
                "session_id": sess_id,
                "current_step": state["current_step"],
                "tool_calls": [],
            })
            return

        # ── 롤백 처리 ──
        if action == "rollback":
            parsed = _parse_user_action(user_input, state["current_step"])
            target = parsed.get("rollback_target") or _prev_step(state["current_step"])

            if target == state["current_step"]:
                # 이미 해당 단계에 있음
                await sse_callback("delta", {"delta": f"이미 {STEP_LABELS.get(target, target)} 단계에 있습니다.\n"})
            else:
                _clear_downstream(state, target)
                state["current_step"] = target
                await sse_callback("step_change", {
                    "step": target,
                    "label": STEP_LABELS.get(target, target),
                    "step_index": STEP_ORDER.index(target),
                    "total_steps": 4,
                    "rollback": True,
                })
                await sse_callback("delta", {
                    "delta": f"**{STEP_LABELS.get(target, target)}** 단계로 돌아갑니다. 다시 진행해주세요.\n",
                })

            _save_session(username, state)
            await sse_callback("awaiting_input", {
                "step": state["current_step"],
                "prompt": f"{STEP_LABELS.get(state['current_step'], '')} 단계를 다시 진행합니다.",
            })
            await sse_callback("done", {
                "ok": True,
                "final": f"{STEP_LABELS.get(target, target)} 단계로 롤백 완료",
                "session_id": sess_id,
                "current_step": state["current_step"],
                "tool_calls": [],
            })
            return

        # ── 단계별 실행 ──
        current = state["current_step"]

        if current == "diagnosis":
            state = await _run_diagnosis(state)

        elif current == "strategy":
            # 전략 방향이 선택되지 않은 경우
            if not state["strategy_direction"]:
                await sse_callback("delta", {
                    "delta": "전략 방향을 먼저 선택해주세요: **마케팅 강화**, **리텐션(이탈방지)**, 또는 **둘 다**\n",
                })
                await sse_callback("awaiting_input", {
                    "step": "strategy",
                    "prompt": "전략 방향을 선택해주세요",
                    "options": ["마케팅 강화", "리텐션(이탈방지)", "둘 다"],
                })
                _save_session(username, state)
                await sse_callback("done", {
                    "ok": True,
                    "final": "전략 방향 선택 대기 중",
                    "session_id": sess_id,
                    "current_step": state["current_step"],
                    "tool_calls": [],
                })
                return
            state = await _run_strategy(state)

        elif current == "plan":
            if action == "confirm" or state.get("plan_confirmed"):
                state = await _run_plan(state)
            else:
                # 확인이 필요함
                await sse_callback("delta", {
                    "delta": "전략을 검토한 후 **확인**을 눌러 실행 계획 수립을 진행하거나, 수정할 부분을 말씀해주세요.\n",
                })
                await sse_callback("awaiting_input", {
                    "step": "plan",
                    "prompt": "전략을 확인하고 실행 계획을 수립할까요?",
                    "options": ["확인", "수정 요청"],
                })
                _save_session(username, state)
                await sse_callback("done", {
                    "ok": True,
                    "final": "전략 확인 대기 중",
                    "session_id": sess_id,
                    "current_step": state["current_step"],
                    "tool_calls": [],
                })
                return

        elif current == "execute":
            if action == "confirm":
                state["plan_confirmed"] = True
                state = await _run_execute(state)
            else:
                await sse_callback("delta", {
                    "delta": "실행 계획을 검토한 후 **승인**을 눌러 자동 조치를 시작하거나, 수정할 부분을 말씀해주세요.\n",
                })
                await sse_callback("awaiting_input", {
                    "step": "execute",
                    "prompt": "실행 계획을 승인하시겠습니까?",
                    "options": ["승인", "수정 요청"],
                })
                _save_session(username, state)
                await sse_callback("done", {
                    "ok": True,
                    "final": "실행 승인 대기 중",
                    "session_id": sess_id,
                    "current_step": state["current_step"],
                    "tool_calls": [],
                })
                return

        elif current == "done":
            await sse_callback("delta", {
                "delta": "컨설팅이 완료되었습니다. 새로운 세션을 시작하려면 **처음부터** 를 입력하세요.\n",
            })
            _save_session(username, state)
            await sse_callback("done", {
                "ok": True,
                "final": "컨설팅 완료",
                "session_id": sess_id,
                "current_step": "done",
                "tool_calls": [],
            })
            return

        # 세션 저장
        _save_session(username, state)

        await sse_callback("done", {
            "ok": True,
            "final": f"{STEP_LABELS.get(state['current_step'], state['current_step'])} 단계 완료",
            "session_id": sess_id,
            "current_step": state["current_step"],
            "tool_calls": [],
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
            "ok": False,
            "final": msg,
            "tool_calls": [],
        })
