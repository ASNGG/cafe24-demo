"""
agent/consulting_graph.py
LangGraph interrupt() 기반 셀러 컨설팅 4단계 워크플로우

- interrupt() + Command(resume=) 으로 단계 전환
- MemorySaver 체크포인터로 세션 자동 관리
- astream_events로 SSE 스트리밍 통합
- Python 3.13: asyncio.to_thread 병렬 도구 실행
"""
import asyncio
import json
import yaml
from typing import TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI

from agent.tools import (
    tool_analyze_seller, tool_predict_seller_churn, tool_get_seller_segment,
    tool_optimize_marketing, tool_generate_retention_message, tool_execute_retention_action,
)
import state as st


# ============================================================
# 프롬프트 로드
# ============================================================
_PROMPT_PATH = Path(__file__).parent / "consulting_prompts.yaml"
_PROMPTS: dict = {}


def _load_prompts() -> dict:
    global _PROMPTS
    if not _PROMPTS:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            _PROMPTS = yaml.safe_load(f)
    return _PROMPTS


def _get_prompt(step: str, kind: str = "system") -> str:
    return _load_prompts().get("steps", {}).get(step, {}).get(kind, "")


def _common_rules() -> str:
    return _load_prompts().get("common_rules", "")


# ============================================================
# State
# ============================================================
class ConsultingState(TypedDict):
    seller_id: str
    api_key: str
    model: str
    # 단계별 결과
    diagnosis_result: dict
    diagnosis_summary: str
    strategy_direction: str
    strategy_result: dict
    strategy_summary: str
    plan_summary: str
    execute_result: dict
    # 사용자 입력 (interrupt resume 값)
    user_input: str


# ============================================================
# 유틸리티
# ============================================================
_ADVANCE_KW = {"다음", "다음 단계", "넘어가", "진행해", "next", "다음 단계로", "다음으로"}
_APPROVE_KW = {"승인", "확인", "네", "ok", "좋아", "실행", "ㅇㅋ", "오케이"}


def _is_advance(text: str) -> bool:
    t = text.strip()
    return t in _ADVANCE_KW or any(t.startswith(kw) for kw in _ADVANCE_KW)


def _is_approve(text: str) -> bool:
    t = text.strip().lower()
    return t in _APPROVE_KW or any(kw in t for kw in _APPROVE_KW)


def _parse_direction(text: str) -> str:
    t = text.lower()
    if "둘 다" in t or "both" in t:
        return "both"
    if any(kw in t for kw in ["마케팅", "marketing"]):
        return "marketing"
    if any(kw in t for kw in ["리텐션", "retention", "이탈"]):
        return "retention"
    return "both"


def _make_llm(state: ConsultingState) -> ChatOpenAI:
    return ChatOpenAI(
        model=state.get("model") or "gpt-4o-mini",
        api_key=state["api_key"],
        streaming=True,
    )


def _fmt_prompt(template: str, state: ConsultingState) -> str:
    """프롬프트 템플릿 변수 치환"""
    replacements = {
        "diagnosis_summary": state.get("diagnosis_summary", ""),
        "strategy_summary": state.get("strategy_summary", ""),
        "plan_summary": state.get("plan_summary", ""),
        "strategy_direction": state.get("strategy_direction", ""),
    }
    for key, val in replacements.items():
        template = template.replace(f"{{{key}}}", val)
    return template


# ============================================================
# 노드 함수
# ============================================================

# ── Step 1: 진단 ──
async def diagnosis_node(state: ConsultingState) -> dict:
    """셀러 진단: 3개 도구 병렬 실행 + LLM 분석 (Python 3.13 asyncio.to_thread)"""
    sid = state["seller_id"]
    # 3개 도구 병렬 실행 — 순차(~3초) → 병렬(~1초)
    analysis, churn, segment = await asyncio.gather(
        asyncio.to_thread(tool_analyze_seller, sid),
        asyncio.to_thread(tool_predict_seller_churn, sid),
        asyncio.to_thread(tool_get_seller_segment, sid),
    )
    tool_results = {
        "seller_analysis": analysis,
        "churn_prediction": churn,
        "segment_info": segment,
    }
    llm = _make_llm(state)
    prompt = f"{_common_rules()}\n\n{_get_prompt('diagnosis', 'system')}"
    resp = await llm.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"셀러 {sid} 진단:\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}"},
    ])
    return {"diagnosis_result": tool_results, "diagnosis_summary": resp.content}


def diagnosis_review(state: ConsultingState) -> dict:
    user_input = interrupt({
        "step": "diagnosis", "step_number": 1, "total": 4,
        "description": "진단 분석 완료",
        "prompt": "진단 결과에 대해 질문하거나, '다음'으로 전략 수립을 시작하세요.",
        "options": ["매출 분석 상세", "이탈 위험 상세", "다음 단계로"],
    })
    return {"user_input": user_input}


async def diagnosis_chat(state: ConsultingState) -> dict:
    llm = _make_llm(state)
    chat_prompt = _get_prompt("diagnosis", "chat") or "셀러 진단 결과를 기반으로 질문에 답하세요."
    await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{chat_prompt}"},
        {"role": "user", "content": f"진단 결과:\n{json.dumps(state.get('diagnosis_result', {}), ensure_ascii=False)}\n\n질문: {state['user_input']}"},
    ])
    return {}


# ── Step 2: 전략 수립 ──
def strategy_direction_node(state: ConsultingState) -> dict:
    choice = interrupt({
        "step": "strategy", "step_number": 2, "total": 4,
        "description": "전략 방향 선택",
        "prompt": "전략 방향을 선택해주세요",
        "options": ["마케팅 강화", "리텐션(이탈방지)", "둘 다"],
    })
    return {"strategy_direction": _parse_direction(choice), "user_input": choice}


async def strategy_node(state: ConsultingState) -> dict:
    sid = state["seller_id"]
    direction = state["strategy_direction"]
    tool_results = {}
    if direction in ("marketing", "both"):
        tool_results["marketing"] = tool_optimize_marketing(seller_id=sid, goal="maximize_roas")
    if direction in ("retention", "both"):
        tool_results["retention"] = tool_generate_retention_message(seller_id=sid, api_key=state.get("api_key", ""))
    llm = _make_llm(state)
    prompt = _fmt_prompt(_get_prompt("strategy", "system"), state)
    resp = await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{prompt}"},
        {"role": "user", "content": json.dumps(tool_results, ensure_ascii=False, indent=2)},
    ])
    return {"strategy_result": tool_results, "strategy_summary": resp.content}


def strategy_review(state: ConsultingState) -> dict:
    user_input = interrupt({
        "step": "strategy", "step_number": 2, "total": 4,
        "description": "전략 수립 완료",
        "prompt": "전략에 대해 질문/수정하거나, '다음'으로 실행 계획을 수립하세요.",
        "options": ["전략 수정", "예산 조정", "다음 단계로"],
    })
    return {"user_input": user_input}


async def strategy_chat(state: ConsultingState) -> dict:
    llm = _make_llm(state)
    chat_prompt = _get_prompt("strategy", "chat") or "전략 결과를 기반으로 질문에 답하세요."
    await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{chat_prompt}"},
        {"role": "user", "content": f"전략 결과:\n{json.dumps(state.get('strategy_result', {}), ensure_ascii=False)}\n\n질문: {state['user_input']}"},
    ])
    return {}


# ── Step 3: 실행 계획 ──
async def plan_node(state: ConsultingState) -> dict:
    llm = _make_llm(state)
    prompt = _fmt_prompt(_get_prompt("plan", "system"), state)
    resp = await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{prompt}"},
        {"role": "user", "content": "이전 진단과 전략을 기반으로 실행 계획을 수립해주세요."},
    ])
    return {"plan_summary": resp.content}


def plan_review(state: ConsultingState) -> dict:
    user_input = interrupt({
        "step": "plan", "step_number": 3, "total": 4,
        "description": "실행 계획 수립 완료",
        "prompt": "실행 계획에 대해 질문/수정하거나, '다음'으로 실행을 시작하세요.",
        "options": ["계획 수정", "일정 조정", "다음 단계로 (실행)"],
    })
    return {"user_input": user_input}


async def plan_chat(state: ConsultingState) -> dict:
    llm = _make_llm(state)
    chat_prompt = _get_prompt("plan", "chat") or "실행 계획을 기반으로 질문에 답하세요."
    await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{chat_prompt}"},
        {"role": "user", "content": f"실행 계획:\n{state.get('plan_summary', '')}\n\n질문: {state['user_input']}"},
    ])
    return {}


# ── Step 4: 실행 ──
def execute_approval(state: ConsultingState) -> dict:
    approval = interrupt({
        "step": "execute", "step_number": 4, "total": 4,
        "description": "실행 승인 대기",
        "prompt": "실행을 승인하시겠습니까?",
        "options": ["승인", "계획 수정", "이전 단계로"],
    })
    return {"user_input": approval}


async def execute_node(state: ConsultingState) -> dict:
    sid = state["seller_id"]
    direction = state.get("strategy_direction", "both")
    tool_results = {}
    if direction in ("retention", "both"):
        tool_results["coupon"] = tool_execute_retention_action(seller_id=sid, action_type="coupon", api_key=state.get("api_key", ""))
        tool_results["manager"] = tool_execute_retention_action(seller_id=sid, action_type="manager_assign", api_key=state.get("api_key", ""))
    if direction in ("marketing", "both"):
        tool_results["marketing"] = tool_optimize_marketing(seller_id=sid, goal="balanced")
    llm = _make_llm(state)
    prompt = _fmt_prompt(_get_prompt("execute", "system"), state)
    resp = await llm.ainvoke([
        {"role": "system", "content": f"{_common_rules()}\n\n{prompt}"},
        {"role": "user", "content": json.dumps(tool_results, ensure_ascii=False, indent=2)},
    ])
    return {"execute_result": tool_results}


# ============================================================
# 라우팅
# ============================================================
def route_diagnosis(state):
    return "strategy_direction" if _is_advance(state.get("user_input", "")) else "diagnosis_chat"


def route_strategy(state):
    return "plan" if _is_advance(state.get("user_input", "")) else "strategy_chat"


def route_plan(state):
    return "execute_approval" if _is_advance(state.get("user_input", "")) else "plan_chat"


def route_execute(state):
    ui = state.get("user_input", "")
    if _is_approve(ui):
        return "execute"
    return "plan"


# ============================================================
# 그래프 빌드
# ============================================================
_checkpointer = MemorySaver()
_compiled_graph = None

# 세션 카운터 (리셋 시 thread_id 변경용)
_session_counters: dict[str, int] = {}


def _get_thread_id(username: str) -> str:
    counter = _session_counters.get(username, 0)
    return f"consulting_{username}_{counter}"


def reset_consulting_session(username: str):
    """컨설팅 세션 초기화 (새 thread_id 발급)"""
    _session_counters[username] = _session_counters.get(username, 0) + 1


def build_consulting_graph():
    g = StateGraph(ConsultingState)

    g.add_node("diagnosis", diagnosis_node)
    g.add_node("diagnosis_review", diagnosis_review)
    g.add_node("diagnosis_chat", diagnosis_chat)
    g.add_node("strategy_direction", strategy_direction_node)
    g.add_node("strategy", strategy_node)
    g.add_node("strategy_review", strategy_review)
    g.add_node("strategy_chat", strategy_chat)
    g.add_node("plan", plan_node)
    g.add_node("plan_review", plan_review)
    g.add_node("plan_chat", plan_chat)
    g.add_node("execute_approval", execute_approval)
    g.add_node("execute", execute_node)

    g.add_edge(START, "diagnosis")
    g.add_edge("diagnosis", "diagnosis_review")
    g.add_conditional_edges("diagnosis_review", route_diagnosis)
    g.add_edge("diagnosis_chat", "diagnosis_review")
    g.add_edge("strategy_direction", "strategy")
    g.add_edge("strategy", "strategy_review")
    g.add_conditional_edges("strategy_review", route_strategy)
    g.add_edge("strategy_chat", "strategy_review")
    g.add_edge("plan", "plan_review")
    g.add_conditional_edges("plan_review", route_plan)
    g.add_edge("plan_chat", "plan_review")
    g.add_conditional_edges("execute_approval", route_execute)
    g.add_edge("execute", END)

    return g.compile(checkpointer=_checkpointer)


def get_consulting_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_consulting_graph()
    return _compiled_graph


# ============================================================
# 활성 세션 확인 (라우팅용)
# ============================================================
def has_active_consulting(username: str) -> bool:
    """사용자의 활성 컨설팅 세션 (pending interrupt) 존재 여부"""
    graph = get_consulting_graph()
    config = {"configurable": {"thread_id": _get_thread_id(username)}}
    try:
        state = graph.get_state(config)
        return bool(state.values) and bool(state.tasks)
    except Exception:
        return False


# ============================================================
# SSE 스트리밍 인터페이스
# ============================================================
_STEP_MAP = {
    "diagnosis": ("diagnosis", 1),
    "diagnosis_review": ("diagnosis", 1),
    "diagnosis_chat": ("diagnosis", 1),
    "strategy_direction": ("strategy", 2),
    "strategy": ("strategy", 2),
    "strategy_review": ("strategy", 2),
    "strategy_chat": ("strategy", 2),
    "plan": ("plan", 3),
    "plan_review": ("plan", 3),
    "plan_chat": ("plan", 3),
    "execute_approval": ("execute", 4),
    "execute": ("execute", 4),
}


async def run_consulting_graph_stream(
    seller_id: str,
    user_input: str,
    username: str,
    sse_callback,
    api_key: str,
    model: str = "gpt-4o-mini",
):
    """LangGraph 기반 컨설팅 SSE 스트리밍"""
    graph = get_consulting_graph()
    thread_id = _get_thread_id(username)
    config = {"configurable": {"thread_id": thread_id}}

    # 리셋 감지
    if any(kw in user_input for kw in ["처음부터", "리셋", "reset"]):
        reset_consulting_session(username)
        thread_id = _get_thread_id(username)
        config = {"configurable": {"thread_id": thread_id}}
        # seller_id가 없으면 이전 세션에서 복구 불가 → 에러
        if not seller_id:
            await sse_callback("delta", {"delta": "컨설팅을 초기화했습니다. 셀러 ID를 포함하여 다시 시작해주세요.\n"})
            await sse_callback("done", {"ok": True})
            return

    # 기존 세션 확인
    existing_state = graph.get_state(config)
    has_session = bool(existing_state.values) and bool(existing_state.tasks)

    if not has_session:
        # 새 세션
        input_data = {"seller_id": seller_id, "api_key": api_key, "model": model}
        await sse_callback("session_info", {"session_id": thread_id})
    else:
        # 기존 세션 resume
        from langgraph.types import Command
        input_data = Command(resume=user_input)

    # 노드 전환 추적
    current_node = None
    final_buf = []

    async for event in graph.astream_events(input_data, config, version="v2"):
        ev_type = event.get("event", "")
        metadata = event.get("metadata", {})

        # 노드 전환 → step_change SSE
        lg_node = metadata.get("langgraph_node")
        if ev_type == "on_chain_start" and lg_node and lg_node != current_node and lg_node in _STEP_MAP:
            current_node = lg_node
            step, num = _STEP_MAP[lg_node]
            await sse_callback("step_change", {
                "step": step, "step_number": num, "total": 4,
                "description": lg_node,
            })

        # LLM 스트리밍 → delta SSE
        if ev_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk:
                content = getattr(chunk, "content", "")
                if content:
                    final_buf.append(content)
                    await sse_callback("delta", {"delta": content})

    # interrupt 확인 → awaiting_input SSE
    final_state = graph.get_state(config)
    if final_state.tasks:
        for task in final_state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_data = task.interrupts[0].value
                if isinstance(interrupt_data, dict):
                    await sse_callback("awaiting_input", interrupt_data)

    await sse_callback("done", {"ok": True, "final": "".join(final_buf)})
