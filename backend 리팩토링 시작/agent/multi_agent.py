"""
agent/multi_agent.py - LangGraph 기반 멀티 에이전트 시스템
==========================================================
카페24 AI 기반 내부 시스템

구조:
- Coordinator (라우터): 사용자 질의 분석 및 적절한 에이전트로 라우팅
- Search Agent: 플랫폼 정보 검색 (쇼핑몰, 카테고리, RAG)
- Analysis Agent: 셀러 분석, ML 예측, 통계
- CS Agent: CS 응답 생성, 품질 평가
- Sub-Agent (서브에이전트): 복합 리텐션 요청 오케스트레이션

에이전트 간 협업:
- 검색 + 분석이 필요한 경우 순차 실행
- 예: "S0001 쇼핑몰 셀러의 이탈 예측" → Search → Analysis
- 복합 리텐션 요청 → Sub-Agent Coordinator → Dispatcher → Retention 사이클
"""
import json
import operator
from typing import TypedDict, Annotated, Sequence, Literal, Any, List, Optional, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    ToolNode = None

from agent.tools import (
    SEARCH_AGENT_TOOLS,
    ANALYSIS_AGENT_TOOLS,
    TRANSLATION_AGENT_TOOLS,
    ALL_TOOLS,
)

try:
    from agent.tools import RETENTION_AGENT_TOOLS
except ImportError:
    RETENTION_AGENT_TOOLS = []
from agent.llm import get_llm, pick_api_key
from agent.router import _keyword_classify, IntentCategory
from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str, format_openai_error, normalize_model_name, json_sanitize
from core.memory import append_memory, memory_messages
import state as st


# ============================================================
# 상태 정의
# ============================================================
class AgentState(TypedDict):
    """멀티 에이전트 그래프의 상태"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    current_agent: str
    tool_calls_log: List[dict]
    iteration: int
    final_response: str
    # 서브에이전트 오케스트레이션 필드
    plan: List[str]            # 서브에이전트 실행 순서
    current_step: int          # 현재 단계 인덱스
    agent_results: List[dict]  # 단계별 결과 누적


# ============================================================
# 에이전트 프롬프트
# ============================================================
COORDINATOR_PROMPT = """당신은 카페24 AI 운영 플랫폼의 코디네이터입니다.
사용자 질의를 분석하여 적절한 전문 에이전트에게 작업을 할당합니다.

## 전문 에이전트:
1. **Search Agent**: 쇼핑몰/카테고리 정보, 플랫폼 RAG 검색
2. **Analysis Agent**: 셀러 분석, 이탈 예측, 이상거래 탐지, KPI 분석
3. **CS Agent**: CS 응답 생성, 품질 평가, 용어집

질의를 분석하고 가장 적합한 에이전트에게 작업을 할당하세요.
"""

SEARCH_AGENT_PROMPT = """당신은 카페24 AI 운영 플랫폼의 **검색 전문가**입니다.

## 담당 업무:
- 쇼핑몰 정보 조회 (get_shop_info, list_shops, get_shop_services)
- 카테고리 정보 조회 (get_category_info, list_categories)
- 플랫폼 지식 RAG 검색 (search_platform_docs, search_platform_lightrag)

## 검색 규칙:
- 쇼핑몰 이름/ID 언급 → get_shop_info
- 쇼핑몰 목록 요청 → list_shops
- 서비스 정보 → get_shop_services
- 카테고리 정보 → get_category_info, list_categories
- 플랫폼 지식 → search_platform_docs 또는 search_platform_lightrag

검색 결과를 바탕으로 정확한 정보를 제공하세요.
검색 결과에 없는 정보는 지어내지 마세요.
"""

ANALYSIS_AGENT_PROMPT = """당신은 카페24 AI 운영 플랫폼의 **분석 전문가**입니다.

## 담당 업무:
- 셀러 분석 (analyze_seller, get_seller_segment, detect_fraud)
- 이탈 예측 (predict_seller_churn, get_churn_prediction) - SHAP 해석 포함
- 이상거래 탐지 (get_segment_statistics, get_fraud_statistics)
- 매출 예측 (predict_shop_revenue, get_shop_performance)
- 마케팅 최적화 (optimize_marketing) - P-PSO 알고리즘
- KPI 분석 (get_trend_analysis, get_cohort_analysis, get_gmv_prediction)
- 대시보드 (get_dashboard_summary)

## 분석 규칙:
- 특정 셀러 분석 → analyze_seller(seller_id)
- 이탈 예측 → predict_seller_churn(seller_id)
- 전체 이탈 현황 → get_churn_prediction()
- 세그먼트 통계 → get_segment_statistics()
- 이상거래 탐지 → get_fraud_statistics()
- 쇼핑몰 성과 → get_shop_performance(shop_id)
- 마케팅 추천 → optimize_marketing(seller_id)

분석 결과와 실행 가능한 인사이트를 제공하세요.
"""

TRANSLATION_AGENT_PROMPT = """당신은 카페24 AI 운영 플랫폼의 **CS 전문가**입니다.

## 담당 업무:
- CS 자동 응답 생성 (auto_reply_cs)
- CS 응답 품질 평가 (check_cs_quality)
- 이커머스 용어집 (get_ecommerce_glossary)
- CS 통계 (get_cs_statistics)
- 문의 분류 (classify_inquiry)

## CS 카테고리:
배송, 환불, 결제, 상품, 계정

## CS 규칙:
- 고객 불만에 공감하며 전문적으로 응대
- 정확한 정책 정보 기반 안내
- 긴급 문의 우선 처리

CS 응답 결과와 품질 평가를 제공하세요.
"""

RETENTION_AGENT_PROMPT = """당신은 카페24 AI 운영 플랫폼의 **셀러 리텐션 전문가**입니다.

## 담당 업무:
- 이탈 위험 셀러 조회 (get_at_risk_sellers) - ML 이탈 예측 + SHAP 분석
- 맞춤 리텐션 메시지 생성 (generate_retention_message) - LLM 기반 메시지 작성
- 리텐션 조치 실행 (execute_retention_action) - 쿠폰, 업그레이드, 매니저 배정
- 셀러 상세 분석 (analyze_seller) - 셀러 정보 및 이탈 위험도 확인
- CS 통계 확인 (get_cs_statistics) - CS 현황 파악

## 분석 규칙:
- 이전 단계의 분석 결과를 반드시 참고하여 전략을 수립
- 이탈 위험 수준(고위험/중위험/저위험)에 따라 맞춤형 전략 제시
- CS 통계가 있으면 불만 패턴을 반영한 전략 수립
- 실행 가능한 구체적 액션을 포함 (action_type: coupon, upgrade_offer, manager_assign, custom_message)

이전 단계 결과를 종합하여 효과적인 리텐션 전략을 제시하세요.
"""

# 서브에이전트 복합 요청 감지용 키워드 패턴
_SUB_AGENT_PATTERNS = [
    ["이탈", "전략"],
    ["이탈", "CS"],
    ["리텐션", "분석"],
    ["이탈", "분석", "발송"],
    ["위험", "전략"],
    ["이탈", "확인", "전략"],
]


# ============================================================
# 에이전트 노드 함수
# ============================================================
def create_agent_executor(llm, tools, system_prompt: str):
    """에이전트 실행기 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm.bind_tools(tools)


def _is_sub_agent_request(text: str) -> bool:
    """복합 리텐션 요청 여부를 키워드 패턴으로 감지"""
    t = text.lower()
    for pattern in _SUB_AGENT_PATTERNS:
        if all(kw in t for kw in pattern):
            return True
    return False


def coordinator_node(state: AgentState, llm) -> dict:
    """코디네이터: 다음 에이전트 결정 (서브에이전트 복합 요청 포함)"""
    messages = state["messages"]

    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    iteration = state.get("iteration", 0)

    if iteration >= 3:
        return {"next_agent": "end", "iteration": iteration + 1}

    # 복합 리텐션 요청 → 서브에이전트 오케스트레이션
    if _is_sub_agent_request(user_message) and RETENTION_AGENT_TOOLS:
        st.logger.info("COORDINATOR sub_agent detected: %s", user_message[:60])
        return {"next_agent": "sub_agent", "iteration": iteration + 1}

    category = _keyword_classify(user_message)

    # IntentCategory → 에이전트 매핑
    _CATEGORY_AGENT_MAP = {
        IntentCategory.CS: "translation",
        IntentCategory.ANALYSIS: "analysis",
        IntentCategory.SELLER: "analysis",
        IntentCategory.DASHBOARD: "analysis",
        IntentCategory.PLATFORM: "search",
        IntentCategory.SHOP: "search",
        IntentCategory.GENERAL: "search",
    }

    next_agent = _CATEGORY_AGENT_MAP.get(category, "search")
    return {"next_agent": next_agent, "iteration": iteration + 1}


def search_agent_node(state: AgentState, llm) -> dict:
    """검색 에이전트"""
    agent = create_agent_executor(llm, SEARCH_AGENT_TOOLS, SEARCH_AGENT_PROMPT)
    result = agent.invoke({"messages": state["messages"]})

    return {
        "messages": [result],
        "current_agent": "search",
        "next_agent": "end",
    }


def analysis_agent_node(state: AgentState, llm) -> dict:
    """분석 에이전트"""
    agent = create_agent_executor(llm, ANALYSIS_AGENT_TOOLS, ANALYSIS_AGENT_PROMPT)
    result = agent.invoke({"messages": state["messages"]})

    return {
        "messages": [result],
        "current_agent": "analysis",
        "next_agent": "end",
    }


def translation_agent_node(state: AgentState, llm) -> dict:
    """번역 에이전트"""
    agent = create_agent_executor(llm, TRANSLATION_AGENT_TOOLS, TRANSLATION_AGENT_PROMPT)
    result = agent.invoke({"messages": state["messages"]})

    return {
        "messages": [result],
        "current_agent": "translation",
        "next_agent": "end",
    }


# ============================================================
# 서브에이전트 오케스트레이션 노드
# ============================================================
def sub_agent_coordinator_node(state: AgentState, llm) -> dict:
    """복합 요청을 분석하여 실행 계획(plan)을 생성"""
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 키워드 기반 plan 생성 (LLM 호출 없이 빠르게 결정)
    t = user_message.lower()
    plan = []

    # 항상 이탈 위험 분석부터
    plan.append("analyze_churn")

    # CS 관련 키워드가 있으면 CS 확인 단계 추가
    if any(kw in t for kw in ["cs", "상담", "문의", "불만"]):
        plan.append("check_cs")

    # 전략/발송/조치 관련 키워드 → 전략 생성
    plan.append("generate_strategy")

    # 실행/발송/자동/조치 → 액션 실행
    if any(kw in t for kw in ["실행", "발송", "자동", "조치", "액션"]):
        plan.append("execute_action")

    st.logger.info("SUB_AGENT_PLAN plan=%s steps=%d", plan, len(plan))

    return {
        "plan": plan,
        "current_step": 0,
        "agent_results": [],
        "current_agent": "sub_agent_coordinator",
    }


def dispatcher_node(state: AgentState) -> dict:
    """plan[current_step]에 따라 다음 서브에이전트 노드로 라우팅"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan):
        # 모든 단계 완료
        return {"next_agent": "end"}

    step_name = plan[current_step]
    st.logger.info(
        "DISPATCHER step=%d/%d action=%s",
        current_step + 1, len(plan), step_name,
    )

    return {"next_agent": step_name, "current_agent": "dispatcher"}


def _dispatch_route(state: AgentState) -> str:
    """dispatcher 조건부 엣지: plan의 현재 단계에 따라 라우팅"""
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    if step >= len(plan):
        return "end"
    return plan[step]


def retention_agent_node(state: AgentState, llm) -> dict:
    """리텐션 에이전트: RETENTION_AGENT_TOOLS로 이탈 방지 작업 수행"""
    # 이전 단계 결과를 컨텍스트로 포함
    agent_results = state.get("agent_results", [])
    context_parts = []
    for i, res in enumerate(agent_results):
        context_parts.append(f"[단계 {i+1} 결과] {res.get('step', '')}: {res.get('summary', '')}")
    context_str = "\n".join(context_parts) if context_parts else "첫 번째 단계입니다."

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_name = plan[current_step] if current_step < len(plan) else "unknown"

    # 원본 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    augmented_prompt = (
        f"{RETENTION_AGENT_PROMPT}\n\n"
        f"## 현재 단계: {step_name} ({current_step + 1}/{len(plan)})\n"
        f"## 이전 단계 결과:\n{context_str}\n\n"
        f"사용자 요청: {user_message}"
    )

    tools = RETENTION_AGENT_TOOLS if RETENTION_AGENT_TOOLS else ANALYSIS_AGENT_TOOLS
    agent = create_agent_executor(llm, tools, augmented_prompt)
    result = agent.invoke({"messages": state["messages"]})

    # 결과를 agent_results에 누적
    new_results = list(agent_results)
    result_summary = result.content[:200] if hasattr(result, "content") and result.content else ""
    new_results.append({"step": step_name, "summary": result_summary})

    return {
        "messages": [result],
        "current_agent": "retention",
        "agent_results": new_results,
        "current_step": current_step + 1,
    }


def _retention_should_continue(state: AgentState) -> str:
    """리텐션 에이전트 후 분기: tool_calls가 있으면 tools, 없으면 dispatcher로 복귀"""
    messages = state["messages"]
    if not messages:
        return "dispatcher"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "dispatcher"


def tool_executor_node(state: AgentState) -> dict:
    """도구 실행 노드"""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "next_agent": "end"}

    tool_calls_log = state.get("tool_calls_log", [])
    new_messages = []
    tool_map = {t.name: t for t in ALL_TOOLS}
    if RETENTION_AGENT_TOOLS:
        tool_map.update({t.name: t for t in RETENTION_AGENT_TOOLS})

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        st.logger.info(
            "MULTI_AGENT_TOOL_CALL agent=%s tool=%s args=%s",
            state.get("current_agent", "unknown"),
            tool_name,
            json.dumps(tool_args, ensure_ascii=False),
        )

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = {"status": "error", "message": safe_str(e)}
                st.logger.exception("TOOL_EXEC_FAIL tool=%s err=%s", tool_name, e)
        else:
            result = {"status": "error", "message": f"도구 '{tool_name}'을 찾을 수 없습니다."}

        tool_calls_log.append({
            "agent": state.get("current_agent", "unknown"),
            "tool": tool_name,
            "args": tool_args,
            "result": result,
        })

        try:
            result_str = json.dumps(json_sanitize(result), ensure_ascii=False)
        except Exception:
            result_str = safe_str(result)

        new_messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

    return {"messages": new_messages, "tool_calls_log": tool_calls_log}


def should_continue(state: AgentState) -> Literal["tools", "end", "coordinator"]:
    """조건부 엣지: 다음 단계 결정"""
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return state.get("next_agent", "end") if state.get("next_agent") != "end" else "end"


def route_to_agent(state: AgentState) -> str:
    """에이전트 라우팅"""
    return state.get("next_agent", "end")


# ============================================================
# 그래프 빌드
# ============================================================
def build_multi_agent_graph(llm):
    """멀티 에이전트 그래프 생성 (서브에이전트 경로 포함)"""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph가 설치되지 않았습니다. 'pip install langgraph'")

    workflow = StateGraph(AgentState)

    # 기존 노드
    workflow.add_node("coordinator", lambda state: coordinator_node(state, llm))
    workflow.add_node("search", lambda state: search_agent_node(state, llm))
    workflow.add_node("analysis", lambda state: analysis_agent_node(state, llm))
    workflow.add_node("translation", lambda state: translation_agent_node(state, llm))
    workflow.add_node("tools", tool_executor_node)

    # 서브에이전트 노드
    workflow.add_node("sub_agent_coordinator", lambda state: sub_agent_coordinator_node(state, llm))
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("retention", lambda state: retention_agent_node(state, llm))

    workflow.set_entry_point("coordinator")

    # coordinator → 에이전트 라우팅 (sub_agent 경로 추가)
    workflow.add_conditional_edges(
        "coordinator",
        route_to_agent,
        {
            "search": "search",
            "analysis": "analysis",
            "translation": "translation",
            "sub_agent": "sub_agent_coordinator",
            "end": END,
        }
    )

    # 기존 에이전트 → tools / end / coordinator
    for agent in ["search", "analysis", "translation"]:
        workflow.add_conditional_edges(
            agent,
            should_continue,
            {"tools": "tools", "end": END, "coordinator": "coordinator"}
        )

    # tools → 호출한 에이전트로 복귀 (retention 추가)
    workflow.add_conditional_edges(
        "tools",
        lambda state: state.get("current_agent", "coordinator"),
        {
            "search": "search",
            "analysis": "analysis",
            "translation": "translation",
            "retention": "retention",
            "coordinator": "coordinator",
        }
    )

    # 서브에이전트 경로: coordinator → sub_agent_coordinator → dispatcher → retention → tools → retention → dispatcher → END
    workflow.add_edge("sub_agent_coordinator", "dispatcher")

    # dispatcher → 각 단계별 에이전트 or END
    _dispatch_targets = {
        "analyze_churn": "retention",
        "check_cs": "retention",
        "generate_strategy": "retention",
        "execute_action": "retention",
        "end": END,
    }
    workflow.add_conditional_edges("dispatcher", _dispatch_route, _dispatch_targets)

    # retention → tools(도구 호출) or dispatcher(다음 단계)
    workflow.add_conditional_edges(
        "retention",
        _retention_should_continue,
        {"tools": "tools", "dispatcher": "dispatcher"}
    )

    return workflow.compile()


# ============================================================
# H15: 모델별 그래프 캐시 (매 요청마다 재빌드 방지)
# ============================================================
_graph_cache: Dict[str, Any] = {}


def _get_cached_graph(llm, model_key: str):
    """모델별로 컴파일된 LangGraph를 캐시하여 반환"""
    if model_key not in _graph_cache:
        _graph_cache[model_key] = build_multi_agent_graph(llm)
        st.logger.info("MULTI_AGENT_GRAPH_BUILD model=%s (cached)", model_key)
    return _graph_cache[model_key]


# ============================================================
# 멀티 에이전트 실행
# ============================================================
def run_multi_agent(req, username: str) -> dict:
    """LangGraph 기반 멀티 에이전트 실행"""
    if not LANGGRAPH_AVAILABLE:
        return {
            "status": "error",
            "message": "langgraph를 설치하세요: pip install langgraph",
            "tool_calls": [],
            "log_file": st.LOG_FILE,
        }

    user_text = safe_str(req.user_input)

    st.logger.info(
        "MULTI_AGENT_START user=%s model=%s input_len=%s",
        username, normalize_model_name(req.model), len(user_text),
    )

    api_key = pick_api_key(req.api_key)
    if not api_key:
        msg = "OpenAI API Key가 없습니다."
        append_memory(username, user_text, msg)
        return {"status": "error", "message": msg, "tool_calls": [], "log_file": st.LOG_FILE}

    try:
        user_temperature = req.temperature if req.temperature is not None else 0.3
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=user_temperature, top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        # H15: 모델별로 컴파일된 그래프 캐시
        graph = _get_cached_graph(llm, normalize_model_name(req.model))

        prev_messages = memory_messages(username)
        messages = []
        for msg in prev_messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_text))

        initial_state = {
            "messages": messages,
            "next_agent": "",
            "current_agent": "",
            "tool_calls_log": [],
            "iteration": 0,
            "final_response": "",
            "plan": [],
            "current_step": 0,
            "agent_results": [],
        }

        final_state = graph.invoke(initial_state)

        final_response = ""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

        if not final_response:
            final_response = "요청을 처리했습니다."

        append_memory(username, user_text, final_response)
        tool_calls_log = final_state.get("tool_calls_log", [])

        agents_used = list(set(tc.get("agent", "") for tc in tool_calls_log))

        st.logger.info(
            "MULTI_AGENT_COMPLETE user=%s agents=%s tools=%s",
            username, agents_used, len(tool_calls_log),
        )

        return {
            "status": "success",
            "response": final_response,
            "tool_calls": tool_calls_log,
            "log_file": st.LOG_FILE,
            "mode": "multi_agent",
            "agents_used": agents_used,
        }

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("MULTI_AGENT_FAIL err=%s", err)

        msg = f"처리 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        append_memory(username, user_text, msg)

        return {
            "status": "error",
            "message": msg if req.debug else "처리 오류가 발생했습니다.",
            "tool_calls": [],
            "log_file": st.LOG_FILE,
            "debug_error": err if req.debug else None,
        }


# ============================================================
# 서브에이전트 스트림 실행 (routes_agent.py에서 호출)
# ============================================================
async def run_sub_agent_stream(req, username: str, sse_callback):
    """서브에이전트 스트리밍 실행 - 각 단계마다 SSE 이벤트 전송

    Args:
        req: AgentRequest (user_input, model, api_key 등)
        username: 사용자명
        sse_callback: async callable(event_type: str, data: dict) -> None
            event_type: "agent_start" | "agent_end" | "tool_start" | "tool_end" | "delta" | "done"
    """
    if not LANGGRAPH_AVAILABLE:
        await sse_callback("done", {"ok": False, "final": "langgraph를 설치하세요.", "tool_calls": []})
        return

    user_text = safe_str(req.user_input)
    api_key = pick_api_key(req.api_key)
    if not api_key:
        await sse_callback("done", {"ok": False, "final": "OpenAI API Key가 없습니다.", "tool_calls": []})
        return

    st.logger.info("SUB_AGENT_STREAM_START user=%s input=%s", username, user_text[:80])

    try:
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=req.temperature if req.temperature is not None else 0.3,
            top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        graph = _get_cached_graph(llm, normalize_model_name(req.model))

        prev_messages = memory_messages(username)
        messages = []
        for msg in prev_messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_text))

        initial_state = {
            "messages": messages,
            "next_agent": "",
            "current_agent": "",
            "tool_calls_log": [],
            "iteration": 0,
            "final_response": "",
            "plan": [],
            "current_step": 0,
            "agent_results": [],
        }

        # 동기 실행 후 단계별 결과 전송
        final_state = graph.invoke(initial_state)

        plan = final_state.get("plan", [])
        agent_results = final_state.get("agent_results", [])
        tool_calls_log = final_state.get("tool_calls_log", [])

        # 각 단계 결과를 SSE로 전송
        for i, result in enumerate(agent_results):
            step_name = result.get("step", f"step_{i}")
            await sse_callback("agent_start", {
                "agent": step_name,
                "step": i + 1,
                "total_steps": len(plan),
            })
            await sse_callback("agent_end", {
                "agent": step_name,
                "step": i + 1,
                "total_steps": len(plan),
                "summary": result.get("summary", ""),
            })

        # 도구 호출 로그 전송
        for tc in tool_calls_log:
            await sse_callback("tool_end", {
                "tool": tc.get("tool", ""),
                "agent": tc.get("agent", ""),
                "status": "success",
            })

        # 최종 응답 추출
        final_response = final_state.get("final_response", "")
        if not final_response:
            for msg in reversed(final_state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_response = msg.content
                    break

        if not final_response:
            final_response = "서브에이전트 처리를 완료했습니다."

        # 최종 응답을 delta로 전송
        await sse_callback("delta", {"delta": final_response})

        append_memory(username, user_text, final_response)
        agents_used = list(set(tc.get("agent", "") for tc in tool_calls_log))

        await sse_callback("done", {
            "ok": True,
            "final": final_response,
            "tool_calls": tool_calls_log,
            "agents_used": agents_used,
            "plan": plan,
            "agent_results": agent_results,
        })

        st.logger.info(
            "SUB_AGENT_STREAM_COMPLETE user=%s steps=%d tools=%d",
            username, len(agent_results), len(tool_calls_log),
        )

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("SUB_AGENT_STREAM_FAIL err=%s", err)
        msg = f"서브에이전트 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        append_memory(username, user_text, msg)
        await sse_callback("done", {
            "ok": False,
            "final": msg if req.debug else "서브에이전트 처리 오류가 발생했습니다.",
            "tool_calls": [],
        })


# M19/I7: 레거시 호환 코드 제거 (AgentType, TaskStatus, MultiAgentSystem 등)
