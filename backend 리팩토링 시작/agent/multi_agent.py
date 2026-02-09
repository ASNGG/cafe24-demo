"""
agent/multi_agent.py - LangGraph 기반 멀티 에이전트 시스템
==========================================================
카페24 AI 기반 내부 시스템

구조:
- Coordinator (라우터): 사용자 질의 분석 및 적절한 에이전트로 라우팅
- Search Agent: 플랫폼 정보 검색 (쇼핑몰, 카테고리, RAG)
- Analysis Agent: 셀러 분석, ML 예측, 통계
- CS Agent: CS 응답 생성, 품질 평가

에이전트 간 협업:
- 검색 + 분석이 필요한 경우 순차 실행
- 예: "S0001 쇼핑몰 셀러의 이탈 예측" → Search → Analysis
"""
import json
import operator
from typing import TypedDict, Annotated, Sequence, Literal, Any, List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

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

from agent.tool_schemas import (
    SEARCH_AGENT_TOOLS,
    ANALYSIS_AGENT_TOOLS,
    TRANSLATION_AGENT_TOOLS,
    ALL_TOOLS,
)
from agent.llm import get_llm, pick_api_key
from agent.intent import detect_intent
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


def coordinator_node(state: AgentState, llm) -> dict:
    """코디네이터: 다음 에이전트 결정"""
    messages = state["messages"]

    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    intents = detect_intent(user_message)
    iteration = state.get("iteration", 0)

    if iteration >= 3:
        return {"next_agent": "end", "iteration": iteration + 1}

    if intents.get("want_translate"):
        return {"next_agent": "translation", "iteration": iteration + 1}
    elif intents.get("want_analytics"):
        return {"next_agent": "analysis", "iteration": iteration + 1}
    elif intents.get("want_rag") or intents.get("want_cookie") or intents.get("want_kingdom"):
        return {"next_agent": "search", "iteration": iteration + 1}
    else:
        return {"next_agent": "search", "iteration": iteration + 1}


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


def tool_executor_node(state: AgentState) -> dict:
    """도구 실행 노드"""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "next_agent": "end"}

    tool_calls_log = state.get("tool_calls_log", [])
    new_messages = []
    tool_map = {t.name: t for t in ALL_TOOLS}

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
                result = {"status": "FAILED", "error": safe_str(e)}
                st.logger.exception("TOOL_EXEC_FAIL tool=%s err=%s", tool_name, e)
        else:
            result = {"status": "FAILED", "error": f"도구 '{tool_name}'을 찾을 수 없습니다."}

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
    """멀티 에이전트 그래프 생성"""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph가 설치되지 않았습니다. 'pip install langgraph'")

    workflow = StateGraph(AgentState)

    workflow.add_node("coordinator", lambda state: coordinator_node(state, llm))
    workflow.add_node("search", lambda state: search_agent_node(state, llm))
    workflow.add_node("analysis", lambda state: analysis_agent_node(state, llm))
    workflow.add_node("translation", lambda state: translation_agent_node(state, llm))
    workflow.add_node("tools", tool_executor_node)

    workflow.set_entry_point("coordinator")

    workflow.add_conditional_edges(
        "coordinator",
        route_to_agent,
        {"search": "search", "analysis": "analysis", "translation": "translation", "end": END}
    )

    for agent in ["search", "analysis", "translation"]:
        workflow.add_conditional_edges(
            agent,
            should_continue,
            {"tools": "tools", "end": END, "coordinator": "coordinator"}
        )

    workflow.add_conditional_edges(
        "tools",
        lambda state: state.get("current_agent", "coordinator"),
        {"search": "search", "analysis": "analysis", "translation": "translation", "coordinator": "coordinator"}
    )

    return workflow.compile()


# ============================================================
# 멀티 에이전트 실행
# ============================================================
def run_multi_agent(req, username: str) -> dict:
    """LangGraph 기반 멀티 에이전트 실행"""
    if not LANGGRAPH_AVAILABLE:
        return {
            "status": "FAILED",
            "response": "langgraph를 설치하세요: pip install langgraph",
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
        return {"status": "FAILED", "response": msg, "tool_calls": [], "log_file": st.LOG_FILE}

    try:
        user_temperature = req.temperature if req.temperature is not None else 0.3
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=user_temperature, top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        graph = build_multi_agent_graph(llm)

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
            "status": "SUCCESS",
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
            "status": "FAILED",
            "response": msg if req.debug else "처리 오류가 발생했습니다.",
            "tool_calls": [],
            "log_file": st.LOG_FILE,
            "debug_error": err if req.debug else None,
        }


# ============================================================
# 레거시 호환 (기존 API 유지)
# ============================================================
class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    TRANSLATOR = "translator"
    ANALYST = "analyst"
    SEARCHER = "searcher"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    task_id: str
    agent_type: AgentType
    description: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class MultiAgentSystem:
    """레거시 호환용 멀티 에이전트 시스템"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.OPENAI_API_KEY
        self.tasks: Dict[str, AgentTask] = {}
        self._task_counter = 0

    def route_request(self, user_request: str, input_data: Optional[Dict] = None) -> Dict[str, Any]:
        """요청 라우팅 (레거시 호환)"""
        from dataclasses import dataclass as dc

        @dc
        class MockReq:
            user_input: str = user_request
            model: str = "gpt-4o-mini"
            api_key: str = self.api_key
            max_tokens: int = 2000
            temperature: float = 0.3
            top_p: float = None
            presence_penalty: float = None
            frequency_penalty: float = None
            seed: int = None
            timeout_ms: int = None
            retries: int = 3
            debug: bool = False

        result = run_multi_agent(MockReq(), "system")
        return result


_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


def process_request(user_request: str, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """레거시 호환"""
    system = get_multi_agent_system()
    return system.route_request(user_request, input_data)
