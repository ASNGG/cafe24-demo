"""
agent/multi_agent.py - LangGraph Supervisor 기반 멀티 에이전트 시스템
====================================================================
카페24 AI 기반 내부 시스템

구조 (langgraph-supervisor 패턴):
- Supervisor: 사용자 질의 분석 → 전문 워커 에이전트에게 위임 → 결과 종합
- 멀티에이전트 Supervisor (8종 워커): churn_analyst, retention_strategist,
  seller_analyst, performance_analyst, fraud_investigator, cs_quality_analyst,
  report_writer, platform_searcher

프로덕션 경로:
- run_multi_agent_stream() → build_multi_agent_supervisor() (SSE 스트리밍)
"""
import json
import time
from typing import Any, Dict
from pathlib import Path

try:
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    create_react_agent = None

try:
    from langgraph_supervisor import create_supervisor
    SUPERVISOR_AVAILABLE = True
except ImportError:
    create_supervisor = None
    SUPERVISOR_AVAILABLE = False

from agent.tools import (
    # 워커용 개별 도구 임포트
    get_at_risk_sellers,
    predict_seller_churn,
    get_churn_prediction,
    generate_retention_message,
    execute_retention_action,
    analyze_seller,
    get_seller_segment,
    detect_fraud,
    get_segment_statistics,
    get_fraud_statistics,
    get_seller_activity_report,
    get_shop_info,
    get_shop_performance,
    get_trend_analysis,
    get_cohort_analysis,
    predict_shop_revenue,
    get_gmv_prediction,
    optimize_marketing,
    get_order_statistics,
    get_dashboard_summary,
    get_cs_statistics,
    auto_reply_cs,
    check_cs_quality,
    classify_inquiry,
    get_ecommerce_glossary,
    search_platform,
    search_platform_lightrag,
    list_shops,
    get_shop_services,
    get_category_info,
    list_categories,
)

from agent.llm import get_llm, pick_api_key
from core.utils import safe_str, format_openai_error, normalize_model_name
from core.memory import append_memory, memory_messages
import state as st


# ============================================================
# 프롬프트 JSON 로드
# ============================================================
_PROMPTS_PATH = Path(__file__).parent / "multi_agent_prompts.json"


def _load_prompts():
    """multi_agent_prompts.json에서 프롬프트 로드"""
    with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_PROMPTS = _load_prompts()

# ============================================================
# Supervisor 워커 정의 (8종)
# ============================================================
MULTI_AGENT_WORKERS = {
    "churn_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["churn_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_at_risk_sellers, predict_seller_churn, get_churn_prediction],
        "description": "이탈 분석 전문가 — ML 이탈 예측 + SHAP 분석",
    },
    "retention_strategist": {
        "prompt": _PROMPTS["multi_agent_workers"]["retention_strategist"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [generate_retention_message, execute_retention_action, get_at_risk_sellers],
        "description": "리텐션 전략가 — 맞춤 메시지 생성 + 자동 조치",
    },
    "seller_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["seller_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [analyze_seller, get_seller_segment, detect_fraud, get_segment_statistics, get_fraud_statistics, get_seller_activity_report],
        "description": "셀러 분석가 — 셀러 종합 분석 + 세그먼트 + 이상거래",
    },
    "performance_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["performance_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_shop_info, get_shop_performance, get_trend_analysis, get_cohort_analysis, predict_shop_revenue, get_gmv_prediction, optimize_marketing, get_order_statistics],
        "description": "성과 분석가 — 매출/KPI/마케팅 분석",
    },
    "fraud_investigator": {
        "prompt": _PROMPTS["multi_agent_workers"]["fraud_investigator"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [detect_fraud, get_fraud_statistics, analyze_seller],
        "description": "이상거래 조사관 — 부정행위 탐지 + 영향 분석",
    },
    "cs_quality_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["cs_quality_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_cs_statistics, auto_reply_cs, check_cs_quality, classify_inquiry, get_ecommerce_glossary],
        "description": "CS 품질 분석가 — CS 통계 + 자동 응답 + 품질 평가",
    },
    "report_writer": {
        "prompt": _PROMPTS["multi_agent_workers"]["report_writer"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_dashboard_summary, get_order_statistics, get_trend_analysis, get_cohort_analysis],
        "description": "리포트 작성가 — 대시보드 + KPI 종합 보고서",
    },
    "platform_searcher": {
        "prompt": _PROMPTS["multi_agent_workers"]["platform_searcher"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [search_platform, search_platform_lightrag, get_shop_info, list_shops, get_shop_services, get_category_info, list_categories, get_ecommerce_glossary],
        "description": "플랫폼 검색가 — RAG 지식 검색 + 쇼핑몰/카테고리 조회",
    },
}

# 멀티에이전트 Supervisor 시스템 프롬프트
MULTI_AGENT_SUPERVISOR_PROMPT = _PROMPTS["supervisors"]["multi_agent"]


# ============================================================
# 멀티에이전트 스트림 실행 (routes_agent.py에서 호출)
# Supervisor 패턴 — 동적 라우팅 + astream_events SSE 스트리밍
# ============================================================
async def run_multi_agent_stream(req, username: str, sse_callback, category=None):
    """Supervisor 기반 멀티에이전트 SSE 스트리밍

    Args:
        req: AgentRequest (user_input, model, api_key 등)
        username: 사용자명
        sse_callback: async callable(event_type: str, data: dict) -> None
            event_type: "agent_start" | "agent_end" | "tool_start" | "tool_end" |
                        "delta" | "done" | "error"
        category: IntentCategory (참고용, Supervisor가 자체 라우팅)
    """
    if not LANGGRAPH_AVAILABLE or not SUPERVISOR_AVAILABLE:
        await sse_callback("done", {
            "ok": False,
            "final": "langgraph / langgraph-supervisor를 설치하세요.",
            "tool_calls": [],
        })
        return

    user_text = safe_str(req.user_input)
    api_key = pick_api_key(req.api_key)
    if not api_key:
        await sse_callback("done", {"ok": False, "final": "OpenAI API Key가 없습니다.", "tool_calls": []})
        return

    st.logger.info("MULTI_AGENT_STREAM_START user=%s input=%s", username, user_text[:80])

    try:
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=True,
            temperature=req.temperature if req.temperature is not None else 0.3,
            top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        model_key = normalize_model_name(req.model)
        supervisor = get_cached_multi_supervisor(llm, model_key)

        # 메시지 히스토리 구성 (멀티턴 지원)
        prev_messages = memory_messages(username)
        input_messages = []
        for msg in prev_messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                input_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                input_messages.append({"role": "assistant", "content": content})
        input_messages.append({"role": "user", "content": user_text})

        # SSE 스트리밍 상태 추적
        tool_calls_log = []
        agent_results = []
        final_buf = []
        current_agent = None
        step_start_time = None
        current_tool = None
        agents_used_set = set()
        tool_fail_counts: Dict[str, int] = {}  # 도구별 실패 횟수 추적
        MAX_TOOL_RETRIES = 3  # 동일 도구 최대 재시도 횟수

        # 워커 이름 집합 (handoff 이벤트 감지용)
        worker_names = set(MULTI_AGENT_WORKERS.keys())

        async for event in supervisor.astream_events(
            {"messages": input_messages},
            version="v2",
            config={"recursion_limit": 40},
        ):
            kind = event.get("event", "")
            data = event.get("data", {})
            metadata = event.get("metadata", {})

            # langgraph_checkpoint_ns 에서 현재 노드 추출
            checkpoint_ns = metadata.get("langgraph_checkpoint_ns", "")
            outer_node = checkpoint_ns.split(":")[0] if checkpoint_ns else ""

            # --- handoff 감지: transfer_to_<worker> 도구 호출 ---
            if kind == "on_tool_start":
                tool_name = event.get("name", "")
                tool_input = data.get("input", {})

                if tool_name.startswith("transfer_to_"):
                    # handoff → agent_start
                    agent_name = tool_name.replace("transfer_to_", "")
                    if agent_name in worker_names:
                        # 이전 워커가 있으면 먼저 종료
                        if current_agent and current_agent != agent_name:
                            elapsed_ms = int((time.time() - step_start_time) * 1000) if step_start_time else 0
                            await sse_callback("agent_end", {
                                "agent": current_agent,
                                "elapsed_ms": elapsed_ms,
                                "description": AGENT_DESCRIPTIONS.get(current_agent, current_agent),
                            })
                        current_agent = agent_name
                        step_start_time = time.time()
                        agents_used_set.add(agent_name)
                        await sse_callback("agent_start", {
                            "agent": agent_name,
                            "description": AGENT_DESCRIPTIONS.get(agent_name, agent_name),
                            "step_detail": MULTI_AGENT_WORKERS.get(agent_name, {}).get("prompt", "")[:100],
                        })
                else:
                    # 일반 도구 호출
                    current_tool = tool_name
                    # 핵심 파라미터만 추출
                    key_params = {}
                    for k in ("seller_id", "shop_id", "threshold", "top_n", "period",
                               "segment", "category", "query", "limit", "mode",
                               "risk_level", "action_type", "days"):
                        if k in tool_input:
                            key_params[k] = tool_input[k]
                    await sse_callback("tool_start", {
                        "tool": tool_name,
                        "agent": current_agent or "",
                        "args": key_params,
                    })

            elif kind == "on_tool_end":
                end_tool_name = event.get("name") or current_tool or "unknown"
                if end_tool_name.startswith("transfer_to_"):
                    pass  # handoff 종료는 스킵
                else:
                    tool_output = data.get("output", {})
                    # ToolMessage content 추출
                    if hasattr(tool_output, "content"):
                        content = tool_output.content
                        if isinstance(content, str):
                            try:
                                tool_output = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                tool_output = {"status": "success", "data": content}
                        elif isinstance(content, (dict, list)):
                            tool_output = content
                        else:
                            tool_output = {"status": "success", "data": safe_str(content)}
                    elif not isinstance(tool_output, (str, dict, list, int, float, bool, type(None))):
                        tool_output = {"status": "success", "data": safe_str(tool_output)}

                    # 결과 미리보기 생성
                    if isinstance(tool_output, dict):
                        result_preview = json.dumps(tool_output, ensure_ascii=False, default=str)[:200]
                    elif isinstance(tool_output, str):
                        result_preview = tool_output[:200]
                    else:
                        result_preview = str(tool_output)[:200]

                    # 도구 실패 횟수 추적
                    is_error = (isinstance(tool_output, dict) and tool_output.get("status") == "error")
                    if is_error:
                        tool_fail_counts[end_tool_name] = tool_fail_counts.get(end_tool_name, 0) + 1

                    tool_calls_log.append({
                        "tool": end_tool_name,
                        "agent": current_agent or "",
                        "result": tool_output,
                    })

                    await sse_callback("tool_end", {
                        "tool": end_tool_name,
                        "agent": current_agent or "",
                        "status": "error" if is_error else "success",
                        "result_preview": result_preview,
                    })
                    current_tool = None

                    # 동일 도구 3회 실패 시 스트림 강제 종료
                    if tool_fail_counts.get(end_tool_name, 0) >= MAX_TOOL_RETRIES:
                        st.logger.warning(
                            "TOOL_RETRY_LIMIT tool=%s fails=%d — aborting stream",
                            end_tool_name, tool_fail_counts[end_tool_name],
                        )
                        error_msg = f"도구 '{end_tool_name}'이(가) {MAX_TOOL_RETRIES}회 연속 실패하여 분석을 중단합니다."
                        final_buf.append(error_msg)
                        await sse_callback("delta", {"delta": "\n\n" + error_msg})
                        break  # astream_events 루프 탈출

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if not chunk or getattr(chunk, "tool_call_chunks", None):
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    # 워커 응답은 스킵 — supervisor 최종 종합 응답만 스트리밍
                    # (워커 진행 상황은 agent_start/end, tool_start/end로 표시)
                    if outer_node and outer_node != "sub_supervisor":
                        continue
                    final_buf.append(content)
                    await sse_callback("delta", {"delta": content})

            # --- 워커 → supervisor 복귀 감지 ---
            if (current_agent
                and outer_node == "sub_supervisor"
                and kind == "on_chat_model_start"):
                elapsed_ms = int((time.time() - step_start_time) * 1000) if step_start_time else 0
                summary = "".join(final_buf[-20:]) if final_buf else ""
                agent_results.append({
                    "agent": current_agent,
                    "summary": summary[:500],
                    "elapsed_ms": elapsed_ms,
                })
                await sse_callback("agent_end", {
                    "agent": current_agent,
                    "elapsed_ms": elapsed_ms,
                    "description": AGENT_DESCRIPTIONS.get(current_agent, current_agent),
                })
                current_agent = None
                step_start_time = None

        # 루프 종료 후: 마지막 워커가 agent_end 없이 끝난 경우
        if current_agent:
            elapsed_ms = int((time.time() - step_start_time) * 1000) if step_start_time else 0
            agent_results.append({
                "agent": current_agent,
                "summary": "".join(final_buf[-20:])[:500] if final_buf else "",
                "elapsed_ms": elapsed_ms,
            })
            await sse_callback("agent_end", {
                "agent": current_agent,
                "elapsed_ms": elapsed_ms,
                "description": AGENT_DESCRIPTIONS.get(current_agent, current_agent),
            })

        # 최종 응답
        final_response = "".join(final_buf).strip() or "멀티에이전트 처리를 완료했습니다."
        append_memory(username, user_text, final_response)

        agents_used = list(agents_used_set)
        await sse_callback("done", {
            "ok": True,
            "final": final_response,
            "tool_calls": tool_calls_log,
            "agents_used": agents_used,
            "agent_results": agent_results,
        })

        st.logger.info(
            "MULTI_AGENT_STREAM_COMPLETE user=%s agents=%s tools=%d",
            username, agents_used, len(tool_calls_log),
        )

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("MULTI_AGENT_STREAM_FAIL err=%s", err)
        msg = f"멀티에이전트 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        append_memory(username, user_text, msg)
        await sse_callback("done", {
            "ok": False,
            "final": msg if req.debug else "멀티에이전트 처리 오류가 발생했습니다.",
            "tool_calls": [],
        })


# 에이전트 설명 헬퍼
AGENT_DESCRIPTIONS = {_wname: _wcfg["description"] for _wname, _wcfg in MULTI_AGENT_WORKERS.items()}


# ============================================================
# 멀티에이전트 Supervisor 빌드 + 캐시
# ============================================================
_multi_supervisor_cache: Dict[str, Any] = {}


def build_multi_agent_supervisor(llm):
    """멀티에이전트용 Supervisor 생성 — 8종 워커 동적 라우팅"""
    if not SUPERVISOR_AVAILABLE:
        raise ImportError("langgraph-supervisor가 설치되지 않았습니다. 'pip install langgraph-supervisor'")

    workers = []
    for name, config in MULTI_AGENT_WORKERS.items():
        agent = create_react_agent(
            model=llm,
            tools=config["tools"],
            name=name,
            prompt=config["prompt"],
        )
        workers.append(agent)

    workflow = create_supervisor(
        agents=workers,
        model=llm,
        prompt=MULTI_AGENT_SUPERVISOR_PROMPT,
        output_mode="full_history",
        supervisor_name="sub_supervisor",
        add_handoff_messages=True,
    )

    return workflow.compile()


def get_cached_multi_supervisor(llm, model_key: str):
    """모델별 멀티에이전트 Supervisor 그래프 캐시"""
    if model_key not in _multi_supervisor_cache:
        _multi_supervisor_cache[model_key] = build_multi_agent_supervisor(llm)
        st.logger.info("MULTI_SUPERVISOR_GRAPH_BUILD model=%s (cached)", model_key)
    return _multi_supervisor_cache[model_key]
