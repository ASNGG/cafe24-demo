"""
agent/multi_agent.py - LangGraph Supervisor 기반 멀티 에이전트 시스템
====================================================================
카페24 AI 기반 내부 시스템

구조 (langgraph-supervisor 패턴):
- Supervisor: 사용자 질의 분석 → 전문 워커 에이전트에게 위임 → 결과 종합
- 멀티에이전트 Supervisor (7종 워커): churn_analyst, retention_strategist,
  seller_analyst, performance_analyst, cs_quality_analyst,
  report_writer, platform_searcher

프로덕션 경로:
- run_multi_agent_stream() → build_multi_agent_supervisor() (SSE 스트리밍)
"""
import json
import time
import contextvars
from typing import Any, Dict
from pathlib import Path

try:
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    create_react_agent = None
    ChatOpenAI = None

try:
    from langgraph_supervisor import create_supervisor
    from langgraph_supervisor.handoff import create_forward_message_tool
    SUPERVISOR_AVAILABLE = True
except ImportError:
    create_supervisor = None
    create_forward_message_tool = None
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
    analyze_data,
)

from agent.llm import get_llm, pick_api_key
from core.utils import safe_str, format_openai_error, normalize_model_name
from core.memory import append_memory, memory_messages
import state as st


# ============================================================
# 프롬프트 YAML 로드 (JSON에서 전환 — 가독성/편집 용이성 향상)
# ============================================================
_PROMPTS_PATH = Path(__file__).parent / "multi_agent_prompts.yaml"


def _load_prompts():
    """multi_agent_prompts.yaml에서 프롬프트 로드"""
    import yaml
    with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_PROMPTS = _load_prompts()

# ============================================================
# 도구 호출 횟수 제한 (요청 단위, contextvars 기반)
# ============================================================
_request_tool_counts = contextvars.ContextVar('tool_counts', default=None)
_MAX_TOOL_INVOKE = 2  # 동일 도구 최대 호출 횟수 (초과 시 즉시 차단 메시지 리턴)


def _wrap_tools_with_limit(tools):
    """각 도구의 func를 래핑하여 호출 횟수 초과 시 즉시 차단 (LangChain 파이프라인 보존)"""
    for t in tools:
        if getattr(t, '_limited', False):
            continue  # 이미 래핑됨
        _orig_func = t.func
        _name = t.name

        def _make_limited(orig_func, tool_name):
            def limited_func(*args, **kwargs):
                counts = _request_tool_counts.get()
                if counts is not None:
                    counts[tool_name] = counts.get(tool_name, 0) + 1
                    if counts[tool_name] > _MAX_TOOL_INVOKE:
                        st.logger.info("TOOL_INVOKE_BLOCKED tool=%s calls=%d", tool_name, counts[tool_name])
                        return {
                            "status": "blocked",
                            "message": f"'{tool_name}' 도구는 이미 {_MAX_TOOL_INVOKE}회 호출되었습니다. 기존 결과를 활용하여 답변하세요."
                        }
                return orig_func(*args, **kwargs)
            return limited_func

        object.__setattr__(t, 'func', _make_limited(_orig_func, _name))
        object.__setattr__(t, '_limited', True)  # 이중 래핑 방지
    return tools


# ============================================================
# Supervisor 워커 정의 (7종) — fraud_investigator → seller_analyst 통합
# ============================================================
MULTI_AGENT_WORKERS = {
    "churn_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["churn_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_at_risk_sellers, predict_seller_churn, get_churn_prediction, get_seller_segment, get_segment_statistics],
        "description": "이탈 분석 전문가 — ML 이탈 예측 + SHAP 분석 + 세그먼트 교차 분석",
    },
    "retention_strategist": {
        "prompt": _PROMPTS["multi_agent_workers"]["retention_strategist"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [generate_retention_message, execute_retention_action, get_at_risk_sellers, analyze_seller],
        "description": "리텐션 전략가 — 맞춤 메시지 생성 + 자동 조치 실행",
    },
    "seller_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["seller_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [analyze_seller, get_seller_segment, detect_fraud, get_segment_statistics, get_fraud_statistics, get_seller_activity_report],
        "description": "셀러 분석가 — 셀러 종합 분석 + 세그먼트 + 이상거래 + CS 비율 분석",
    },
    "performance_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["performance_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_shop_info, get_shop_performance, get_trend_analysis, get_cohort_analysis, predict_shop_revenue, get_gmv_prediction, optimize_marketing, get_order_statistics, list_shops, analyze_data],
        "description": "성과 분석가 — 특정 쇼핑몰 성과/매출/마케팅/코호트 개별 분석",
    },
    "cs_quality_analyst": {
        "prompt": _PROMPTS["multi_agent_workers"]["cs_quality_analyst"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_cs_statistics, auto_reply_cs, check_cs_quality, classify_inquiry, analyze_data],
        "description": "CS 품질 분석가 — CS 통계 + 자동 응답 + 품질 평가",
    },
    "report_writer": {
        "prompt": _PROMPTS["multi_agent_workers"]["report_writer"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [get_dashboard_summary, get_order_statistics, get_trend_analysis, get_cohort_analysis, get_cs_statistics, get_segment_statistics, get_fraud_statistics, analyze_data],
        "description": "리포트 작성가 — 대시보드 + KPI + CS/셀러/이상거래 종합 보고서",
    },
    "platform_searcher": {
        "prompt": _PROMPTS["multi_agent_workers"]["platform_searcher"]["prompt"] + _PROMPTS["common_rules"],
        "tools": [search_platform, search_platform_lightrag, list_shops, get_shop_services, get_category_info, list_categories, get_ecommerce_glossary],
        "description": "플랫폼 검색가 — RAG 지식 검색 + 카테고리/서비스 조회",
    },
}

# 멀티에이전트 Supervisor 시스템 프롬프트
MULTI_AGENT_SUPERVISOR_PROMPT = _PROMPTS["supervisors"]["multi_agent"]


# ============================================================
# 쿼리 디컴포지션 — 복합 질문을 서브 질문으로 분리 (코드 기반)
# ============================================================
import re as _re

# 엔티티 ID 패턴 (SEL0001, S0001 등)
_ENTITY_RE = _re.compile(r'\b(SEL\d+|S\d{4,})\b', _re.IGNORECASE)

# 복합 패턴 마커 — 빠른 존재 체크용 (정규식 분리 전 게이트)
_COMPOUND_MARKERS = [
    "하고 ", "해주고 ", "주고 ", "받고 ", "면서 ",
    "한 다음 ", "그리고 ",
    "한후 ", "한 후 ", "후에 ", " 후 ",
    "한뒤 ", "한 뒤 ", " 뒤 ",
]

# 분리 정규식 — 긴 패턴 우선 매칭 (교대순서 중요)
_SPLIT_RE = _re.compile(
    r'(?:'
    r'한\s*다음\s|'       # "한 다음"
    r'한\s*후에?\s|'      # "한후", "한 후", "한 후에"
    r'한\s*뒤에?\s|'      # "한뒤", "한 뒤", "한 뒤에"
    r'그리고\s|'          # "그리고"
    r'하고\s|'            # "하고" (분석하고, 진단하고)
    r'해\s*주고\s|'       # "해주고", "해 주고"
    r'주고\s|'            # "알려주고", "돌려주고"
    r'받고\s|'            # "진단 받고"
    r'면서\s|'            # "면서"
    r'\s후\s|'            # " 후 "
    r'\s뒤\s'             # " 뒤 "
    r')'
)

# 워커 영역별 키워드 — 도메인 감지에 len(kw)² 가중치 점수 적용
# 긴 키워드 = 더 구체적 = 높은 점수 → 모호한 단문 키워드보다 정확한 복합 키워드 우선
_DOMAIN_KEYWORDS = {
    "churn": [
        "이탈 예측", "이탈 확률", "이탈 위험", "이탈 분석",
        "이탈", "churn",
    ],
    "retention": [
        "리텐션 전략", "리텐션 메시지", "리텐션 실행",
        "리텐션", "retention", "전략 실행", "조치 실행",
        "쿠폰", "매니저 배정",
    ],
    "seller": [
        "셀러 종합 진단", "셀러 분석", "셀러 진단", "셀러 정보", "셀러 활동",
        "종합 진단", "세그먼트", "이상거래", "부정행위", "fraud", "컨설팅",
        "셀러",
    ],
    "performance": [
        "쇼핑몰 매출", "매출 분석", "마케팅 최적화", "마케팅 예산",
        "매출 예측", "코호트", "GMV", "트렌드", "성과",
    ],
    "cs": [
        "CS 통계", "CS 품질", "자동 응답",
        "CS", "상담", "문의",
    ],
    "report": [
        "대시보드 요약", "종합 보고", "전체 현황",
        "대시보드", "KPI", "리포트", "보고서", "요약",
    ],
    "platform": [
        "이탈방지 정책", "카페24 정책", "운영 가이드", "쇼핑몰 운영",
        "보안정책", "결제수단", "설정 방법",
        "카페24", "정책", "수수료", "가이드", "용어",
        "플랫폼", "검색",
    ],
}

# 엔티티 전파 불필요 도메인 (셀러/쇼핑몰 ID가 검색 노이즈가 되는 영역)
_ENTITY_FREE_DOMAINS = frozenset({"platform", "report"})


def _detect_domain(text: str) -> str | None:
    """텍스트에서 워커 영역 감지 — 키워드 길이² 가중치 합산, 최고 점수 도메인 반환

    예) "카페24 이탈방지 정책 검색해줘"
        churn:    "이탈"(2²=4)                              = 4
        platform: "이탈방지 정책"(7²=49)+"카페24"(3²=9)+... = 66  ← 승
    """
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text:
                score += len(kw) ** 2
        if score > 0:
            scores[domain] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def _decompose_query(user_text: str) -> list:
    """복합 질문을 코드 기반으로 분리 (LLM 호출 없음, <1ms)

    처리 파이프라인:
    1. 복합 패턴 존재 여부 빠른 게이트
    2. 정규식 분리
    3. 짧은 파트 병합 (3자 이하 → 이전 파트에 붙임)
    4. 도메인 감지 + None 도메인 계승 + 인접 동일 도메인 병합
    5. 도메인 다양성 체크 (단일 도메인이면 분리 안 함)
    6. 엔티티 ID 도메인 인식 기반 전파 (platform/report에는 전파 안 함)
    """
    # 1. 빠른 게이트
    if not any(m in user_text for m in _COMPOUND_MARKERS):
        return [user_text]

    # 2. 정규식 분리
    parts = _SPLIT_RE.split(user_text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return [user_text]

    # 3. 짧은 파트 병합 (조사/어미만 남은 잔여물)
    merged = []
    for p in parts:
        if len(p) <= 3 and merged:
            merged[-1] += " " + p
        else:
            merged.append(p)
    parts = merged
    if len(parts) < 2:
        return [user_text]

    # 4. 도메인 감지
    domains = [_detect_domain(p) for p in parts]

    # 4-1. None 도메인 → 이전 파트 도메인 계승
    for i in range(len(domains)):
        if domains[i] is None and i > 0:
            domains[i] = domains[i - 1]

    # 4-2. 인접 동일 도메인 파트 병합 (같은 워커가 처리할 내용을 하나로)
    merged_parts: list[str] = []
    merged_domains: list[str | None] = []
    for p, d in zip(parts, domains):
        if merged_domains and d == merged_domains[-1] and d is not None:
            merged_parts[-1] += " " + p
        else:
            merged_parts.append(p)
            merged_domains.append(d)
    parts = merged_parts
    domains = merged_domains
    if len(parts) < 2:
        return [user_text]

    # 5. 도메인 다양성 체크
    unique_domains = set(d for d in domains if d is not None)
    if len(unique_domains) <= 1:
        return [user_text]

    # 6. 엔티티 전파 — 도메인 인식 기반
    entity_ids = list(dict.fromkeys(_ENTITY_RE.findall(user_text)))  # 중복 제거, 순서 유지
    if entity_ids:
        entity_prefix = " ".join(entity_ids)
        for i, (part, domain) in enumerate(zip(parts, domains)):
            if _ENTITY_RE.search(part):
                continue  # 이미 엔티티 포함
            if domain in _ENTITY_FREE_DOMAINS:
                continue  # 엔티티가 검색 노이즈가 되는 도메인
            parts[i] = f"{entity_prefix} {part}"

    st.logger.info(
        "QUERY_DECOMPOSED original=%s sub_queries=%s domains=%s",
        user_text[:80], parts, domains,
    )
    return parts


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
        # ── 쿼리 디컴포지션: 복합 질문을 서브 질문으로 분리 ──
        sub_queries = _decompose_query(user_text)

        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=True,
            temperature=req.temperature if req.temperature is not None else 0.3,
            top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        model_key = normalize_model_name(req.model)
        supervisor = get_cached_multi_supervisor(llm, model_key, api_key=api_key)

        # 메시지 히스토리 구성 (멀티턴 지원)
        prev_messages = memory_messages(username)
        base_messages = []
        for msg in prev_messages:
            role, content = msg.get("role", ""), msg.get("content", "")
            if role == "user":
                base_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                base_messages.append({"role": "assistant", "content": content})

        # 서브쿼리별 순차 실행 목록 (1개면 원본 그대로)
        query_list = sub_queries if len(sub_queries) >= 2 else [user_text]

        # SSE 스트리밍 상태 추적 (전체 세션 공유)
        tool_calls_log = []
        agent_results = []
        final_buf = []
        agents_used_set = set()
        MAX_TOOL_RETRIES = 5
        MAX_TOOL_CALLS = 2

        # 워커 이름 집합 (handoff 이벤트 감지용)
        worker_names = set(MULTI_AGENT_WORKERS.keys())

        # ── 서브쿼리별 순차 실행 ──
        for sq_idx, sq_text in enumerate(query_list):
            # 서브쿼리마다 도구 카운터/차단 상태 리셋
            _request_tool_counts.set({})
            tool_fail_counts: Dict[str, int] = {}
            tool_total_counts: Dict[str, int] = {}
            blocked_tools: set = set()
            current_agent = None
            step_start_time = None
            current_tool = None

            # 서브쿼리 간 구분자 삽입
            if sq_idx > 0:
                separator = "\n\n---\n\n"
                final_buf.append(separator)
                await sse_callback("delta", {"delta": separator})

            if len(query_list) >= 2:
                st.logger.info("SUB_QUERY_START idx=%d query=%s", sq_idx + 1, sq_text[:60])

            input_messages = base_messages + [{"role": "user", "content": sq_text}]

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

                    if tool_name == "forward_message":
                        pass
                    elif tool_name.startswith("transfer_to_"):
                        agent_name = tool_name.replace("transfer_to_", "")
                        if agent_name in worker_names:
                            if current_agent and current_agent != agent_name:
                                elapsed_ms = int((time.time() - step_start_time) * 1000) if step_start_time else 0
                                await sse_callback("agent_end", {
                                    "agent": current_agent,
                                    "elapsed_ms": elapsed_ms,
                                    "description": AGENT_DESCRIPTIONS.get(current_agent, current_agent),
                                })
                            if agents_used_set:
                                separator = "\n\n---\n\n"
                                final_buf.append(separator)
                                await sse_callback("delta", {"delta": separator})
                            current_agent = agent_name
                            step_start_time = time.time()
                            agents_used_set.add(agent_name)
                            await sse_callback("agent_start", {
                                "agent": agent_name,
                                "description": AGENT_DESCRIPTIONS.get(agent_name, agent_name),
                                "step_detail": MULTI_AGENT_WORKERS.get(agent_name, {}).get("prompt", "")[:100],
                            })
                    else:
                        if tool_name in blocked_tools:
                            current_tool = None
                            continue
                        current_tool = tool_name
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
                    if end_tool_name in ("forward_message",) or end_tool_name.startswith("transfer_to_"):
                        pass
                    elif end_tool_name in blocked_tools:
                        tool_total_counts[end_tool_name] = tool_total_counts.get(end_tool_name, 0) + 1
                        if tool_total_counts[end_tool_name] % 4 == 0:
                            st.logger.warning(
                                "BLOCKED_TOOL_REPEAT tool=%s calls=%d — 계속 무시 중",
                                end_tool_name, tool_total_counts[end_tool_name],
                            )
                        continue
                    else:
                        tool_output = data.get("output", {})
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

                        if isinstance(tool_output, dict):
                            result_preview = json.dumps(tool_output, ensure_ascii=False, default=str)[:200]
                        elif isinstance(tool_output, str):
                            result_preview = tool_output[:200]
                        else:
                            result_preview = str(tool_output)[:200]

                        is_error = (isinstance(tool_output, dict) and tool_output.get("status") == "error")
                        if is_error:
                            tool_fail_counts[end_tool_name] = tool_fail_counts.get(end_tool_name, 0) + 1
                        tool_total_counts[end_tool_name] = tool_total_counts.get(end_tool_name, 0) + 1

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

                        if tool_total_counts.get(end_tool_name, 0) >= MAX_TOOL_CALLS:
                            blocked_tools.add(end_tool_name)
                            st.logger.warning(
                                "TOOL_BLOCKED tool=%s calls=%d — 이후 호출 UI 숨김",
                                end_tool_name, tool_total_counts[end_tool_name],
                            )

                        if tool_fail_counts.get(end_tool_name, 0) >= MAX_TOOL_RETRIES:
                            st.logger.warning(
                                "TOOL_RETRY_LIMIT tool=%s fails=%d — aborting stream",
                                end_tool_name, tool_fail_counts[end_tool_name],
                            )
                            error_msg = f"도구 '{end_tool_name}'이(가) {MAX_TOOL_RETRIES}회 연속 실패하여 분석을 중단합니다."
                            final_buf.append(error_msg)
                            await sse_callback("delta", {"delta": "\n\n" + error_msg})
                            break

                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if not chunk or getattr(chunk, "tool_call_chunks", None):
                        continue
                    content = getattr(chunk, "content", "")
                    if isinstance(content, str) and content:
                        if not outer_node or outer_node == "sub_supervisor":
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

            # 서브쿼리 astream 종료: 마지막 워커 agent_end 처리
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

        # 최종 응답 (전체 서브쿼리 완료 후)
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
_MULTI_SUPERVISOR_CACHE_MAX = 3  # Supervisor 캐시 최대 크기 (메모리 최적화)


def build_multi_agent_supervisor(llm, api_key: str = ""):
    """멀티에이전트용 Supervisor 생성 — 7종 워커 동적 라우팅
    전체 워커 동일 LLM (gpt-5-mini) 사용"""
    if not SUPERVISOR_AVAILABLE:
        raise ImportError("langgraph-supervisor가 설치되지 않았습니다. 'pip install langgraph-supervisor'")

    workers = []
    for name, config in MULTI_AGENT_WORKERS.items():
        # 도구 호출 횟수 제한 래핑 적용
        limited_tools = _wrap_tools_with_limit(config["tools"])
        agent = create_react_agent(
            model=llm,
            tools=limited_tools,
            name=name,
            prompt=config["prompt"],
        )
        workers.append(agent)

    # forward_message: 워커 응답을 Supervisor 재해석 없이 직접 전달 (공식 패턴)
    supervisor_tools = []
    if create_forward_message_tool:
        supervisor_tools.append(create_forward_message_tool("sub_supervisor"))

    workflow = create_supervisor(
        agents=workers,
        model=llm,
        prompt=MULTI_AGENT_SUPERVISOR_PROMPT,
        output_mode="full_history",
        supervisor_name="sub_supervisor",
        add_handoff_messages=True,
        tools=supervisor_tools,
    )

    return workflow.compile()


def get_cached_multi_supervisor(llm, model_key: str, api_key: str = ""):
    """모델별 멀티에이전트 Supervisor 그래프 캐시 (FIFO, 최대 3개)"""
    if model_key not in _multi_supervisor_cache:
        # 캐시 크기 제한: 최대 개수 초과 시 가장 오래된 항목부터 제거 (FIFO)
        if len(_multi_supervisor_cache) >= _MULTI_SUPERVISOR_CACHE_MAX:
            oldest_key = next(iter(_multi_supervisor_cache))
            del _multi_supervisor_cache[oldest_key]
            st.logger.info("MULTI_SUPERVISOR_CACHE_EVICT model=%s (limit=%d)", oldest_key, _MULTI_SUPERVISOR_CACHE_MAX)
        _multi_supervisor_cache[model_key] = build_multi_agent_supervisor(llm, api_key=api_key)
        st.logger.info("MULTI_SUPERVISOR_GRAPH_BUILD model=%s (cached, size=%d)", model_key, len(_multi_supervisor_cache))
    return _multi_supervisor_cache[model_key]
