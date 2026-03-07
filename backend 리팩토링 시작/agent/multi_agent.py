"""
agent/multi_agent.py - LangGraph Supervisor 기반 멀티 에이전트 시스템
====================================================================
카페24 AI 기반 내부 시스템

구조 (langgraph-supervisor 패턴):
- Supervisor: 사용자 질의 분석 → 전문 워커 에이전트에게 위임 → 결과 종합
- 일반 질문용 Supervisor (3종 워커): search_agent, analysis_agent, cs_agent
- 멀티에이전트 Supervisor (8종 워커): churn_analyst, retention_strategist,
  seller_analyst, performance_analyst, fraud_investigator, cs_quality_analyst,
  report_writer, platform_searcher

프로덕션 경로:
- run_multi_agent_stream() → build_multi_agent_supervisor() (SSE 스트리밍)
- build_supervisor_graph() → 일반 질문용 Supervisor
"""
import json
import time
from typing import Any, Dict

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
    SEARCH_AGENT_TOOLS,
    ANALYSIS_AGENT_TOOLS,
    TRANSLATION_AGENT_TOOLS,
    # 멀티에이전트 워커용 개별 도구 임포트
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
# 에이전트 프롬프트
# ============================================================
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
# 워커 공통 응답 규칙
# ============================================================
_WORKER_COMMON_RULES = """
## 응답 규칙 (반드시 준수)
- 도구 결과의 핵심 수치(건수, 금액, 비율, 점수 등)를 **구체적으로 언급**하세요
- **마크다운 표·볼드·리스트**를 활용하여 가독성 높게 정리하세요
- "성공적으로 조회했습니다", "확인했습니다" 같은 **형식적 응답은 절대 금지**
- 숫자만 나열하지 말고, **"그래서 뭐가 문제이고, 어떻게 해야 하는가"**를 포함하세요
- 데이터가 충분하면 **최소 3개 이상의 인사이트**를 제공하세요
- 금액은 **₩ + 천 단위 콤마** 사용, 큰 금액은 **억/만원 단위**로 환산

## 분석 관점 (필수)
1. **추세 파악**: 증가/감소/정체 패턴
2. **이상값 발견**: 평균에서 크게 벗어나는 값
3. **비교 분석**: 항목 간 차이
4. **원인 추론**: 왜 이런 패턴이 나타나는지 가설 제시
5. **실행 제안**: 구체적인 액션 아이템
"""

# ============================================================
# 멀티에이전트 Supervisor 워커 정의 (8종)
# ============================================================
MULTI_AGENT_WORKERS = {
    "churn_analyst": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **셀러 이탈 분석 전문가**입니다.\n"
            "ML 모델(RandomForest)과 SHAP 분석으로 이탈 위험을 예측하고 주요 원인을 파악합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 이탈 확률, 위험 등급(high/medium/low)을 명확히 구분하세요\n"
            "- SHAP top_factors를 **원인 순위 표**로 정리하세요\n"
            "- 마지막 접속일, 매출 규모, 환불률 등 핵심 지표를 비교하세요\n"
            "- 이탈 확률이 높은 셀러의 **공통 패턴**을 찾아 보고하세요\n"
            "- 즉각적인 리텐션 조치가 필요한 셀러를 **우선순위**로 추천하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [get_at_risk_sellers, predict_seller_churn, get_churn_prediction],
        "description": "이탈 분석 전문가 — ML 이탈 예측 + SHAP 분석",
    },
    "retention_strategist": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **셀러 리텐션 전략 전문가**입니다.\n"
            "이탈 위험 셀러에게 맞춤 메시지를 생성하고, 리텐션 조치를 제안합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 셀러 상황(매출, 접속일, 환불률)에 맞는 **맞춤 전략**을 제안하세요\n"
            "- 조치 유형별(쿠폰/업그레이드/매니저 배정) **예상 효과**를 언급하세요\n"
            "- 리텐션 메시지는 **톤앤매너**를 셀러 등급에 맞게 조절하세요\n"
            "- 긴급도에 따라 **즉시/단기/중기** 조치로 분류하세요\n\n"
            "## 조치 실행 규칙 (매우 중요)\n"
            "- `generate_retention_message`로 추천 조치를 먼저 확인하세요\n"
            "- 추천 조치 목록과 예상 효과를 **먼저 사용자에게 제시**하세요\n"
            "- `execute_retention_action`은 사용자가 **명시적으로 '실행해', '적용해', '진행해'**라고 요청한 경우에만 호출하세요\n"
            "- 사용자가 '전략 실행해줘', '조치 해줘'처럼 실행을 요청한 경우 → **긴급도가 가장 높은 1개만** 자동 실행하고, 나머지는 제안으로 남기세요\n"
            "- 여러 조치를 한꺼번에 실행하지 마세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [generate_retention_message, execute_retention_action, get_at_risk_sellers],
        "description": "리텐션 전략가 — 맞춤 메시지 생성 + 자동 조치",
    },
    "seller_analyst": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **셀러 종합 분석 전문가**입니다.\n"
            "셀러의 활동, 세그먼트, 이상거래, 성과를 종합 분석합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 세그먼트 간 **셀러 수, 평균 매출, 주문 수, 환불률**을 비교 표로 정리하세요\n"
            "- 세그먼트별 특징과 **관리 전략**을 제시하세요 (성장형 vs 휴면 vs 파워)\n"
            "- 개별 셀러 분석 시 **강점/약점/기회/위협**을 구분하세요\n"
            "- 동일 세그먼트 내 상위/하위 셀러 간 차이를 분석하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [analyze_seller, get_seller_segment, detect_fraud, get_segment_statistics, get_fraud_statistics, get_seller_activity_report],
        "description": "셀러 분석가 — 셀러 종합 분석 + 세그먼트 + 이상거래",
    },
    "performance_analyst": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **쇼핑몰 성과 및 KPI 분석 전문가**입니다.\n"
            "매출 트렌드, 코호트 분석, GMV 예측, 마케팅 최적화를 수행합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 매출/주문/전환율의 **기간별 추세**(전월 대비, 전년 대비)를 분석하세요\n"
            "- 코호트 리텐션에서 **이탈 급감 구간**을 찾아 원인을 추론하세요\n"
            "- 마케팅 채널별 **ROI, CPA, ROAS**를 비교하세요\n"
            "- GMV 예측 시 **성장률, ARPU, 티어별 분포**를 함께 제시하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [get_shop_info, get_shop_performance, get_trend_analysis, get_cohort_analysis, predict_shop_revenue, get_gmv_prediction, optimize_marketing, get_order_statistics],
        "description": "성과 분석가 — 매출/KPI/마케팅 분석",
    },
    "fraud_investigator": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **이상거래 조사 전문가**입니다.\n"
            "부정 거래 패턴을 탐지하고 영향도를 분석합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 이상 탐지 건수, **위험 점수 분포**, 영향 금액을 정리하세요\n"
            "- 이상 유형별(환불 사기, 가짜 주문, 비정상 패턴) **발생 빈도**를 분류하세요\n"
            "- 고위험 셀러의 **구체적 이상 행동 패턴**을 설명하세요\n"
            "- **즉시 차단, 모니터링, 경고** 등 대응 방안을 제시하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [detect_fraud, get_fraud_statistics, analyze_seller],
        "description": "이상거래 조사관 — 부정행위 탐지 + 영향 분석",
    },
    "cs_quality_analyst": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **CS 품질 분석 전문가**입니다.\n"
            "CS 문의 통계, 자동 분류, 감성 분석, 자동 응답 생성을 담당합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- 카테고리별 **티켓 수, 만족도, 평균 해결 시간**을 비교 표로 정리하세요\n"
            "- 만족도가 낮거나 해결 시간이 긴 **병목 카테고리**를 지적하세요\n"
            "- 우수 카테고리 vs 취약 카테고리의 **차이 원인**을 분석하세요\n"
            "- CS 품질 개선을 위한 **구체적 우선순위 액션**을 제안하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [get_cs_statistics, auto_reply_cs, check_cs_quality, classify_inquiry, get_ecommerce_glossary],
        "description": "CS 품질 분석가 — CS 통계 + 자동 응답 + 품질 평가",
    },
    "report_writer": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **운영 리포트 전문가**입니다.\n"
            "대시보드 현황, 주문 통계, KPI를 종합하여 보고서를 작성합니다.\n\n"
            "## 역할별 분석 지침\n"
            "- **경영진이 바로 의사결정할 수 있는 수준**의 보고서를 작성하세요\n"
            "- 핵심 KPI를 **요약 표**로 먼저 제시하고, 상세 분석을 이어가세요\n"
            "- 전월/전주 대비 **변화량과 변화율(%)**을 함께 표기하세요\n"
            "- 긍정/부정 트렌드를 구분하여 **주의 필요 항목**을 별도 표기하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [get_dashboard_summary, get_order_statistics, get_trend_analysis, get_cohort_analysis],
        "description": "리포트 작성가 — 대시보드 + KPI 종합 보고서",
    },
    "platform_searcher": {
        "prompt": (
            "당신은 카페24 AI 운영 플랫폼의 **플랫폼 지식 검색 전문가**입니다.\n"
            "카페24 플랫폼 정책, 기능, 운영 가이드, FAQ를 RAG(검색증강생성)로 검색합니다.\n"
            "쇼핑몰/카테고리 정보 조회와 이커머스 용어 설명도 담당합니다.\n\n"
            "**반드시 search_platform 또는 search_platform_lightrag 도구를 호출하여 검색한 뒤 답변하세요.**\n"
            "절대 도구 없이 직접 답변하지 마세요.\n\n"
            "## 역할별 분석 지침\n"
            "- RAG 결과 전체를 꼼꼼히 읽고, 유사한 표현도 찾으세요\n"
            "- RAG 결과에 정보가 있는데 '모르겠습니다'라고 답변하지 마세요\n"
            "- RAG 결과에 **없는 정보를 절대 지어내지 마세요** (할루시네이션 금지)\n"
            "- 숫자를 묻는 질문에는 RAG 결과에서 실제로 나열된 항목을 세어서 답하세요\n"
            + _WORKER_COMMON_RULES
        ),
        "tools": [search_platform, search_platform_lightrag, get_shop_info, list_shops, get_shop_services, get_category_info, list_categories, get_ecommerce_glossary],
        "description": "플랫폼 검색가 — RAG 지식 검색 + 쇼핑몰/카테고리 조회",
    },
}

# 멀티에이전트 Supervisor 시스템 프롬프트
MULTI_AGENT_SUPERVISOR_PROMPT = """당신은 카페24 AI 운영 플랫폼의 **멀티에이전트 Supervisor**입니다.
사용자의 요청을 분석하여 적절한 전문 워커 에이전트에게 작업을 위임하고, 결과를 종합합니다.

## 전문 워커 에이전트 (8종):
1. **churn_analyst**: 셀러 이탈 분석 — ML 이탈 예측, SHAP 분석, 위험 셀러 조회
2. **retention_strategist**: 리텐션 전략 — 맞춤 메시지 생성, 쿠폰/업그레이드/매니저 배정 조치
3. **seller_analyst**: 셀러 종합 분석 — 활동, 세그먼트, 이상거래, 성과 분석
4. **performance_analyst**: 쇼핑몰 성과/KPI — 매출 트렌드, 코호트, GMV 예측, 마케팅 최적화
5. **fraud_investigator**: 이상거래 조사 — 부정 거래 탐지, 영향도 분석
6. **cs_quality_analyst**: CS 품질 — 문의 통계, 자동 분류, 감성 분석, 자동 응답
7. **report_writer**: 운영 리포트 — 대시보드, 주문 통계, KPI 종합 보고서
8. **platform_searcher**: 플랫폼 지식 검색 — RAG 기반 정책/기능/FAQ 검색, 쇼핑몰/카테고리 조회

## 라우팅 판단 기준:
- 이탈/위험 셀러/고위험 → churn_analyst
- 이탈 방지 전략/메시지/조치 → retention_strategist
- 셀러 종합 진단/세그먼트 통계 → seller_analyst
- 쇼핑몰 성과/매출/마케팅/코호트/트렌드/GMV → performance_analyst
- 이상거래/부정행위/비정상 → fraud_investigator
- CS 품질/상담/감성/CS 통계 → cs_quality_analyst
- 대시보드/KPI/리포트/전체 현황 → report_writer
- 플랫폼 정책/기능/FAQ/설정 방법/용어 → platform_searcher (반드시 RAG 검색 필수)

## 워크플로우 규칙:
- 복합 요청("~하고 ~도 해줘")은 여러 워커에게 순차적으로 위임
- 간단한 후속 질문은 직접 답변 가능 (워커 위임 불필요)

## 대화 맥락 유지 (매우 중요):
- 이전 대화에서 특정 쇼핑몰/셀러를 언급했으면, 후속 질문도 **그 대상 기준**으로 처리
- 새로운 대상이 언급될 때까지 이전 대상 유지

## 최종 응답 규칙 (매우 중요 — 반드시 준수!):
- 워커가 반환한 데이터를 **반드시 구체적으로 분석**하여 사용자에게 전달하세요
- 핵심 수치(건수, 금액, 비율, 점수, 순위 등)를 **구체적으로 언급**하세요
- **마크다운 표·볼드·리스트**를 활용하여 가독성 높게 정리하세요
- "성공적으로 조회했습니다", "확인했습니다" 같은 **형식적 한 줄 응답은 절대 금지**
- 숫자만 나열하지 말고, **"그래서 뭐가 문제이고, 어떻게 해야 하는가"**를 포함하세요
- 데이터에서 **인사이트를 도출**하고, 주목할 점이나 개선 방향을 제안하세요
- 워커가 반환한 데이터를 누락하지 말고 **상세하게 정리**하세요
- 금액은 **₩ + 천 단위 콤마**, 큰 금액은 **억/만원 단위**로 환산
- 데이터가 충분하면 **최소 3개 이상의 인사이트**를 제공하세요
"""


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


# M19/I7: 레거시 호환 코드 제거 (AgentType, TaskStatus, MultiAgentSystem 등)


# ============================================================
# Supervisor 패턴 (일반 질문용)
# ============================================================
SUPERVISOR_PROMPT = """당신은 카페24 AI 운영 플랫폼의 Supervisor입니다.
사용자 질의를 분석하여 적절한 전문 에이전트에게 작업을 위임하고, 결과를 종합하여 답변합니다.

## 전문 에이전트:
1. **search_agent**: 쇼핑몰/카테고리 정보 조회, 플랫폼 RAG 문서 검색
   - 쇼핑몰 이름/ID 조회, 목록/서비스 정보, 카테고리 조회
   - 플랫폼 정책/가이드/용어 검색
2. **analysis_agent**: 셀러 분석, ML 예측, 매출/KPI 분석
   - 셀러 상세분석, 세그먼트, 이상거래 탐지, 이탈 예측
   - 쇼핑몰 성과/매출 예측, 마케팅 최적화
   - 대시보드, 코호트, 트렌드, GMV 예측
3. **cs_agent**: CS 응답 생성, 품질 평가, 문의 분류
   - CS 자동 응답, 품질 평가, 이커머스 용어집, CS 통계

## 판단 기준:
- 쇼핑몰/카테고리 정보, 플랫폼 정책/기능 질문 → search_agent
- 셀러 분석, 통계, 예측, 성과, 마케팅 → analysis_agent
- CS 문의, 상담, 품질 평가 → cs_agent
- 복합 질문이면 순차적으로 여러 에이전트에게 위임
- 간단한 인사/일반 대화는 직접 답변 (에이전트 위임 불필요)

## 종합 규칙:
- 에이전트 결과를 받으면 충분한지 검토
- 추가 정보가 필요하면 다른 에이전트에게 추가 위임
- 최종적으로 모든 결과를 종합하여 사용자에게 한국어로 답변
"""


def build_supervisor_graph(llm):
    """langgraph-supervisor 기반 Supervisor 그래프 생성"""
    search_agent = create_react_agent(
        model=llm,
        tools=SEARCH_AGENT_TOOLS,
        name="search_agent",
        prompt=SEARCH_AGENT_PROMPT,
    )

    analysis_agent = create_react_agent(
        model=llm,
        tools=ANALYSIS_AGENT_TOOLS,
        name="analysis_agent",
        prompt=ANALYSIS_AGENT_PROMPT,
    )

    cs_agent = create_react_agent(
        model=llm,
        tools=TRANSLATION_AGENT_TOOLS,
        name="cs_agent",
        prompt=TRANSLATION_AGENT_PROMPT,
    )

    workflow = create_supervisor(
        agents=[search_agent, analysis_agent, cs_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
        supervisor_name="supervisor",
        add_handoff_messages=True,
    )

    return workflow.compile()


# Supervisor 그래프 모델별 캐시 (기존 _graph_cache와 별도 관리)
_supervisor_cache: Dict[str, Any] = {}


def get_cached_supervisor(llm, model_key: str):
    """모델별 Supervisor 그래프 캐시"""
    if model_key not in _supervisor_cache:
        _supervisor_cache[model_key] = build_supervisor_graph(llm)
        st.logger.info("SUPERVISOR_GRAPH_BUILD model=%s (cached)", model_key)
    return _supervisor_cache[model_key]


# Intent → Agent 직접 매핑 (supervisor 우회용)
INTENT_AGENT_MAP = {
    "shop": "search_agent",
    "seller": "analysis_agent",
    "analysis": "analysis_agent",
    "cs": "cs_agent",
    "dashboard": "analysis_agent",
    "retention": "analysis_agent",
    # platform, general → supervisor 판단 필요
}

# 개별 워커 에이전트 캐시 (supervisor 우회 직접 호출용)
_worker_cache: Dict[str, Any] = {}


def get_cached_worker(llm, model_key: str, agent_name: str):
    """개별 워커 에이전트 캐시 반환 (supervisor 우회 직접 호출)"""
    cache_key = f"{model_key}:{agent_name}"
    if cache_key not in _worker_cache:
        if agent_name == "search_agent":
            agent = create_react_agent(model=llm, tools=SEARCH_AGENT_TOOLS, name="search_agent", prompt=SEARCH_AGENT_PROMPT)
        elif agent_name == "analysis_agent":
            agent = create_react_agent(model=llm, tools=ANALYSIS_AGENT_TOOLS, name="analysis_agent", prompt=ANALYSIS_AGENT_PROMPT)
        elif agent_name == "cs_agent":
            agent = create_react_agent(model=llm, tools=TRANSLATION_AGENT_TOOLS, name="cs_agent", prompt=TRANSLATION_AGENT_PROMPT)
        else:
            return None
        _worker_cache[cache_key] = agent
        st.logger.info("WORKER_AGENT_BUILD agent=%s model=%s (cached)", agent_name, model_key)
    return _worker_cache[cache_key]


# 에이전트 설명 헬퍼
AGENT_DESCRIPTIONS = {
    "search_agent": "검색 에이전트 — 쇼핑몰/카테고리/플랫폼 정보 검색",
    "analysis_agent": "분석 에이전트 — 셀러 분석, ML 예측, KPI 분석",
    "cs_agent": "CS 에이전트 — CS 응답 생성, 품질 평가",
}
# 멀티에이전트 워커 설명도 추가
for _wname, _wcfg in MULTI_AGENT_WORKERS.items():
    AGENT_DESCRIPTIONS[_wname] = _wcfg["description"]


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
