"""
agent/router.py - CAFE24 AI 운영 플랫폼 LLM Router (의도 분류 전용)
============================================================
CAFE24 AI 운영 플랫폼

질문을 먼저 분류한 뒤, 해당 카테고리의 도구만 Executor에 노출합니다.

분류 우선순위:
1. 키워드 기반 분류 (가장 빠름, 비용 없음)
2. LLM Router (gpt-4o-mini, fallback)

References:
- https://www.anthropic.com/research/building-effective-agents
- https://github.com/aurelio-labs/semantic-router
"""
import re
from typing import Optional
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str
from agent.intent import (
    ANALYSIS_KEYWORDS, PLATFORM_KEYWORDS, SHOP_KEYWORDS,
    SELLER_KEYWORDS, CS_KEYWORDS, DASHBOARD_KEYWORDS, GENERAL_KEYWORDS,
    RETENTION_KEYWORDS,
)
import state as st


# ============================================================
# 카테고리 정의
# ============================================================
class IntentCategory(str, Enum):
    """질문 의도 카테고리"""
    ANALYSIS = "analysis"       # 매출, GMV, 이탈, DAU, 코호트, 트렌드
    PLATFORM = "platform"       # 플랫폼 정책, 기능, 운영 가이드
    SHOP = "shop"               # 쇼핑몰 정보, 서비스, 성과, 매출
    SELLER = "seller"           # 셀러 분석, 세그먼트, 부정행위 탐지
    CS = "cs"                   # CS 자동응답, 품질 검사, 문의 분류
    RETENTION = "retention"     # 이탈 방지, 리텐션 전략, 위험 셀러 관리
    DASHBOARD = "dashboard"     # 대시보드, 전체 현황
    GENERAL = "general"         # 일반 대화, 인사


# 카테고리별 도구 매핑
CATEGORY_TOOLS = {
    IntentCategory.ANALYSIS: [
        "get_churn_prediction",
        "get_gmv_prediction",
        "get_trend_analysis",
        "get_cohort_analysis",
    ],
    IntentCategory.PLATFORM: [
        "search_platform",
        "search_platform_lightrag",
    ],
    IntentCategory.SHOP: [
        "get_shop_info",
        "list_shops",
        "get_shop_services",
        "get_shop_performance",
        "predict_shop_revenue",
        "optimize_marketing",
        "get_category_info",         # 카테고리/업종 조회
        "list_categories",           # 카테고리 목록 조회
        "get_dashboard_summary",     # 쇼핑몰 전체 현황/분포 요약
        "search_platform",           # 쇼핑몰 정보도 RAG에서 검색 (DB에 없는 쇼핑몰 지원)
        "search_platform_lightrag",  # LightRAG 검색
    ],
    IntentCategory.SELLER: [
        "analyze_seller",
        "predict_seller_churn",
        "get_seller_segment",
        "detect_fraud",
        "get_fraud_statistics",     # 이상거래/부정행위 전체 통계
        "get_segment_statistics",
        "optimize_marketing",       # 셀러별 마케팅 최적화
        "get_shop_performance",     # 셀러 쇼핑몰 성과
        "predict_shop_revenue",     # 셀러 매출 예측
    ],
    IntentCategory.CS: [
        "auto_reply_cs",
        "check_cs_quality",
        "get_ecommerce_glossary",
        "get_cs_statistics",
        "classify_inquiry",
    ],
    IntentCategory.RETENTION: [
        "get_at_risk_sellers",
        "generate_retention_message",
        "execute_retention_action",
        "analyze_seller",
        "get_cs_statistics",
    ],
    IntentCategory.DASHBOARD: [
        "get_dashboard_summary",
        "get_segment_statistics",
        "get_cs_statistics",
        "get_order_statistics",
    ],
    IntentCategory.GENERAL: [],  # 도구 없이 대화만
}


# ============================================================
# 키워드 기반 빠른 분류 (LLM 호출 없이)
# cross-6: 키워드는 intent.py에서 단일 소스로 import
# ============================================================


def _keyword_classify(text: str) -> Optional[IntentCategory]:
    """
    키워드 기반 빠른 분류 (LLM 호출 없이)

    Returns:
        IntentCategory or None (불확실한 경우)
    """
    t = text.lower()

    # 우선순위: 셀러ID감지 > 리텐션 > 분석 > 셀러 > 쇼핑몰 > CS > 대시보드 > 플랫폼 > 일반

    # 0. 셀러 ID(SEL0001)가 포함되면 SELLER 우선 (분석보다 높은 우선순위)
    if re.search(r'SEL\d{1,6}', text, re.IGNORECASE):
        return IntentCategory.SELLER

    # 0.5. 리텐션 키워드 (ANALYSIS보다 우선 - 이탈 방지/위험 셀러 관리)
    if any(kw in t for kw in RETENTION_KEYWORDS):
        return IntentCategory.RETENTION

    # 1. 분석 키워드 (셀러 ID 없는 일반 분석)
    if any(kw in t for kw in ANALYSIS_KEYWORDS):
        return IntentCategory.ANALYSIS

    # 2. 셀러 분석 키워드
    if any(kw in t for kw in SELLER_KEYWORDS):
        return IntentCategory.SELLER

    # 3. 쇼핑몰 관련 (성과, 정보, 마케팅)
    if any(kw in t for kw in SHOP_KEYWORDS):
        return IntentCategory.SHOP

    # 4. CS 관련
    if any(kw in t for kw in CS_KEYWORDS):
        return IntentCategory.CS

    # 5. 대시보드 관련
    if any(kw in t for kw in DASHBOARD_KEYWORDS):
        return IntentCategory.DASHBOARD

    # 6. 플랫폼 관련 (정책, 기능, 용어)
    if any(kw in t for kw in PLATFORM_KEYWORDS):
        return IntentCategory.PLATFORM

    # 7. 일반 대화
    if any(kw in t for kw in GENERAL_KEYWORDS):
        return IntentCategory.GENERAL

    # 불확실한 경우 None 반환 → LLM Router 사용
    return None


# ============================================================
# LLM Router (분류 전용)
# ============================================================
ROUTER_SYSTEM_PROMPT = """당신은 질문 분류 전문가입니다.
사용자 질문을 분석하여 **정확히 하나의 카테고리**를 반환하세요.

## 카테고리 정의

| 카테고리 | 설명 | 예시 질문 |
|----------|------|----------|
| analysis | 매출, GMV, 이탈, DAU, 코호트, 트렌드, KPI | "GMV 성장률", "이탈 현황", "활성 셀러 수 알려줘" |
| platform | 플랫폼 정책, 기능, 운영 가이드, 용어 | "정산 주기가 뭐야?", "카페24 배송비 정책" |
| shop | 쇼핑몰 정보/목록/분포, 카테고리 목록, 서비스, 성과, 마케팅 | "S0001 쇼핑몰 정보", "쇼핑몰 플랜별 분포", "Premium 등급 쇼핑몰 목록", "카테고리 전체 목록", "패션 카테고리 쇼핑몰 현황" |
| seller | 특정 셀러 분석, 세그먼트, 부정행위 탐지 | "SEL0001 분석", "셀러 세그먼트 통계" |
| cs | CS 자동응답, 품질 검사, 문의 분류, 용어집 | "환불 문의 응답해줘", "CS 품질 분석" |
| retention | 이탈 방지, 리텐션 전략, 위험 셀러 관리, 맞춤 메시지 | "이탈 위험 셀러 분석", "리텐션 전략 실행" |
| dashboard | 대시보드, 전체 현황, 요약 통계 | "대시보드 보여줘", "전체 현황" |
| general | 일반 대화, 인사, 도움말 | "안녕", "뭐해?" |

## 규칙

1. **반드시 하나의 카테고리만** 반환
2. 복합 질문은 **핵심 의도** 기준으로 분류
3. 숫자/통계 관련 → analysis 우선
4. 플랫폼 지식/정책 → platform
5. 특정 쇼핑몰 (S0001 등) → shop
6. 쇼핑몰 목록/분포/현황, 카테고리 목록/정보, 플랜별/등급별/티어별 → shop
7. 특정 셀러 (SEL0001 등) → seller
8. 고객 문의/상담 → cs
9. 이탈 방지/리텐션 전략/위험 셀러 → retention

## 출력 형식

카테고리명만 반환하세요. 예: analysis"""


# M13: LLM Router용 ChatOpenAI 모듈 캐시
_router_llm_cache: dict = {}

# M17: 카테고리 문자열 매핑 (공통)
_CATEGORY_MAP = {
    "analysis": IntentCategory.ANALYSIS,
    "platform": IntentCategory.PLATFORM,
    "shop": IntentCategory.SHOP,
    "seller": IntentCategory.SELLER,
    "cs": IntentCategory.CS,
    "retention": IntentCategory.RETENTION,
    "dashboard": IntentCategory.DASHBOARD,
    "general": IntentCategory.GENERAL,
}


def route_intent_llm(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> IntentCategory:
    """
    M17: 동기 함수로 변환 (async 오버헤드 제거)
    LLM을 사용한 의도 분류 (키워드 분류 실패 시)

    Args:
        text: 사용자 질문
        api_key: OpenAI API 키
        model: 사용할 모델 (기본: gpt-4o-mini)

    Returns:
        IntentCategory
    """
    try:
        cache_key = f"{model}:{api_key[:8] if api_key else ''}"
        if cache_key not in _router_llm_cache:
            _router_llm_cache[cache_key] = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=0,
                max_tokens=20,
            )
        llm = _router_llm_cache[cache_key]

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"질문: {text}"),
        ]

        response = llm.invoke(messages)
        category_str = safe_str(response.content).strip().lower()
        category = _CATEGORY_MAP.get(category_str, IntentCategory.GENERAL)

        st.logger.info(
            "ROUTER_LLM_CLASSIFY query=%s result=%s",
            text[:50], category.value,
        )

        return category

    except Exception as e:
        st.logger.warning("ROUTER_LLM_FAIL err=%s, fallback=general", safe_str(e))
        return IntentCategory.GENERAL


def get_tools_for_category(category: IntentCategory) -> list[str]:
    """카테고리에 해당하는 도구 이름 목록 반환"""
    return CATEGORY_TOOLS.get(category, [])


# ============================================================
# 편의 함수
# ============================================================
def classify_and_get_tools(
    text: str,
    api_key: str,
    use_llm_fallback: bool = True,
    **kwargs,  # 하위 호환성 (use_semantic_router 등 무시)
) -> tuple[IntentCategory, list[str]]:
    """
    질문 분류 및 도구 목록 반환 (원스톱)

    분류 순서:
    1. 키워드 기반 분류 (가장 빠름, 비용 없음)
    2. LLM Router (gpt-4o-mini, fallback)

    Args:
        text: 사용자 질문
        api_key: OpenAI API 키
        use_llm_fallback: 키워드 분류 실패 시 LLM 사용 여부

    Returns:
        (카테고리, 도구 이름 목록)
    """
    # 1단계: 키워드 기반 분류 (가장 빠름)
    category = _keyword_classify(text)

    if category is not None:
        st.logger.info(
            "ROUTER_KEYWORD query=%s category=%s",
            text[:40], category.value,
        )
        tools = get_tools_for_category(category)
        return category, tools

    # 2단계: LLM Router (키워드 분류 실패 시 fallback)
    if use_llm_fallback and api_key:
        category = route_intent_llm(text, api_key)
        st.logger.info(
            "ROUTER_LLM query=%s category=%s",
            text[:40], category.value,
        )
    else:
        # 기본값: GENERAL
        category = IntentCategory.GENERAL
        st.logger.info(
            "ROUTER_DEFAULT query=%s category=general",
            text[:40],
        )

    tools = get_tools_for_category(category)
    return category, tools
