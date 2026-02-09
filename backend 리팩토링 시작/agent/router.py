"""
agent/router.py - CAFE24 AI 운영 플랫폼 LLM Router (의도 분류 전용)
============================================================
CAFE24 AI 운영 플랫폼

질문을 먼저 분류한 뒤, 해당 카테고리의 도구만 Executor에 노출합니다.

분류 우선순위:
1. 키워드 기반 분류 (가장 빠름, 비용 없음)
2. Semantic Router (임베딩 유사도, LLM 호출 없음)
3. LLM Router (gpt-4o-mini, fallback)

References:
- https://www.anthropic.com/research/building-effective-agents
- https://github.com/aurelio-labs/semantic-router
"""
import re
from typing import Literal, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str
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
# ============================================================
_ANALYSIS_KEYWORDS = [
    "매출", "수익", "revenue", "arpu", "arppu", "과금", "성장률",
    "gmv", "거래액", "총거래", "거래 금액",
    "이탈", "churn", "고위험", "중위험", "저위험", "이탈률", "이탈 요인",
    "코호트", "cohort", "리텐션", "retention", "잔존", "week1", "week4",
    "트렌드", "trend", "kpi", "dau", "mau", "wau", "지표", "변화율",
    "활성 셀러", "신규 가입", "가입 추이", "전환율", "변화 분석", "추이 분석",
    "신규 쇼핑몰", "주문량", "주문 수", "결제 분석",
]

_PLATFORM_KEYWORDS = [
    # 핵심 플랫폼 키워드 (RAG 검색 필수)
    "플랫폼", "정책", "기능", "운영", "가이드", "도움말", "사용법",
    "정산", "정산 주기", "수수료", "배송비 정책", "반품 정책", "환불 정책",
    "앱스토어", "앱 연동", "API", "개발자", "테마", "디자인",
    "쇼핑몰 개설", "멀티쇼핑몰", "해외 배송", "글로벌 커머스",
    # 질문 패턴 (플랫폼 지식 요청)
    "뜻", "용어", "설명", "정의", "개념", "meaning", "definition",
    "뭐야", "무엇", "어떤", "알려줘", "어떻게 됐", "왜 그런",
]

_SHOP_KEYWORDS = [
    "쇼핑몰 정보", "쇼핑몰 서비스", "쇼핑몰 성과", "쇼핑몰 매출",
    "쇼핑몰 목록", "쇼핑몰 리스트", "쇼핑몰 현황", "쇼핑몰 분포",
    "쇼핑몰 플랜", "쇼핑몰 등급", "쇼핑몰 티어",
    "쇼핑몰 수", "쇼핑몰 개수", "쇼핑몰 몇", "쇼핑몰 통계",
    "전체 쇼핑몰", "총 쇼핑몰", "쇼핑몰 총",
    "샵 정보", "샵 성과", "샵 매출", "샵 목록", "shop",
    "카테고리별", "업종별", "티어별", "플랜별", "등급별",
    "카테고리 목록", "카테고리 전체", "카테고리 정보", "카테고리 현황",
    "업종 목록", "업종 전체", "업종 정보", "업종 현황",
    "패션", "뷰티", "식품", "가전", "리빙", "디지털",
    "프리미엄", "스탠다드", "베이직", "엔터프라이즈",
    "premium", "standard", "basic", "enterprise",
    "마케팅", "광고", "ROAS", "마케팅 최적화", "광고 전략",
]

_SELLER_KEYWORDS = [
    "셀러 분석", "셀러 정보", "셀러 이탈", "셀러 예측", "판매자 분석", "입점업체",
    "세그먼트", "군집",
    # 세그먼트 이름 (CSV 기준: 성장형/휴면/우수/파워/관리필요)
    "성장형 셀러", "휴면 셀러", "우수 셀러", "파워 셀러", "관리 필요 셀러",
    "성장형", "휴면", "관리 필요",
    # 이상/부정행위
    "이상 셀러", "부정행위", "이상 탐지", "이상거래", "이상 거래", "어뷰징", "사기", "비정상", "허위 주문", "리뷰 조작",
    "fraud", "anomaly", "부정 거래", "사기 탐지", "사기 거래",
    "SEL0", "SEL1", "SEL2", "SEL3", "SEL4", "SEL5", "SEL6", "SEL7", "SEL8", "SEL9",  # 셀러 ID 패턴
]

_CS_KEYWORDS = [
    "CS", "고객 상담", "문의", "상담", "자동 응답", "자동응답",
    "환불", "반품", "교환", "배송 문의", "결제 문의",
    "CS 품질", "상담 품질", "용어집",
    "문의 분류", "카테고리 분류", "분류해", "티켓", "응답 생성",
    "결제 오류", "카드 오류", "배송 지연", "환불 요청", "교환 요청",
]

_DASHBOARD_KEYWORDS = [
    "대시보드", "dashboard", "전체 현황", "요약",
    "셀러 활동", "활동 현황", "주문 현황", "정산 현황",
    "운영 이벤트", "이벤트 통계", "이벤트 현황", "주문 이벤트", "정산 이벤트",
]

_GENERAL_KEYWORDS = [
    "안녕", "하이", "헬로", "hi", "hello",
    "고마워", "감사", "thanks",
    "뭐해", "누구", "자기소개",
]


def _keyword_classify(text: str) -> Optional[IntentCategory]:
    """
    키워드 기반 빠른 분류 (LLM 호출 없이)

    Returns:
        IntentCategory or None (불확실한 경우)
    """
    t = text.lower()

    # 우선순위: 셀러ID감지 > 분석 > 셀러 > 쇼핑몰 > CS > 플랫폼 > 대시보드 > 일반

    # 0. 셀러 ID(SEL0001)가 포함되면 SELLER 우선 (분석보다 높은 우선순위)
    if re.search(r'SEL\d{1,6}', text, re.IGNORECASE):
        return IntentCategory.SELLER

    # 1. 분석 키워드 (셀러 ID 없는 일반 분석)
    if any(kw in t for kw in _ANALYSIS_KEYWORDS):
        return IntentCategory.ANALYSIS

    # 2. 셀러 분석 키워드
    if any(kw in t for kw in _SELLER_KEYWORDS):
        # 셀러 관련 키워드 체크
        if re.search(r'SEL\d{1,6}', text, re.IGNORECASE):
            return IntentCategory.SELLER
        if any(kw in t for kw in ["셀러", "판매자", "입점업체", "세그먼트", "부정행위", "이상", "사기",
                                    "성장형", "휴면", "우수", "파워", "관리 필요"]):
            return IntentCategory.SELLER

    # 3. 쇼핑몰 관련 (성과, 정보, 마케팅)
    if any(kw in t for kw in _SHOP_KEYWORDS):
        return IntentCategory.SHOP

    # 4. CS 관련
    if any(kw in t for kw in _CS_KEYWORDS):
        return IntentCategory.CS

    # 5. 대시보드 관련
    if any(kw in t for kw in _DASHBOARD_KEYWORDS):
        return IntentCategory.DASHBOARD

    # 6. 플랫폼 관련 (정책, 기능, 용어)
    if any(kw in t for kw in _PLATFORM_KEYWORDS):
        return IntentCategory.PLATFORM

    # 7. 일반 대화
    if any(kw in t for kw in _GENERAL_KEYWORDS):
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

## 출력 형식

카테고리명만 반환하세요. 예: analysis"""


async def route_intent_llm(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",  # 빠르고 저렴한 모델 사용
) -> IntentCategory:
    """
    LLM을 사용한 의도 분류 (키워드 분류 실패 시)

    Args:
        text: 사용자 질문
        api_key: OpenAI API 키
        model: 사용할 모델 (기본: gpt-4o-mini)

    Returns:
        IntentCategory
    """
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0,  # 결정론적 분류
            max_tokens=20,  # 카테고리명만 반환
        )

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"질문: {text}"),
        ]

        response = llm.invoke(messages)
        category_str = safe_str(response.content).strip().lower()

        # 카테고리 매핑
        category_map = {
            "analysis": IntentCategory.ANALYSIS,
            "platform": IntentCategory.PLATFORM,
            "shop": IntentCategory.SHOP,
            "seller": IntentCategory.SELLER,
            "cs": IntentCategory.CS,
            "dashboard": IntentCategory.DASHBOARD,
            "general": IntentCategory.GENERAL,
        }

        category = category_map.get(category_str, IntentCategory.GENERAL)

        st.logger.info(
            "ROUTER_LLM_CLASSIFY query=%s result=%s",
            text[:50], category.value,
        )

        return category

    except Exception as e:
        st.logger.warning("ROUTER_LLM_FAIL err=%s, fallback=general", safe_str(e))
        return IntentCategory.GENERAL


def route_intent_sync(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> IntentCategory:
    """
    동기 버전의 의도 분류

    1. 키워드 기반 빠른 분류 시도
    2. 실패 시 LLM Router 사용
    """
    # 1단계: 키워드 기반 빠른 분류
    category = _keyword_classify(text)

    if category is not None:
        st.logger.info(
            "ROUTER_KEYWORD_CLASSIFY query=%s result=%s",
            text[:50], category.value,
        )
        return category

    # 2단계: LLM Router (키워드 분류 실패 시)
    # 동기 환경에서는 asyncio.run 사용
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 이벤트 루프가 실행 중이면 새 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    route_intent_llm(text, api_key, model)
                )
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(route_intent_llm(text, api_key, model))
    except Exception:
        return asyncio.run(route_intent_llm(text, api_key, model))


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
        category = route_intent_sync(text, api_key)
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
