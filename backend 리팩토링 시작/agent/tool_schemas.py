"""
agent/tool_schemas.py - CAFE24 AI 운영 플랫폼 LLM Tool Calling 도구 정의
===================================================================
CAFE24 AI 운영 플랫폼

LangChain @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 도구들을 정의합니다.
"""
from typing import Optional
from langchain_core.tools import tool

from agent.tools import (
    tool_get_shop_info,
    tool_list_shops,
    tool_get_shop_services,
    tool_get_category_info,
    tool_list_categories,
    tool_auto_reply_cs,
    tool_check_cs_quality,
    tool_get_ecommerce_glossary,
    tool_analyze_seller,
    tool_get_seller_segment,
    tool_detect_fraud,
    tool_get_segment_statistics,
    tool_get_fraud_statistics,
    tool_get_order_statistics,
    tool_get_seller_activity_report,
    tool_classify_inquiry,
    tool_get_cs_statistics,
    tool_get_dashboard_summary,
    # ML 모델 예측 도구
    tool_predict_seller_churn,
    tool_predict_shop_revenue,
    tool_get_shop_performance,
    tool_optimize_marketing,
    # 분석 도구
    tool_get_churn_prediction,
    tool_get_cohort_analysis,
    tool_get_trend_analysis,
    tool_get_gmv_prediction,
)
from rag.service import tool_rag_search
from rag.light_rag import lightrag_search_sync, get_lightrag_status
import state as st


# ============================================================
# 쇼핑몰(샵) 도구
# ============================================================
@tool
def get_shop_info(shop_id: str) -> dict:
    """
    특정 쇼핑몰(샵)의 상세 정보를 조회합니다.

    Args:
        shop_id: 쇼핑몰 ID (예: S0001, S0042)

    Returns:
        쇼핑몰의 이름, 카테고리, 티어, 지역, 매출 현황 등 상세 정보
    """
    return tool_get_shop_info(shop_id)


@tool
def list_shops(
    category: Optional[str] = None,
    tier: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    """
    쇼핑몰(샵) 목록을 조회합니다. 카테고리, 티어, 지역으로 필터링할 수 있습니다.

    Args:
        category: 카테고리 필터 (예: 패션, 뷰티, 식품, 가전, 리빙, 디지털)
        tier: 티어 필터 (예: 프리미엄, 스탠다드, 베이직, 엔터프라이즈)
        region: 지역 필터 (예: 국내, 해외, 글로벌)

    Returns:
        필터링된 쇼핑몰 목록
    """
    return tool_list_shops(category=category, tier=tier, region=region)


@tool
def get_shop_services(shop_id: str) -> dict:
    """
    특정 쇼핑몰의 이용 중인 서비스/앱 정보를 조회합니다.

    Args:
        shop_id: 쇼핑몰 ID (예: S0001)

    Returns:
        이용 중인 서비스, 앱, 부가 기능 목록
    """
    return tool_get_shop_services(shop_id)


# ============================================================
# 카테고리/업종 도구
# ============================================================
@tool
def get_category_info(category_id: str) -> dict:
    """
    특정 카테고리(업종)의 상세 정보를 조회합니다.

    Args:
        category_id: 카테고리 ID 또는 이름 (예: CAT001, 패션)

    Returns:
        카테고리의 이름, 설명, 소속 쇼핑몰 수 등
    """
    return tool_get_category_info(category_id)


@tool
def list_categories() -> dict:
    """
    모든 카테고리(업종) 목록을 조회합니다.

    Returns:
        카테고리 목록과 기본 정보
    """
    return tool_list_categories()


# ============================================================
# CS(고객 상담) 도구
# ============================================================
@tool
def auto_reply_cs(
    inquiry_text: str,
    category: str = "general",
) -> dict:
    """
    CS 문의에 대한 자동 응답을 생성합니다.
    카페24 플랫폼 정책과 이커머스 용어를 반영합니다.

    Args:
        inquiry_text: 고객 문의 텍스트
        category: 문의 카테고리 (general, order, delivery, refund, payment, product, account 등)

    Returns:
        자동 생성된 CS 응답 텍스트 및 관련 정책 안내
    """
    return tool_auto_reply_cs(inquiry_text, category)


@tool
def check_cs_quality(
    ticket_category: str,
    seller_tier: str,
    sentiment_score: float,
    order_value: float,
    is_repeat: bool = False,
    text_length: int = 0,
) -> dict:
    """
    CS 응답 품질을 평가합니다.

    Args:
        ticket_category: 문의 카테고리 (order, delivery, refund, payment 등)
        seller_tier: 셀러 티어 (프리미엄, 스탠다드, 베이직)
        sentiment_score: 고객 감성 점수 (-1.0 ~ 1.0)
        order_value: 주문 금액
        is_repeat: 반복 문의 여부
        text_length: 응답 텍스트 길이

    Returns:
        품질 등급 (excellent/good/acceptable/needs_review), 우선순위, 권장사항
    """
    return tool_check_cs_quality(ticket_category, seller_tier, sentiment_score, order_value, is_repeat, text_length)


@tool
def get_ecommerce_glossary(term: Optional[str] = None) -> dict:
    """
    카페24 이커머스 용어집을 조회합니다.

    Args:
        term: 검색할 용어 (선택사항, 예: GMV, ARPU, 정산, 배송, 환불 등)
              지정하지 않으면 전체 용어 목록을 반환합니다.

    Returns:
        이커머스 플랫폼 용어와 설명
    """
    return tool_get_ecommerce_glossary(term=term)


@tool
def get_cs_statistics() -> dict:
    """
    CS 상담 데이터 통계를 조회합니다.

    Returns:
        카테고리별/채널별 CS 통계, 응답 품질 분포, 평균 처리 시간
    """
    return tool_get_cs_statistics()


# ============================================================
# 셀러 분석 도구
# ============================================================
@tool
def analyze_seller(seller_id: str) -> dict:
    """
    ★ 특정 셀러 분석 질문은 이 도구를 사용하세요! ★

    특정 셀러의 운영 패턴을 분석합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "SEL0001 셀러 분석해줘", "이 셀러 정보 알려줘"
    - "SEL0042 셀러 세그먼트가 뭐야?"
    - "특정 셀러 운영 패턴"

    Args:
        seller_id: 셀러 ID (예: SEL0001, SEL0042)

    Returns:
        셀러 세그먼트, 운영 지표(주문 수, GMV, 반품률, CS 건수, 정산), 이상 여부
    """
    return tool_analyze_seller(seller_id)


@tool
def get_seller_segment(seller_id: str) -> dict:
    """
    셀러 피처를 기반으로 세그먼트를 분류합니다.

    Args:
        seller_id: 셀러 ID (예: SEL0001)

    Returns:
        세그먼트 분류 결과 (파워셀러/성장셀러/일반셀러/신규셀러/휴면셀러)
    """
    return tool_get_seller_segment(seller_id)


@tool
def detect_fraud(seller_id: str) -> dict:
    """
    셀러 또는 주문의 이상/부정행위 여부를 탐지합니다.

    Args:
        seller_id: 셀러 ID (예: SEL0001)

    Returns:
        이상 여부, 이상 점수, 위험 수준, 의심 유형 (허위 주문, 리뷰 조작, 비정상 환불 등)
    """
    return tool_detect_fraud(seller_id)


@tool
def get_segment_statistics() -> dict:
    """
    셀러 세그먼트별 통계를 조회합니다.

    Returns:
        세그먼트별 셀러 수, 평균 GMV, 평균 주문 수, 이상 비율
    """
    return tool_get_segment_statistics()


@tool
def get_fraud_statistics() -> dict:
    """
    전체 부정행위/이상 탐지 통계를 조회합니다.

    Returns:
        이상 셀러 수, 이상 비율, 유형별 분포, 이상 셀러 샘플
    """
    return tool_get_fraud_statistics()


@tool
def get_seller_activity_report(seller_id: str, days: int = 30) -> dict:
    """
    특정 셀러의 활동 리포트를 생성합니다.

    Args:
        seller_id: 셀러 ID
        days: 조회할 기간 (기본값: 30일)

    Returns:
        활동 요약, 주문/배송/CS/정산 집계
    """
    return tool_get_seller_activity_report(seller_id, days)


# ============================================================
# 주문/거래 도구
# ============================================================
@tool
def get_order_statistics(event_type: Optional[str] = None, days: int = 30) -> dict:
    """
    운영 이벤트 통계를 조회합니다.

    Args:
        event_type: 이벤트 타입 필터. 가능한 값: order_received(주문), payment_settled(정산), refund_processed(환불), cs_ticket(CS문의), login(로그인), marketing_campaign(마케팅), product_listed(상품등록), product_updated(상품수정). None이면 전체 이벤트.
        days: 조회할 기간 (기본값: 30일)

    Returns:
        이벤트 타입별 집계, 일별 추이
    """
    return tool_get_order_statistics(event_type, days)


# ============================================================
# 문의 분류 도구
# ============================================================
@tool
def classify_inquiry(text: str) -> dict:
    """
    CS 문의 텍스트의 카테고리를 분류합니다.

    Args:
        text: 분류할 문의 텍스트

    Returns:
        예측 카테고리 (주문, 배송, 환불, 결제, 상품, 계정 등), 신뢰도, 상위 3개 카테고리
    """
    return tool_classify_inquiry(text)


# ============================================================
# RAG 검색 도구
# ============================================================
@tool
def search_platform(query: str, top_k: int = 5) -> dict:
    """
    카페24 플랫폼/정책/기능/운영에 대한 '지식' 질문에만 사용합니다.
    (기본 임베딩 검색, LightRAG보다 가벼움)

    ⚠️ 사용 금지 케이스 (다른 도구 사용):
    - 셀러/이탈 분석 → get_churn_prediction, analyze_seller
    - 매출/GMV → get_gmv_prediction
    - 코호트/리텐션 → get_cohort_analysis
    - DAU/KPI → get_trend_analysis
    - 쇼핑몰 성과/매출 → get_shop_performance

    ✅ 사용해야 하는 케이스:
    - "카페24 정산 주기란?", "배송비 정책이 뭐야?"
    - "쇼핑몰 개설 방법", "카페24 앱스토어란?"

    Args:
        query: 검색 질의 (플랫폼/정책/기능 관련)
        top_k: 검색 결과 개수 (기본값: 5)

    Returns:
        관련 문서 스니펫과 출처
    """
    return tool_rag_search(query, top_k=top_k, api_key=st.OPENAI_API_KEY)


@tool
def search_platform_lightrag(query: str, mode: str = "hybrid", top_k: int = 5) -> dict:
    """
    카페24 플랫폼/정책/기능/운영에 대한 '지식' 질문에만 사용합니다.

    ⚠️ 사용 금지 케이스 (다른 도구 사용):
    - 셀러 통계/이탈 예측 → get_churn_prediction 사용
    - 매출/GMV 분석 → get_gmv_prediction 사용
    - 코호트/리텐션 → get_cohort_analysis 사용
    - DAU/KPI 트렌드 → get_trend_analysis 사용
    - 특정 셀러 분석 → analyze_seller, predict_seller_churn 사용
    - 쇼핑몰 성과/매출 → get_shop_performance 사용

    ✅ 사용해야 하는 케이스:
    - "카페24 정산 주기란?", "배송비 정책이 뭐야?"
    - "쇼핑몰 개설 절차", "카페24 앱스토어 연동 방법"
    - "멀티쇼핑몰 관리 기능", "해외 배송 정책"

    검색 모드:
    - "local": 구체적인 엔티티/기능 중심 (예: "카페24 정산 주기")
    - "global": 추상적인 테마/개념 중심 (예: "카페24 플랫폼 전체 구조")
    - "hybrid": local + global 조합 (권장, 기본값)

    Args:
        query: 검색 질의 (플랫폼/정책/기능 관련)
        mode: 검색 모드 ("local", "global", "hybrid", "naive")
        top_k: 검색 결과 개수 (기본값: 5)

    Returns:
        관련 문서 및 엔티티 정보
    """
    # LightRAG 상태 확인
    status = get_lightrag_status()
    if not status.get("ready"):
        return {
            "status": "FAILED",
            "error": "LightRAG가 준비되지 않았습니다. 먼저 인덱스를 빌드해주세요.",
            "lightrag_status": status
        }

    return lightrag_search_sync(query, mode=mode)  # top_k는 state.LIGHTRAG_CONFIG에서 관리


# ============================================================
# 대시보드 도구
# ============================================================
@tool
def get_dashboard_summary() -> dict:
    """
    대시보드 요약 정보를 조회합니다.

    Returns:
        쇼핑몰/셀러/CS/주문/정산 통계 요약
    """
    return tool_get_dashboard_summary()


# ============================================================
# ML 모델 예측 도구
# ============================================================
@tool
def predict_seller_churn(seller_id: str) -> dict:
    """
    특정 셀러의 이탈 확률을 예측합니다.
    ML 모델(LightGBM)과 SHAP Explainer를 사용하여 예측 및 주요 이탈 요인을 분석합니다.

    사용 예시:
    - "SEL0001 셀러 이탈 예측해줘" → predict_seller_churn(seller_id="SEL0001")
    - "이 셀러가 이탈할 확률은?" → predict_seller_churn(seller_id="SEL0001")

    Args:
        seller_id: 셀러 ID (예: SEL0001)

    Returns:
        이탈 확률(%), 위험 수준(HIGH/MEDIUM/LOW), 주요 이탈 요인, 권장 조치
    """
    return tool_predict_seller_churn(seller_id)


@tool
def predict_shop_revenue(shop_id: str) -> dict:
    """
    쇼핑몰의 운영 데이터를 기반으로 예상 매출을 예측합니다.
    LightGBM 회귀 모델을 사용합니다.

    사용 예시:
    - 특정 쇼핑몰의 예상 월매출 계산
    - 운영 지표 변경 시 매출 변화 시뮬레이션

    Args:
        shop_id: 쇼핑몰 ID (예: S0001)

    Returns:
        예측 월매출, 성장률, 주요 매출 기여 요인 분석
    """
    return tool_predict_shop_revenue(shop_id)


@tool
def get_shop_performance(shop_id: str) -> dict:
    """
    특정 쇼핑몰의 현재 운영 데이터를 기반으로 성과를 분석합니다.
    실제 운영 데이터와 ML 모델을 결합하여 분석합니다.

    사용 예시:
    - "S0001 쇼핑몰 성과 어때?" → get_shop_performance(shop_id="S0001")
    - "S0042 매출 분석해줘" → get_shop_performance(shop_id="S0042")

    Args:
        shop_id: 쇼핑몰 ID (예: S0001, S0042)

    Returns:
        쇼핑몰 정보, 실제 매출, 예측 매출, 성과 등급, 주요 지표
    """
    return tool_get_shop_performance(shop_id)


@tool
def optimize_marketing(
    seller_id: str,
    budget: Optional[float] = None,
    goal: str = "maximize_revenue",
) -> dict:
    """
    셀러의 데이터를 분석하여 최적의 마케팅 전략을 제안합니다.
    P-PSO(Phasor Particle Swarm Optimization) 알고리즘을 사용합니다.

    사용 예시:
    - "마케팅 예산 어디에 쓰면 좋을까?" → optimize_marketing(seller_id="SEL0001")
    - "매출 최대화 마케팅 추천해줘" → optimize_marketing(seller_id="SEL0001", goal="maximize_revenue")
    - "효율적인 광고 전략 알려줘" → optimize_marketing(seller_id="SEL0001", goal="maximize_roas")

    Args:
        seller_id: 셀러 ID (예: SEL0001)
        budget: 마케팅 예산 (선택사항, 없으면 현재 예산 기준)
        goal: 최적화 목표
            - maximize_revenue: 매출 최대화 (기본값)
            - maximize_roas: 광고 수익률 최대화
            - balanced: 균형 잡힌 마케팅

    Returns:
        마케팅 채널별 투자 추천 (최대 10개), 예상 GMV 증가, 예상 ROAS, 필요 예산
    """
    return tool_optimize_marketing(seller_id, budget, goal)


# ============================================================
# 분석 도구 (Analysis Tools)
# ============================================================
@tool
def get_churn_prediction(risk_level: str = None, limit: int = None) -> dict:
    """
    ★ 셀러 이탈 관련 질문은 이 도구를 사용하세요! ★

    전체 셀러의 이탈 예측 분석을 조회합니다.
    고위험/중위험/저위험 이탈 셀러 수와 주요 이탈 요인을 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "이탈 셀러 현황", "이탈 예측 분석"
    - "고위험 셀러 몇 명이야?", "중위험 이탈 셀러"
    - "이탈률 알려줘", "이탈 요인이 뭐야?"
    - "어떤 셀러가 이탈할 것 같아?"

    Args:
        risk_level: 특정 위험 등급만 필터 ("high", "medium", "low")
            - "high": 고위험 셀러만 조회
            - "medium": 중위험 셀러만 조회
            - "low": 저위험 셀러만 조회
        limit: 상세 셀러 목록 반환 시 최대 개수 (기본값: 10)

    사용 예시:
    - "이탈 예측 분석 보여줘" → get_churn_prediction()
    - "고위험 이탈 셀러 목록" → get_churn_prediction(risk_level="high")
    - "중위험 이탈 셀러 현황" → get_churn_prediction(risk_level="medium")

    Returns:
        고위험/중위험/저위험 셀러 수, 예상 이탈률, 주요 이탈 요인 5개, 인사이트
    """
    return tool_get_churn_prediction(risk_level=risk_level, limit=limit)


@tool
def get_cohort_analysis(month: str = None) -> dict:
    """
    ★ 코호트/리텐션 관련 질문은 이 도구를 사용하세요! ★

    코호트 리텐션 분석을 조회합니다.
    월별 코호트의 Week1/Week2/Week4/Week8/Week12 리텐션율을 반환합니다.
    데이터 범위: 2024-07 ~ 2024-12

    ✅ 이 도구를 사용해야 하는 질문:
    - "코호트 분석 보여줘", "리텐션 현황 알려줘"
    - "Week4 리텐션 얼마야?", "Week1 잔존율은?"
    - "2024-11 코호트 분석", "11월 리텐션 현황"
    - "전체 코호트 평균 리텐션"

    Args:
        month: 특정 월 필터 (예: "2024-11", "2024-07"). 미지정 시 전체 코호트 반환.

    Returns:
        코호트별 주차 리텐션율, 전체 평균 리텐션, 인사이트
    """
    return tool_get_cohort_analysis(month=month)


@tool
def get_trend_analysis(start_date: str = None, end_date: str = None, days: int = None) -> dict:
    """
    ★ DAU/KPI/트렌드 관련 질문은 이 도구를 사용하세요! ★

    트렌드 KPI 분석을 조회합니다.
    주요 지표(활성 셀러 수, ARPU, 신규 가입, 주문 수 등)의 변화율과 상관관계를 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "활성 셀러 수 얼마야?", "오늘 활성 셀러 수"
    - "트렌드 분석 보여줘", "KPI 변화율"
    - "신규 가입자 수", "주문량 분석"
    - "지표 변화 추이", "최근 7일 활성 셀러"

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        days: 최근 N일 분석 (기본값: 7일)

    Returns:
        KPI별 현재/이전 값, 변화율, 주요 상관관계, 인사이트
    """
    return tool_get_trend_analysis(start_date=start_date, end_date=end_date, days=days)


@tool
def get_gmv_prediction(days: int = None, start_date: str = None, end_date: str = None) -> dict:
    """
    ★ 매출/GMV 관련 질문은 이 도구를 사용하세요! ★

    GMV(총 거래액) 예측 분석을 조회합니다.
    예상 GMV, ARPU/ARPPU, 셀러 티어별 거래 분포를 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "GMV 예측 보여줘", "이번 달 예상 거래액은?"
    - "ARPU 얼마야?", "ARPPU 분석해줘"
    - "프리미엄 셀러 몇 명?", "셀러 티어별 거래 분포"
    - "매출 성장률은?", "GMV 분석해줘"

    Args:
        days: 최근 N일 기준 분석 (기본값: 30일)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        예상 월간 GMV, 성장률, ARPU/ARPPU, 셀러 티어별 거래 분포, 인사이트
    """
    return tool_get_gmv_prediction(days=days, start_date=start_date, end_date=end_date)


# ============================================================
# 에이전트별 도구 분류 (Multi-Agent용)
# ============================================================

# 검색 에이전트 도구: 플랫폼 정보 검색 (쇼핑몰, 카테고리, RAG)
SEARCH_AGENT_TOOLS = [
    get_shop_info,
    list_shops,
    get_shop_services,
    get_category_info,
    list_categories,
    search_platform,
    search_platform_lightrag,
]

# 분석 에이전트 도구: 셀러 분석, ML 예측, 통계
ANALYSIS_AGENT_TOOLS = [
    analyze_seller,
    get_seller_segment,
    detect_fraud,
    predict_seller_churn,
    predict_shop_revenue,
    get_shop_performance,
    optimize_marketing,
    # 분석
    get_churn_prediction,
    get_cohort_analysis,
    get_trend_analysis,
    get_gmv_prediction,
    get_dashboard_summary,
    get_segment_statistics,
    get_fraud_statistics,
    get_order_statistics,
    get_seller_activity_report,
]

# CS 에이전트 도구: CS 응답, 품질 평가, 문의 분류
CS_AGENT_TOOLS = [
    auto_reply_cs,
    check_cs_quality,
    get_ecommerce_glossary,
    get_cs_statistics,
    classify_inquiry,
]
TRANSLATION_AGENT_TOOLS = CS_AGENT_TOOLS  # multi_agent.py 호환 alias

# ============================================================
# 모든 도구 리스트 (LLM에 바인딩할 때 사용)
# ============================================================
ALL_TOOLS = [
    # 쇼핑몰 정보
    get_shop_info,
    list_shops,
    get_shop_services,
    # 카테고리 정보
    get_category_info,
    list_categories,
    # CS (고객 상담)
    auto_reply_cs,
    check_cs_quality,
    get_ecommerce_glossary,
    get_cs_statistics,
    # 셀러 분석
    analyze_seller,
    get_seller_segment,
    detect_fraud,
    get_segment_statistics,
    get_fraud_statistics,
    get_seller_activity_report,
    # 주문/거래
    get_order_statistics,
    # 문의 분류
    classify_inquiry,
    # RAG 검색
    search_platform,
    search_platform_lightrag,
    # 대시보드
    get_dashboard_summary,
    # ML 모델 예측
    predict_seller_churn,
    predict_shop_revenue,
    get_shop_performance,
    optimize_marketing,
    # 분석 도구
    get_churn_prediction,
    get_cohort_analysis,
    get_trend_analysis,
    get_gmv_prediction,
]
