"""
agent/intent.py - CAFE24 AI 운영 플랫폼 인텐트 감지 및 도구 라우팅
============================================================
CAFE24 AI 운영 플랫폼

사용자 입력을 분석하여 적절한 도구를 실행합니다.
"""
import time
from typing import Optional, Dict, Any, Tuple

from core.constants import RAG_DOCUMENTS, SUMMARY_TRIGGERS
from core.utils import safe_str
from agent.tools import (
    tool_get_shop_info,
    tool_list_shops,
    tool_get_category_info,
    tool_list_categories,
    tool_auto_reply_cs,
    tool_get_ecommerce_glossary,
    tool_analyze_seller,
    tool_get_segment_statistics,
    tool_get_order_statistics,
    tool_classify_inquiry,
    tool_get_dashboard_summary,
)
from rag.service import tool_rag_search
import state as st


# ============================================================
# 요약 트리거 / 컨텍스트 재활용
# ============================================================
def _has_summary_trigger(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k.lower() in t for k in SUMMARY_TRIGGERS)


def set_last_context(username: str, context_id: Optional[str], results: Dict[str, Any], user_text: str, mode: str) -> None:
    if not username:
        return
    if not isinstance(results, dict) or len(results) == 0:
        return
    with st.LAST_CONTEXT_LOCK:
        st.LAST_CONTEXT_STORE[username] = {
            "context_id": safe_str(context_id).strip() if context_id else "",
            "results": results,
            "user_text": safe_str(user_text),
            "ts": time.time(),
            "mode": safe_str(mode),
        }


def get_last_context(username: str) -> Optional[Dict[str, Any]]:
    if not username:
        return None
    with st.LAST_CONTEXT_LOCK:
        return st.LAST_CONTEXT_STORE.get(username)


def can_reuse_last_context(username: str, context_id: Optional[str], user_text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not _has_summary_trigger(user_text):
        return (False, None)

    ctx = get_last_context(username)
    if not ctx or not isinstance(ctx.get("results"), dict) or len(ctx["results"]) == 0:
        return (False, None)

    ts = float(ctx.get("ts") or 0.0)
    if ts > 0 and (time.time() - ts) > st.LAST_CONTEXT_TTL_SEC:
        return (False, None)

    # 컨텍스트 ID가 일치하면 재사용
    req_id = safe_str(context_id).strip() if context_id else ""
    last_id = safe_str(ctx.get("context_id")).strip()

    if req_id and req_id == last_id:
        return (True, ctx)

    # 이전 결과가 있으면 재사용
    r = ctx.get("results") or {}
    context_keys = ("get_shop_info", "analyze_seller", "auto_reply_cs", "rag_search", "dashboard")
    if any(isinstance(r.get(k), dict) for k in context_keys):
        return (True, ctx)

    return (False, None)


# ============================================================
# 인텐트 감지
# ============================================================
def detect_intent(user_text: str) -> Dict[str, bool]:
    t = (user_text or "").strip().lower()

    # RAG 트리거
    rag_triggers = ["뜻", "용어", "설명", "정의", "개념", "meaning", "definition", "플랫폼", "정책", "가이드"]

    # 문서 키워드 검색
    has_doc_keyword = False
    for _, doc in RAG_DOCUMENTS.items():
        for kw in doc.get("keywords", []):
            kw2 = (kw or "").strip().lower()
            if kw2 and (kw2 in t):
                has_doc_keyword = True
                break
        if has_doc_keyword:
            break

    # 대시보드/현황
    status_triggers = ["현황", "대시보드", "dashboard", "통계", "요약", "summary"]
    force_full_status = any(x in t for x in status_triggers)

    # 분석/예측 관련 키워드 (RAG 불필요)
    analytics_keywords = [
        "이탈", "churn", "코호트", "cohort", "리텐션", "retention", "잔존",
        "트렌드", "trend", "kpi", "dau", "mau", "wau",
        "매출", "revenue", "arpu", "arppu", "과금", "성장률",
        "gmv", "거래액", "예측", "predict",
        "마케팅", "광고", "roas", "마케팅 최적화",
    ]
    want_analytics = any(kw in t for kw in analytics_keywords)

    return {
        # 쇼핑몰 관련
        "want_shop_info": ("쇼핑몰" in t or "샵" in t) and (("정보" in t) or ("누구" in t) or ("알려" in t)),
        "want_shop_list": ("쇼핑몰" in t or "샵" in t) and (("목록" in t) or ("리스트" in t) or ("전체" in t)),

        # 카테고리 관련
        "want_category_info": ("카테고리" in t or "업종" in t) and (("정보" in t) or ("어디" in t) or ("알려" in t)),
        "want_category_list": ("카테고리" in t or "업종" in t) and (("목록" in t) or ("리스트" in t) or ("전체" in t)),

        # CS 관련
        "want_cs_reply": ("문의" in t or "상담" in t or "cs" in t) and ("응답" in t or "답변" in t or "자동" in t),
        "want_glossary": ("용어집" in t) or ("용어" in t and "이커머스" in t) or ("용어" in t and "커머스" in t),

        # 셀러 분석
        "want_seller_analysis": ("셀러" in t or "판매자" in t or "입점" in t) and ("분석" in t or "정보" in t),
        "want_segment": ("세그먼트" in t) or ("군집" in t) or ("분류" in t and "셀러" in t),
        "want_fraud_detection": ("부정행위" in t) or ("사기" in t) or ("어뷰징" in t) or ("비정상" in t) or ("허위" in t) or ("리뷰 조작" in t),

        # 주문/거래 통계
        "want_order_stats": ("주문" in t and ("통계" in t or "현황" in t)) or ("거래" in t and "통계" in t) or ("배송" in t and "현황" in t),

        # 문의 분류
        "want_classify": ("분류" in t and ("문의" in t or "텍스트" in t)) or ("카테고리" in t and "분류" in t),

        # RAG 검색
        "want_rag": any(x in t for x in rag_triggers) or has_doc_keyword,

        # 대시보드
        "want_dashboard": force_full_status,

        # 분석/예측 (RAG 불필요)
        "want_analytics": want_analytics,
    }


# ============================================================
# 셀러 ID 추출 (SEL0001 패턴)
# ============================================================
def extract_seller_id(user_text: str) -> Optional[str]:
    """텍스트에서 셀러 ID를 추출합니다."""
    if not user_text:
        return None

    import re
    pattern = r'SEL\d{1,6}'
    match = re.search(pattern, user_text.upper())
    if match:
        return match.group()

    return None


# ============================================================
# 쇼핑몰 ID 추출 (S0001 패턴)
# ============================================================
def extract_shop_id(user_text: str) -> Optional[str]:
    """텍스트에서 쇼핑몰 ID를 추출합니다."""
    if not user_text:
        return None

    import re
    pattern = r'S\d{4,6}'
    match = re.search(pattern, user_text.upper())
    if match:
        return match.group()

    return None


# ============================================================
# 주문 ID 추출 (O 패턴)
# ============================================================
def extract_order_id(user_text: str) -> Optional[str]:
    """텍스트에서 주문 ID를 추출합니다."""
    if not user_text:
        return None

    import re
    pattern = r'O\d{4,8}'
    match = re.search(pattern, user_text.upper())
    if match:
        return match.group()

    return None


# ============================================================
# CS 문의 카테고리 추출
# ============================================================
def extract_cs_category(user_text: str) -> Optional[str]:
    """텍스트에서 CS 문의 카테고리를 추출합니다."""
    if not user_text:
        return None

    category_map = {
        "주문": "order", "order": "order", "주문 관련": "order",
        "배송": "delivery", "delivery": "delivery", "배송 관련": "delivery", "택배": "delivery",
        "환불": "refund", "refund": "refund", "환불 관련": "refund",
        "반품": "refund", "교환": "refund",
        "결제": "payment", "payment": "payment", "결제 관련": "payment", "카드": "payment",
        "상품": "product", "product": "product", "상품 관련": "product",
        "계정": "account", "account": "account", "로그인": "account", "비밀번호": "account",
    }

    t = user_text.lower()
    for keyword, cat_code in category_map.items():
        if keyword in t:
            return cat_code

    return "general"  # 기본값: general


# ============================================================
# 결정적 도구 실행 파이프라인
# ============================================================
def run_deterministic_tools(user_text: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    """사용자 입력에 따라 적절한 도구를 실행합니다."""
    intents = detect_intent(user_text)
    results: Dict[str, Any] = {}

    # RAG 검색 (플랫폼 정보 질문)
    if intents.get("want_rag"):
        results["rag_search"] = tool_rag_search(
            user_text,
            top_k=min(5, st.RAG_MAX_TOPK),
            api_key=""
        )

    # 대시보드 요약
    if intents.get("want_dashboard"):
        results["dashboard"] = tool_get_dashboard_summary()
        return results

    # 쇼핑몰 정보
    if intents.get("want_shop_info"):
        shop_id = extract_shop_id(user_text)
        if shop_id:
            results["get_shop_info"] = tool_get_shop_info(shop_id)
        else:
            # ID를 못 찾으면 목록 반환
            results["list_shops"] = tool_list_shops()
        return results

    if intents.get("want_shop_list"):
        # 카테고리/티어/지역 필터 추출
        category = None
        tier = None
        region = None

        category_keywords = ["패션", "뷰티", "식품", "가전", "리빙", "디지털"]
        for c in category_keywords:
            if c in user_text:
                category = c
                break

        tier_keywords = ["프리미엄", "스탠다드", "베이직", "엔터프라이즈"]
        for ti in tier_keywords:
            if ti in user_text:
                tier = ti
                break

        region_keywords = {"국내": "국내", "해외": "해외", "글로벌": "글로벌"}
        for rk, rv in region_keywords.items():
            if rk in user_text:
                region = rv
                break

        results["list_shops"] = tool_list_shops(category=category, tier=tier, region=region)
        return results

    # 카테고리 정보
    if intents.get("want_category_info") or intents.get("want_category_list"):
        results["list_categories"] = tool_list_categories()
        return results

    # CS 자동 응답
    if intents.get("want_cs_reply"):
        cs_category = extract_cs_category(user_text)
        results["cs_reply_context"] = {
            "status": "SUCCESS",
            "action": "CS_REPLY",
            "category": cs_category,
            "message": f"'{cs_category}' 카테고리 CS 응답을 준비합니다.",
        }
        return results

    if intents.get("want_glossary"):
        results["ecommerce_glossary"] = tool_get_ecommerce_glossary()
        return results

    # 셀러 분석
    if intents.get("want_seller_analysis"):
        seller_id = extract_seller_id(user_text)
        if seller_id:
            results["analyze_seller"] = tool_analyze_seller(seller_id)
        else:
            results["segment_statistics"] = tool_get_segment_statistics()
        return results

    if intents.get("want_segment"):
        results["segment_statistics"] = tool_get_segment_statistics()
        return results

    # 주문/거래 통계
    if intents.get("want_order_stats"):
        results["order_statistics"] = tool_get_order_statistics()
        return results

    # 문의 분류
    if intents.get("want_classify"):
        results["classify_context"] = {
            "status": "SUCCESS",
            "action": "CLASSIFY",
            "message": "문의 분류를 위해 분류할 텍스트를 입력해주세요.",
        }
        return results

    return results
