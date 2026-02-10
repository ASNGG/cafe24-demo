"""
CAFE24 AI 운영 플랫폼 - 에이전트 도구 모음
==========================================
카페24 AI 기반 내부 시스템 개발 프로젝트

주요 기능:
1. 쇼핑몰/셀러/상품 데이터 조회 및 분석
2. CS 문의 자동 분류 및 응답 품질 예측
3. 이상거래 탐지 및 셀러 이탈 예측
4. 매출/GMV 예측 및 마케팅 최적화
5. RAG 기반 플랫폼 지식 검색
"""

from typing import Optional, List, Dict, Any
import json
import numpy as np
import pandas as pd

from core.constants import (
    PLAN_TIERS,
    SHOP_CATEGORIES,
    SELLER_REGIONS,
    PAYMENT_METHODS,
    ORDER_STATUSES,
    ECOMMERCE_TERMS,
    FEATURE_COLS_CS_QUALITY,
    FEATURE_COLS_SELLER_SEGMENT,
    FEATURE_COLS_CHURN,
    FEATURE_LABELS,
    ML_MODEL_INFO,
    SELLER_SEGMENT_NAMES,
    CS_TICKET_CATEGORIES,
    CS_PRIORITY_GRADES,
)
from core.utils import safe_str, safe_int, safe_float
import state as st


# ============================================================
# 1. 쇼핑몰 정보 조회
# ============================================================
def tool_get_shop_info(shop_id: str) -> dict:
    """쇼핑몰 정보를 조회합니다. shop_id 또는 쇼핑몰명으로 검색 가능합니다."""
    if st.SHOPS_DF is None:
        return {"status": "error", "message": "쇼핑몰 데이터가 로드되지 않았습니다."}

    shop = st.SHOPS_DF[st.SHOPS_DF["shop_id"] == shop_id]
    if shop.empty:
        # 이름으로도 검색 시도
        name_col = "shop_name" if "shop_name" in st.SHOPS_DF.columns else "name"
        shop = st.SHOPS_DF[st.SHOPS_DF[name_col].str.contains(shop_id, na=False)]

    if shop.empty:
        return {"status": "error", "message": f"쇼핑몰 '{shop_id}'를 찾을 수 없습니다."}

    row = shop.iloc[0]

    # SHOP_PERFORMANCE_DF에서 성과 데이터 조인
    perf = None
    if st.SHOP_PERFORMANCE_DF is not None:
        perf_match = st.SHOP_PERFORMANCE_DF[st.SHOP_PERFORMANCE_DF["shop_id"] == safe_str(row.get("shop_id"))]
        if not perf_match.empty:
            perf = perf_match.iloc[0]

    name_col = "shop_name" if "shop_name" in row.index else "name"
    return {
        "status": "success",
        "shop_id": safe_str(row.get("shop_id")),
        "name": safe_str(row.get(name_col)),
        "plan_tier": safe_str(row.get("plan_tier")),
        "category": safe_str(row.get("category")),
        "region": safe_str(row.get("region")),
        "open_date": safe_str(row.get("open_date")),
        "monthly_revenue": safe_int(perf.get("monthly_revenue")) if perf is not None else 0,
        "product_count": safe_int(row.get("product_count", 0)),
        "monthly_orders": safe_int(perf.get("monthly_orders")) if perf is not None else 0,
        "avg_order_value": safe_int(perf.get("avg_order_value")) if perf is not None else 0,
        "visitor_count": safe_int(perf.get("monthly_visitors")) if perf is not None else 0,
        "conversion_rate": safe_float(perf.get("conversion_rate")) if perf is not None else 0.0,
        "review_score": safe_float(perf.get("review_score")) if perf is not None else 0.0,
        "return_rate": safe_float(perf.get("return_rate")) if perf is not None else 0.0,
        "customer_retention_rate": safe_float(perf.get("customer_retention_rate")) if perf is not None else 0.0,
        "shop_status": safe_str(row.get("status")),
    }


def _get_segment_name(cluster: int) -> str:
    """CSV segment_name 컬럼에서 클러스터 번호에 해당하는 이름 반환"""
    if st.SELLER_ANALYTICS_DF is not None and "segment_name" in st.SELLER_ANALYTICS_DF.columns:
        match = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["cluster"] == cluster]
        if not match.empty:
            return str(match.iloc[0]["segment_name"])
    return SELLER_SEGMENT_NAMES.get(cluster, f"세그먼트 {cluster}")


def tool_list_shops(
    category: Optional[str] = None,
    plan_tier: Optional[str] = None,
    tier: Optional[str] = None,
    region: Optional[str] = None,
) -> dict:
    """쇼핑몰 목록을 조회합니다. 업종/플랜/지역으로 필터링 가능합니다."""
    if st.SHOPS_DF is None:
        return {"status": "error", "message": "쇼핑몰 데이터가 로드되지 않았습니다."}

    # tier와 plan_tier 둘 다 지원
    effective_tier = plan_tier or tier

    df = st.SHOPS_DF.copy()

    if category:
        df = df[df["category"].str.contains(category, na=False, case=False)]
    if effective_tier:
        df = df[df["plan_tier"].str.contains(effective_tier, na=False, case=False)]
    if region:
        df = df[df["region"].str.contains(region, na=False, case=False)]

    # SHOP_PERFORMANCE_DF 조인 (월매출 등)
    perf_map = {}
    if st.SHOP_PERFORMANCE_DF is not None:
        for _, p in st.SHOP_PERFORMANCE_DF.iterrows():
            perf_map[p.get("shop_id")] = p

    name_col = "shop_name" if "shop_name" in df.columns else "name"
    shops = []
    for _, row in df.iterrows():
        sid = safe_str(row.get("shop_id"))
        perf = perf_map.get(sid)
        shops.append({
            "shop_id": sid,
            "name": safe_str(row.get(name_col)),
            "plan_tier": safe_str(row.get("plan_tier")),
            "category": safe_str(row.get("category")),
            "region": safe_str(row.get("region")),
            "monthly_revenue": safe_int(perf.get("monthly_revenue")) if perf is not None else 0,
            "monthly_orders": safe_int(perf.get("monthly_orders")) if perf is not None else 0,
            "shop_status": safe_str(row.get("status")),
        })

    return {
        "status": "success",
        "total": len(shops),
        "filters": {"category": category, "plan_tier": effective_tier, "region": region},
        "shops": shops,
    }


def tool_get_shop_services(shop_id: str) -> dict:
    """쇼핑몰에서 이용 중인 서비스 목록을 조회합니다."""
    if not st.SHOP_SERVICE_MAP:
        # SERVICES_DF에서 직접 조회
        if st.SERVICES_DF is None:
            return {"status": "error", "message": "서비스 데이터가 로드되지 않았습니다."}

        services = st.SERVICES_DF[st.SERVICES_DF["shop_id"] == shop_id]
        if services.empty:
            return {"status": "error", "message": f"쇼핑몰 '{shop_id}'의 서비스 정보를 찾을 수 없습니다."}

        service_list = []
        for _, row in services.iterrows():
            service_list.append({
                "service_id": safe_str(row.get("service_id")),
                "service_name": safe_str(row.get("service_name")),
                "service_type": safe_str(row.get("service_type")),
                "service_status": safe_str(row.get("status")),
                "monthly_fee": safe_int(row.get("monthly_fee")),
                "description": safe_str(row.get("description")),
            })

        return {
            "status": "success",
            "shop_id": shop_id,
            "total_services": len(service_list),
            "services": service_list,
        }

    # 캐시된 매핑에서 조회
    if shop_id not in st.SHOP_SERVICE_MAP:
        return {"status": "error", "message": f"쇼핑몰 '{shop_id}'의 서비스 정보를 찾을 수 없습니다."}

    shop_services = st.SHOP_SERVICE_MAP[shop_id]
    return {
        "status": "success",
        "shop_id": shop_id,
        "total_services": len(shop_services) if isinstance(shop_services, list) else 1,
        "services": shop_services if isinstance(shop_services, list) else [shop_services],
    }


# ============================================================
# 2. 상품 카테고리 정보 조회
# ============================================================
def tool_get_category_info(category_id: str) -> dict:
    """상품 카테고리 정보를 조회합니다."""
    if st.CATEGORIES_DF is None:
        return {"status": "error", "message": "카테고리 데이터가 로드되지 않았습니다."}

    # cat_id 또는 category_id 컬럼 지원
    id_col = "cat_id" if "cat_id" in st.CATEGORIES_DF.columns else "category_id"
    name_col = "name_ko" if "name_ko" in st.CATEGORIES_DF.columns else "name"

    category = st.CATEGORIES_DF[st.CATEGORIES_DF[id_col] == category_id]
    if category.empty:
        # 이름으로도 검색 시도
        category = st.CATEGORIES_DF[st.CATEGORIES_DF[name_col].str.contains(category_id, na=False, case=False)]

    if category.empty:
        return {"status": "error", "message": f"카테고리 '{category_id}'를 찾을 수 없습니다."}

    row = category.iloc[0]
    parent_col = "parent_cat" if "parent_cat" in row.index else "parent_id"
    desc_col = "description_ko" if "description_ko" in row.index else "description"

    return {
        "status": "success",
        "category_id": safe_str(row.get(id_col)),
        "name": safe_str(row.get(name_col)),
        "name_en": safe_str(row.get("name_en", "")),
        "parent_id": safe_str(row.get(parent_col)),
        "description": safe_str(row.get(desc_col)),
        "description_en": safe_str(row.get("description_en", "")),
    }


def tool_list_categories() -> dict:
    """모든 상품 카테고리 목록을 조회합니다."""
    if st.CATEGORIES_DF is None:
        return {"status": "error", "message": "카테고리 데이터가 로드되지 않았습니다."}

    id_col = "cat_id" if "cat_id" in st.CATEGORIES_DF.columns else "category_id"
    name_col = "name_ko" if "name_ko" in st.CATEGORIES_DF.columns else "name"
    parent_col = "parent_cat" if "parent_cat" in st.CATEGORIES_DF.columns else "parent_id"
    desc_col = "description_ko" if "description_ko" in st.CATEGORIES_DF.columns else "description"

    categories = []
    for _, row in st.CATEGORIES_DF.iterrows():
        categories.append({
            "category_id": safe_str(row.get(id_col)),
            "name": safe_str(row.get(name_col)),
            "name_en": safe_str(row.get("name_en", "")),
            "parent_id": safe_str(row.get(parent_col)),
            "description": safe_str(row.get(desc_col)),
        })

    return {
        "status": "success",
        "total": len(categories),
        "categories": categories,
    }


# ============================================================
# 3. CS 관련 도구
# ============================================================
def tool_auto_reply_cs(
    inquiry_text: str,
    inquiry_category: str = "기타",
    seller_tier: str = "Basic",
    order_id: Optional[str] = None,
) -> dict:
    """
    CS 문의에 대한 자동 응답 초안을 생성합니다.
    LLM을 사용하여 카페24 플랫폼 정책에 맞는 답변을 작성합니다.
    """
    if inquiry_category not in CS_TICKET_CATEGORIES:
        # 가장 유사한 카테고리 매칭 시도
        matched = False
        for cat in CS_TICKET_CATEGORIES:
            if cat in inquiry_category or inquiry_category in cat:
                inquiry_category = cat
                matched = True
                break
        if not matched:
            inquiry_category = "기타"

    # CS 응답 컨텍스트 생성 (실제 LLM 호출은 agent/runner.py에서 처리)
    cs_context = {
        "inquiry_text": inquiry_text,
        "inquiry_category": inquiry_category,
        "seller_tier": seller_tier,
        "order_id": order_id,
        "platform": "CAFE24",
        "priority_guide": CS_PRIORITY_GRADES,
        "category_guide": CS_TICKET_CATEGORIES,
    }

    return {
        "status": "success",
        "action": "CS_AUTO_REPLY",
        "context": cs_context,
        "message": f"'{inquiry_text[:50]}...' 문의에 대한 자동 응답을 생성합니다. 카테고리: {inquiry_category}, 셀러 등급: {seller_tier}.",
    }


def tool_check_cs_quality(
    ticket_category: str,
    seller_tier: str,
    sentiment_score: float,
    order_value: float,
    is_repeat_issue: bool = False,
    text_length: int = 100,
) -> dict:
    """CS 티켓의 우선순위/긴급도를 예측합니다."""
    if st.CS_QUALITY_MODEL is None:
        return {"status": "error", "message": "CS 품질 예측 모델이 로드되지 않았습니다."}

    # 피처 인코딩
    try:
        category_encoded = st.LE_TICKET_CATEGORY.transform([ticket_category])[0] if st.LE_TICKET_CATEGORY else 0
    except (ValueError, AttributeError):
        category_encoded = 0

    try:
        tier_encoded = st.LE_SELLER_TIER.transform([seller_tier])[0] if st.LE_SELLER_TIER else 0
    except (ValueError, AttributeError):
        tier_encoded = 0

    features = {
        "ticket_category_encoded": category_encoded,
        "seller_tier_encoded": tier_encoded,
        "sentiment_score": sentiment_score,
        "order_value": order_value,
        "is_repeat_issue": int(is_repeat_issue),
        "text_length": text_length,
    }

    # 모델 예측
    try:
        X = pd.DataFrame([features])[FEATURE_COLS_CS_QUALITY]
        pred = st.CS_QUALITY_MODEL.predict(X)[0]
        proba = st.CS_QUALITY_MODEL.predict_proba(X)[0]

        priority_grade = st.LE_CS_PRIORITY.inverse_transform([pred])[0] if st.LE_CS_PRIORITY else "normal"

        grade_info = CS_PRIORITY_GRADES.get(priority_grade, {})

        return {
            "status": "success",
            "ticket_category": ticket_category,
            "seller_tier": seller_tier,
            "predicted_priority": priority_grade,
            "priority_description": grade_info.get("description", ""),
            "confidence": float(max(proba)),
            "is_repeat_issue": is_repeat_issue,
            "recommendations": _get_cs_recommendations(priority_grade, is_repeat_issue, ticket_category),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_cs_recommendations(priority: str, is_repeat: bool, category: str) -> List[str]:
    """CS 우선순위에 따른 권장사항을 반환합니다."""
    recommendations = []

    if priority == "urgent":
        recommendations.append("즉시 담당자 배정이 필요합니다.")
        recommendations.append("셀러에게 1시간 이내 초기 응답을 보내주세요.")
    elif priority == "high":
        recommendations.append("우선 처리 대상입니다. 4시간 이내 응답을 권장합니다.")
    elif priority == "normal":
        recommendations.append("일반 처리 흐름을 따릅니다.")
    elif priority == "low":
        recommendations.append("낮은 우선순위입니다. FAQ 자동 답변을 활용하세요.")

    if is_repeat:
        recommendations.append("반복 문의입니다. 근본 원인 해결이 필요합니다. 이전 티켓 히스토리를 확인하세요.")

    if category == "환불":
        recommendations.append("환불 정책 및 처리 기간을 명확히 안내하세요.")
    elif category == "배송":
        recommendations.append("배송 추적 정보를 확인하고 예상 도착일을 안내하세요.")
    elif category == "결제":
        recommendations.append("결제 오류 로그를 확인하고 대안 결제 수단을 안내하세요.")

    return recommendations


def tool_get_ecommerce_glossary(term: Optional[str] = None) -> dict:
    """이커머스 용어집을 조회합니다. 특정 용어를 검색하거나 전체 목록을 반환합니다."""
    terms = []

    if term:
        # 특정 용어 검색
        term_upper = term.upper()
        if term_upper in ECOMMERCE_TERMS:
            info = ECOMMERCE_TERMS[term_upper]
            terms.append({
                "term": term_upper,
                "english": info.get("en", ""),
                "description": info.get("desc", ""),
            })
        elif term in ECOMMERCE_TERMS:
            info = ECOMMERCE_TERMS[term]
            terms.append({
                "term": term,
                "english": info.get("en", ""),
                "description": info.get("desc", ""),
            })
        else:
            # 부분 매칭 시도
            for key, info in ECOMMERCE_TERMS.items():
                if term.lower() in key.lower() or term.lower() in info.get("desc", "").lower():
                    terms.append({
                        "term": key,
                        "english": info.get("en", ""),
                        "description": info.get("desc", ""),
                    })
    else:
        # 전체 용어집
        for key, info in ECOMMERCE_TERMS.items():
            terms.append({
                "term": key,
                "english": info.get("en", ""),
                "description": info.get("desc", ""),
            })

    if not terms:
        return {"status": "error", "message": f"'{term}' 관련 용어를 찾을 수 없습니다."}

    return {
        "status": "success",
        "total": len(terms),
        "search_term": term,
        "terms": terms,
    }


# ============================================================
# 4. 셀러 분석 도구
# ============================================================
def tool_analyze_seller(seller_id: str) -> dict:
    """셀러의 운영 데이터 및 성과를 분석합니다."""
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "셀러 분석 데이터가 로드되지 않았습니다."}

    seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if seller.empty:
        return {"status": "error", "message": f"셀러 '{seller_id}'를 찾을 수 없습니다."}

    row = seller.iloc[0]

    return {
        "status": "success",
        "seller_id": seller_id,
        "performance": {
            "total_orders": safe_int(row.get("total_orders")),
            "total_revenue": safe_int(row.get("total_revenue")),
            "product_count": safe_int(row.get("product_count")),
            "avg_order_value": safe_int(row.get("avg_order_value")),
            "conversion_rate": safe_float(row.get("conversion_rate")),
            "repeat_purchase_rate": safe_float(row.get("repeat_purchase_rate")),
            "monthly_growth_rate": safe_float(row.get("monthly_growth_rate")),
        },
        "operations": {
            "cs_tickets": safe_int(row.get("cs_tickets")),
            "refund_rate": safe_float(row.get("refund_rate")),
            "avg_response_time": safe_float(row.get("avg_response_time")),
            "days_since_last_login": safe_int(row.get("days_since_last_login")),
            "days_since_register": safe_int(row.get("days_since_register")),
        },
        "marketing": {
            "ad_spend": safe_int(row.get("ad_spend")),
            "roas": safe_float(row.get("roas")),
        },
        "plan_tier": safe_str(row.get("plan_tier")),
    }


def tool_get_seller_segment(seller_id_or_features) -> dict:
    """셀러 ID 또는 피처 딕셔너리를 기반으로 세그먼트를 분류합니다 (K-Means 클러스터링)."""

    # seller_id(str)로 호출된 경우 → SELLER_ANALYTICS_DF에서 조회
    if isinstance(seller_id_or_features, str):
        seller_id = seller_id_or_features
        if st.SELLER_ANALYTICS_DF is None:
            return {"status": "error", "message": "셀러 분석 데이터가 로드되지 않았습니다."}

        seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
        if seller.empty:
            return {"status": "error", "message": f"셀러 '{seller_id}'를 찾을 수 없습니다."}

        row = seller.iloc[0]

        # 이미 cluster 컬럼이 있으면 바로 반환
        if "cluster" in row.index and pd.notna(row.get("cluster")):
            cluster = int(row["cluster"])
            segment_name = safe_str(row.get("segment_name", "")) or _get_segment_name(cluster)
            return {
                "status": "success",
                "seller_id": seller_id,
                "segment": {
                    "cluster": cluster,
                    "name": segment_name,
                },
            }

        # cluster가 없으면 모델로 예측
        seller_features = {col: float(row.get(col, 0)) for col in FEATURE_COLS_SELLER_SEGMENT if col in row.index}
    else:
        seller_features = seller_id_or_features
        seller_id = None

    if st.SELLER_SEGMENT_MODEL is None:
        return {"status": "error", "message": "셀러 세그먼트 모델이 로드되지 않았습니다."}

    try:
        X = pd.DataFrame([seller_features])[FEATURE_COLS_SELLER_SEGMENT].fillna(0)
        X_scaled = st.SCALER_CLUSTER.transform(X) if st.SCALER_CLUSTER else X
        cluster = int(st.SELLER_SEGMENT_MODEL.predict(X_scaled)[0])

        result = {
            "status": "success",
            "segment": {
                "cluster": cluster,
                "name": _get_segment_name(cluster),
            },
        }
        if seller_id:
            result["seller_id"] = seller_id
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_detect_fraud(seller_id: Optional[str] = None, transaction_features: Optional[dict] = None) -> dict:
    """이상거래를 탐지합니다. 셀러 ID 또는 거래 피처를 기반으로 분석합니다."""
    if st.FRAUD_DETECTION_MODEL is None:
        return {"status": "error", "message": "이상거래 탐지 모델이 로드되지 않았습니다."}

    try:
        # 셀러 ID로 기존 이상거래 데이터 조회
        # CSV 컬럼: seller_id, anomaly_score, anomaly_type, detected_date, details
        if seller_id and st.FRAUD_DETAILS_DF is not None:
            fraud_records = st.FRAUD_DETAILS_DF[st.FRAUD_DETAILS_DF["seller_id"] == seller_id]
            if not fraud_records.empty:
                records = []
                for _, row in fraud_records.iterrows():
                    score = safe_float(row.get("anomaly_score", 0))
                    records.append({
                        "seller_id": safe_str(row.get("seller_id")),
                        "anomaly_score": score,
                        "anomaly_type": safe_str(row.get("anomaly_type")),
                        "detected_date": safe_str(row.get("detected_date")),
                        "details": safe_str(row.get("details")),
                    })
                max_score = max(r["anomaly_score"] for r in records)
                return {
                    "status": "success",
                    "seller_id": seller_id,
                    "fraud_records": records,
                    "total_flags": len(records),
                    "risk_level": "HIGH" if max_score >= 0.9 else "MEDIUM" if max_score >= 0.7 else "LOW",
                }

        # 거래 피처로 실시간 탐지
        if transaction_features:
            feature_cols = ["order_amount", "order_frequency", "refund_rate",
                            "review_anomaly_score", "payment_failure_rate"]
            X = pd.DataFrame([transaction_features])[feature_cols].fillna(0)

            pred = int(st.FRAUD_DETECTION_MODEL.predict(X)[0])
            score = float(st.FRAUD_DETECTION_MODEL.decision_function(X)[0])

            is_fraud = pred == -1

            return {
                "status": "success",
                "is_fraud": is_fraud,
                "anomaly_score": score,
                "risk_level": "HIGH" if is_fraud and score < -0.2 else "MEDIUM" if is_fraud else "LOW",
                "recommendation": "비정상 거래 패턴이 감지되었습니다. 해당 셀러의 거래 내역을 상세 조사하세요." if is_fraud else "정상적인 거래 패턴입니다.",
            }

        return {"status": "error", "message": "seller_id 또는 transaction_features를 제공해주세요."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_segment_statistics() -> dict:
    """셀러 세그먼트별 통계를 조회합니다."""
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "셀러 분석 데이터가 로드되지 않았습니다."}

    try:
        df = st.SELLER_ANALYTICS_DF.copy()

        # 세그먼트 분류가 되어 있는 경우
        if "cluster" in df.columns or "segment" in df.columns:
            cluster_col = "cluster" if "cluster" in df.columns else "segment"
            stats = []
            for cluster in sorted(df[cluster_col].unique()):
                segment_sellers = df[df[cluster_col] == cluster]
                if segment_sellers.empty:
                    continue
                name = _get_segment_name(int(cluster))
                stats.append({
                    "cluster": int(cluster),
                    "name": name,
                    "seller_count": len(segment_sellers),
                    "avg_revenue": safe_float(segment_sellers["total_revenue"].mean()),
                    "avg_orders": safe_float(segment_sellers["total_orders"].mean()),
                    "avg_products": safe_float(segment_sellers["product_count"].mean()),
                    "avg_refund_rate": safe_float(segment_sellers["refund_rate"].mean()),
                })

            return {
                "status": "success",
                "total_sellers": len(df),
                "segments": stats,
            }

        # 세그먼트 분류가 안 되어 있으면 모델로 분류
        if st.SELLER_SEGMENT_MODEL is not None and st.SCALER_CLUSTER is not None:
            X = df[FEATURE_COLS_SELLER_SEGMENT].fillna(0)
            X_scaled = st.SCALER_CLUSTER.transform(X)
            df["cluster"] = st.SELLER_SEGMENT_MODEL.predict(X_scaled)

            stats = []
            for cluster in sorted(df["cluster"].unique()):
                segment_sellers = df[df["cluster"] == cluster]
                if segment_sellers.empty:
                    continue
                name = _get_segment_name(int(cluster))
                stats.append({
                    "cluster": int(cluster),
                    "name": name,
                    "seller_count": len(segment_sellers),
                    "avg_revenue": safe_float(segment_sellers["total_revenue"].mean()),
                    "avg_orders": safe_float(segment_sellers["total_orders"].mean()),
                    "avg_products": safe_float(segment_sellers["product_count"].mean()),
                    "avg_refund_rate": safe_float(segment_sellers["refund_rate"].mean()),
                })

            return {
                "status": "success",
                "total_sellers": len(df),
                "segments": stats,
            }

        return {"status": "error", "message": "세그먼트 분류 데이터 또는 모델이 없습니다."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_fraud_statistics() -> dict:
    """전체 이상거래 통계를 조회합니다."""
    if st.FRAUD_DETAILS_DF is None:
        return {"status": "error", "message": "이상거래 데이터가 로드되지 않았습니다."}

    df = st.FRAUD_DETAILS_DF
    total_records = len(df)

    # fraud_details.csv 컬럼: seller_id, anomaly_score, anomaly_type, detected_date, details
    # 모든 레코드가 이미 탐지된 이상거래

    # 이상거래 유형별 분포
    anomaly_type_dist = {}
    if "anomaly_type" in df.columns:
        anomaly_type_dist = df["anomaly_type"].value_counts().to_dict()

    # 이상 점수 통계
    avg_score = safe_float(df["anomaly_score"].mean()) if "anomaly_score" in df.columns else 0
    max_score = safe_float(df["anomaly_score"].max()) if "anomaly_score" in df.columns else 0
    min_score = safe_float(df["anomaly_score"].min()) if "anomaly_score" in df.columns else 0

    # 고위험 (anomaly_score >= 0.9)
    high_risk_count = len(df[df["anomaly_score"] >= 0.9]) if "anomaly_score" in df.columns else 0
    unique_sellers = df["seller_id"].nunique() if "seller_id" in df.columns else 0

    # 위험도 높은 순 샘플 (최대 10건)
    sorted_df = df.sort_values("anomaly_score", ascending=False) if "anomaly_score" in df.columns else df
    fraud_samples = []
    for _, row in sorted_df.head(10).iterrows():
        fraud_samples.append({
            "seller_id": safe_str(row.get("seller_id")),
            "anomaly_score": safe_float(row.get("anomaly_score")),
            "anomaly_type": safe_str(row.get("anomaly_type")),
            "detected_date": safe_str(row.get("detected_date")),
            "details": safe_str(row.get("details")),
        })

    return {
        "status": "success",
        "total_anomalies": total_records,
        "unique_sellers": unique_sellers,
        "high_risk_count": high_risk_count,
        "high_risk_rate": round(high_risk_count / total_records * 100, 1) if total_records > 0 else 0,
        "anomaly_type_distribution": anomaly_type_dist,
        "anomaly_score_stats": {"avg": avg_score, "max": max_score, "min": min_score},
        "top_anomalies": fraud_samples,
    }


# ============================================================
# 5. 주문/운영 로그 분석 도구
# ============================================================
def tool_get_order_statistics(event_type: Optional[str] = None, days: int = 30) -> dict:
    """주문/운영 이벤트 통계를 조회합니다."""
    if st.OPERATION_LOGS_DF is None:
        return {"status": "error", "message": "운영 로그 데이터가 로드되지 않았습니다."}

    df = st.OPERATION_LOGS_DF.copy()
    # CSV 컬럼: log_id, seller_id, event_type, event_date, details_json
    date_col = "event_date" if "event_date" in df.columns else "timestamp"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # 최근 N일 필터
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    df = df[df[date_col] >= cutoff]

    # event_type 별칭 매핑 (LLM이 축약형으로 보내는 경우 처리)
    _EVENT_TYPE_ALIAS = {
        "order": "order_received",
        "payment": "payment_settled",
        "settlement": "payment_settled",
        "refund": "refund_processed",
        "delivery": "order_received",
        "cancel": "refund_processed",
        "cs": "cs_ticket",
        "marketing": "marketing_campaign",
        "product": "product_listed",
    }
    if event_type:
        event_type = _EVENT_TYPE_ALIAS.get(event_type.lower(), event_type)
        df = df[df["event_type"] == event_type]

    # 이벤트 타입별 집계
    event_counts = df["event_type"].value_counts().to_dict()

    # 일별 추이
    daily_counts = df.groupby(df[date_col].dt.date).size().to_dict()
    daily_counts = {str(k): v for k, v in daily_counts.items()}

    # details_json에서 주문 금액 합산 시도
    total_amount = 0
    if "details_json" in df.columns:
        import json as _json
        for val in df["details_json"].dropna():
            try:
                parsed = _json.loads(val) if isinstance(val, str) else val
                if isinstance(parsed, dict) and "order_amount" in parsed:
                    total_amount += int(parsed["order_amount"])
            except (ValueError, TypeError, _json.JSONDecodeError):
                pass

    # 셀러별 이벤트 수
    seller_event_counts = df["seller_id"].value_counts().head(10).to_dict() if "seller_id" in df.columns else {}

    return {
        "status": "success",
        "period": f"최근 {days}일",
        "total_events": len(df),
        "total_amount": total_amount,
        "event_type_filter": event_type,
        "event_counts": event_counts,
        "top_sellers_by_events": seller_event_counts,
        "daily_trend": daily_counts,
    }


def tool_get_seller_activity_report(seller_id: str, days: int = 30) -> dict:
    """특정 셀러의 활동 리포트를 생성합니다."""
    if st.SELLER_ACTIVITY_DF is None and st.OPERATION_LOGS_DF is None:
        return {"status": "error", "message": "셀러 활동 데이터가 로드되지 않았습니다."}

    # SELLER_ACTIVITY_DF 우선 사용
    # CSV 컬럼: seller_id, date, orders_processed, products_updated, cs_handled, revenue
    if st.SELLER_ACTIVITY_DF is not None:
        df = st.SELLER_ACTIVITY_DF[st.SELLER_ACTIVITY_DF["seller_id"] == seller_id].copy()
        if df.empty:
            return {"status": "error", "message": f"셀러 '{seller_id}'의 활동 로그를 찾을 수 없습니다."}

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff = df["date"].max() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]

        active_days = df["date"].dt.date.nunique()
        total_orders = safe_int(df["orders_processed"].sum()) if "orders_processed" in df.columns else 0
        total_revenue = safe_int(df["revenue"].sum()) if "revenue" in df.columns else 0
        total_cs = safe_int(df["cs_handled"].sum()) if "cs_handled" in df.columns else 0

        event_summary = {
            "주문처리": total_orders,
            "상품업데이트": safe_int(df["products_updated"].sum()) if "products_updated" in df.columns else 0,
            "CS처리": total_cs,
        }

        return {
            "status": "success",
            "seller_id": seller_id,
            "period": f"최근 {days}일",
            "total_events": total_orders + total_cs,
            "total_amount": total_revenue,
            "active_days": active_days,
            "event_summary": event_summary,
            "avg_events_per_day": round((total_orders + total_cs) / max(active_days, 1), 2),
        }

    # OPERATION_LOGS_DF 폴백
    # CSV 컬럼: log_id, seller_id, event_type, event_date, details_json
    df = st.OPERATION_LOGS_DF[st.OPERATION_LOGS_DF["seller_id"] == seller_id].copy()
    if df.empty:
        return {"status": "error", "message": f"셀러 '{seller_id}'의 활동 로그를 찾을 수 없습니다."}

    date_col = "event_date" if "event_date" in df.columns else "timestamp"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    df = df[df[date_col] >= cutoff]

    event_summary = df["event_type"].value_counts().to_dict()
    active_days = df[date_col].dt.date.nunique()

    # details_json에서 주문 금액 합산 시도
    total_amount = 0
    if "details_json" in df.columns:
        import json as _json
        for val in df["details_json"].dropna():
            try:
                parsed = _json.loads(val) if isinstance(val, str) else val
                if isinstance(parsed, dict) and "order_amount" in parsed:
                    total_amount += int(parsed["order_amount"])
            except (ValueError, TypeError, _json.JSONDecodeError):
                pass

    return {
        "status": "success",
        "seller_id": seller_id,
        "period": f"최근 {days}일",
        "total_events": len(df),
        "total_amount": total_amount,
        "active_days": active_days,
        "event_summary": event_summary,
        "avg_events_per_day": round(len(df) / max(active_days, 1), 2),
    }


# ============================================================
# 6. 문의 분류 도구
# ============================================================
def tool_classify_inquiry(text: str) -> dict:
    """CS 문의 텍스트를 카테고리별로 자동 분류합니다 (TF-IDF + RandomForest)."""
    if st.INQUIRY_CLASSIFICATION_MODEL is None or st.TFIDF_VECTORIZER is None:
        return {"status": "error", "message": "문의 분류 모델이 로드되지 않았습니다."}

    try:
        X = st.TFIDF_VECTORIZER.transform([text])
        pred = st.INQUIRY_CLASSIFICATION_MODEL.predict(X)[0]
        proba = st.INQUIRY_CLASSIFICATION_MODEL.predict_proba(X)[0]

        category = st.LE_INQUIRY_CATEGORY.inverse_transform([pred])[0] if st.LE_INQUIRY_CATEGORY else "기타"

        # 상위 3개 카테고리 확률
        top_indices = np.argsort(proba)[::-1][:3]
        categories = st.LE_INQUIRY_CATEGORY.classes_ if st.LE_INQUIRY_CATEGORY else []

        top_categories = []
        for idx in top_indices:
            if idx < len(categories):
                top_categories.append({
                    "category": categories[idx],
                    "probability": float(proba[idx]),
                })

        return {
            "status": "success",
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_category": category,
            "confidence": float(max(proba)),
            "top_categories": top_categories,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================
# 7. RAG 검색 도구
# ============================================================
def tool_search_platform(query: str, top_k: int = 5) -> dict:
    """플랫폼 지식베이스를 검색합니다 (카페24 정책, 가이드, FAQ 등)."""
    # 실제 RAG 검색은 rag/service.py에서 처리
    # 여기서는 인터페이스만 정의
    return {
        "status": "success",
        "action": "RAG_SEARCH",
        "query": query,
        "top_k": top_k,
        "message": f"플랫폼 지식베이스에서 '{query}' 관련 정보를 검색합니다.",
    }


def tool_search_platform_lightrag(query: str, mode: str = "hybrid") -> dict:
    """
    LightRAG로 플랫폼 지식베이스를 검색합니다. (경량 GraphRAG)

    듀얼 레벨 검색 지원:
    - local: 엔티티 중심 검색 (구체적인 질문에 적합)
      예: "카페24 환불 정책은?"
    - global: 테마 중심 검색 (추상적인 질문에 적합)
      예: "카페24 플랫폼의 수수료 체계는?"
    - hybrid: local + global 조합 (권장)
    - naive: 기본 검색

    Args:
        query: 검색 쿼리
        mode: 검색 모드 ("naive", "local", "global", "hybrid")
    """
    return {
        "status": "success",
        "action": "LIGHTRAG_SEARCH",
        "query": query,
        "mode": mode,
        "message": f"LightRAG로 '{query}' 관련 정보를 검색합니다. (모드: {mode})",
    }


# ============================================================
# 8. CS 통계 도구
# ============================================================
def tool_get_cs_statistics() -> dict:
    """CS 문의 통계를 조회합니다."""
    if st.CS_STATS_DF is None:
        return {"status": "error", "message": "CS 통계 데이터가 로드되지 않았습니다."}

    df = st.CS_STATS_DF
    # CSV 컬럼: category, total_tickets, avg_resolution_hours, satisfaction_score

    # 전체 티켓 수 합산
    total_tickets = int(df["total_tickets"].sum()) if "total_tickets" in df.columns else len(df)

    # 카테고리별 통계
    cat_col = "category" if "category" in df.columns else "ticket_category"
    by_category = {}
    if cat_col in df.columns and "total_tickets" in df.columns:
        for _, row in df.iterrows():
            by_category[str(row[cat_col])] = {
                "total_tickets": int(row["total_tickets"]),
                "avg_resolution_hours": safe_float(row.get("avg_resolution_hours", 0)),
                "satisfaction_score": safe_float(row.get("satisfaction_score", 0)),
            }
    elif cat_col in df.columns:
        by_category = df[cat_col].value_counts().to_dict()

    # 평균 만족도 점수
    avg_satisfaction = safe_float(df["satisfaction_score"].mean()) if "satisfaction_score" in df.columns else 0

    # 평균 해결 시간
    avg_resolution = safe_float(df["avg_resolution_hours"].mean()) if "avg_resolution_hours" in df.columns else 0

    return {
        "status": "success",
        "total_tickets": total_tickets,
        "avg_satisfaction_score": round(avg_satisfaction, 2),
        "avg_resolution_hours": round(avg_resolution, 1),
        "by_category": by_category,
    }


# ============================================================
# 9. 이탈 예측 분석
# ============================================================
def tool_get_churn_prediction(risk_level: str = None, limit: int = None) -> dict:
    """셀러 이탈 예측 분석을 조회합니다. 고위험/중위험/저위험 이탈 셀러 수와 주요 이탈 요인을 반환합니다.

    Args:
        risk_level: 특정 위험 등급만 필터 ("high", "medium", "low")
        limit: 상세 셀러 목록 반환 시 최대 개수 (기본값: 10)
    """
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "셀러 분석 데이터가 없습니다."}

    try:
        df = st.SELLER_ANALYTICS_DF.copy()
        original_total = len(df)

        # 특정 위험 등급 필터링
        filtered_sellers = []
        if risk_level:
            risk_level = risk_level.lower()
            if risk_level not in ['high', 'medium', 'low']:
                return {"status": "error", "message": "risk_level은 'high', 'medium', 'low' 중 하나여야 합니다."}

            if 'churn_risk_level' in df.columns:
                if risk_level == 'medium' and 'churn_probability' in df.columns:
                    df = df[(df['churn_probability'] > 0.3) & (df['churn_probability'] <= 0.7)]
                else:
                    df = df[df['churn_risk_level'] == risk_level]
            elif 'is_churned' in df.columns:
                if risk_level == 'high':
                    df = df[df['is_churned'] == 1]
                elif risk_level == 'low':
                    df = df[df['is_churned'] == 0]

            # 상세 셀러 목록 (limit 적용)
            max_sellers = limit if limit and limit > 0 else 10
            if 'seller_id' in df.columns:
                for _, row in df.head(max_sellers).iterrows():
                    seller_info = {"seller_id": row['seller_id']}
                    if 'churn_probability' in df.columns:
                        seller_info['churn_probability'] = f"{row['churn_probability'] * 100:.1f}%"
                    if 'total_revenue' in df.columns:
                        seller_info['total_revenue'] = safe_int(row['total_revenue'])
                    filtered_sellers.append(seller_info)

        total = len(df) if not risk_level else original_total

        # 실제 데이터에서 이탈 위험 분류
        full_df = st.SELLER_ANALYTICS_DF.copy()
        if 'churn_risk_level' in full_df.columns:
            high_risk = len(full_df[full_df['churn_risk_level'] == 'high'])
            low_risk = len(full_df[full_df['churn_risk_level'] == 'low'])
            if 'churn_probability' in full_df.columns:
                medium_mask = (full_df['churn_probability'] > 0.3) & (full_df['churn_probability'] <= 0.7)
                medium_risk = len(full_df[medium_mask])
            else:
                medium_risk = 0
        elif 'is_churned' in full_df.columns:
            high_risk = len(full_df[full_df['is_churned'] == 1])
            low_risk = len(full_df[full_df['is_churned'] == 0])
            medium_risk = 0
        else:
            high_risk = int(total * 0.085)
            medium_risk = int(total * 0.142)
            low_risk = total - high_risk - medium_risk

        # SHAP 기반 이탈 요인 분석
        shap_cols = [c for c in df.columns if c.startswith('shap_')]
        top_factors = []
        if shap_cols:
            factor_names = {
                'shap_total_orders': '주문 수 감소',
                'shap_total_revenue': '매출 하락',
                'shap_product_count': '상품 등록 감소',
                'shap_cs_tickets': 'CS 문의 증가',
                'shap_refund_rate': '환불률 증가',
                'shap_avg_response_time': '응답 시간 지연',
                'shap_days_since_last_login': '장기 미접속',
                'shap_days_since_register': '가입 후 경과 일수',
                'shap_plan_tier_encoded': '플랜 등급',
            }
            shap_importance = {col: df[col].abs().mean() for col in shap_cols}
            sorted_factors = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            total_importance = sum(v for _, v in sorted_factors) or 1
            for col, val in sorted_factors:
                pct = round(val / total_importance * 100)
                factor_name = factor_names.get(col, col.replace('shap_', ''))
                top_factors.append({"factor": factor_name, "importance": f"{pct}%"})
        else:
            top_factors = [
                {"factor": "장기 미접속", "importance": "32%"},
                {"factor": "매출 하락", "importance": "25%"},
                {"factor": "상품 등록 감소", "importance": "18%"},
                {"factor": "CS 문의 급증", "importance": "15%"},
                {"factor": "환불률 상승", "importance": "10%"},
            ]

        churn_rate = round(high_risk / original_total * 100, 1) if original_total > 0 else 0
        top_factor_name = top_factors[0]['factor'] if top_factors else '활동 감소'

        result = {
            "status": "success",
            "prediction_type": "셀러 이탈 예측",
            "summary": {
                "total_sellers": original_total,
                "high_risk_count": high_risk,
                "medium_risk_count": medium_risk,
                "low_risk_count": low_risk,
                "predicted_churn_rate": churn_rate,
            },
            "top_factors": top_factors,
        }

        # 특정 위험 등급 필터 적용 시 상세 정보 추가
        if risk_level and filtered_sellers:
            level_names = {'high': '고위험', 'medium': '중위험', 'low': '저위험'}
            result["filtered"] = {
                "risk_level": risk_level,
                "count": len(df),
                "sellers": filtered_sellers
            }

        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_cohort_analysis(cohort: str = None, month: str = None) -> dict:
    """셀러 코호트 리텐션 분석을 조회합니다. 주간 리텐션율과 트렌드를 반환합니다.

    CSV 형식: cohort_month, week1, week2, week4, week8, week12 (와이드 포맷)

    Args:
        cohort: 사용하지 않음 (호환성 유지)
        month: 특정 월 필터 (예: "2024-11", "2025-01")
    """
    if st.COHORT_RETENTION_DF is None:
        return {"status": "error", "message": "코호트 리텐션 데이터가 없습니다."}

    try:
        df = st.COHORT_RETENTION_DF.copy()

        # 월 필터링 (cohort 또는 month 사용)
        filter_month = month or cohort
        if filter_month:
            # "2024-11 W1" 같은 형식에서 월만 추출
            import re as _re
            m = _re.search(r'(\d{4}-\d{2})', filter_month)
            if m:
                filter_month = m.group(1)
            df = df[df['cohort_month'].astype(str).str.contains(filter_month, case=False, na=False)]
            if len(df) == 0:
                available = st.COHORT_RETENTION_DF['cohort_month'].tolist()
                return {"status": "error", "message": f"'{filter_month}' 코호트를 찾을 수 없습니다. 사용 가능: {available}"}

        # 와이드 포맷(week1,week2,week4,...) 컬럼 감지
        week_cols = [c for c in df.columns if c.startswith("week")]

        # 코호트별 리텐션 데이터 구성
        retention = {}
        for _, row in df.iterrows():
            cohort_name = safe_str(row.get("cohort_month"))
            weeks = {}
            for wc in week_cols:
                val = safe_float(row.get(wc))
                weeks[wc] = f"{val:.1f}%"
            retention[cohort_name] = {"weeks": weeks}

        # 전체 평균 리텐션 계산
        avg_retention = {}
        for wc in week_cols:
            vals = pd.to_numeric(df[wc], errors="coerce").dropna()
            avg_retention[wc] = round(vals.mean(), 1) if len(vals) > 0 else 0

        return {
            "status": "success",
            "analysis_type": "셀러 코호트 리텐션 분석",
            "total_cohorts": len(retention),
            "retention": retention,
            "avg_retention": {k: f"{v}%" for k, v in avg_retention.items()},
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_trend_analysis(start_date: str = None, end_date: str = None, days: int = None) -> dict:
    """플랫폼 트렌드 KPI 분석을 조회합니다. GMV, 주문수, 활성 셀러 등 주요 지표의 변화율을 반환합니다.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        days: 최근 N일 분석 (start_date/end_date 대신 사용 가능)
    """
    if st.DAILY_METRICS_DF is None:
        return {"status": "error", "message": "일별 지표 데이터가 없습니다."}

    try:
        df = st.DAILY_METRICS_DF.copy()

        # 날짜 컬럼 파싱
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # 날짜 필터링
        full_df = df.copy()  # fallback용 원본 보관
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                if 'date' in df.columns:
                    df = df[(df['date'] >= start) & (df['date'] <= end)]
                    # 날짜 필터링 결과가 부족하면 전체 데이터로 fallback
                    if len(df) < 2:
                        df = full_df
            except Exception:
                pass
        elif days and days > 0:
            df = df.tail(days * 2)  # 비교를 위해 2배 기간 가져오기

        # 최근 기간 vs 이전 기간 비교
        compare_days = days if days and days > 0 else 7
        if len(df) >= compare_days * 2:
            recent = df.tail(compare_days)
            previous = df.iloc[-(compare_days * 2):-compare_days]
        elif len(df) >= 2:
            mid = len(df) // 2
            recent = df.tail(mid)
            previous = df.head(mid)
        else:
            return {"status": "error", "message": "데이터가 충분하지 않습니다."}

        def calc_change(curr, prev):
            if prev == 0:
                return 0
            return round((curr - prev) / prev * 100, 1)

        def format_change(val):
            if val > 0:
                return f"+{val}%"
            return f"{val}%"

        # 컬럼명 호환 (daily_metrics.csv: total_gmv, active_shops, new_signups, cs_tickets_open 등)
        def _col(df_, name, fallbacks=None):
            if name in df_.columns:
                return name
            for fb in (fallbacks or []):
                if fb in df_.columns:
                    return fb
            return None

        gmv_col = _col(recent, 'total_gmv', ['gmv'])
        shops_col = _col(recent, 'active_shops', ['active_sellers'])
        orders_col = _col(recent, 'total_orders')
        signups_col = _col(recent, 'new_signups', ['new_sellers'])
        cs_open_col = _col(recent, 'cs_tickets_open', ['cs_tickets'])
        cs_resolved_col = _col(recent, 'cs_tickets_resolved')
        settlement_col = _col(recent, 'avg_settlement_time')
        sessions_col = _col(recent, 'total_sessions')

        # KPI 계산 - GMV
        gmv_curr = int(recent[gmv_col].mean()) if gmv_col else 0
        gmv_prev = int(previous[gmv_col].mean()) if gmv_col else 0
        gmv_change = calc_change(gmv_curr, gmv_prev)

        # 활성 쇼핑몰 수
        shops_curr = int(recent[shops_col].mean()) if shops_col else 0
        shops_prev = int(previous[shops_col].mean()) if shops_col else 0
        shops_change = calc_change(shops_curr, shops_prev)

        # 총 주문수
        orders_curr = int(recent[orders_col].mean()) if orders_col else 0
        orders_prev = int(previous[orders_col].mean()) if orders_col else 0
        orders_change = calc_change(orders_curr, orders_prev)

        # 신규 가입
        signups_curr = int(recent[signups_col].mean()) if signups_col else 0
        signups_prev = int(previous[signups_col].mean()) if signups_col else 0
        signups_change = calc_change(signups_curr, signups_prev)

        # 정산 소요시간
        settlement_curr = round(recent[settlement_col].mean(), 1) if settlement_col else 0
        settlement_prev = round(previous[settlement_col].mean(), 1) if settlement_col else 0
        settlement_change = calc_change(settlement_curr, settlement_prev)

        # CS 티켓 (open)
        cs_curr = int(recent[cs_open_col].mean()) if cs_open_col else 0
        cs_prev = int(previous[cs_open_col].mean()) if cs_open_col else 0
        cs_change = calc_change(cs_curr, cs_prev)

        # CS 해결률
        cs_resolved_curr = int(recent[cs_resolved_col].sum()) if cs_resolved_col else 0
        cs_open_sum_curr = int(recent[cs_open_col].sum()) if cs_open_col else 1
        cs_rate_curr = round(cs_resolved_curr / max(cs_open_sum_curr, 1) * 100, 1)

        cs_resolved_prev = int(previous[cs_resolved_col].sum()) if cs_resolved_col else 0
        cs_open_sum_prev = int(previous[cs_open_col].sum()) if cs_open_col else 1
        cs_rate_prev = round(cs_resolved_prev / max(cs_open_sum_prev, 1) * 100, 1)
        cs_rate_change = calc_change(cs_rate_curr, cs_rate_prev)

        # 세션
        sessions_curr = int(recent[sessions_col].mean()) if sessions_col else 0
        sessions_prev = int(previous[sessions_col].mean()) if sessions_col else 0
        sessions_change = calc_change(sessions_curr, sessions_prev)

        # 상관관계 계산
        correlations = []
        if len(df) >= 7 and shops_col and gmv_col and orders_col:
            corr_shops_gmv = df[shops_col].corr(df[gmv_col])
            corr_orders_gmv = df[orders_col].corr(df[gmv_col])
            correlations = [
                {"var1": "활성 쇼핑몰", "var2": "GMV", "correlation": round(corr_shops_gmv, 2),
                 "strength": "강함" if abs(corr_shops_gmv) > 0.7 else "중간"},
                {"var1": "주문수", "var2": "GMV", "correlation": round(corr_orders_gmv, 2),
                 "strength": "강함" if abs(corr_orders_gmv) > 0.7 else "중간"},
            ]

        # 매출 포맷팅
        def format_revenue(val):
            if val >= 100000000:
                return f"₩{val / 100000000:.1f}억"
            elif val >= 10000:
                return f"₩{val / 10000:.0f}만"
            else:
                return f"₩{val:,}"

        insight_parts = [f"일평균 GMV {format_revenue(gmv_curr)}으로 전기간 대비 {format_change(gmv_change)} 변화."]
        if signups_change < 0:
            insight_parts.append(f"신규 가입 {format_change(signups_change)} 감소 주의.")
        if cs_rate_change > 0:
            insight_parts.append(f"CS 해결률 {format_change(cs_rate_change)} 개선.")

        return {
            "status": "success",
            "analysis_type": "플랫폼 트렌드 분석",
            "period": f"최근 {len(recent)}일 vs 이전 {len(previous)}일",
            "kpis": {
                "GMV": {"current": format_revenue(gmv_curr), "previous": format_revenue(gmv_prev), "change": format_change(gmv_change)},
                "활성쇼핑몰": {"current": shops_curr, "previous": shops_prev, "change": format_change(shops_change)},
                "주문수": {"current": orders_curr, "previous": orders_prev, "change": format_change(orders_change)},
                "신규가입": {"current": signups_curr, "previous": signups_prev, "change": format_change(signups_change)},
                "정산소요시간": {"current": f"{settlement_curr}일", "previous": f"{settlement_prev}일", "change": format_change(settlement_change)},
                "CS해결률": {"current": f"{cs_rate_curr}%", "previous": f"{cs_rate_prev}%", "change": format_change(cs_rate_change)},
                "CS티켓": {"current": cs_curr, "previous": cs_prev, "change": format_change(cs_change)},
                "세션수": {"current": sessions_curr, "previous": sessions_prev, "change": format_change(sessions_change)},
            },
            "correlations": correlations,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_gmv_prediction(days: int = None, start_date: str = None, end_date: str = None) -> dict:
    """GMV(총 거래액) 예측 분석을 조회합니다. 예상 GMV, 주문 트렌드, 셀러 매출 분포를 반환합니다.

    Args:
        days: 최근 N일 기준 분석 (기본값: 30일)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
    """
    if st.DAILY_METRICS_DF is None:
        return {"status": "error", "message": "일별 지표 데이터가 없습니다."}

    try:
        metrics_df = st.DAILY_METRICS_DF.copy()

        # 날짜 컬럼 파싱
        if 'date' in metrics_df.columns:
            metrics_df['date'] = pd.to_datetime(metrics_df['date'])

        # 날짜 필터링
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                if 'date' in metrics_df.columns:
                    metrics_df = metrics_df[(metrics_df['date'] >= start) & (metrics_df['date'] <= end)]
            except Exception:
                pass

        # 분석 기간 설정 (기본 30일)
        analyze_days = days if days and days > 0 else 30

        # 최근 N일 vs 이전 N일
        if len(metrics_df) >= analyze_days * 2:
            recent = metrics_df.tail(analyze_days)
            previous = metrics_df.iloc[-(analyze_days * 2):-analyze_days]
        elif len(metrics_df) >= 14:
            mid = len(metrics_df) // 2
            recent = metrics_df.tail(mid)
            previous = metrics_df.head(mid)
        else:
            recent = metrics_df
            previous = metrics_df

        # 컬럼명 호환
        gmv_col = 'total_gmv' if 'total_gmv' in recent.columns else 'gmv'
        shops_col = 'active_shops' if 'active_shops' in recent.columns else 'active_sellers'

        # 월 GMV 예측 (최근 일평균 * 30)
        daily_avg_gmv = recent[gmv_col].mean() if gmv_col in recent.columns else 0
        monthly_gmv = int(daily_avg_gmv * 30)
        prev_monthly = int(previous[gmv_col].mean() * 30) if gmv_col in previous.columns else 0
        growth_rate = round((monthly_gmv - prev_monthly) / prev_monthly * 100, 1) if prev_monthly > 0 else 0

        # 평균 주문 금액 (AOV) - daily_metrics에 없으면 GMV/주문수로 계산
        if 'avg_order_value' in recent.columns:
            avg_order_value = int(recent['avg_order_value'].mean())
        elif 'total_orders' in recent.columns and gmv_col in recent.columns:
            total_orders_sum = recent['total_orders'].sum()
            avg_order_value = int(recent[gmv_col].sum() / max(total_orders_sum, 1))
        else:
            avg_order_value = 0

        # 총 주문수
        total_orders = int(recent['total_orders'].sum()) if 'total_orders' in recent.columns else 0

        # 활성 쇼핑몰
        avg_active_sellers = int(recent[shops_col].mean()) if shops_col in recent.columns else 0

        # 셀러 등급별 매출 분포 (SELLER_ANALYTICS_DF 기반)
        tier_distribution = {}
        if st.SELLER_ANALYTICS_DF is not None and 'plan_tier' in st.SELLER_ANALYTICS_DF.columns:
            seller_df = st.SELLER_ANALYTICS_DF
            for tier in PLAN_TIERS:
                tier_sellers = seller_df[seller_df['plan_tier'] == tier]
                if len(tier_sellers) > 0:
                    tier_distribution[tier] = {
                        "seller_count": len(tier_sellers),
                        "avg_revenue": safe_int(tier_sellers['total_revenue'].mean()),
                        "total_revenue": safe_int(tier_sellers['total_revenue'].sum()),
                    }

        # 전환율 (daily_metrics에 없을 수 있음)
        avg_conversion = round(recent['conversion_rate'].mean(), 2) if 'conversion_rate' in recent.columns else 0.0

        # 매출 포맷팅
        def format_revenue(val):
            if val >= 100000000:
                return f"₩{val / 100000000:.1f}억"
            elif val >= 10000:
                return f"₩{val / 10000:.0f}만"
            else:
                return f"₩{val:,}"

        growth_str = f"+{growth_rate}%" if growth_rate > 0 else f"{growth_rate}%"

        return {
            "status": "success",
            "prediction_type": "GMV 예측",
            "monthly_forecast": {
                "predicted_gmv": format_revenue(monthly_gmv),
                "growth_rate": growth_str,
                "daily_avg": format_revenue(int(daily_avg_gmv)),
            },
            "platform_metrics": {
                "avg_order_value": f"₩{avg_order_value:,}",
                "total_orders": total_orders,
                "active_sellers": avg_active_sellers,
                "conversion_rate": f"{avg_conversion}%",
            },
            "tier_distribution": tier_distribution,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_dashboard_summary() -> dict:
    """대시보드 요약 정보를 조회합니다. 플랫폼 전체 운영 현황을 반환합니다."""

    summary = {
        "status": "success",
    }

    # 쇼핑몰 통계 (프론트엔드: shop_stats.total, shop_stats.by_tier)
    if st.SHOPS_DF is not None:
        shops_df = st.SHOPS_DF
        by_tier = shops_df["plan_tier"].value_counts().to_dict() if "plan_tier" in shops_df.columns else {}
        summary["shop_stats"] = {
            "total": len(shops_df),
            "by_tier": by_tier,
            "by_plan_tier": by_tier,
            "by_category": shops_df["category"].value_counts().to_dict() if "category" in shops_df.columns else {},
            "by_region": shops_df["region"].value_counts().to_dict() if "region" in shops_df.columns else {},
        }

    # 셀러 세그먼트 통계 (프론트엔드: seller_stats.total, .anomaly_count, .segments)
    if st.SELLER_ANALYTICS_DF is not None:
        seller_df = st.SELLER_ANALYTICS_DF
        anomaly_count = int(seller_df["is_anomaly"].sum()) if "is_anomaly" in seller_df.columns else 0
        seller_stats = {
            "total": len(seller_df),
            "anomaly_count": anomaly_count,
        }

        if "cluster" in seller_df.columns:
            raw_segments = seller_df["cluster"].value_counts().to_dict()
            seller_stats["segments"] = {
                _get_segment_name(k): v for k, v in raw_segments.items()
            }
        elif "plan_tier" in seller_df.columns:
            seller_stats["by_plan_tier"] = seller_df["plan_tier"].value_counts().to_dict()

        summary["seller_stats"] = seller_stats
    elif st.SELLERS_DF is not None:
        sellers_df = st.SELLERS_DF
        summary["seller_stats"] = {
            "total": len(sellers_df),
            "by_plan_tier": sellers_df["plan_tier"].value_counts().to_dict() if "plan_tier" in sellers_df.columns else {},
            "by_region": sellers_df["region"].value_counts().to_dict() if "region" in sellers_df.columns else {},
        }

    # CS 통계 (프론트엔드: cs_stats.total, .avg_satisfaction, .by_category)
    # CSV 컬럼: category, total_tickets, avg_resolution_hours, satisfaction_score
    if st.CS_STATS_DF is not None:
        cs_df = st.CS_STATS_DF
        cat_col = "category" if "category" in cs_df.columns else "ticket_category"
        total_tickets = int(cs_df["total_tickets"].sum()) if "total_tickets" in cs_df.columns else len(cs_df)
        avg_satisfaction = round(float(cs_df["satisfaction_score"].mean()), 1) if "satisfaction_score" in cs_df.columns else 0
        avg_resolution = round(float(cs_df["avg_resolution_hours"].mean()), 1) if "avg_resolution_hours" in cs_df.columns else 0

        # by_category: {카테고리: 건수} 형태
        by_category = {}
        if cat_col in cs_df.columns and "total_tickets" in cs_df.columns:
            for _, row in cs_df.iterrows():
                by_category[str(row[cat_col])] = int(row["total_tickets"])
        elif cat_col in cs_df.columns:
            by_category = cs_df[cat_col].value_counts().to_dict()

        summary["cs_stats"] = {
            "total": total_tickets,
            "total_tickets": total_tickets,
            "avg_satisfaction": avg_satisfaction,
            "avg_resolution_hours": avg_resolution,
            "by_category": by_category,
        }

    # 주문/이벤트 통계 (프론트엔드: order_stats.total, .by_type)
    # CSV 컬럼: log_id, seller_id, event_type, event_date, details_json
    if st.OPERATION_LOGS_DF is not None and "event_type" in st.OPERATION_LOGS_DF.columns:
        logs_df = st.OPERATION_LOGS_DF
        event_counts = logs_df["event_type"].value_counts().to_dict()
        summary["order_stats"] = {
            "total": len(logs_df),
            "total_events": len(logs_df),
            "by_type": event_counts,
        }

    # 이상거래 통계
    # CSV 컬럼: seller_id, anomaly_score, anomaly_type, detected_date, details
    if st.FRAUD_DETAILS_DF is not None:
        fraud_df = st.FRAUD_DETAILS_DF
        high_risk_count = len(fraud_df[fraud_df["anomaly_score"] >= 0.9]) if "anomaly_score" in fraud_df.columns else 0
        by_type = fraud_df["anomaly_type"].value_counts().to_dict() if "anomaly_type" in fraud_df.columns else {}
        summary["fraud_stats"] = {
            "total_records": len(fraud_df),
            "high_risk_count": high_risk_count,
            "fraud_rate": round(high_risk_count / len(fraud_df) * 100, 1) if len(fraud_df) > 0 else 0,
            "by_type": by_type,
        }

    # 일별 플랫폼 지표 (최근 14일)
    # CSV 컬럼: date, active_shops, total_gmv, new_signups, total_orders, avg_settlement_time, cs_tickets_open, cs_tickets_resolved, fraud_alerts
    if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
        recent_df = st.DAILY_METRICS_DF.tail(14)
        daily_list = []
        for _, row in recent_df.iterrows():
            date_val = safe_str(row.get("date"))
            # 날짜를 MM/DD 형식으로 변환
            try:
                date_display = pd.to_datetime(date_val).strftime("%m/%d")
            except Exception:
                date_display = date_val
            daily_list.append({
                "date": date_display,
                "gmv": safe_int(row.get("total_gmv", row.get("gmv", 0))),
                "active_sellers": safe_int(row.get("active_shops", row.get("active_sellers", 0))),
                "total_orders": safe_int(row.get("total_orders", 0)),
                "new_sellers": safe_int(row.get("new_signups", row.get("new_sellers", 0))),
            })
        summary["daily_metrics"] = daily_list
        # 프론트엔드 GMV 차트용: daily_gmv 배열 (date, gmv)
        summary["daily_gmv"] = [{"date": d["date"], "gmv": d["gmv"]} for d in daily_list]
    else:
        summary["daily_metrics"] = []
        summary["daily_gmv"] = []

    return summary


# ============================================================
# 10. ML 모델 예측 도구
# ============================================================
def tool_predict_seller_churn(seller_id: str) -> dict:
    """
    특정 셀러의 이탈 확률을 예측합니다.
    SELLER_CHURN_MODEL(RandomForest)과 SHAP Explainer를 사용하여 예측 및 설명을 제공합니다.
    """
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "셀러 분석 데이터가 로드되지 않았습니다."}

    # 셀러 데이터 조회
    seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if seller.empty:
        return {"status": "error", "message": f"셀러 '{seller_id}'를 찾을 수 없습니다."}

    row = seller.iloc[0]

    # 이탈 예측 모델이 없으면 휴리스틱 사용
    if st.SELLER_CHURN_MODEL is None:
        total_orders = safe_int(row.get("total_orders", 0))
        total_revenue = safe_int(row.get("total_revenue", 0))
        days_since_last = safe_int(row.get("days_since_last_login", 0))
        refund_rate = safe_float(row.get("refund_rate", 0))
        cs_tickets = safe_int(row.get("cs_tickets", 0))

        # 이탈 위험 점수 계산
        churn_score = 0.3  # 기본값
        if days_since_last > 14:
            churn_score += 0.25
        elif days_since_last > 7:
            churn_score += 0.15
        if total_orders < 10:
            churn_score += 0.1
        if total_revenue < 100000:
            churn_score += 0.1
        if refund_rate > 10:
            churn_score += 0.1
        if cs_tickets > 20:
            churn_score += 0.05

        churn_score = min(max(churn_score, 0.05), 0.95)
        risk_level = "HIGH" if churn_score > 0.6 else "MEDIUM" if churn_score > 0.3 else "LOW"

        return {
            "status": "success",
            "seller_id": seller_id,
            "churn_probability": round(churn_score * 100, 1),
            "risk_level": risk_level,
            "model_used": "heuristic",
            "top_factors": [
                {"factor": f"마지막 접속 {days_since_last}일 전", "importance": 30},
                {"factor": f"총 주문 수 {total_orders}건", "importance": 25},
                {"factor": f"총 매출 ₩{total_revenue:,}", "importance": 20},
                {"factor": f"환불률 {refund_rate}%", "importance": 15},
                {"factor": f"CS 문의 {cs_tickets}건", "importance": 10},
            ],
            "recommendation": _get_seller_churn_recommendation(risk_level, days_since_last, total_revenue),
        }

    # 실제 모델 사용
    try:
        # 피처 준비
        feature_cols = FEATURE_COLS_CHURN
        X = pd.DataFrame([{col: safe_float(row.get(col, 0)) for col in feature_cols}])

        # plan_tier 인코딩 처리
        if "plan_tier_encoded" in feature_cols and "plan_tier" in row.index:
            tier_map = {tier: i for i, tier in enumerate(PLAN_TIERS)}
            X["plan_tier_encoded"] = tier_map.get(row.get("plan_tier", "Basic"), 0)

        # 예측
        churn_prob = st.SELLER_CHURN_MODEL.predict_proba(X)[0][1]  # 이탈 확률
        churn_pred = st.SELLER_CHURN_MODEL.predict(X)[0]

        risk_level = "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.3 else "LOW"

        # SHAP 설명
        top_factors = []
        if st.SHAP_EXPLAINER_CHURN is not None:
            try:
                explainer = st.SHAP_EXPLAINER_CHURN

                if hasattr(explainer, 'shap_values'):
                    shap_result = explainer.shap_values(X)
                    if isinstance(shap_result, list) and len(shap_result) == 2:
                        shap_vals = np.array(shap_result[1])[0]
                    elif isinstance(shap_result, np.ndarray):
                        if shap_result.ndim == 3:
                            shap_vals = shap_result[0, :, 1]
                        else:
                            shap_vals = shap_result[0]
                    else:
                        shap_vals = np.array(shap_result)[0]
                else:
                    shap_result = explainer(X)
                    if hasattr(shap_result, 'values'):
                        shap_vals = shap_result.values[0]
                    else:
                        shap_vals = np.array(shap_result)[0]

                feature_importance = list(zip(feature_cols, np.abs(shap_vals)))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                for feat, imp in feature_importance[:5]:
                    top_factors.append({
                        "factor": FEATURE_LABELS.get(feat, feat),
                        "importance": round(float(imp) * 100, 1),
                    })
            except Exception as e:
                print(f"[SHAP Error] {type(e).__name__}: {e}")

        if not top_factors:
            top_factors = [
                {"factor": "마지막 접속 후 일수", "importance": 30},
                {"factor": "총 매출", "importance": 25},
                {"factor": "주문 수", "importance": 20},
                {"factor": "환불률", "importance": 15},
                {"factor": "플랜 등급", "importance": 10},
            ]

        days_since_last = safe_int(row.get("days_since_last_login", 0))
        total_revenue = safe_int(row.get("total_revenue", 0))

        return {
            "status": "success",
            "seller_id": seller_id,
            "churn_probability": round(churn_prob * 100, 1),
            "risk_level": risk_level,
            "will_churn": bool(churn_pred),
            "model_used": "random_forest",
            "top_factors": top_factors,
            "recommendation": _get_seller_churn_recommendation(risk_level, days_since_last, total_revenue),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_seller_churn_recommendation(risk_level: str, days_since_last: int, total_revenue: int) -> str:
    """셀러 이탈 위험에 따른 권장사항"""
    if risk_level == "HIGH":
        if days_since_last > 14:
            return "긴급! 14일 이상 미접속 셀러입니다. 전담 매니저 배정 및 복귀 인센티브(수수료 할인, 광고 크레딧) 제공을 권장합니다."
        return "높은 이탈 위험. 1:1 운영 컨설팅 및 플랜 업그레이드 혜택 안내를 권장합니다."
    elif risk_level == "MEDIUM":
        if total_revenue < 500000:
            return "중간 이탈 위험. 매출 부진 셀러입니다. 마케팅 교육 프로그램 및 프로모션 참여 유도를 권장합니다."
        return "중간 이탈 위험. 정기적인 운영 현황 리포트 발송 및 신규 기능 안내를 권장합니다."
    return "낮은 이탈 위험. 현재 운영 상태가 양호합니다. 플랜 업그레이드 및 추가 서비스 교차 판매 기회를 모색하세요."


def tool_predict_shop_revenue(shop_id: str) -> dict:
    """
    쇼핑몰의 다음달 매출을 예측합니다.
    REVENUE_PREDICTION_MODEL(LightGBM)을 사용합니다.
    """
    if st.SHOP_PERFORMANCE_DF is None:
        return {"status": "error", "message": "쇼핑몰 성과 데이터가 로드되지 않았습니다."}

    # 쇼핑몰 검색
    shop = st.SHOP_PERFORMANCE_DF[st.SHOP_PERFORMANCE_DF["shop_id"] == shop_id]
    if shop.empty:
        shop = st.SHOP_PERFORMANCE_DF[st.SHOP_PERFORMANCE_DF["name"].str.contains(shop_id, na=False)]

    if shop.empty:
        return {"status": "error", "message": f"쇼핑몰 '{shop_id}'를 찾을 수 없습니다."}

    row = shop.iloc[0]

    # 실제 다음달 매출 (데이터에 있는 경우)
    actual_next_revenue = safe_int(row.get("next_month_revenue", 0))

    # 현재 성과 지표
    total_revenue = safe_int(row.get("total_revenue"))
    total_orders = safe_int(row.get("total_orders"))
    unique_customers = safe_int(row.get("unique_customers"))
    avg_order_value = safe_int(row.get("avg_order_value"))
    revenue_growth = safe_float(row.get("revenue_growth"))
    conversion_rate = safe_float(row.get("conversion_rate"))
    review_score = safe_float(row.get("review_score"))
    refund_rate = safe_float(row.get("refund_rate"))

    # 모델 예측
    predicted_revenue = actual_next_revenue
    if st.REVENUE_PREDICTION_MODEL is not None:
        try:
            feature_dict = {
                "total_revenue": total_revenue,
                "total_orders": total_orders,
                "unique_customers": unique_customers,
                "avg_order_value": avg_order_value,
                "revenue_growth": revenue_growth,
                "conversion_rate": conversion_rate,
                "review_score": review_score,
                "refund_rate": refund_rate,
            }
            X = pd.DataFrame([feature_dict])
            predicted_revenue = int(st.REVENUE_PREDICTION_MODEL.predict(X)[0])
        except Exception:
            predicted_revenue = actual_next_revenue if actual_next_revenue > 0 else int(total_revenue * (1 + revenue_growth / 100))

    # 매출 등급
    revenue_tier = _get_revenue_tier(predicted_revenue)

    # 매출 포맷팅
    def format_revenue(val):
        if val >= 100000000:
            return f"₩{val / 100000000:.1f}억"
        elif val >= 10000:
            return f"₩{val / 10000:.0f}만"
        else:
            return f"₩{val:,}"

    return {
        "status": "success",
        "shop_id": safe_str(row.get("shop_id")),
        "name": safe_str(row.get("name")),
        "category": safe_str(row.get("category")),
        "region": safe_str(row.get("region")),
        "predicted_revenue": format_revenue(predicted_revenue),
        "predicted_revenue_raw": predicted_revenue,
        "revenue_tier": revenue_tier,
        "current_performance": {
            "total_revenue": format_revenue(total_revenue),
            "total_orders": total_orders,
            "unique_customers": unique_customers,
            "avg_order_value": f"₩{avg_order_value:,}",
            "revenue_growth": f"{revenue_growth}%",
            "conversion_rate": f"{conversion_rate}%",
            "review_score": review_score,
            "refund_rate": f"{refund_rate}%",
        },
        "analysis": _analyze_shop_performance(row, predicted_revenue),
    }


def _get_revenue_tier(revenue: int) -> str:
    """매출 규모에 따른 등급 반환"""
    if revenue >= 50000000:
        return "S (최상위 매출)"
    elif revenue >= 20000000:
        return "A (상위 매출)"
    elif revenue >= 10000000:
        return "B (평균 매출)"
    elif revenue >= 5000000:
        return "C (하위 매출)"
    return "D (최하위 매출)"


def _analyze_shop_performance(row, predicted_revenue: int) -> str:
    """쇼핑몰 성과 분석"""
    analysis = []

    conversion_rate = safe_float(row.get("conversion_rate"))
    review_score = safe_float(row.get("review_score"))
    refund_rate = safe_float(row.get("refund_rate"))
    revenue_growth = safe_float(row.get("revenue_growth"))

    if conversion_rate > 3:
        analysis.append("전환율이 우수하여 트래픽 유입 확대가 효과적일 수 있습니다")
    elif conversion_rate < 1:
        analysis.append("전환율 개선이 필요합니다. 상품 상세 페이지 최적화를 권장합니다")

    if review_score > 4.5:
        analysis.append("고객 리뷰 평점이 매우 높아 브랜드 신뢰도가 우수합니다")
    elif review_score < 3.5:
        analysis.append("리뷰 평점이 낮습니다. CS 품질 개선 및 상품 품질 점검이 필요합니다")

    if refund_rate > 5:
        analysis.append("환불률이 높습니다. 상품 설명 정확성 및 배송 품질 점검을 권장합니다")

    if revenue_growth > 10:
        analysis.append("높은 매출 성장률을 보이고 있습니다")
    elif revenue_growth < -5:
        analysis.append("매출이 감소 추세입니다. 마케팅 전략 재검토가 필요합니다")

    if not analysis:
        analysis.append("전반적으로 안정적인 운영 상태입니다")

    return ". ".join(analysis) + "."


def tool_get_shop_performance(shop_id: str) -> dict:
    """
    특정 쇼핑몰의 성과 KPI를 조회합니다.
    """
    if st.SHOP_PERFORMANCE_DF is None:
        return {"status": "error", "message": "쇼핑몰 성과 데이터가 로드되지 않았습니다."}

    # 쇼핑몰 검색
    shop = st.SHOP_PERFORMANCE_DF[st.SHOP_PERFORMANCE_DF["shop_id"] == shop_id]
    if shop.empty:
        shop = st.SHOP_PERFORMANCE_DF[st.SHOP_PERFORMANCE_DF["name"].str.contains(shop_id, na=False)]

    if shop.empty:
        return {"status": "error", "message": f"쇼핑몰 '{shop_id}'를 찾을 수 없습니다."}

    row = shop.iloc[0]

    total_revenue = safe_int(row.get("total_revenue"))

    def format_revenue(val):
        if val >= 100000000:
            return f"₩{val / 100000000:.1f}억"
        elif val >= 10000:
            return f"₩{val / 10000:.0f}만"
        else:
            return f"₩{val:,}"

    return {
        "status": "success",
        "shop_id": safe_str(row.get("shop_id")),
        "name": safe_str(row.get("name")),
        "category": safe_str(row.get("category")),
        "region": safe_str(row.get("region")),
        "performance": {
            "total_revenue": format_revenue(total_revenue),
            "total_revenue_raw": total_revenue,
            "total_orders": safe_int(row.get("total_orders")),
            "unique_customers": safe_int(row.get("unique_customers")),
            "avg_order_value": safe_int(row.get("avg_order_value")),
            "revenue_growth": safe_float(row.get("revenue_growth")),
            "conversion_rate": safe_float(row.get("conversion_rate")),
            "review_score": safe_float(row.get("review_score")),
            "refund_rate": safe_float(row.get("refund_rate")),
        },
        "revenue_tier": _get_revenue_tier(total_revenue),
    }


def tool_optimize_marketing(
    seller_id: str,
    goal: str = "maximize_roas",
    total_budget: float = None,
) -> dict:
    """
    셀러의 마케팅 예산을 분석하여 최적의 광고 투자 전략을 제안합니다.
    P-PSO(Phasor Particle Swarm Optimization) 알고리즘 사용.

    Args:
        seller_id: 셀러 ID
        goal: 최적화 목표 ('maximize_roas', 'maximize_revenue', 'balanced')
        total_budget: 총 마케팅 예산 (None이면 셀러 매출의 10%로 자동 산정)

    Returns:
        채널별 예산 배분 추천, 예상 ROAS, 매출 예측 (프론트엔드 호환 포맷)
    """
    # 채널→캠페인 타입 매핑
    CAMPAIGN_TYPE_MAP = {
        "search_ads": "cpc",
        "display_ads": "display",
        "social_media": "social",
        "email_marketing": "email",
        "influencer": "social",
        "content_marketing": "content",
    }

    try:
        # 셀러 매출 기반 예산 산정
        if total_budget is None:
            seller_revenue = 0
            if st.SELLERS_DF is not None:
                row = st.SELLERS_DF[st.SELLERS_DF["seller_id"] == seller_id]
                if not row.empty:
                    seller_revenue = float(row.iloc[0].get("total_revenue", 0))
            if st.SELLER_ANALYTICS_DF is not None and seller_revenue == 0:
                row = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
                if not row.empty:
                    seller_revenue = float(row.iloc[0].get("total_revenue", 0))
            total_budget = max(500_000, seller_revenue * 0.1)  # 매출 10% 또는 최소 50만

        if st.MARKETING_OPTIMIZER_AVAILABLE:
            from ml.marketing_optimizer import MarketingOptimizer
            optimizer = MarketingOptimizer(seller_id, total_budget=total_budget, goal=goal)
            result = optimizer.optimize(max_iterations=200)

            if "error" in result:
                return {"status": "error", "message": result["error"]}

            # optimizer allocation → 프론트엔드 recommendations 변환
            recommendations = []
            total_cvr_gain = 0.0
            for rec in result.get("allocation", []):
                channel = rec.get("channel", "")
                campaign_type = CAMPAIGN_TYPE_MAP.get(channel, "cpc")
                budget_val = rec.get("budget", 0)
                uplift = rec.get("expected_revenue_uplift", 0)
                cvr_gain = round(uplift * 100, 1)
                total_cvr_gain += cvr_gain
                eff = rec.get("efficiency_score", 0)
                max_eff = max(r.get("efficiency_score", 1) for r in result.get("allocation", [{"efficiency_score": 1}]))
                norm_eff = eff / max_eff if max_eff > 0 else 0

                recommendations.append({
                    "channel": channel,
                    "channel_name": rec.get("channel_name", channel),
                    "campaign_type": campaign_type,
                    "from_budget": f"₩{int(budget_val * 0.7):,}",
                    "to_budget": f"₩{int(budget_val):,}",
                    "cvr_gain": cvr_gain,
                    "efficiency": round(norm_eff, 3),
                    "cost": {"ad_spend": int(budget_val)},
                    "expected_roas": rec.get("expected_roas", 0),
                    "expected_revenue": rec.get("expected_revenue", 0),
                })

            return {
                "status": "success",
                "seller_id": seller_id,
                "goal": goal,
                "total_cvr_gain": round(total_cvr_gain, 1),
                "optimization_method": result.get("optimization_method", "P-PSO"),
                "recommendations": sorted(recommendations, key=lambda x: x["cvr_gain"], reverse=True),
                "summary": {
                    "total_budget": int(total_budget),
                    "total_expected_revenue": result.get("total_expected_revenue", 0),
                    "overall_roas": result.get("overall_roas", 0),
                },
                "budget_usage": result.get("budget_usage", {}),
            }
        else:
            return {"status": "error", "message": "마케팅 최적화 모듈이 로드되지 않았습니다."}

    except Exception as e:
        st.logger.exception("마케팅 최적화 실패")
        return {"status": "error", "message": str(e)}


# ============================================================
# 도구 레지스트리
# ============================================================
AVAILABLE_TOOLS = {
    # 쇼핑몰 정보
    "get_shop_info": tool_get_shop_info,
    "list_shops": tool_list_shops,
    "get_shop_services": tool_get_shop_services,

    # 카테고리 정보
    "get_category_info": tool_get_category_info,
    "list_categories": tool_list_categories,

    # CS 관련
    "auto_reply_cs": tool_auto_reply_cs,
    "check_cs_quality": tool_check_cs_quality,
    "get_ecommerce_glossary": tool_get_ecommerce_glossary,
    "get_cs_statistics": tool_get_cs_statistics,

    # 셀러 분석
    "analyze_seller": tool_analyze_seller,
    "get_seller_segment": tool_get_seller_segment,
    "detect_fraud": tool_detect_fraud,
    "get_segment_statistics": tool_get_segment_statistics,
    "get_fraud_statistics": tool_get_fraud_statistics,

    # 주문/운영 로그
    "get_order_statistics": tool_get_order_statistics,
    "get_seller_activity_report": tool_get_seller_activity_report,

    # 문의 분류
    "classify_inquiry": tool_classify_inquiry,

    # RAG 검색
    "search_platform": tool_search_platform,
    "search_platform_lightrag": tool_search_platform_lightrag,

    # 대시보드/통계
    "get_dashboard_summary": tool_get_dashboard_summary,

    # 예측 분석
    "get_churn_prediction": tool_get_churn_prediction,
    "get_gmv_prediction": tool_get_gmv_prediction,
    "get_cohort_analysis": tool_get_cohort_analysis,
    "get_trend_analysis": tool_get_trend_analysis,

    # ML 모델 예측
    "predict_seller_churn": tool_predict_seller_churn,
    "predict_shop_revenue": tool_predict_shop_revenue,
    "get_shop_performance": tool_get_shop_performance,
    "optimize_marketing": tool_optimize_marketing,
}
