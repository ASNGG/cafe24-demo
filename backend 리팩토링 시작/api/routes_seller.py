"""
api/routes_seller.py - 셀러 관련 API
"""
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from core.utils import safe_str, json_sanitize, get_revenue_r2
from agent.tools import (
    tool_analyze_seller, tool_get_seller_segment, tool_detect_fraud,
    tool_get_segment_statistics, tool_get_seller_activity_report,
    tool_get_shop_performance, tool_predict_seller_churn,
    tool_predict_shop_revenue,
)
import state as st
from api.common import verify_credentials, error_response


router = APIRouter(prefix="/api", tags=["seller"])


@router.get("/sellers/autocomplete")
def sellers_autocomplete(q: str = "", limit: int = 8, user: dict = Depends(verify_credentials)):
    if st.SELLERS_DF is None:
        return error_response("셀러 데이터 없음")
    q = q.strip().upper()
    if not q:
        return {"status": "success", "users": []}
    df = st.SELLERS_DF
    mask = df["seller_id"].str.upper().str.contains(q, na=False)
    if "shop_name" in df.columns:
        mask |= df["shop_name"].str.upper().str.contains(q, na=False)
    matched = df[mask].head(limit)
    users = [{"id": r["seller_id"], "name": r["seller_id"]} for r in matched[["seller_id"]].to_dict("records")]
    return {"status": "success", "users": users}


@router.get("/sellers/search")
def search_seller(q: str, days: int = 7, user: dict = Depends(verify_credentials)):
    """셀러 검색"""
    if st.SELLERS_DF is None or st.SELLER_ANALYTICS_DF is None:
        return error_response("셀러 데이터가 로드되지 않았습니다.")
    if days not in [7, 30, 90]:
        days = 7
    q = q.strip().upper()
    seller_row = st.SELLERS_DF[st.SELLERS_DF["seller_id"] == q]
    if seller_row.empty:
        seller_row = st.SELLERS_DF[st.SELLERS_DF["seller_id"].str.contains(q, case=False, na=False)]
    if seller_row.empty:
        return error_response("셀러를 찾을 수 없습니다.")

    seller_data = seller_row.iloc[0].to_dict()
    seller_id = seller_data["seller_id"]
    analytics_row = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if not analytics_row.empty:
        analytics = analytics_row.iloc[0].to_dict()
        for k, v in analytics.items():
            if k != "seller_id":
                seller_data[k] = v
    seller_data["segment"] = seller_data.get("segment_name", f"세그먼트 {seller_data.get('cluster', 0)}")

    activity = []
    period_stats = {"total_revenue": 0, "total_orders": 0, "active_days": 0, "total_cs": 0, "total_refunds": 0}
    if st.SELLER_ACTIVITY_DF is not None:
        seller_activity = st.SELLER_ACTIVITY_DF[st.SELLER_ACTIVITY_DF["seller_id"] == seller_id]
        if not seller_activity.empty:
            seller_activity = seller_activity.tail(days)
            period_stats["active_days"] = len(seller_activity)
            # 컬럼 존재 여부 사전 확인
            rev_col = "revenue" if "revenue" in seller_activity.columns else "daily_revenue"
            ord_col = "orders_processed" if "orders_processed" in seller_activity.columns else "daily_orders"
            cs_col = "cs_handled" if "cs_handled" in seller_activity.columns else "cs_tickets"
            prod_col = "products_updated" if "products_updated" in seller_activity.columns else "product_count"
            # 벡터화로 합계 계산
            period_stats["total_revenue"] = int(seller_activity[rev_col].fillna(0).sum()) if rev_col in seller_activity.columns else 0
            period_stats["total_orders"] = int(seller_activity[ord_col].fillna(0).sum()) if ord_col in seller_activity.columns else 0
            period_stats["total_cs"] = int(seller_activity[cs_col].fillna(0).sum()) if cs_col in seller_activity.columns else 0
            # activity 리스트 빌드 (itertuples)
            for t in seller_activity.itertuples(index=False):
                revenue = int(getattr(t, rev_col, 0) or 0)
                orders = int(getattr(t, ord_col, 0) or 0)
                products = int(getattr(t, prod_col, 0) or 0)
                activity.append({"date": getattr(t, "date", ""), "revenue": revenue, "orders": orders, "product_count": products})

    def _percentile_score(value, col_name):
        if st.SELLER_ANALYTICS_DF is None or col_name not in st.SELLER_ANALYTICS_DF.columns:
            return 0
        col = st.SELLER_ANALYTICS_DF[col_name].dropna()
        if len(col) == 0:
            return 0
        rank = (col < value).sum() / len(col) * 100
        return min(100, max(0, int(rank)))

    stats = {
        "매출력": _percentile_score(seller_data.get("total_revenue", 0), "total_revenue"),
        "주문량": _percentile_score(seller_data.get("total_orders", 0), "total_orders"),
        "상품력": _percentile_score(seller_data.get("product_count", 0), "product_count"),
        "CS대응": max(0, 100 - _percentile_score(seller_data.get("cs_tickets", 0), "cs_tickets")),
        "성장성": _percentile_score(period_stats["total_orders"] / max(1, days) if days > 0 else 0, "total_orders"),
    }

    top_shops = []
    if st.SHOPS_DF is not None:
        shop_id = seller_data["seller_id"].replace("SEL", "S")
        matched = st.SHOPS_DF[st.SHOPS_DF["shop_id"] == shop_id]
        if len(matched) > 0:
            name_col = "shop_name" if "shop_name" in matched.columns else "shop_id"
            top_shops = matched.head(2)[name_col].fillna("").tolist()

    model_predictions = {}
    churn_prob = seller_data.get("churn_probability")
    churn_risk = seller_data.get("churn_risk")
    if churn_prob is not None:
        risk_label = "높음" if float(churn_prob) > 0.7 else "보통" if float(churn_prob) > 0.4 else "낮음"
        shap_factors = []
        shap_cols = [c for c in seller_data.keys() if str(c).startswith("shap_")]
        if shap_cols:
            shap_items = [(c.replace("shap_", ""), abs(float(seller_data.get(c, 0)))) for c in shap_cols]
            shap_items.sort(key=lambda x: x[1], reverse=True)
            factor_names = {"total_orders": "총 주문수", "total_revenue": "총 매출", "product_count": "상품 수", "cs_tickets": "CS 문의", "refund_rate": "환불률", "avg_response_time": "평균 응답시간", "days_since_last_login": "마지막 로그인", "days_since_register": "가입 경과일", "plan_tier_encoded": "플랜 등급"}
            for fname, fval in shap_items[:5]:
                shap_factors.append({"factor": factor_names.get(fname, fname), "importance": round(fval, 4)})
        model_predictions["churn"] = {"model": "셀러 이탈 예측 (RandomForest+SHAP)", "probability": round(float(churn_prob) * 100, 1), "risk_level": risk_label, "risk_code": int(churn_risk) if churn_risk is not None else (2 if float(churn_prob) > 0.7 else 1 if float(churn_prob) > 0.4 else 0), "factors": shap_factors}

    anomaly_score = seller_data.get("anomaly_score")
    is_anomaly = seller_data.get("is_anomaly")
    if anomaly_score is not None:
        model_predictions["fraud"] = {"model": "이상거래 탐지 (Isolation Forest)", "anomaly_score": round(float(anomaly_score), 4), "is_anomaly": bool(is_anomaly) if is_anomaly is not None else float(anomaly_score) > 0.7, "risk_level": "위험" if float(anomaly_score) > 0.7 else "주의" if float(anomaly_score) > 0.5 else "정상"}

    cluster = seller_data.get("cluster", 0)
    seg_name = seller_data.get("segment_name", f"세그먼트 {cluster}")
    model_predictions["segment"] = {"model": "셀러 세그먼트 (K-Means)", "cluster": int(cluster), "segment_name": seg_name}

    cs_score_val = seller_data.get("cs_quality_score")
    cs_grade_val = seller_data.get("cs_quality_grade")
    refund_rate = float(seller_data.get("refund_rate", 0))
    avg_resp = float(seller_data.get("avg_response_time", 0))
    if cs_score_val is not None:
        model_predictions["cs_quality"] = {"model": "CS 응답 품질 (RandomForest)", "score": int(cs_score_val), "grade": str(cs_grade_val) if cs_grade_val else ("우수" if int(cs_score_val) >= 80 else "보통" if int(cs_score_val) >= 50 else "개선필요"), "refund_rate": round(refund_rate, 4), "avg_response_time": round(avg_resp, 1)}

    predicted_rev = seller_data.get("predicted_revenue")
    rev_growth = seller_data.get("revenue_growth_rate")
    if predicted_rev is not None and float(predicted_rev) > 0:
        model_predictions["revenue"] = {"model": "매출 예측 (LightGBM)", "predicted_next_month": int(float(predicted_rev)), "growth_rate": round(float(rev_growth), 1) if rev_growth is not None else 0.0, "confidence": round(r2_score_val * 100, 1) if (r2_score_val := get_revenue_r2()) is not None else None}

    seller_obj = {
        "id": seller_data["seller_id"], "segment": seller_data["segment"],
        "plan_tier": seller_data.get("plan_tier", "Standard"),
        "monthly_revenue": period_stats["total_revenue"], "total_revenue": period_stats["total_revenue"],
        "product_count": seller_data.get("product_count", 0), "order_count": period_stats["total_orders"],
        "shops_count": seller_data.get("product_count", 0), "region": seller_data.get("region", "서울"),
        "is_anomaly": seller_data.get("is_anomaly", False), "top_shops": top_shops,
        "stats": stats, "activity": activity, "model_predictions": model_predictions,
        "period_stats": {"days": days, "total_revenue": period_stats["total_revenue"], "total_orders": period_stats["total_orders"], "active_days": period_stats["active_days"], "avg_daily_revenue": round(period_stats["total_revenue"] / max(1, period_stats["active_days"]), 1), "total_cs": period_stats["total_cs"]},
    }
    return json_sanitize({"status": "success", "days": days, "user": seller_obj, "seller": seller_obj})


@router.get("/sellers/analyze/{seller_id}")
def analyze_seller(seller_id: str, user: dict = Depends(verify_credentials)):
    return tool_analyze_seller(seller_id)


@router.post("/sellers/segment")
def get_seller_segment(seller_features: dict, user: dict = Depends(verify_credentials)):
    return tool_get_seller_segment(seller_features)


@router.post("/sellers/fraud")
def detect_seller_fraud(seller_features: dict, user: dict = Depends(verify_credentials)):
    return tool_detect_fraud(seller_features=seller_features)


@router.get("/sellers/segments/statistics")
def get_segment_stats(user: dict = Depends(verify_credentials)):
    return tool_get_segment_statistics()


@router.get("/users/segments/{segment_name}/details")
def get_segment_details(segment_name: str, user: dict = Depends(verify_credentials)):
    try:
        if st.SELLER_ANALYTICS_DF is None:
            return error_response("셀러 분석 데이터 없음")
        df = st.SELLER_ANALYTICS_DF
        if "segment_name" in df.columns:
            seg = df[df["segment_name"] == segment_name]
        else:
            return error_response(f"알 수 없는 세그먼트: {segment_name}")
        total = len(df)
        count = len(seg)
        return json_sanitize({
            "status": "success", "segment": segment_name, "count": count,
            "percentage": round(count / max(total, 1) * 100, 1),
            "avg_monthly_revenue": int(seg["total_revenue"].mean()) if "total_revenue" in seg.columns else 0,
            "avg_product_count": int(seg["product_count"].mean()) if "product_count" in seg.columns else 0,
            "avg_order_count": int(seg["total_orders"].mean()) if "total_orders" in seg.columns else 0,
            "top_activities": [], "retention_rate": None,
        })
    except Exception as e:
        return error_response(safe_str(e))


@router.get("/sellers/{seller_id}/activity")
def get_seller_activity(seller_id: str, days: int = 30, user: dict = Depends(verify_credentials)):
    return tool_get_seller_activity_report(seller_id, days)


@router.get("/sellers/performance")
def get_sellers_performance(user: dict = Depends(verify_credentials)):
    try:
        if st.SELLERS_DF is None or st.SELLERS_DF.empty:
            return error_response("셀러 데이터 없음")
        cols = ["seller_id", "plan_tier", "segment"]
        available_cols = [c for c in cols if c in st.SELLERS_DF.columns]
        top100 = st.SELLERS_DF.head(100)
        sellers = [
            {"id": r.get("seller_id", ""), "name": r.get("seller_id", ""), "plan_tier": r.get("plan_tier", "Standard"), "segment": r.get("segment", "알 수 없음")}
            for r in top100[available_cols].to_dict("records")
        ]
        return {"status": "success", "sellers": sellers}
    except Exception as e:
        st.logger.error(f"셀러 목록 조회 오류: {e}")
        return error_response(str(e))
