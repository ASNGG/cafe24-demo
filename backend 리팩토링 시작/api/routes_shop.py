"""
api/routes_shop.py - 쇼핑몰/카테고리/대시보드/분석/통계
"""
import time as _time
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from core.utils import safe_str, json_sanitize, get_revenue_r2
from core.constants import CS_TICKET_CATEGORIES, CS_PRIORITY_GRADES
from agent.tools import (
    tool_get_shop_info, tool_list_shops, tool_get_shop_services,
    tool_list_categories, tool_get_category_info,
    tool_get_order_statistics, tool_classify_inquiry,
    tool_get_dashboard_summary, tool_get_cs_statistics,
)
import state as st
from api.common import verify_credentials, TextClassifyRequest


router = APIRouter(prefix="/api", tags=["shop"])

# ── set_index 캐싱 ──
_perf_indexed = None       # shop_id 인덱싱된 SHOP_PERFORMANCE_DF 캐시
_perf_indexed_id = None    # 원본 DF id()로 변경 감지

def _get_perf_indexed():
    """SHOP_PERFORMANCE_DF의 set_index('shop_id') 결과를 캐싱"""
    global _perf_indexed, _perf_indexed_id
    if st.SHOP_PERFORMANCE_DF is None:
        return None
    cur_id = id(st.SHOP_PERFORMANCE_DF)
    if _perf_indexed is not None and _perf_indexed_id == cur_id:
        return _perf_indexed
    _perf_indexed = st.SHOP_PERFORMANCE_DF.set_index("shop_id")
    _perf_indexed_id = cur_id
    return _perf_indexed

# ── 인사이트 캐싱 ──
_insights_cache = None
_insights_cache_ts = 0.0
_INSIGHTS_CACHE_TTL = 60  # 60초


# ============================================================
# 쇼핑몰 API
# ============================================================
@router.get("/shops")
def get_shops(
    plan_tier: Optional[str] = None,
    category: Optional[str] = None,
    user: dict = Depends(verify_credentials)
):
    """쇼핑몰 목록 조회 (성과 데이터 포함)"""
    result = tool_list_shops(plan_tier=plan_tier, category=category)
    perf = _get_perf_indexed()
    if result.get("status") == "success" and perf is not None:
        for shop in result.get("shops", []):
            sid = shop.get("shop_id", "")
            if sid in perf.index:
                row = perf.loc[sid]
                shop["usage"] = int(min(100, max(0, float(row.get("customer_retention_rate", 0)) * 100)))
                shop["cvr"] = float(row.get("conversion_rate", 0))
                shop["popularity"] = int(min(100, max(0, float(row.get("review_score", 0)) * 20)))
    return result


@router.get("/shops/{shop_id}")
def get_shop(shop_id: str, user: dict = Depends(verify_credentials)):
    return tool_get_shop_info(shop_id)


@router.get("/shops/{shop_id}/services")
def get_shop_services(shop_id: str, user: dict = Depends(verify_credentials)):
    return tool_get_shop_services(shop_id)


# ============================================================
# 카테고리 API
# ============================================================
@router.get("/categories")
def get_categories(user: dict = Depends(verify_credentials)):
    return tool_list_categories()


@router.get("/categories/{category_id}")
def get_category(category_id: str, user: dict = Depends(verify_credentials)):
    return tool_get_category_info(category_id)


# ============================================================
# 주문/운영 통계 API
# ============================================================
@router.get("/orders/statistics")
def get_order_stats(
    event_type: Optional[str] = None,
    days: int = 30,
    user: dict = Depends(verify_credentials)
):
    return tool_get_order_statistics(event_type=event_type, days=days)


# ============================================================
# 텍스트 분류 API
# ============================================================
@router.post("/classify/inquiry")
def classify_inquiry(req: TextClassifyRequest, user: dict = Depends(verify_credentials)):
    return tool_classify_inquiry(req.text)


# ============================================================
# 대시보드 API
# ============================================================
@router.get("/dashboard/summary")
def get_dashboard_summary(user: dict = Depends(verify_credentials)):
    return tool_get_dashboard_summary()


@router.get("/dashboard/insights")
def get_dashboard_insights(user: dict = Depends(verify_credentials)):
    """AI 인사이트 - 실제 데이터 기반 동적 생성 (60초 캐싱)"""
    global _insights_cache, _insights_cache_ts

    # 캐시 히트
    now = _time.time()
    if _insights_cache is not None and (now - _insights_cache_ts) < _INSIGHTS_CACHE_TTL:
        return _insights_cache

    insights = []

    try:
        activity_col = "active_shops" if st.DAILY_METRICS_DF is not None and "active_shops" in st.DAILY_METRICS_DF.columns else "dau"
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 14 and activity_col in st.DAILY_METRICS_DF.columns:
            df = st.DAILY_METRICS_DF
            recent_7 = df.tail(7)[activity_col].mean()
            prev_7 = df.tail(14).head(7)[activity_col].mean()
            dau_change = round((recent_7 - prev_7) / max(prev_7, 1) * 100, 1)
            if dau_change > 5:
                insights.append({"type": "positive", "icon": "arrow_up", "title": "활성 쇼핑몰 상승 추세", "description": f"최근 7일간 활성 쇼핑몰이 {dau_change}% 증가했습니다. 긍정적인 성장세입니다."})
            elif dau_change < -5:
                insights.append({"type": "warning", "icon": "arrow_down", "title": "활성 쇼핑몰 하락 주의", "description": f"최근 7일간 활성 쇼핑몰이 {abs(dau_change)}% 감소했습니다. 원인 분석이 필요합니다."})
            else:
                insights.append({"type": "neutral", "icon": "stable", "title": "활성 쇼핑몰 안정적", "description": f"최근 7일간 활성 쇼핑몰 변화가 {dau_change:+.1f}%로 안정적입니다."})

        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            cohort_df = st.COHORT_RETENTION_DF
            for _, row in cohort_df.iloc[::-1].iterrows():
                week2 = row.get("week2")
                if week2 is not None and not pd.isna(week2):
                    week2_val = float(week2)
                    if week2_val < 50:
                        insights.append({"type": "warning", "icon": "retention", "title": "리텐션 개선 필요", "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표(50%) 대비 낮습니다. 온보딩 개선을 권장합니다."})
                    elif week2_val >= 65:
                        insights.append({"type": "positive", "icon": "retention", "title": "리텐션 우수", "description": f"Week 2 리텐션이 {week2_val:.0f}%로 매우 우수합니다."})
                    else:
                        insights.append({"type": "neutral", "icon": "retention", "title": "리텐션 양호", "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표 수준입니다."})
                    break

        quality_col = "satisfaction_score" if st.CS_STATS_DF is not None and "satisfaction_score" in st.CS_STATS_DF.columns else "avg_quality"
        category_col = "category" if st.CS_STATS_DF is not None and "category" in st.CS_STATS_DF.columns else "lang_name"
        if st.CS_STATS_DF is not None and len(st.CS_STATS_DF) > 0 and quality_col in st.CS_STATS_DF.columns:
            avg_quality = st.CS_STATS_DF[quality_col].mean()
            best_row = st.CS_STATS_DF.loc[st.CS_STATS_DF[quality_col].idxmax()]
            best_name = best_row.get(category_col, "일반")
            if avg_quality >= 90:
                insights.append({"type": "positive", "icon": "translation", "title": "CS 만족도 우수", "description": f"{best_name} 카테고리 만족도가 {best_row[quality_col]:.1f}점으로 목표치를 초과 달성했습니다."})
            elif avg_quality < 80:
                insights.append({"type": "warning", "icon": "translation", "title": "CS 만족도 개선 필요", "description": f"평균 CS 만족도가 {avg_quality:.1f}점입니다. 개선이 필요합니다."})
            else:
                insights.append({"type": "neutral", "icon": "translation", "title": "CS 만족도 양호", "description": f"평균 CS 만족도가 {avg_quality:.1f}점으로 양호합니다. {best_name} 카테고리가 {best_row[quality_col]:.1f}점으로 가장 높습니다."})

        if st.SELLER_ANALYTICS_DF is not None and "is_anomaly" in st.SELLER_ANALYTICS_DF.columns:
            anomaly_count = int(st.SELLER_ANALYTICS_DF["is_anomaly"].sum())
            total_users = len(st.SELLER_ANALYTICS_DF)
            anomaly_rate = round(anomaly_count / max(total_users, 1) * 100, 1)
            if anomaly_rate > 5:
                insights.append({"type": "warning", "icon": "anomaly", "title": "이상 유저 주의", "description": f"이상 행동 유저가 {anomaly_count}명({anomaly_rate}%)입니다. 모니터링 강화가 필요합니다."})

        if not insights:
            insights.append({"type": "neutral", "icon": "info", "title": "데이터 분석 중", "description": "충분한 데이터가 수집되면 AI 인사이트가 제공됩니다."})

        result = json_sanitize({"status": "success", "insights": insights[:3]})
        _insights_cache = result
        _insights_cache_ts = _time.time()
        return result

    except Exception as e:
        st.logger.exception("인사이트 생성 실패")
        return {"status": "error", "message": safe_str(e), "insights": []}


@router.get("/dashboard/alerts")
def get_dashboard_alerts(limit: int = 5, user: dict = Depends(verify_credentials)):
    """실시간 알림 - ANOMALY_DETAILS_DF 기반 이상 행동 알림"""
    try:
        alerts = []
        anomaly_df = st.FRAUD_DETAILS_DF

        if anomaly_df is not None and len(anomaly_df) > 0:
            df = anomaly_df.copy()
            date_col = "detected_date" if "detected_date" in df.columns else "detected_at"
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.sort_values(date_col, ascending=False)

            for _, row in df.head(limit).iterrows():
                user_id = str(row.get("seller_id", row.get("user_id", "Unknown")))
                anomaly_type = str(row.get("anomaly_type", "이상 행동"))
                if "severity" in row.index:
                    severity = str(row.get("severity", "medium")).lower()
                else:
                    score = float(row.get("anomaly_score", 0))
                    severity = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
                detail = str(row.get("details", row.get("detail", "")))
                detected_val = row.get(date_col)

                time_ago = "방금 전"
                if pd.notna(detected_val):
                    now = datetime.now()
                    diff = now - detected_val
                    if diff.days > 0:
                        time_ago = f"{diff.days}일 전"
                    elif diff.seconds >= 3600:
                        time_ago = f"{diff.seconds // 3600}시간 전"
                    elif diff.seconds >= 60:
                        time_ago = f"{diff.seconds // 60}분 전"

                color_type = "red" if severity == "high" else "orange" if severity == "medium" else "yellow"
                alerts.append({"user_id": user_id, "type": anomaly_type, "severity": severity, "color": color_type, "detail": detail if detail else anomaly_type, "time_ago": time_ago})

        return json_sanitize({"status": "success", "alerts": alerts, "total_count": len(anomaly_df) if anomaly_df is not None else 0})

    except Exception as e:
        st.logger.exception("알림 조회 실패")
        return {"status": "error", "message": safe_str(e), "alerts": []}


# ============================================================
# 공통 헬퍼
# ============================================================
def _success_response(**kwargs) -> dict:
    """표준 성공 응답 래퍼"""
    return {"status": "success", **kwargs}


def _extract_shap_values(shap_raw) -> np.ndarray:
    """SHAP 결과에서 ndarray를 추출 (다양한 반환 형태 대응)"""
    if hasattr(shap_raw, 'values'):
        return shap_raw.values
    if isinstance(shap_raw, list) and len(shap_raw) == 2:
        return shap_raw[1]
    if isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3:
            return shap_raw[:, :, 1]
        return shap_raw
    return shap_raw


def _compute_correlation() -> list:
    """DAILY_METRICS_DF에서 지표 간 상관관계를 계산"""
    correlation = []
    if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 7:
        corr_cols = ["active_shops", "total_gmv", "total_orders", "new_signups"]
        corr_labels = {"active_shops": "활성 쇼핑몰", "total_gmv": "GMV", "total_orders": "주문수", "new_signups": "신규가입"}
        avail = [c for c in corr_cols if c in st.DAILY_METRICS_DF.columns]
        if len(avail) >= 2:
            corr_matrix = st.DAILY_METRICS_DF[avail].corr()
            for i in range(len(avail)):
                for j in range(i + 1, len(avail)):
                    correlation.append({"var1": corr_labels.get(avail[i], avail[i]), "var2": corr_labels.get(avail[j], avail[j]), "correlation": round(float(corr_matrix.iloc[i, j]), 3)})
    return correlation


# ============================================================
# 분석 헬퍼
# ============================================================
def _classify_severity(filtered_df, anomaly_count: int):
    """이상 탐지 데이터의 severity별 건수 분류"""
    if "severity" in filtered_df.columns:
        severity_counts = filtered_df["severity"].value_counts().to_dict()
        return int(severity_counts.get("high", 0)), int(severity_counts.get("medium", 0)), int(severity_counts.get("low", 0))
    if "anomaly_score" in filtered_df.columns:
        high = int((filtered_df["anomaly_score"] > 0.8).sum())
        medium = int(((filtered_df["anomaly_score"] > 0.5) & (filtered_df["anomaly_score"] <= 0.8)).sum())
        return high, medium, max(0, anomaly_count - high - medium)
    return 0, 0, anomaly_count


def _build_anomaly_trend(filtered_df, date_col: str, days: int, reference_date):
    """이상 탐지 트렌드 데이터 생성"""
    trend = []
    if date_col not in filtered_df.columns or len(filtered_df) == 0:
        return trend
    if days == 7:
        filtered_df = filtered_df.copy()
        filtered_df["date_str"] = filtered_df[date_col].dt.strftime("%m/%d")
        daily_counts = filtered_df.groupby("date_str").size().to_dict()
        for i in range(7):
            d = reference_date - timedelta(days=6 - i)
            date_str = d.strftime("%m/%d")
            trend.append({"date": date_str, "count": int(daily_counts.get(date_str, 0))})
    else:
        bucket_size = 5 if days == 30 else 15
        num_buckets = 6
        for i in range(num_buckets):
            start_day = days - (i + 1) * bucket_size
            end_day = days - i * bucket_size
            start_date = reference_date - timedelta(days=end_day)
            end_date = reference_date - timedelta(days=start_day)
            period_df = filtered_df[(filtered_df[date_col] >= start_date) & (filtered_df[date_col] < end_date)]
            label_offset = 2 if days == 30 else 7
            label = (reference_date - timedelta(days=end_day - label_offset)).strftime("%m/%d")
            trend.append({"date": label, "count": len(period_df)})
    return trend


def _build_recent_alerts(filtered_df, date_col: str, reference_date, count: int):
    """최근 이상 알림 목록 생성"""
    recent_df = filtered_df.nlargest(count, date_col) if date_col in filtered_df.columns else filtered_df.head(count)
    alerts = []
    for _, row in recent_df.iterrows():
        if date_col in row.index and pd.notna(row[date_col]):
            time_diff = reference_date - row[date_col]
            if time_diff.days > 0:
                time_str = f"{time_diff.days}일 전"
            elif time_diff.seconds >= 3600:
                time_str = f"{time_diff.seconds // 3600}시간 전"
            else:
                time_str = f"{max(1, time_diff.seconds // 60)}분 전"
        else:
            time_str = "최근"
        if "severity" in row.index:
            sev = str(row.get("severity", "medium"))
        elif "anomaly_score" in row.index:
            score = float(row.get("anomaly_score", 0))
            sev = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
        else:
            sev = "medium"
        alerts.append({"id": str(row.get("seller_id", row.get("user_id", "M000000"))), "type": str(row.get("anomaly_type", "알 수 없음")), "severity": sev, "detail": str(row.get("details", row.get("detail", "이상 패턴 감지"))), "time": time_str})
    return alerts


# ============================================================
# 분석 API (이상탐지, 이탈예측, 코호트, 트렌드 KPI, 상관관계, 통계)
# ============================================================
@router.get("/analysis/anomaly")
def get_anomaly_analysis(days: int = 7, user: dict = Depends(verify_credentials)):
    """이상탐지 분석 데이터"""
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "유저 분석 데이터가 없습니다."}
    if days not in [7, 30, 90]:
        days = 7
    try:
        df = st.SELLER_ANALYTICS_DF
        total_users = len(df)
        anomaly_df = st.FRAUD_DETAILS_DF
        today = datetime.now()

        if anomaly_df is not None and len(anomaly_df) > 0:
            anomaly_df = anomaly_df.copy()
            date_col = "detected_date" if "detected_date" in anomaly_df.columns else "detected_at"
            if date_col in anomaly_df.columns:
                anomaly_df[date_col] = pd.to_datetime(anomaly_df[date_col], errors="coerce")
                latest_date = anomaly_df[date_col].max()
                reference_date = latest_date if pd.notna(latest_date) else today
                cutoff_date = reference_date - timedelta(days=days)
                filtered_df = anomaly_df[anomaly_df[date_col] >= cutoff_date]
            else:
                filtered_df = anomaly_df
                reference_date = today

            anomaly_count = len(filtered_df)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0
            high_risk, medium_risk, low_risk = _classify_severity(filtered_df, anomaly_count)

            by_type = []
            id_col = "seller_id" if "seller_id" in filtered_df.columns else "user_id"
            if "anomaly_type" in filtered_df.columns:
                agg_dict = {id_col: "count"}
                if "severity" in filtered_df.columns:
                    agg_dict["severity"] = "first"
                elif "anomaly_score" in filtered_df.columns:
                    agg_dict["anomaly_score"] = "mean"
                type_severity = filtered_df.groupby("anomaly_type").agg(agg_dict).reset_index()
                if "severity" in type_severity.columns:
                    type_severity.columns = ["type", "count", "severity"]
                elif "anomaly_score" in type_severity.columns:
                    type_severity.columns = ["type", "count", "avg_score"]
                    type_severity["severity"] = type_severity["avg_score"].apply(lambda x: "high" if x > 0.8 else "medium" if x > 0.5 else "low")
                else:
                    type_severity.columns = ["type", "count"]
                    type_severity["severity"] = "medium"
                for _, row in type_severity.iterrows():
                    by_type.append({"type": row["type"], "count": int(row["count"]), "severity": row["severity"]})
                by_type.sort(key=lambda x: x["count"], reverse=True)

            trend = _build_anomaly_trend(filtered_df, date_col, days, reference_date)
            alert_count = {7: 4, 30: 6, 90: 8}.get(days, 4)
            recent_alerts = _build_recent_alerts(filtered_df, date_col, reference_date, alert_count)
        else:
            anomaly_users = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else df.iloc[:0]
            anomaly_count = len(anomaly_users)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0
            high_risk, medium_risk, low_risk = 0, 0, anomaly_count
            by_type, trend, recent_alerts = [], [], []

        return json_sanitize({
            "status": "success",
            "data_source": "ANOMALY_DETAILS_DF" if (st.FRAUD_DETAILS_DF is not None and len(st.FRAUD_DETAILS_DF) > 0) else "USER_ANALYTICS_DF",
            "summary": {"total_users": total_users, "anomaly_count": anomaly_count, "anomaly_rate": anomaly_rate, "high_risk": high_risk, "medium_risk": medium_risk, "low_risk": low_risk},
            "by_type": by_type, "trend": trend, "recent_alerts": recent_alerts,
        })
    except Exception as e:
        st.logger.error(f"이상탐지 분석 오류: {e}")
        return {"status": "error", "message": safe_str(e)}


@router.get("/analysis/prediction/churn")
def get_churn_prediction(days: int = 7, user: dict = Depends(verify_credentials)):
    """이탈 예측 분석 (실제 ML 모델 + SHAP 기반)"""
    if days not in [7, 30, 90]:
        days = 7
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "유저 분석 데이터가 없습니다."}
    try:
        df = st.SELLER_ANALYTICS_DF.copy()
        total = len(df)
        model_accuracy = None
        top_factors = []
        high_risk_count = medium_risk_count = low_risk_count = 0
        available_features = []
        feature_names_kr = {}

        if st.SELLER_CHURN_MODEL is not None:
            config = st.CHURN_MODEL_CONFIG or {}
            features = config.get("features", ["total_orders", "total_revenue", "product_count", "cs_tickets", "refund_rate", "avg_response_time"])
            feature_names_kr = config.get("feature_names_kr", {"total_orders": "총 주문 수", "total_revenue": "총 매출", "product_count": "등록 상품 수", "cs_tickets": "CS 문의 수", "refund_rate": "환불률", "avg_response_time": "평균 응답 시간"})
            model_accuracy = (config.get("model_accuracy") or 0) * 100
            available_features = [f for f in features if f in df.columns]
            if available_features:
                X = df[available_features].fillna(0)
                churn_proba = st.SELLER_CHURN_MODEL.predict_proba(X)[:, 1]
                df["churn_probability"] = churn_proba
                high_threshold = {7: 0.7, 30: 0.6, 90: 0.5}.get(days, 0.7)
                medium_threshold = {7: 0.4, 30: 0.35, 90: 0.3}.get(days, 0.4)
                high_risk_count = int((churn_proba >= high_threshold).sum())
                medium_risk_count = int(((churn_proba >= medium_threshold) & (churn_proba < high_threshold)).sum())
                low_risk_count = total - high_risk_count - medium_risk_count

                if st.SHAP_EXPLAINER_CHURN is not None:
                    try:
                        shap_values = np.array(_extract_shap_values(st.SHAP_EXPLAINER_CHURN.shap_values(X)))
                        shap_importance = np.abs(shap_values).mean(axis=0)
                        total_imp = shap_importance.sum()
                        if total_imp > 0:
                            shap_importance = shap_importance / total_imp
                        sorted_indices = np.argsort(shap_importance)[::-1]
                        for idx in sorted_indices[:5]:
                            feat = available_features[idx]
                            top_factors.append({"factor": feature_names_kr.get(feat, feat), "importance": round(float(shap_importance[idx]), 3)})
                    except Exception as e:
                        st.logger.warning(f"SHAP 분석 실패: {e}")

                if not top_factors and hasattr(st.SELLER_CHURN_MODEL, "feature_importances_"):
                    importances = st.SELLER_CHURN_MODEL.feature_importances_
                    sorted_indices = importances.argsort()[::-1]
                    for idx in sorted_indices[:5]:
                        feat = available_features[idx]
                        top_factors.append({"factor": feature_names_kr.get(feat, feat), "importance": round(float(importances[idx]), 3)})

        if not top_factors:
            high_risk_count = medium_risk_count = 0
            low_risk_count = total

        high_risk_users = []
        user_sample_count = min(3 + days // 30 * 2, 7)
        if "churn_probability" in df.columns:
            high_risk_df = df.nlargest(user_sample_count, "churn_probability")
            for _, row in high_risk_df.iterrows():
                user_id = row.get("seller_id", row.get("user_id", "M000000"))
                cluster = int(row.get("cluster", 0))
                prob = int(row["churn_probability"] * 100)
                user_factors = []
                if st.SHAP_EXPLAINER_CHURN is not None and available_features:
                    try:
                        user_X = row[available_features].values.reshape(1, -1)
                        user_shap = np.array(_extract_shap_values(st.SHAP_EXPLAINER_CHURN.shap_values(user_X)))
                        if user_shap.ndim > 1:
                            user_shap = user_shap[0]
                        user_shap = user_shap.flatten()
                        sorted_idx = np.abs(user_shap).argsort()[::-1]
                        for idx in sorted_idx[:3]:
                            feat = available_features[idx]
                            shap_val = user_shap[idx]
                            user_factors.append({"factor": feature_names_kr.get(feat, feat), "direction": "위험" if shap_val > 0 else "양호", "impact": round(abs(float(shap_val)), 3)})
                    except Exception:
                        pass
                high_risk_users.append({"id": user_id, "name": user_id, "segment": row.get("segment_name", f"세그먼트 {cluster}"), "probability": prob, "last_active": None, "factors": user_factors if user_factors else None})

        revenue_data = None
        engagement_data = None
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            recent = st.DAILY_METRICS_DF.tail(days)
            gmv_col = "total_gmv" if "total_gmv" in recent.columns else None
            shops_col = "active_shops" if "active_shops" in recent.columns else None
            if gmv_col:
                current_gmv = float(recent[gmv_col].iloc[-1]) if len(recent) > 0 else 0
                prev_gmv = float(recent[gmv_col].iloc[0]) if len(recent) > 1 else current_gmv
                growth = round((current_gmv - prev_gmv) / max(1, prev_gmv) * 100, 1) if prev_gmv > 0 else 0
                active = int(recent[shops_col].mean()) if shops_col else total
                arpu = int(current_gmv / max(1, active))
                whale_count = dolphin_count = minnow_count = 0
                if st.SELLER_ANALYTICS_DF is not None and "total_revenue" in st.SELLER_ANALYTICS_DF.columns:
                    rev_col = st.SELLER_ANALYTICS_DF["total_revenue"].dropna()
                    if len(rev_col) > 0:
                        q90 = rev_col.quantile(0.90)
                        q70 = rev_col.quantile(0.70)
                        whale_count = int((rev_col >= q90).sum())
                        dolphin_count = int(((rev_col >= q70) & (rev_col < q90)).sum())
                        minnow_count = int((rev_col < q70).sum())
                revenue_data = {"predicted_monthly": int(current_gmv * 30 / max(1, days)), "growth_rate": growth, "predicted_arpu": arpu, "predicted_arppu": None, "confidence": None, "whale_count": whale_count, "dolphin_count": dolphin_count, "minnow_count": minnow_count}
            if shops_col:
                active_avg = int(recent[shops_col].mean())
                mau = None
                stickiness = None
                if "active_shops" in st.DAILY_METRICS_DF.columns:
                    recent_30 = st.DAILY_METRICS_DF.tail(30)
                    if len(recent_30) > 0:
                        mau = int(recent_30["active_shops"].max())
                        stickiness = int(active_avg / max(1, mau) * 100) if mau else None
                avg_session = None
                sessions_per_day = None
                dm = st.DAILY_METRICS_DF.tail(days)
                if "avg_session_minutes" in dm.columns:
                    avg_session = round(float(dm["avg_session_minutes"].mean()), 1)
                if "total_sessions" in dm.columns and "active_shops" in dm.columns:
                    avg_shops = dm["active_shops"].mean()
                    sessions_per_day = round(float(dm["total_sessions"].mean() / max(1, avg_shops)), 1)
                engagement_data = {"predicted_dau": active_avg, "predicted_mau": mau, "stickiness": stickiness, "avg_session": avg_session, "sessions_per_day": sessions_per_day}

        return json_sanitize({
            "status": "success",
            "model_available": st.SELLER_CHURN_MODEL is not None,
            "shap_available": st.SHAP_EXPLAINER_CHURN is not None,
            "churn": {"high_risk_count": high_risk_count, "medium_risk_count": medium_risk_count, "low_risk_count": low_risk_count, "predicted_churn_rate": round(high_risk_count / total * 100, 1) if total > 0 else 0, "model_accuracy": round(model_accuracy, 1), "top_factors": top_factors, "high_risk_users": high_risk_users},
            "revenue": revenue_data, "engagement": engagement_data,
        })
    except Exception as e:
        st.logger.error(f"이탈 예측 API 오류: {e}")
        return {"status": "error", "message": safe_str(e)}


@router.get("/analysis/prediction/churn/user/{user_id}")
def get_user_churn_prediction(user_id: str, user: dict = Depends(verify_credentials)):
    """개별 사용자 이탈 예측 + SHAP 분석"""
    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "error", "message": "유저 분석 데이터가 없습니다."}
    try:
        df = st.SELLER_ANALYTICS_DF.copy()
        id_col = "seller_id" if "seller_id" in df.columns else "user_id"
        user_row = df[df[id_col] == user_id]
        if user_row.empty:
            return {"status": "error", "message": f"유저 {user_id}를 찾을 수 없습니다."}
        user_row = user_row.iloc[0]
        config = st.CHURN_MODEL_CONFIG or {}
        features = config.get("features", ["total_orders", "total_revenue", "product_count", "cs_tickets", "refund_rate", "avg_response_time"])
        feature_names_kr = config.get("feature_names_kr", {"total_orders": "총 주문 수", "total_revenue": "총 매출", "product_count": "등록 상품 수", "cs_tickets": "CS 문의 수", "refund_rate": "환불률", "avg_response_time": "평균 응답 시간"})
        available_features = [f for f in features if f in df.columns]
        if st.SELLER_CHURN_MODEL is None:
            return {"status": "error", "message": "이탈 예측 모델이 로드되지 않았습니다."}
        if not available_features:
            return {"status": "error", "message": "필요한 feature가 데이터에 없습니다."}
        user_X = user_row[available_features].values.reshape(1, -1)
        churn_proba = st.SELLER_CHURN_MODEL.predict_proba(user_X)[0, 1]
        if churn_proba >= 0.7:
            risk_level, risk_label = "high", "고위험"
        elif churn_proba >= 0.4:
            risk_level, risk_label = "medium", "중위험"
        else:
            risk_level, risk_label = "low", "저위험"
        shap_factors = []
        if st.SHAP_EXPLAINER_CHURN is not None:
            try:
                user_shap = np.array(_extract_shap_values(st.SHAP_EXPLAINER_CHURN.shap_values(user_X)))
                if user_shap.ndim > 1:
                    user_shap = user_shap[0]
                user_shap = user_shap.flatten()
                for i, feat in enumerate(available_features):
                    shap_val = float(user_shap[i])
                    feature_val = float(user_row[feat])
                    shap_factors.append({"feature": feat, "feature_kr": feature_names_kr.get(feat, feat), "shap_value": round(shap_val, 4), "feature_value": round(feature_val, 2), "direction": "위험" if shap_val > 0 else "양호"})
                shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            except Exception as e:
                st.logger.warning(f"SHAP 분석 실패: {e}")
        cluster = int(user_row.get("cluster", 0))
        return json_sanitize({"status": "success", "user_id": user_id, "user_name": user_id, "segment": user_row.get("segment_name", f"세그먼트 {cluster}"), "churn_probability": round(float(churn_proba) * 100, 1), "risk_level": risk_level, "risk_label": risk_label, "shap_factors": shap_factors, "model_accuracy": round((config.get("model_accuracy") or 0) * 100, 1) if config.get("model_accuracy") else None, "shap_available": st.SHAP_EXPLAINER_CHURN is not None})
    except Exception as e:
        st.logger.error(f"개별 유저 이탈 예측 오류: {e}")
        return {"status": "error", "message": safe_str(e)}


@router.get("/analysis/cohort/retention")
def get_cohort_retention(days: int = 7, user: dict = Depends(verify_credentials)):
    """코호트 리텐션 분석"""
    if days not in [7, 30, 90]:
        days = 7
    weeks = max(1, min(13, days // 7))
    try:
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            raw_data = st.COHORT_RETENTION_DF.tail(weeks).to_dict("records")
            cohort_data = []
            for row in raw_data:
                entry = {"cohort": row.get("cohort_month", row.get("cohort", "unknown")), "week0": 100}
                for col in ["week1", "week2", "week4", "week8", "week12"]:
                    if col in row and row[col] is not None and not (isinstance(row[col], float) and pd.isna(row[col])):
                        entry[col] = round(float(row[col]), 1)
                cohort_data.append(entry)
        else:
            cohort_data = []

        ltv_by_cohort = []
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            if st.SELLER_ANALYTICS_DF is not None and "predicted_ltv" in st.SELLER_ANALYTICS_DF.columns:
                if st.SELLERS_DF is not None and "join_date" in st.SELLERS_DF.columns:
                    merged = st.SELLERS_DF[["seller_id", "join_date"]].merge(st.SELLER_ANALYTICS_DF[["seller_id", "predicted_ltv"]], on="seller_id", how="inner")
                    merged["cohort_month"] = pd.to_datetime(merged["join_date"], errors="coerce").dt.to_period("M").astype(str)
                    cohort_grp = merged.groupby("cohort_month").agg(ltv=("predicted_ltv", "mean"), users=("seller_id", "count")).reset_index().sort_values("cohort_month", ascending=False).head(6)
                    ltv_by_cohort = [{"cohort": row["cohort_month"], "ltv": int(row["ltv"]), "users": int(row["users"])} for _, row in cohort_grp.iterrows()]

        conversion = []
        if st.CONVERSION_FUNNEL_DF is not None and len(st.CONVERSION_FUNNEL_DF) > 0:
            conversion = st.CONVERSION_FUNNEL_DF.to_dict("records")

        return json_sanitize({"status": "success", "retention": cohort_data, "ltv_by_cohort": ltv_by_cohort, "conversion": conversion})
    except Exception as e:
        return {"status": "error", "message": safe_str(e)}


@router.get("/analysis/trend/kpis")
def get_trend_kpis(days: int = 7, user: dict = Depends(verify_credentials)):
    """트렌드 KPI 분석"""
    try:
        if days not in [7, 30, 90]:
            days = 7
        if st.DAILY_METRICS_DF is None or len(st.DAILY_METRICS_DF) == 0:
            return {"status": "error", "message": "일별 지표 데이터가 없습니다."}

        df = st.DAILY_METRICS_DF.copy()
        recent_df = df.tail(min(days, len(df)))
        daily_metrics = []
        for _, r in recent_df.iterrows():
            d_str = str(r.get("date", ""))
            daily_metrics.append({"date": d_str[-5:].replace("-", "/") if len(d_str) >= 5 else d_str, "dau": int(r.get("active_shops", 0)), "new_users": int(r.get("new_signups", 0)), "sessions": int(r.get("total_sessions", r.get("active_shops", 0) * 3)), "active_shops": int(r.get("active_shops", 0)), "total_gmv": int(r.get("total_gmv", 0)), "total_orders": int(r.get("total_orders", 0))})

        n = len(recent_df)
        prev_start = max(0, len(df) - n * 2)
        prev_end = len(df) - n
        prev_df = df.iloc[prev_start:prev_end] if prev_end > prev_start else recent_df

        def _avg(frame, col, default=0):
            return float(frame[col].mean()) if col in frame.columns else default

        def _sum(frame, col, default=0):
            return float(frame[col].sum()) if col in frame.columns else default

        active_shops = int(_avg(recent_df, "active_shops"))
        active_shops_prev = int(_avg(prev_df, "active_shops"))
        total_gmv = int(_avg(recent_df, "total_gmv"))
        total_gmv_prev = int(_avg(prev_df, "total_gmv"))
        new_signups = int(_avg(recent_df, "new_signups"))
        new_signups_prev = int(_avg(prev_df, "new_signups"))
        settlement_time = round(_avg(recent_df, "avg_settlement_time"), 1)
        settlement_time_prev = round(_avg(prev_df, "avg_settlement_time"), 1)
        total_orders = int(_avg(recent_df, "total_orders"))
        total_orders_prev = int(_avg(prev_df, "total_orders"))
        cs_open = _sum(recent_df, "cs_tickets_open")
        cs_resolved = _sum(recent_df, "cs_tickets_resolved")
        cs_rate = round(cs_resolved / max(cs_open, 1) * 100, 1)
        cs_open_prev = _sum(prev_df, "cs_tickets_open")
        cs_resolved_prev = _sum(prev_df, "cs_tickets_resolved")
        cs_rate_prev = round(cs_resolved_prev / max(cs_open_prev, 1) * 100, 1)

        def _change(cur, prev):
            return round((cur - prev) / max(abs(prev), 1) * 100, 1)

        kpis = [
            {"name": "활성 쇼핑몰", "current": active_shops, "previous": active_shops_prev, "trend": "up" if active_shops >= active_shops_prev else "down", "change": _change(active_shops, active_shops_prev)},
            {"name": "일 GMV", "current": total_gmv, "previous": total_gmv_prev, "trend": "up" if total_gmv >= total_gmv_prev else "down", "change": _change(total_gmv, total_gmv_prev)},
            {"name": "신규가입", "current": new_signups, "previous": new_signups_prev, "trend": "up" if new_signups >= new_signups_prev else "down", "change": _change(new_signups, new_signups_prev)},
            {"name": "총 주문수", "current": total_orders, "previous": total_orders_prev, "trend": "up" if total_orders >= total_orders_prev else "down", "change": _change(total_orders, total_orders_prev)},
            {"name": "정산소요시간", "current": settlement_time, "previous": settlement_time_prev, "trend": "down" if settlement_time <= settlement_time_prev else "up", "change": _change(settlement_time, settlement_time_prev)},
            {"name": "CS해결률", "current": cs_rate, "previous": cs_rate_prev, "trend": "up" if cs_rate >= cs_rate_prev else "down", "change": _change(cs_rate, cs_rate_prev)},
        ]

        forecast = []
        if "active_shops" in st.DAILY_METRICS_DF.columns:
            recent = st.DAILY_METRICS_DF.tail(14)
            if len(recent) >= 3:
                vals = recent["active_shops"].values
                nn = len(vals)
                x = np.arange(nn)
                slope = (np.mean(x * vals) - np.mean(x) * np.mean(vals)) / max(1, np.var(x))
                intercept = np.mean(vals) - slope * np.mean(x)
                std_err = np.std(vals - (slope * x + intercept))
                last_date = pd.to_datetime(recent["date"].iloc[-1], errors="coerce")
                for i in range(1, 6):
                    pred_val = int(slope * (nn + i) + intercept)
                    pred_date = (last_date + timedelta(days=i)).strftime("%m/%d") if last_date is not pd.NaT else f"D+{i}"
                    forecast.append({"date": pred_date, "predicted_dau": max(0, pred_val), "lower": max(0, int(pred_val - 1.5 * std_err)), "upper": int(pred_val + 1.5 * std_err)})

        correlation = _compute_correlation()

        return json_sanitize({"status": "success", "kpis": kpis, "daily_metrics": daily_metrics, "correlation": correlation, "forecast": forecast})
    except Exception as e:
        return {"status": "error", "message": safe_str(e)}


@router.get("/analysis/correlation")
def get_correlation_analysis(user: dict = Depends(verify_credentials)):
    """지표 상관관계 분석"""
    return json_sanitize({"status": "success", "correlation": _compute_correlation()})


@router.get("/stats/summary")
def get_summary_stats(days: int = 7, user: dict = Depends(verify_credentials)):
    """통계 요약 (분석 패널용)"""
    if days not in [7, 30, 90]:
        days = 7
    summary = {
        "status": "success",
        "days": days,
        "shops_count": len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0,
        "categories_count": len(st.CATEGORIES_DF) if st.CATEGORIES_DF is not None else 0,
        "sellers_count": len(st.SELLERS_DF) if st.SELLERS_DF is not None else 0,
        "cs_stats_count": len(st.CS_STATS_DF) if st.CS_STATS_DF is not None else 0,
        "operation_logs_count": len(st.OPERATION_LOGS_DF) if st.OPERATION_LOGS_DF is not None else 0,
    }

    if st.SELLER_ACTIVITY_DF is not None and len(st.SELLER_ACTIVITY_DF) > 0:
        try:
            activity_df = st.SELLER_ACTIVITY_DF.tail(100 * days)
            active_users = activity_df["seller_id"].nunique()
            summary["active_users_in_period"] = active_users
            summary["active_user_ratio"] = round(active_users / summary["sellers_count"] * 100, 1) if summary["sellers_count"] > 0 else 0
        except Exception:
            pass

    if st.CS_STATS_DF is not None and "avg_quality" in st.CS_STATS_DF.columns:
        summary["avg_cs_quality"] = round(float(st.CS_STATS_DF["avg_quality"].mean()), 1)

    if st.SHOPS_DF is not None and "plan_tier" in st.SHOPS_DF.columns:
        summary["plan_tier_stats"] = st.SHOPS_DF["plan_tier"].value_counts().to_dict()

    if st.SELLER_ANALYTICS_DF is not None and "segment_name" in st.SELLER_ANALYTICS_DF.columns:
        raw_segments = st.SELLER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        seg_name_map = st.SELLER_ANALYTICS_DF.drop_duplicates("cluster").set_index("cluster")["segment_name"].to_dict()
        summary["user_segments"] = {seg_name_map.get(k, f"세그먼트 {k}"): v for k, v in raw_segments.items()}
    elif st.SELLER_ANALYTICS_DF is not None and "cluster" in st.SELLER_ANALYTICS_DF.columns:
        raw_segments = st.SELLER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        summary["user_segments"] = {f"세그먼트 {k}": v for k, v in raw_segments.items()}

        if st.SELLER_ACTIVITY_DF is not None and len(st.SELLER_ACTIVITY_DF) > 0:
            try:
                activity_df = st.SELLER_ACTIVITY_DF.copy().tail(100 * days)
                rev_col = "revenue" if "revenue" in activity_df.columns else "daily_revenue"
                ord_col = "orders_processed" if "orders_processed" in activity_df.columns else "daily_orders"
                cs_col = "cs_handled" if "cs_handled" in activity_df.columns else "cs_tickets"
                user_activity = activity_df.groupby("seller_id").agg({rev_col: "sum", ord_col: "sum", cs_col: "sum"}).reset_index()
                user_activity.columns = ["seller_id", "daily_revenue", "daily_orders", "cs_tickets"]
                user_activity = user_activity.merge(st.SELLER_ANALYTICS_DF[["seller_id", "cluster"]], on="seller_id", how="left")
                segment_metrics = {}
                seg_name_map = st.SELLER_ANALYTICS_DF.drop_duplicates("cluster").set_index("cluster")["segment_name"].to_dict() if "segment_name" in st.SELLER_ANALYTICS_DF.columns else {}
                analytics_df = st.SELLER_ANALYTICS_DF
                for cluster in raw_segments.keys():
                    name = seg_name_map.get(cluster, f"세그먼트 {cluster}")
                    seg_analytics = analytics_df[analytics_df["cluster"] == cluster]
                    cnt = int(raw_segments.get(cluster, 0))
                    avg_rev = int(seg_analytics["total_revenue"].mean()) if not seg_analytics.empty and "total_revenue" in seg_analytics.columns else 0
                    avg_prod = int(seg_analytics["product_count"].mean()) if not seg_analytics.empty and "product_count" in seg_analytics.columns else 0
                    avg_ord = int(seg_analytics["total_orders"].mean()) if not seg_analytics.empty and "total_orders" in seg_analytics.columns else 0
                    retention = 0
                    if not seg_analytics.empty and "churn_probability" in seg_analytics.columns:
                        retention = int((seg_analytics["churn_probability"] < 0.5).sum() / len(seg_analytics) * 100)
                    segment_metrics[name] = {"count": cnt, "avg_monthly_revenue": avg_rev, "avg_product_count": avg_prod, "avg_order_count": avg_ord, "retention": retention}
                summary["segment_metrics"] = segment_metrics
            except Exception as e:
                st.logger.warning(f"세그먼트 지표 계산 실패: {e}")

    if st.SHOPS_DF is not None and "category" in st.SHOPS_DF.columns:
        summary["category_shops"] = st.SHOPS_DF["category"].value_counts().to_dict()

    if st.CS_STATS_DF is not None and len(st.CS_STATS_DF) > 0:
        stats_list = []
        for _, row in st.CS_STATS_DF.iterrows():
            stats_list.append({"category": str(row.get("category", "기타")), "lang_name": str(row.get("category", "기타")), "total_count": int(row.get("total_tickets", 0)), "avg_quality": round(float(row.get("satisfaction_score", 0)) * 20, 1), "avg_resolution_hours": float(row.get("avg_resolution_hours", 0)), "pending_count": 0})
        summary["cs_stats_detail"] = stats_list

    date_col_logs = "event_date" if st.OPERATION_LOGS_DF is not None and "event_date" in st.OPERATION_LOGS_DF.columns else "timestamp"
    if st.OPERATION_LOGS_DF is not None and date_col_logs in st.OPERATION_LOGS_DF.columns:
        try:
            dfl = st.OPERATION_LOGS_DF.copy()
            dfl["date"] = pd.to_datetime(dfl[date_col_logs], errors="coerce").dt.strftime("%m/%d")
            daily = dfl.groupby("date")["seller_id"].nunique().tail(7)
            new_signups_map = {}
            if st.DAILY_METRICS_DF is not None and "new_signups" in st.DAILY_METRICS_DF.columns and "date" in st.DAILY_METRICS_DF.columns:
                dm = st.DAILY_METRICS_DF.copy()
                dm["_date_key"] = pd.to_datetime(dm["date"], errors="coerce").dt.strftime("%m/%d")
                new_signups_map = dict(zip(dm["_date_key"], dm["new_signups"].fillna(0).astype(int)))
            summary["daily_trend"] = [{"date": date, "active_users": int(count), "new_users": int(new_signups_map.get(date, 0))} for date, count in daily.items()]
        except Exception:
            pass

    return json_sanitize(summary)
