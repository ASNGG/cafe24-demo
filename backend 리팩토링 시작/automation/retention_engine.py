"""
automation/retention_engine.py - 셀러 이탈 방지 자동 조치 엔진
=============================================================
ML 이탈 예측 → LLM 맞춤 메시지 생성 → 자동 조치 실행
카페24 PRO Marketing 패턴: 데이터 분석 → AI 판단 → 자동 실행
"""
import time
import uuid
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from core.constants import FEATURE_COLS_CHURN, FEATURE_LABELS, PLAN_TIERS
from core.utils import safe_str, safe_int, safe_float
from agent.llm import get_llm, invoke_with_retry, pick_api_key
from automation.action_logger import log_action, save_retention_action
import state as st


def get_at_risk_sellers(threshold: float = 0.6, limit: int = 20) -> List[Dict]:
    """
    이탈 위험 셀러 목록을 반환합니다.
    SELLER_CHURN_MODEL로 예측하고 SHAP으로 요인을 분석합니다.
    threshold 이상인 셀러만 필터링, 확률 높은 순 정렬.
    """
    if st.SELLER_ANALYTICS_DF is None:
        st.logger.warning("RETENTION get_at_risk_sellers: SELLER_ANALYTICS_DF is None")
        return []

    df = st.SELLER_ANALYTICS_DF.copy()
    if df.empty:
        return []

    results = []

    # ML 모델이 있는 경우
    if st.SELLER_CHURN_MODEL is not None:
        try:
            feature_cols = FEATURE_COLS_CHURN
            X = pd.DataFrame([
                {col: safe_float(row.get(col, 0)) for col in feature_cols}
                for _, row in df.iterrows()
            ])

            # plan_tier 인코딩 처리
            if "plan_tier_encoded" in feature_cols and "plan_tier" in df.columns:
                tier_map = {tier: i for i, tier in enumerate(PLAN_TIERS)}
                X["plan_tier_encoded"] = df["plan_tier"].map(
                    lambda t: tier_map.get(t, 0)
                )

            # 이탈 확률 예측
            proba = st.SELLER_CHURN_MODEL.predict_proba(X)[:, 1]

            # SHAP 분석 (전체 배치)
            shap_values_all = None
            if st.SHAP_EXPLAINER_CHURN is not None:
                try:
                    explainer = st.SHAP_EXPLAINER_CHURN
                    if hasattr(explainer, "shap_values"):
                        shap_result = explainer.shap_values(X)
                        if isinstance(shap_result, list) and len(shap_result) == 2:
                            shap_values_all = np.array(shap_result[1])
                        elif isinstance(shap_result, np.ndarray):
                            if shap_result.ndim == 3:
                                shap_values_all = shap_result[:, :, 1]
                            else:
                                shap_values_all = shap_result
                        else:
                            shap_values_all = np.array(shap_result)
                    else:
                        shap_result = explainer(X)
                        if hasattr(shap_result, "values"):
                            shap_values_all = shap_result.values
                        else:
                            shap_values_all = np.array(shap_result)
                except Exception as e:
                    st.logger.error("RETENTION SHAP batch error: %s", str(e))

            for idx, (_, row) in enumerate(df.iterrows()):
                prob = float(proba[idx])
                if prob < threshold:
                    continue

                # SHAP top factors
                top_factors = []
                if shap_values_all is not None:
                    shap_vals = np.abs(shap_values_all[idx])
                    feat_imp = list(zip(feature_cols, shap_vals))
                    feat_imp.sort(key=lambda x: x[1], reverse=True)
                    for feat, imp in feat_imp[:5]:
                        top_factors.append({
                            "factor": FEATURE_LABELS.get(feat, feat),
                            "importance": round(float(imp) * 100, 1),
                        })

                if not top_factors:
                    top_factors = _default_factors(row)

                results.append({
                    "seller_id": safe_str(row.get("seller_id", "")),
                    "churn_probability": round(prob * 100, 1),
                    "risk_level": "high" if prob > 0.7 else "medium",
                    "top_factors": top_factors,
                    "seller_info": {
                        "total_orders": safe_int(row.get("total_orders", 0)),
                        "total_revenue": safe_int(row.get("total_revenue", 0)),
                        "days_since_last_login": safe_int(row.get("days_since_last_login", 0)),
                        "refund_rate": safe_float(row.get("refund_rate", 0)),
                        "product_count": safe_int(row.get("product_count", 0)),
                    },
                })

        except Exception as e:
            st.logger.error("RETENTION ML prediction error: %s", str(e))
            results = _heuristic_at_risk(df, threshold)
    else:
        # 모델이 없으면 휴리스틱 사용
        results = _heuristic_at_risk(df, threshold)

    # 확률 높은 순 정렬 + limit
    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return results[:limit]


def _heuristic_at_risk(df: pd.DataFrame, threshold: float) -> List[Dict]:
    """ML 모델이 없을 때 휴리스틱으로 이탈 위험 셀러를 산출합니다."""
    results = []
    for _, row in df.iterrows():
        total_orders = safe_int(row.get("total_orders", 0))
        total_revenue = safe_int(row.get("total_revenue", 0))
        days_since_last = safe_int(row.get("days_since_last_login", 0))
        refund_rate = safe_float(row.get("refund_rate", 0))
        cs_tickets = safe_int(row.get("cs_tickets", 0))

        score = 0.3
        if days_since_last > 14:
            score += 0.25
        elif days_since_last > 7:
            score += 0.15
        if total_orders < 10:
            score += 0.1
        if total_revenue < 100000:
            score += 0.1
        if refund_rate > 10:
            score += 0.1
        if cs_tickets > 20:
            score += 0.05
        score = min(max(score, 0.05), 0.95)

        if score < threshold:
            continue

        results.append({
            "seller_id": safe_str(row.get("seller_id", "")),
            "churn_probability": round(score * 100, 1),
            "risk_level": "high" if score > 0.7 else "medium",
            "top_factors": _default_factors(row),
            "seller_info": {
                "total_orders": total_orders,
                "total_revenue": total_revenue,
                "days_since_last_login": days_since_last,
                "refund_rate": refund_rate,
                "product_count": safe_int(row.get("product_count", 0)),
            },
        })
    return results


def _default_factors(row) -> List[Dict]:
    """기본 이탈 요인 목록을 생성합니다."""
    return [
        {"factor": f"마지막 접속 {safe_int(row.get('days_since_last_login', 0))}일 전", "importance": 30},
        {"factor": f"총 주문 수 {safe_int(row.get('total_orders', 0))}건", "importance": 25},
        {"factor": f"총 매출 {safe_int(row.get('total_revenue', 0)):,}원", "importance": 20},
        {"factor": f"환불률 {safe_float(row.get('refund_rate', 0))}%", "importance": 15},
        {"factor": f"CS 문의 {safe_int(row.get('cs_tickets', 0))}건", "importance": 10},
    ]


def generate_retention_message(seller_id: str, api_key: str = "") -> Dict:
    """
    특정 셀러에 대한 맞춤 리텐션 메시지를 LLM으로 생성합니다.
    할인 쿠폰, 프리미엄 업그레이드, 전담 매니저 배정 등 추천 포함.
    """
    if st.SELLER_ANALYTICS_DF is None:
        return {"seller_id": seller_id, "message": "", "recommended_actions": [],
                "urgency": "unknown", "error": "셀러 분석 데이터가 로드되지 않았습니다."}

    seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if seller.empty:
        return {"seller_id": seller_id, "message": "", "recommended_actions": [],
                "urgency": "unknown", "error": f"셀러 '{seller_id}'를 찾을 수 없습니다."}

    row = seller.iloc[0]

    # 이탈 요인 분석
    churn_info = _analyze_single_seller(row)

    # LLM 프롬프트 구성
    system_prompt = (
        "당신은 카페24 플랫폼의 셀러 리텐션 전문가입니다. "
        "이탈 위험이 있는 셀러에게 보낼 맞춤 메시지와 추천 조치를 생성하세요.\n"
        "반드시 JSON 형식으로만 응답하세요:\n"
        '{"message": "셀러에게 보낼 메시지", '
        '"recommended_actions": ["조치1", "조치2", "조치3"], '
        '"urgency": "high 또는 medium"}'
    )

    user_prompt = (
        f"셀러 정보:\n"
        f"- 셀러 ID: {seller_id}\n"
        f"- 이탈 확률: {churn_info['churn_probability']}%\n"
        f"- 위험 등급: {churn_info['risk_level']}\n"
        f"- 총 주문: {safe_int(row.get('total_orders', 0))}건\n"
        f"- 총 매출: {safe_int(row.get('total_revenue', 0)):,}원\n"
        f"- 마지막 접속: {safe_int(row.get('days_since_last_login', 0))}일 전\n"
        f"- 환불률: {safe_float(row.get('refund_rate', 0))}%\n"
        f"- 등록 상품: {safe_int(row.get('product_count', 0))}개\n"
        f"- CS 문의: {safe_int(row.get('cs_tickets', 0))}건\n"
        f"- 주요 이탈 요인: {', '.join(f['factor'] for f in churn_info['top_factors'][:3])}\n\n"
        f"위 정보를 바탕으로 이 셀러의 이탈을 방지하기 위한 맞춤 메시지와 "
        f"구체적인 추천 조치(할인 쿠폰, 프리미엄 업그레이드, 전담 매니저 배정 등)를 생성하세요."
    )

    resolved_key = pick_api_key(api_key)
    if not resolved_key:
        return {"seller_id": seller_id, "message": "", "recommended_actions": [],
                "urgency": churn_info["risk_level"], "error": "API 키가 설정되지 않았습니다."}

    try:
        llm = get_llm(
            model="gpt-4o-mini",
            api_key=resolved_key,
            max_tokens=1000,
            streaming=False,
            temperature=0.7,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response_text = invoke_with_retry(llm, messages)

        # JSON 파싱 시도
        import json
        try:
            parsed = json.loads(response_text)
            return {
                "seller_id": seller_id,
                "message": safe_str(parsed.get("message", response_text)),
                "recommended_actions": parsed.get("recommended_actions", []),
                "urgency": safe_str(parsed.get("urgency", churn_info["risk_level"])),
            }
        except json.JSONDecodeError:
            return {
                "seller_id": seller_id,
                "message": response_text,
                "recommended_actions": [],
                "urgency": churn_info["risk_level"],
            }

    except Exception as e:
        st.logger.error("RETENTION generate_message error seller=%s: %s", seller_id, str(e))
        return {"seller_id": seller_id, "message": "", "recommended_actions": [],
                "urgency": churn_info["risk_level"], "error": str(e)}


def _analyze_single_seller(row) -> Dict:
    """단일 셀러의 이탈 분석 결과를 반환합니다."""
    if st.SELLER_CHURN_MODEL is not None:
        try:
            feature_cols = FEATURE_COLS_CHURN
            X = pd.DataFrame([{col: safe_float(row.get(col, 0)) for col in feature_cols}])

            if "plan_tier_encoded" in feature_cols and "plan_tier" in row.index:
                tier_map = {tier: i for i, tier in enumerate(PLAN_TIERS)}
                X["plan_tier_encoded"] = tier_map.get(row.get("plan_tier", "Basic"), 0)

            prob = float(st.SELLER_CHURN_MODEL.predict_proba(X)[0][1])
            risk_level = "high" if prob > 0.7 else "medium" if prob > 0.3 else "low"

            top_factors = []
            if st.SHAP_EXPLAINER_CHURN is not None:
                try:
                    explainer = st.SHAP_EXPLAINER_CHURN
                    if hasattr(explainer, "shap_values"):
                        shap_result = explainer.shap_values(X)
                        if isinstance(shap_result, list) and len(shap_result) == 2:
                            shap_vals = np.array(shap_result[1])[0]
                        elif isinstance(shap_result, np.ndarray):
                            shap_vals = shap_result[0, :, 1] if shap_result.ndim == 3 else shap_result[0]
                        else:
                            shap_vals = np.array(shap_result)[0]
                    else:
                        shap_result = explainer(X)
                        shap_vals = shap_result.values[0] if hasattr(shap_result, "values") else np.array(shap_result)[0]

                    feat_imp = sorted(zip(feature_cols, np.abs(shap_vals)), key=lambda x: x[1], reverse=True)
                    for feat, imp in feat_imp[:5]:
                        top_factors.append({
                            "factor": FEATURE_LABELS.get(feat, feat),
                            "importance": round(float(imp) * 100, 1),
                        })
                except Exception as e:
                    st.logger.error("RETENTION SHAP single error: %s", str(e))

            if not top_factors:
                top_factors = _default_factors(row)

            return {
                "churn_probability": round(prob * 100, 1),
                "risk_level": risk_level,
                "top_factors": top_factors,
            }
        except Exception as e:
            st.logger.error("RETENTION _analyze_single_seller ML error: %s", str(e))

    # 휴리스틱 폴백
    days_since_last = safe_int(row.get("days_since_last_login", 0))
    total_orders = safe_int(row.get("total_orders", 0))
    total_revenue = safe_int(row.get("total_revenue", 0))
    refund_rate = safe_float(row.get("refund_rate", 0))
    cs_tickets = safe_int(row.get("cs_tickets", 0))

    score = 0.3
    if days_since_last > 14:
        score += 0.25
    elif days_since_last > 7:
        score += 0.15
    if total_orders < 10:
        score += 0.1
    if total_revenue < 100000:
        score += 0.1
    if refund_rate > 10:
        score += 0.1
    if cs_tickets > 20:
        score += 0.05
    score = min(max(score, 0.05), 0.95)

    return {
        "churn_probability": round(score * 100, 1),
        "risk_level": "high" if score > 0.7 else "medium" if score > 0.3 else "low",
        "top_factors": _default_factors(row),
    }


def execute_retention_action(seller_id: str, action_type: str, api_key: str = "") -> Dict:
    """
    리텐션 조치를 실행합니다 (시뮬레이션).
    action_type: "coupon" | "upgrade_offer" | "manager_assign" | "custom_message"
    """
    valid_actions = {"coupon", "upgrade_offer", "manager_assign", "custom_message"}
    if action_type not in valid_actions:
        return {
            "status": "error",
            "message": f"지원하지 않는 조치 유형입니다: {action_type}. "
                       f"사용 가능: {', '.join(sorted(valid_actions))}",
        }

    action_id = str(uuid.uuid4())[:8]
    timestamp = time.time()

    # 조치별 상세 내용 시뮬레이션
    action_details = {
        "coupon": {
            "description": "할인 쿠폰 발급",
            "detail": f"셀러 {seller_id}에게 수수료 30% 할인 쿠폰(30일간 유효) 발급 완료",
            "coupon_code": f"RET-{action_id.upper()}",
            "discount_rate": 30,
            "validity_days": 30,
        },
        "upgrade_offer": {
            "description": "프리미엄 플랜 업그레이드 제안",
            "detail": f"셀러 {seller_id}에게 프리미엄 플랜 3개월 무료 체험 제안 발송 완료",
            "offer_plan": "Premium",
            "free_months": 3,
        },
        "manager_assign": {
            "description": "전담 매니저 배정",
            "detail": f"셀러 {seller_id}에게 전담 운영 매니저 배정 완료 (24시간 내 첫 연락 예정)",
            "manager_id": f"MGR-{str(uuid.uuid4())[:4].upper()}",
            "contact_deadline_hours": 24,
        },
        "custom_message": {
            "description": "맞춤 리텐션 메시지 발송",
            "detail": f"셀러 {seller_id}에게 AI 생성 맞춤 리텐션 메시지 발송 완료",
        },
    }

    detail = action_details[action_type]

    # 커스텀 메시지인 경우 LLM으로 메시지 생성
    if action_type == "custom_message":
        msg_result = generate_retention_message(seller_id, api_key=api_key)
        if msg_result.get("message"):
            detail["message_content"] = msg_result["message"]
            detail["recommended_actions"] = msg_result.get("recommended_actions", [])

    # 액션 로깅
    log_entry = log_action(
        action_type=f"retention_{action_type}",
        target_id=seller_id,
        detail=detail,
        status="success",
    )

    # 리텐션 히스토리 저장
    retention_record = {
        "action_id": action_id,
        "seller_id": seller_id,
        "action_type": action_type,
        "description": detail["description"],
        "detail": detail["detail"],
        "timestamp": timestamp,
        "log_id": log_entry.get("id", ""),
    }
    save_retention_action(retention_record)

    st.logger.info(
        "RETENTION_ACTION executed action_id=%s type=%s seller=%s",
        action_id, action_type, seller_id,
    )

    return {
        "status": "success",
        "action_id": action_id,
        "action_type": action_type,
        "seller_id": seller_id,
        "detail": detail["detail"],
    }
