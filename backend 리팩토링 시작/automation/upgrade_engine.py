"""
automation/upgrade_engine.py - 셀러 플랜 업그레이드 자동 추천 엔진
================================================================
규칙 기반 업그레이드 후보 탐지 → LLM 맞춤 메시지 생성 → 자동 조치 실행
카페24 PRO Marketing 패턴: 데이터 분석 → AI 판단 → 자동 실행
"""
import json
import time
import uuid
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from core.constants import PLAN_TIERS
from core.utils import safe_str, safe_int, safe_float
from agent.llm import get_llm, invoke_with_retry, pick_api_key
from automation.action_logger import (
    log_action,
    create_pipeline_run,
    update_pipeline_step,
    complete_pipeline_run,
)
import state as st


# ── 플랜별 업그레이드 임계값 ──
# {현재 플랜: (다음 플랜, 매출 기준, 주문수 기준)}
_UPGRADE_THRESHOLDS = {
    "Basic": ("Standard", 5_000_000, 100),
    "Standard": ("Premium", 20_000_000, 500),
    "Premium": ("Enterprise", 50_000_000, 2000),
}


def _compute_upgrade_score(
    row,
    revenue_threshold: int,
    order_threshold: int,
) -> float:
    """
    업그레이드 점수 계산 (0~100).
    매출과 주문수를 각각 임계값 대비 비율로 환산하여 가중 합산.
    - 매출 가중치 60%, 주문수 가중치 40%
    """
    total_revenue = safe_int(row.get("total_revenue", 0))
    total_orders = safe_int(row.get("total_orders", 0))

    # 임계값 대비 비율 (최대 200%까지 허용 후 100점으로 클리핑)
    revenue_ratio = min(total_revenue / max(revenue_threshold, 1), 2.0)
    order_ratio = min(total_orders / max(order_threshold, 1), 2.0)

    # 가중 합산 → 0~100 범위
    raw_score = (revenue_ratio * 0.6 + order_ratio * 0.4) * 50
    return round(min(max(raw_score, 0), 100), 1)


def _build_reasons(row, revenue_threshold: int, order_threshold: int) -> List[Dict]:
    """업그레이드 추천 사유 목록을 생성합니다."""
    reasons = []
    total_revenue = safe_int(row.get("total_revenue", 0))
    total_orders = safe_int(row.get("total_orders", 0))

    if total_revenue >= revenue_threshold:
        reasons.append({
            "factor": "매출 기준 달성",
            "value": f"{total_revenue:,}원 (기준: {revenue_threshold:,}원)",
        })
    if total_orders >= order_threshold:
        reasons.append({
            "factor": "주문수 기준 달성",
            "value": f"{total_orders:,}건 (기준: {order_threshold:,}건)",
        })

    # 추가 성장 지표
    product_count = safe_int(row.get("product_count", 0))
    if product_count > 50:
        reasons.append({
            "factor": "상품 등록 수 우수",
            "value": f"{product_count}개",
        })

    refund_rate = safe_float(row.get("refund_rate", 0))
    if refund_rate < 3:
        reasons.append({
            "factor": "낮은 환불률",
            "value": f"{refund_rate}%",
        })

    days_since_last = safe_int(row.get("days_since_last_login", 0))
    if days_since_last <= 3:
        reasons.append({
            "factor": "활발한 플랫폼 활동",
            "value": f"최근 접속 {days_since_last}일 전",
        })

    return reasons


def get_upgrade_candidates(limit: int = 20) -> List[Dict]:
    """
    규칙 기반으로 플랜 업그레이드 후보 셀러를 탐지합니다.
    - Basic → Standard: 매출 >= 5,000,000 or 주문수 >= 100
    - Standard → Premium: 매출 >= 20,000,000 or 주문수 >= 500
    - Premium → Enterprise: 매출 >= 50,000,000 or 주문수 >= 2000
    - Enterprise는 최고 티어이므로 제외
    """
    run_id = create_pipeline_run("upgrade", ["detect", "analyze"])
    update_pipeline_step(run_id, "detect", "processing")

    if st.SELLER_ANALYTICS_DF is None:
        st.logger.warning("UPGRADE get_upgrade_candidates: SELLER_ANALYTICS_DF is None")
        update_pipeline_step(run_id, "detect", "complete", {"count": 0})
        update_pipeline_step(run_id, "analyze", "complete")
        complete_pipeline_run(run_id)
        return []

    df = st.SELLER_ANALYTICS_DF.copy()
    if df.empty:
        update_pipeline_step(run_id, "detect", "complete", {"count": 0})
        update_pipeline_step(run_id, "analyze", "complete")
        complete_pipeline_run(run_id)
        return []

    results = []

    # 벡터화를 위해 숫자 컬럼 미리 변환
    revenue_col = pd.to_numeric(df.get("total_revenue", 0), errors="coerce").fillna(0)
    orders_col = pd.to_numeric(df.get("total_orders", 0), errors="coerce").fillna(0)

    for current_plan, (next_plan, rev_thresh, ord_thresh) in _UPGRADE_THRESHOLDS.items():
        # 현재 플랜이 current_plan인 셀러 필터링
        plan_mask = df.get("plan_tier", pd.Series(dtype=str)) == current_plan
        if not plan_mask.any():
            continue

        # 매출 또는 주문수 기준 충족
        qualify_mask = plan_mask & (
            (revenue_col >= rev_thresh) | (orders_col >= ord_thresh)
        )
        qualified_indices = np.where(qualify_mask)[0]

        for idx in qualified_indices:
            row = df.iloc[idx]
            score = _compute_upgrade_score(row, rev_thresh, ord_thresh)
            reasons = _build_reasons(row, rev_thresh, ord_thresh)

            results.append({
                "seller_id": safe_str(row.get("seller_id", "")),
                "current_plan": current_plan,
                "recommended_plan": next_plan,
                "upgrade_score": score,
                "reasons": reasons,
                "seller_info": {
                    "total_orders": safe_int(row.get("total_orders", 0)),
                    "total_revenue": safe_int(row.get("total_revenue", 0)),
                    "days_since_last_login": safe_int(row.get("days_since_last_login", 0)),
                    "refund_rate": safe_float(row.get("refund_rate", 0)),
                    "cs_tickets": safe_int(row.get("cs_tickets", 0)),
                    "product_count": safe_int(row.get("product_count", 0)),
                },
            })

    # 점수 높은 순 정렬 + limit
    results.sort(key=lambda x: x["upgrade_score"], reverse=True)

    update_pipeline_step(run_id, "detect", "complete", {"count": len(results)})
    update_pipeline_step(run_id, "analyze", "complete")
    complete_pipeline_run(run_id)

    st.logger.info("UPGRADE get_upgrade_candidates: %d명 탐지", len(results[:limit]))
    return results[:limit]


def generate_upgrade_message(seller_id: str, api_key: str = "") -> Dict:
    """
    특정 셀러에 대한 맞춤 플랜 업그레이드 추천 메시지를 LLM으로 생성합니다.
    매출 성장에 맞는 상위 플랜의 혜택을 안내합니다.
    """
    run_id = create_pipeline_run("upgrade", ["detect", "analyze", "generate"])
    update_pipeline_step(run_id, "detect", "processing")

    if st.SELLER_ANALYTICS_DF is None:
        update_pipeline_step(run_id, "detect", "error")
        complete_pipeline_run(run_id)
        return {
            "seller_id": seller_id,
            "message": "",
            "benefits": [],
            "urgency": "unknown",
            "error": "셀러 분석 데이터가 로드되지 않았습니다.",
        }

    seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if seller.empty:
        update_pipeline_step(run_id, "detect", "error")
        complete_pipeline_run(run_id)
        return {
            "seller_id": seller_id,
            "message": "",
            "benefits": [],
            "urgency": "unknown",
            "error": f"셀러 '{seller_id}'를 찾을 수 없습니다.",
        }

    row = seller.iloc[0]
    current_plan = safe_str(row.get("plan_tier", "Basic"))

    # 업그레이드 대상 분석
    threshold_info = _UPGRADE_THRESHOLDS.get(current_plan)
    if threshold_info is None:
        # Enterprise(최고 티어) 등 업그레이드 불가
        update_pipeline_step(run_id, "detect", "complete")
        update_pipeline_step(run_id, "analyze", "complete")
        update_pipeline_step(run_id, "generate", "complete")
        complete_pipeline_run(run_id)
        return {
            "seller_id": seller_id,
            "message": "",
            "benefits": [],
            "urgency": "low",
            "error": f"현재 플랜({current_plan})은 이미 최고 티어이거나 업그레이드 대상이 아닙니다.",
        }

    next_plan, rev_thresh, ord_thresh = threshold_info
    score = _compute_upgrade_score(row, rev_thresh, ord_thresh)

    update_pipeline_step(run_id, "detect", "complete")
    update_pipeline_step(run_id, "analyze", "processing")

    # LLM 프롬프트 구성
    system_prompt = (
        "당신은 카페24 플랫폼의 셀러 성장 컨설턴트입니다. "
        "셀러의 매출 성장 데이터를 분석하여 최적의 플랜 업그레이드를 추천하세요.\n"
        "반드시 JSON 형식으로만 응답하세요:\n"
        '{"message": "셀러에게 보낼 업그레이드 추천 메시지", '
        '"benefits": ["상위 플랜 혜택1", "상위 플랜 혜택2", "상위 플랜 혜택3"], '
        '"urgency": "high 또는 medium 또는 low"}'
    )

    user_prompt = (
        f"셀러 정보:\n"
        f"- 셀러 ID: {seller_id}\n"
        f"- 현재 플랜: {current_plan}\n"
        f"- 추천 플랜: {next_plan}\n"
        f"- 업그레이드 점수: {score}/100\n"
        f"- 총 주문: {safe_int(row.get('total_orders', 0))}건\n"
        f"- 총 매출: {safe_int(row.get('total_revenue', 0)):,}원\n"
        f"- 마지막 접속: {safe_int(row.get('days_since_last_login', 0))}일 전\n"
        f"- 환불률: {safe_float(row.get('refund_rate', 0))}%\n"
        f"- 등록 상품: {safe_int(row.get('product_count', 0))}개\n"
        f"- CS 문의: {safe_int(row.get('cs_tickets', 0))}건\n\n"
        f"업그레이드 기준:\n"
        f"- 매출 기준: {rev_thresh:,}원\n"
        f"- 주문수 기준: {ord_thresh:,}건\n\n"
        f"위 정보를 바탕으로 이 셀러에게 {next_plan} 플랜으로의 업그레이드를 추천하는 "
        f"맞춤 메시지와 구체적인 혜택(API 호출 한도 증가, 프리미엄 디자인 템플릿, "
        f"전담 매니저, 수수료 우대 등)을 생성하세요."
    )

    update_pipeline_step(run_id, "analyze", "complete")
    update_pipeline_step(run_id, "generate", "processing")

    resolved_key = pick_api_key(api_key)
    if not resolved_key:
        update_pipeline_step(run_id, "generate", "error")
        complete_pipeline_run(run_id)
        return {
            "seller_id": seller_id,
            "message": "",
            "benefits": [],
            "urgency": "medium" if score >= 50 else "low",
            "error": "API 키가 설정되지 않았습니다.",
        }

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
        try:
            parsed = json.loads(response_text)
            result = {
                "seller_id": seller_id,
                "current_plan": current_plan,
                "recommended_plan": next_plan,
                "upgrade_score": score,
                "message": safe_str(parsed.get("message", response_text)),
                "benefits": parsed.get("benefits", []),
                "urgency": safe_str(parsed.get("urgency", "medium")),
            }
        except json.JSONDecodeError:
            result = {
                "seller_id": seller_id,
                "current_plan": current_plan,
                "recommended_plan": next_plan,
                "upgrade_score": score,
                "message": response_text,
                "benefits": [],
                "urgency": "medium" if score >= 50 else "low",
            }

        update_pipeline_step(run_id, "generate", "complete")
        complete_pipeline_run(run_id)
        return result

    except Exception as e:
        st.logger.error("UPGRADE generate_message error seller=%s: %s", seller_id, str(e))
        update_pipeline_step(run_id, "generate", "error", {"error": str(e)})
        complete_pipeline_run(run_id)
        return {
            "seller_id": seller_id,
            "message": "",
            "benefits": [],
            "urgency": "medium" if score >= 50 else "low",
            "error": str(e),
        }


def execute_upgrade_action(
    seller_id: str,
    action_type: str,
    api_key: str = "",
) -> Dict:
    """
    업그레이드 조치를 실행합니다 (시뮬레이션).
    action_type: "upgrade_recommend" | "benefit_info" | "consultation_request" | "custom_message"
    """
    run_id = create_pipeline_run("upgrade_action", ["execute", "log"])
    update_pipeline_step(run_id, "execute", "processing")

    valid_actions = {"upgrade_recommend", "benefit_info", "consultation_request", "custom_message"}
    if action_type not in valid_actions:
        update_pipeline_step(run_id, "execute", "error")
        complete_pipeline_run(run_id)
        return {
            "status": "error",
            "message": f"지원하지 않는 조치 유형입니다: {action_type}. "
                       f"사용 가능: {', '.join(sorted(valid_actions))}",
        }

    action_id = str(uuid.uuid4())[:8]
    timestamp = time.time()

    # 셀러 정보 조회 (플랜 정보 포함)
    current_plan = "Unknown"
    recommended_plan = "Unknown"
    if st.SELLER_ANALYTICS_DF is not None:
        seller = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
        if not seller.empty:
            current_plan = safe_str(seller.iloc[0].get("plan_tier", "Unknown"))
            threshold_info = _UPGRADE_THRESHOLDS.get(current_plan)
            if threshold_info:
                recommended_plan = threshold_info[0]

    # 조치별 상세 내용 시뮬레이션
    action_details = {
        "upgrade_recommend": {
            "description": "플랜 업그레이드 추천 발송",
            "detail": f"셀러 {seller_id}에게 {current_plan} → {recommended_plan} 플랜 업그레이드 추천 메시지 발송 완료",
            "current_plan": current_plan,
            "recommended_plan": recommended_plan,
        },
        "benefit_info": {
            "description": "상위 플랜 혜택 안내",
            "detail": f"셀러 {seller_id}에게 {recommended_plan} 플랜 혜택 상세 안내 발송 완료",
            "recommended_plan": recommended_plan,
            "benefits_sent": True,
        },
        "consultation_request": {
            "description": "업그레이드 상담 예약",
            "detail": f"셀러 {seller_id}의 플랜 업그레이드 전문 상담 예약 완료 (48시간 내 연락 예정)",
            "consultant_id": f"CON-{str(uuid.uuid4())[:4].upper()}",
            "contact_deadline_hours": 48,
        },
        "custom_message": {
            "description": "AI 맞춤 업그레이드 메시지 발송",
            "detail": f"셀러 {seller_id}에게 AI 생성 맞춤 업그레이드 추천 메시지 발송 완료",
        },
    }

    detail = action_details[action_type]

    # 커스텀 메시지인 경우 LLM으로 메시지 생성
    if action_type == "custom_message":
        msg_result = generate_upgrade_message(seller_id, api_key=api_key)
        if msg_result.get("message"):
            detail["message_content"] = msg_result["message"]
            detail["benefits"] = msg_result.get("benefits", [])

    # 액션 로깅
    log_entry = log_action(
        action_type=f"upgrade_{action_type}",
        target_id=seller_id,
        detail=detail,
        status="success",
    )

    st.logger.info(
        "UPGRADE_ACTION executed action_id=%s type=%s seller=%s",
        action_id, action_type, seller_id,
    )

    update_pipeline_step(run_id, "execute", "complete", {"action_type": action_type})
    update_pipeline_step(run_id, "log", "complete")
    complete_pipeline_run(run_id)

    return {
        "status": "success",
        "action_id": action_id,
        "action_type": action_type,
        "seller_id": seller_id,
        "detail": detail["detail"],
        "pipeline_run_id": run_id,
    }
