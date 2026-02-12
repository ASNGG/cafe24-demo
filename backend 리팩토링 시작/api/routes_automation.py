"""
api/routes_automation.py - 자동화 엔진 API 라우터
================================================
탐지 → 자동 실행 3대 기능:
  1. 셀러 이탈 방지 자동 조치
  2. CS FAQ 자동 생성
  3. 운영 리포트 자동 생성
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.common import verify_credentials
from core.utils import safe_str
import state as st

from automation import action_logger
from automation import retention_engine
from automation import faq_engine
from automation import report_engine

router = APIRouter(prefix="/api/automation", tags=["automation"])


# ============================================================
# Pydantic 요청 모델
# ============================================================
class RetentionMessageRequest(BaseModel):
    seller_id: str
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True


class RetentionExecuteRequest(BaseModel):
    seller_id: str
    action_type: str = Field("custom_message", description="coupon | upgrade_offer | manager_assign | custom_message")
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True


class FaqGenerateRequest(BaseModel):
    category: Optional[str] = None
    count: int = Field(5, ge=1, le=20)
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True


class FaqUpdateRequest(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None


class ReportGenerateRequest(BaseModel):
    report_type: str = Field("daily", description="daily | weekly | monthly")
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True


# ============================================================
# 1. 셀러 이탈 방지 자동 조치
# ============================================================
@router.get("/retention/at-risk")
def get_at_risk_sellers(
    threshold: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    user=Depends(verify_credentials),
):
    """이탈 위험 셀러 목록 조회"""
    try:
        sellers = retention_engine.get_at_risk_sellers(threshold=threshold, limit=limit)
        return {"status": "success", "sellers": sellers, "total": len(sellers)}
    except Exception as e:
        st.logger.error("RETENTION_AT_RISK_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.post("/retention/message")
def generate_retention_message(
    req: RetentionMessageRequest,
    user=Depends(verify_credentials),
):
    """리텐션 메시지 생성"""
    try:
        result = retention_engine.generate_retention_message(
            seller_id=req.seller_id,
            api_key=req.api_key,
        )
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("RETENTION_MESSAGE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.post("/retention/execute")
def execute_retention_action(
    req: RetentionExecuteRequest,
    user=Depends(verify_credentials),
):
    """이탈 방지 조치 실행"""
    try:
        result = retention_engine.execute_retention_action(
            seller_id=req.seller_id,
            action_type=req.action_type,
            api_key=req.api_key,
        )
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("RETENTION_EXECUTE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.get("/retention/history")
def get_retention_history(
    limit: int = Query(50, ge=1, le=200),
    user=Depends(verify_credentials),
):
    """리텐션 조치 이력 조회"""
    try:
        history = action_logger.get_retention_history(limit=limit)
        return {"status": "success", "total": len(history), "history": history}
    except Exception as e:
        st.logger.error("RETENTION_HISTORY_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


# ============================================================
# 2. CS FAQ 자동 생성
# ============================================================
@router.post("/faq/analyze")
def analyze_cs_patterns(
    user=Depends(verify_credentials),
):
    """CS 문의 패턴 분석"""
    try:
        result = faq_engine.analyze_cs_patterns()
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_ANALYZE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.post("/faq/generate")
def generate_faq(
    req: FaqGenerateRequest,
    user=Depends(verify_credentials),
):
    """FAQ 자동 생성"""
    try:
        result = faq_engine.generate_faq_items(
            category=req.category,
            count=req.count,
            api_key=req.api_key,
        )
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_GENERATE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.get("/faq/list")
def list_faqs(
    status: Optional[str] = Query(None, description="draft | approved | all"),
    user=Depends(verify_credentials),
):
    """FAQ 목록 조회"""
    try:
        result = faq_engine.list_faqs(status=status)
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_LIST_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.put("/faq/{faq_id}/approve")
def approve_faq(
    faq_id: str,
    user=Depends(verify_credentials),
):
    """FAQ 승인"""
    try:
        result = faq_engine.approve_faq(faq_id=faq_id)
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_APPROVE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.put("/faq/{faq_id}")
def update_faq(
    faq_id: str,
    req: FaqUpdateRequest,
    user=Depends(verify_credentials),
):
    """FAQ 수정"""
    try:
        result = faq_engine.update_faq(
            faq_id=faq_id,
            question=req.question,
            answer=req.answer,
        )
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_UPDATE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.delete("/faq/{faq_id}")
def delete_faq(
    faq_id: str,
    user=Depends(verify_credentials),
):
    """FAQ 삭제"""
    try:
        result = faq_engine.delete_faq_item(faq_id=faq_id)
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("FAQ_DELETE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


# ============================================================
# 3. 운영 리포트 자동 생성
# ============================================================
@router.post("/report/generate")
def generate_report(
    req: ReportGenerateRequest,
    user=Depends(verify_credentials),
):
    """운영 리포트 자동 생성"""
    try:
        result = report_engine.generate_report(
            report_type=req.report_type,
            api_key=req.api_key,
        )
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("REPORT_GENERATE_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.get("/report/history")
def get_report_history(
    limit: int = Query(20, ge=1, le=100),
    user=Depends(verify_credentials),
):
    """리포트 생성 이력 조회"""
    try:
        result = report_engine.get_history(limit=limit)
        return {"status": "success", **result}
    except Exception as e:
        st.logger.error("REPORT_HISTORY_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


# ============================================================
# 4. 공통 - 액션 로그
# ============================================================
@router.get("/actions/log")
def get_actions_log(
    action_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user=Depends(verify_credentials),
):
    """자동화 액션 로그 조회"""
    try:
        logs = action_logger.get_action_log(action_type=action_type, limit=limit)
        return {"status": "success", "total": len(logs), "logs": logs}
    except Exception as e:
        st.logger.error("ACTION_LOG_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))


@router.get("/actions/stats")
def get_actions_stats(
    user=Depends(verify_credentials),
):
    """자동화 액션 통계"""
    try:
        stats = action_logger.get_action_stats()
        return {"status": "success", **stats}
    except Exception as e:
        st.logger.error("ACTION_STATS_ERROR: %s", safe_str(e))
        raise HTTPException(status_code=500, detail=safe_str(e))
