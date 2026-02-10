"""
process_miner/routes.py
프로세스 마이너 전용 API 라우터
- /api/process-miner/discover   : 프로세스 패턴 발견
- /api/process-miner/bottlenecks: 병목 분석
- /api/process-miner/recommend  : AI 자동화 추천
- /api/process-miner/predict    : 다음 활동 예측
- /api/process-miner/anomalies  : 이상 프로세스 탐지
- /api/process-miner/dashboard  : 전체 통계 대시보드
"""

import random

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import state as st
from core.utils import json_sanitize

from process_miner.event_generator import generate_event_logs
from process_miner.miner import discover_process
from process_miner.bottleneck import analyze_bottlenecks
from process_miner.recommender import recommend_automation
from process_miner.predictor import predict_next_activity
from process_miner.anomaly_detector import detect_anomalies

# ---------------------------------------------------------------------------
# 라우터
# ---------------------------------------------------------------------------
pm_router = APIRouter(prefix="/api/process-miner", tags=["process-miner"])

# ---------------------------------------------------------------------------
# Pydantic 요청 모델
# ---------------------------------------------------------------------------
VALID_PROCESS_TYPES = {"order", "cs", "settlement", "all"}


class PMDiscoverRequest(BaseModel):
    process_type: str = Field(default="order", description="order / cs / settlement / all")
    n_cases: int = Field(default=200, ge=10, le=5000)


class PMBottleneckRequest(BaseModel):
    process_type: str = Field(default="order", description="order / cs / settlement / all")
    n_cases: int = Field(default=200, ge=10, le=5000)


class PMRecommendRequest(BaseModel):
    process_type: str = Field(default="order", description="order / cs / settlement")
    n_cases: int = Field(default=200, ge=10, le=5000)


class PMPredictRequest(BaseModel):
    process_type: str = Field(default="order", description="order / cs / settlement")
    n_cases: int = Field(default=200, ge=10, le=5000)
    case_id: str = Field(default="", description="예측 대상 케이스 ID (빈 문자열이면 랜덤 선택)")


class PMAnomalyRequest(BaseModel):
    process_type: str = Field(default="order", description="order / cs / settlement / all")
    n_cases: int = Field(default=200, ge=10, le=5000)


# ---------------------------------------------------------------------------
# POST /discover
# ---------------------------------------------------------------------------
@pm_router.post("/discover")
async def pm_discover(body: PMDiscoverRequest):
    """프로세스 패턴 발견"""
    try:
        ptype = body.process_type if body.process_type in VALID_PROCESS_TYPES else "order"
        st.logger.info("PM_DISCOVER start type=%s n=%d", ptype, body.n_cases)

        events = generate_event_logs(process_type=ptype, n_cases=body.n_cases)
        result = discover_process(events)

        st.logger.info(
            "PM_DISCOVER done patterns=%d",
            len(result.get("top_patterns", [])),
        )
        return JSONResponse(content={
            "status": "success",
            "process_type": ptype,
            "n_cases": body.n_cases,
            "data": json_sanitize(result),
        })
    except Exception as e:
        st.logger.exception("PM_DISCOVER error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /bottlenecks
# ---------------------------------------------------------------------------
@pm_router.post("/bottlenecks")
async def pm_bottlenecks(body: PMBottleneckRequest):
    """병목 분석"""
    try:
        ptype = body.process_type if body.process_type in VALID_PROCESS_TYPES else "order"
        st.logger.info("PM_BOTTLENECK start type=%s n=%d", ptype, body.n_cases)

        events = generate_event_logs(process_type=ptype, n_cases=body.n_cases)
        result = analyze_bottlenecks(events)

        st.logger.info(
            "PM_BOTTLENECK done bottlenecks=%d",
            len(result.get("bottlenecks", [])),
        )
        return JSONResponse(content={
            "status": "success",
            "process_type": ptype,
            "n_cases": body.n_cases,
            "data": json_sanitize(result),
        })
    except Exception as e:
        st.logger.exception("PM_BOTTLENECK error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------
@pm_router.post("/recommend")
async def pm_recommend(body: PMRecommendRequest):
    """AI 자동화 추천 (discover + bottleneck → LLM 분석)"""
    try:
        ptype = body.process_type
        if ptype not in {"order", "cs", "settlement"}:
            ptype = "order"
        st.logger.info("PM_RECOMMEND start type=%s n=%d", ptype, body.n_cases)

        events = generate_event_logs(process_type=ptype, n_cases=body.n_cases)

        discovery_result = discover_process(events)
        bottleneck_result = analyze_bottlenecks(events)

        recommendation = await recommend_automation(
            discovery_result=discovery_result,
            bottleneck_result=bottleneck_result,
            process_type=ptype,
        )

        st.logger.info(
            "PM_RECOMMEND done recs=%d saving=%s%%",
            len(recommendation.get("recommendations", [])),
            recommendation.get("estimated_time_saving_percent", "?"),
        )
        return JSONResponse(content={
            "status": "success",
            "process_type": ptype,
            "n_cases": body.n_cases,
            "data": json_sanitize(recommendation),
        })
    except Exception as e:
        st.logger.exception("PM_RECOMMEND error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@pm_router.post("/predict")
async def pm_predict(body: PMPredictRequest):
    """다음 활동 예측"""
    try:
        ptype = body.process_type
        if ptype not in {"order", "cs", "settlement"}:
            ptype = "order"
        st.logger.info("PM_PREDICT start type=%s n=%d", ptype, body.n_cases)

        events = generate_event_logs(process_type=ptype, n_cases=body.n_cases)

        case_id = body.case_id
        if not case_id:
            case_id = random.choice(list({e["case_id"] for e in events}))

        result = predict_next_activity(
            events, case_id=case_id, process_type=ptype, n_cases=body.n_cases,
        )

        st.logger.info("PM_PREDICT done case_id=%s", case_id)
        return JSONResponse(content={
            "status": "success",
            "process_type": ptype,
            "n_cases": body.n_cases,
            "data": json_sanitize(result),
        })
    except Exception as e:
        st.logger.exception("PM_PREDICT error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# POST /anomalies
# ---------------------------------------------------------------------------
@pm_router.post("/anomalies")
async def pm_anomalies(body: PMAnomalyRequest):
    """이상 프로세스 탐지"""
    try:
        ptype = body.process_type if body.process_type in VALID_PROCESS_TYPES else "order"
        st.logger.info("PM_ANOMALIES start type=%s n=%d", ptype, body.n_cases)

        events = generate_event_logs(process_type=ptype, n_cases=body.n_cases)
        result = detect_anomalies(events)

        st.logger.info(
            "PM_ANOMALIES done anomalies=%d",
            len(result.get("anomalies", [])),
        )
        return JSONResponse(content={
            "status": "success",
            "process_type": ptype,
            "n_cases": body.n_cases,
            "data": json_sanitize(result),
        })
    except Exception as e:
        st.logger.exception("PM_ANOMALIES error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------
@pm_router.get("/dashboard")
async def pm_dashboard():
    """3가지 프로세스(order/cs/settlement) 요약 통계 대시보드"""
    try:
        st.logger.info("PM_DASHBOARD start")
        dashboard: dict = {}

        for ptype in ("order", "cs", "settlement"):
            try:
                events = generate_event_logs(process_type=ptype, n_cases=100)
                disc = discover_process(events)
                bn = analyze_bottlenecks(events)

                patterns = disc.get("top_patterns", [])
                bottlenecks = bn.get("bottlenecks", [])

                dashboard[ptype] = {
                    "total_cases": disc.get("total_cases", 0),
                    "unique_patterns": len(patterns),
                    "top_pattern": (
                        " → ".join(patterns[0].get("sequence", []))
                        if patterns
                        else None
                    ),
                    "top_pattern_pct": (
                        round(float(patterns[0].get("ratio", 0)) * 100, 1)
                        if patterns
                        else 0
                    ),
                    "bottleneck_count": len(bottlenecks),
                    "worst_bottleneck": (
                        f"{bottlenecks[0]['from_step']} → {bottlenecks[0]['to_step']}"
                        if bottlenecks
                        else None
                    ),
                    "worst_avg_minutes": (
                        float(bottlenecks[0].get("avg_duration_minutes", 0))
                        if bottlenecks
                        else 0
                    ),
                }
            except Exception as e:
                st.logger.warning("PM_DASHBOARD %s failed: %s", ptype, e)
                dashboard[ptype] = {"error": str(e)}

        st.logger.info("PM_DASHBOARD done")
        return JSONResponse(content={
            "status": "success",
            "data": json_sanitize(dashboard),
        })
    except Exception as e:
        st.logger.exception("PM_DASHBOARD error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )
