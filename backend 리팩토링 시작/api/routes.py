"""
api/routes.py - CAFE24 AI 운영 플랫폼 FastAPI 라우트 정의
===================================================
카페24 AI 기반 내부 시스템

주요 기능:
1. 카페24 이커머스 데이터 API
2. LLM 기반 CS 서비스
3. 멀티 에이전트 대화
4. RAG 기반 지식 검색
"""
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, HTTPException, Depends, status, Request, UploadFile, File, BackgroundTasks, Body, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_READER = None  # Lazy loading
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None

from core.constants import DEFAULT_SYSTEM_PROMPT, ECOMMERCE_TERMS, CS_TICKET_CATEGORIES, CS_PRIORITY_GRADES
from core.utils import safe_str, safe_int, json_sanitize
from core.memory import clear_memory, append_memory
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
    tool_get_order_statistics,
    tool_get_seller_activity_report,
    tool_classify_inquiry,
    tool_search_platform,
    tool_get_cs_statistics,
    tool_get_dashboard_summary,
    tool_get_churn_prediction,
    tool_predict_seller_churn,
    tool_predict_shop_revenue,
    tool_get_shop_performance,
    tool_optimize_marketing,
    AVAILABLE_TOOLS,
)
from agent.llm import (
    build_langchain_messages, get_llm, chunk_text, pick_api_key,
)
from agent.runner import run_agent
from rag.service import (
    rag_build_or_load_index, tool_rag_search, _rag_list_files,
    rag_search_hybrid, BM25_AVAILABLE, RERANKER_AVAILABLE, KNOWLEDGE_GRAPH
)
from rag.light_rag import (
    lightrag_search_sync, lightrag_search_dual_sync,
    build_lightrag_from_rag_docs, get_lightrag_status, clear_lightrag,
    LIGHTRAG_AVAILABLE
)
from rag.k2rag import (
    k2rag_search_sync, index_documents as k2rag_index,
    get_status as k2rag_get_status, update_config as k2rag_update_config,
    load_from_existing_rag as k2rag_load_existing, summarize_text as k2rag_summarize
)
import state as st

router = APIRouter(prefix="/api")
security = HTTPBasic()


def _get_revenue_r2():
    """매출 예측 모델 R2 스코어 반환 (churn_model_config.json 패턴 활용)"""
    try:
        cfg_path = st.BASE_DIR / "revenue_model_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("r2_score")
    except Exception:
        pass
    return None


# ============================================================
# Pydantic 모델
# ============================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class ShopRequest(BaseModel):
    shop_id: str


class CategoryRequest(BaseModel):
    category_id: str


class UserRequest(BaseModel):
    user_id: str


class CsReplyRequest(BaseModel):
    text: str
    ticket_category: str = Field("일반", description="문의 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")


class CsQualityRequest(BaseModel):
    ticket_category: str = Field("일반", description="CS 티켓 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")
    sentiment_score: float = Field(0.0, description="감성 점수 (-1.0 ~ 1.0)")
    order_value: float = Field(50000, description="주문 금액")
    is_repeat_issue: bool = Field(False, description="반복 문의 여부")
    text_length: int = Field(100, description="문의 텍스트 길이")


class CsPipelineRequest(BaseModel):
    """CS 자동화 파이프라인 요청"""
    inquiry_text: str = Field(..., description="고객 문의 텍스트")
    seller_tier: str = Field("Standard", description="셀러 등급")
    order_id: Optional[str] = Field(None, description="주문 ID")
    order_value: float = Field(50000, description="주문 금액")
    is_repeat_issue: bool = Field(False, description="반복 문의 여부")
    confidence_threshold: float = Field(0.75, description="자동 처리 신뢰도 임계값 (0.0~1.0)")


class CsPipelineAnswerRequest(BaseModel):
    """CS 파이프라인 답변 생성 요청"""
    inquiry_text: str = Field(..., description="고객 문의 텍스트")
    inquiry_category: str = Field("기타", description="문의 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")
    order_id: Optional[str] = Field(None, description="주문 ID")
    rag_mode: str = Field("rag", description="RAG 모드: rag | lightrag | k2rag")
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class TextClassifyRequest(BaseModel):
    text: str


class RagRequest(BaseModel):
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(st.RAG_DEFAULT_TOPK, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class AgentRequest(BaseModel):
    user_input: str = Field(..., alias="user_input")
    api_key: str = Field("", alias="apiKey")
    model: str = Field("gpt-4o-mini", alias="model")
    max_tokens: int = Field(8000, alias="maxTokens")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="systemPrompt")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="topP")
    presence_penalty: Optional[float] = Field(None, alias="presencePenalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequencyPenalty")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: Optional[int] = Field(None, alias="timeoutMs")
    retries: Optional[int] = Field(None, alias="retries")
    stream: Optional[bool] = Field(None, alias="stream")
    rag_mode: str = Field("rag", alias="ragMode")  # 'rag' | 'lightrag' | 'k2rag' | 'auto'
    agent_mode: str = Field("single", alias="agentMode")  # 'single' | 'multi' (LangGraph)
    debug: bool = Field(True, alias="debug")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class UserCreateRequest(BaseModel):
    user_id: str
    name: str
    password: str
    role: str


class RagReloadRequest(BaseModel):
    api_key: str = Field("", alias="apiKey")
    force: bool = Field(True, alias="force")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class DeleteFileRequest(BaseModel):
    filename: str
    api_key: str = Field("", alias="apiKey")
    skip_reindex: bool = Field(False, alias="skipReindex")  # 다중 삭제 시 재빌드 건너뛰기
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class HybridSearchRequest(BaseModel):
    """Hybrid Search 요청 모델"""
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(5, alias="topK")
    use_reranking: bool = Field(True, alias="useReranking")
    use_kg: bool = Field(False, alias="useKg")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagSearchRequest(BaseModel):
    """LightRAG 검색 요청 모델"""
    query: str
    mode: str = Field("hybrid", description="검색 모드: naive, local, global, hybrid")
    top_k: int = Field(5, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagBuildRequest(BaseModel):
    """LightRAG 빌드 요청 모델"""
    force_rebuild: bool = Field(False, alias="forceRebuild")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagSearchRequest(BaseModel):
    """K2RAG 검색 요청 모델"""
    query: str
    top_k: int = Field(10, alias="topK")
    use_kg: bool = Field(True, alias="useKg", description="Knowledge Graph 사용")
    use_summary: bool = Field(True, alias="useSummary", description="요약 사용")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagConfigRequest(BaseModel):
    """K2RAG 설정 요청 모델"""
    hybrid_lambda: Optional[float] = Field(None, alias="hybridLambda", description="Hybrid 가중치 (0.0-1.0)")
    top_k: Optional[int] = Field(None, alias="topK")
    use_summarization: Optional[bool] = Field(None, alias="useSummarization")
    use_knowledge_graph: Optional[bool] = Field(None, alias="useKnowledgeGraph")
    llm_model: Optional[str] = Field(None, alias="llmModel")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class MarketingOptimizeRequest(BaseModel):
    """마케팅 예산 최적화 요청 모델"""
    seller_id: Optional[str] = Field(None, description="셀러 ID (프론트엔드 호환)")
    top_n: int = Field(10, description="상위 N개 결과")
    target_shops: Optional[List[str]] = Field(None, alias="targetShops", description="대상 쇼핑몰 ID 리스트")
    budget_constraints: Optional[dict] = Field(None, alias="budgetConstraints", description="예산 제약 (예: {'total': 1000000})")
    max_iterations: int = Field(10, alias="maxIterations", description="PSO 최대 반복 횟수")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


# ============================================================
# 인증
# ============================================================
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 실패",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": username, "role": st.USERS[username]["role"], "name": st.USERS[username]["name"]}


# ============================================================
# 유틸
# ============================================================
def sse_pack(event: str, data: dict) -> str:
    """SSE 이벤트 포맷으로 직렬화 (LangChain 객체도 안전하게 처리)"""
    safe_data = json_sanitize(data)
    return f"event: {event}\ndata: {json.dumps(safe_data, ensure_ascii=False)}\n\n"


# ============================================================
# 헬스 체크
# ============================================================
@router.get("/health")
def health():
    st.logger.info("HEALTH_CHECK")
    return {
        "status": "SUCCESS",
        "message": "ok",
        "log_file": st.LOG_FILE,
        "pid": os.getpid(),
        "platform": "CAFE24 AI Platform",
        "models_ready": bool(
            st.CS_QUALITY_MODEL is not None and
            st.SELLER_SEGMENT_MODEL is not None and
            st.FRAUD_DETECTION_MODEL is not None
        ),
        "data_ready": {
            "shops": st.SHOPS_DF is not None and len(st.SHOPS_DF) > 0,
            "categories": st.CATEGORIES_DF is not None and len(st.CATEGORIES_DF) > 0,
            "sellers": st.SELLERS_DF is not None and len(st.SELLERS_DF) > 0,
            "operation_logs": st.OPERATION_LOGS_DF is not None and len(st.OPERATION_LOGS_DF) > 0,
        },
    }


# ============================================================
# 로그인
# ============================================================
@router.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(status_code=401, detail="인증 실패")
    user = st.USERS[username]
    clear_memory(username)
    return {"status": "SUCCESS", "username": username, "user_name": user["name"], "user_role": user["role"]}


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
    if result.get("status") == "SUCCESS" and st.SHOP_PERFORMANCE_DF is not None:
        perf = st.SHOP_PERFORMANCE_DF.set_index("shop_id")
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
    """특정 쇼핑몰 정보 조회"""
    return tool_get_shop_info(shop_id)


@router.get("/shops/{shop_id}/services")
def get_shop_services(shop_id: str, user: dict = Depends(verify_credentials)):
    """쇼핑몰 서비스 정보 조회"""
    return tool_get_shop_services(shop_id)


# ============================================================
# 카테고리 API
# ============================================================
@router.get("/categories")
def get_categories(user: dict = Depends(verify_credentials)):
    """카테고리 목록 조회"""
    return tool_list_categories()


@router.get("/categories/{category_id}")
def get_category(category_id: str, user: dict = Depends(verify_credentials)):
    """특정 카테고리 정보 조회"""
    return tool_get_category_info(category_id)


# ============================================================
# CS API
# ============================================================
@router.post("/cs/reply")
def cs_auto_reply(req: CsReplyRequest, user: dict = Depends(verify_credentials)):
    """CS 자동 응답 생성"""
    return tool_auto_reply_cs(
        inquiry_text=req.text,
        inquiry_category=req.ticket_category,
        seller_tier=req.seller_tier,
    )


@router.post("/cs/quality")
def check_cs_quality_route(req: CsQualityRequest, user: dict = Depends(verify_credentials)):
    """CS 응답 품질 / 우선순위 평가"""
    return tool_check_cs_quality(
        ticket_category=req.ticket_category,
        seller_tier=req.seller_tier,
        sentiment_score=req.sentiment_score,
        order_value=req.order_value,
        is_repeat_issue=req.is_repeat_issue,
        text_length=req.text_length,
    )


@router.get("/cs/glossary")
def get_ecommerce_glossary(term: Optional[str] = None, user: dict = Depends(verify_credentials)):
    """이커머스 용어집 조회"""
    return tool_get_ecommerce_glossary(term=term)


@router.get("/cs/statistics")
def get_cs_stats(user: dict = Depends(verify_credentials)):
    """CS 통계 조회"""
    return tool_get_cs_statistics()


# ============================================================
# CS 자동화 파이프라인 API
# ============================================================
@router.post("/cs/pipeline")
def cs_pipeline(req: CsPipelineRequest, user: dict = Depends(verify_credentials)):
    """
    CS 자동화 파이프라인 - 5단계 순차 처리
    접수(분류) → 검토(우선순위) → 답변(컨텍스트) → 회신(채널) → 개선(통계)
    """
    result = {"status": "SUCCESS", "steps": {}}

    # ===== Step 1: 접수 (자동 분류) =====
    step_classify = tool_classify_inquiry(req.inquiry_text)
    result["steps"]["classify"] = step_classify

    if step_classify.get("status") != "SUCCESS":
        result["status"] = "PARTIAL"
        return result

    predicted_category = step_classify.get("predicted_category", "기타")
    confidence = step_classify.get("confidence", 0.0)

    # ===== Step 2: 검토 (우선순위 + 자동/수동 분기) =====
    negative_words = ["화나", "불만", "짜증", "최악", "실망", "환불", "사기", "안돼", "못써"]
    text_lower = req.inquiry_text
    sentiment = -0.4 if any(w in text_lower for w in negative_words) else 0.1

    is_auto = confidence >= req.confidence_threshold
    routing = "auto" if is_auto else "manual"

    priority_result = tool_check_cs_quality(
        ticket_category=predicted_category,
        seller_tier=req.seller_tier,
        sentiment_score=sentiment,
        order_value=req.order_value,
        is_repeat_issue=req.is_repeat_issue,
        text_length=len(req.inquiry_text),
    )

    result["steps"]["review"] = {
        "confidence": confidence,
        "threshold": req.confidence_threshold,
        "routing": routing,
        "predicted_category": predicted_category,
        "sentiment_score": sentiment,
        "priority": priority_result,
    }

    # ===== Step 3: 답변 (CS 응답 컨텍스트) =====
    answer_context = tool_auto_reply_cs(
        inquiry_text=req.inquiry_text,
        inquiry_category=predicted_category,
        seller_tier=req.seller_tier,
        order_id=req.order_id,
    )
    result["steps"]["answer"] = answer_context

    # ===== Step 4: 회신 (채널 목록 - mock) =====
    result["steps"]["reply"] = {
        "status": "READY",
        "channels": ["이메일", "카카오톡", "SMS", "인앱 알림"],
        "selected_channel": None,
        "message": "회신 채널을 선택하면 n8n 워크플로우로 자동 전송됩니다.",
    }

    # ===== Step 5: 개선 (통계) =====
    stats = tool_get_cs_statistics()
    result["steps"]["improve"] = {
        "statistics": stats,
        "pipeline_meta": {
            "classification_model_accuracy": 0.82,
            "auto_routing_rate": f"{req.confidence_threshold * 100:.0f}% 이상 자동",
            "categories": CS_TICKET_CATEGORIES,
            "priority_grades": list(CS_PRIORITY_GRADES.keys()),
        },
    }

    return result


@router.post("/cs/pipeline/answer")
async def cs_pipeline_answer(req: CsPipelineAnswerRequest, request: Request, user: dict = Depends(verify_credentials)):
    """CS 파이프라인 Step 3 - RAG 기반 답변 초안 생성 (SSE 스트리밍)"""

    async def gen():
        try:
            api_key = pick_api_key(req.api_key)
            if not api_key:
                yield f"data: {json.dumps({'type': 'error', 'data': 'API 키가 설정되지 않았습니다.'})}\n\n"
                return

            # 1. RAG 검색으로 정책 문서 컨텍스트 확보
            rag_query = f"카페24 {req.inquiry_category} 문의 정책 가이드: {req.inquiry_text}"
            context_text = ""
            source_count = 0

            try:
                if req.rag_mode == "lightrag" and LIGHTRAG_AVAILABLE:
                    rag_result = lightrag_search_sync(rag_query, mode="hybrid")
                elif req.rag_mode == "k2rag":
                    rag_result = k2rag_search_sync(rag_query)
                else:
                    rag_result = rag_search_hybrid(rag_query, top_k=3, api_key=api_key)

                if rag_result and rag_result.get("status") == "SUCCESS":
                    results = rag_result.get("results", [])
                    source_count = len(results)
                    snippets = []
                    for r in results[:3]:
                        txt = r.get("text", "") or r.get("content", "") or r.get("snippet", "")
                        if txt:
                            snippets.append(txt[:500])
                    context_text = "\n\n".join(snippets)
            except Exception as e:
                st.logger.warning("CS Pipeline RAG search failed: %s", e)

            yield f"data: {json.dumps({'type': 'rag_context', 'data': {'source_count': source_count, 'context_preview': context_text[:300] if context_text else '(검색 결과 없음)'}})}\n\n"

            # 2. LLM으로 답변 초안 생성 (스트리밍)
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            system_prompt = f"""당신은 카페24 이커머스 플랫폼의 셀러 지원 전문 상담원입니다.
셀러(쇼핑몰 운영자)가 카페24 플랫폼에 보낸 문의에 대해 정중하고 전문적인 답변 초안을 작성하세요.

문의 카테고리: {req.inquiry_category}
셀러 등급: {req.seller_tier}
관련 ID: {req.order_id or '없음'}

{f'참고 정책/가이드:{chr(10)}{context_text}' if context_text else ''}

답변 작성 원칙:
1. 셀러의 어려움에 공감하는 표현으로 시작
2. 카페24 관리자 페이지 기준으로 구체적인 해결 방법 제시
3. 필요 시 관련 설정 경로(메뉴 위치) 안내
4. 처리 기한 또는 예상 소요 시간 명시
5. 추가 기술지원 채널(개발자센터, 파트너 지원) 안내로 마무리
6. 전문적이고 신뢰감 있는 톤 유지"""

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key,
                temperature=0.3,
                streaming=True,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"셀러 문의: {req.inquiry_text}"),
            ]

            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            st.logger.error("CS Pipeline Answer Error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── n8n 연동: 회신 워크플로우 (job_id 기반 + SSE + 콜백) ──
_cs_job_queues: dict = {}  # job_id → asyncio.Queue (프로덕션에선 Redis Pub/Sub)


@router.post("/cs/send-reply")
async def cs_send_reply(request: Request, user: dict = Depends(verify_credentials)):
    """회신 전송 작업 시작 — job_id 발급 + n8n 트리거 (또는 시뮬레이션)"""
    body = await request.json()
    inquiries = body.get("inquiries", [])

    job_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _cs_job_queues[job_id] = queue

    # webhook 트리거 즉시 완료
    await queue.put({"type": "step", "data": {"node": "webhook", "status": "completed", "detail": "트리거 완료"}})

    n8n_url = os.environ.get("N8N_WEBHOOK_URL", "")
    callback_base = os.environ.get("N8N_CALLBACK_URL", "")

    if n8n_url:
        asyncio.create_task(_n8n_trigger(job_id, n8n_url, callback_base, inquiries, queue))
    else:
        asyncio.create_task(_simulate_workflow(job_id, inquiries, queue))

    return {"status": "SUCCESS", "job_id": job_id}


async def _n8n_trigger(job_id: str, n8n_url: str, callback_base: str, inquiries: list, queue: asyncio.Queue):
    """n8n webhook 호출 후 단계별 이벤트 재생"""
    try:
        import httpx
        all_channels = sorted({ch for inq in inquiries for ch in inq.get("channels", [])})

        st.logger.info("[n8n] job=%s calling %s channels=%s", job_id, n8n_url, all_channels)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(n8n_url, json={
                "job_id": job_id,
                "inquiries": inquiries,
                "channels": all_channels,
            })

        st.logger.info("[n8n] job=%s status=%s body=%s", job_id, resp.status_code, resp.text[:300])

        if resp.status_code >= 400:
            await queue.put({"type": "error", "data": f"n8n 호출 실패: HTTP {resp.status_code}"})
            await queue.put(None)
            return

        # n8n 성공 → 단계별 이벤트 재생
        await _replay_steps_from_n8n(inquiries, all_channels, queue)

    except Exception as e:
        st.logger.error("[n8n] job=%s error: %s", job_id, e)
        await queue.put({"type": "error", "data": f"n8n 연결 실패: {str(e)[:80]}"})
        await queue.put(None)


async def _replay_steps_from_n8n(inquiries: list, all_channels: list, queue: asyncio.Queue):
    """n8n 실행 완료 후 단계별 SSE 이벤트 재생 (콜백 없을 때)"""
    try:
        # 답변 검증
        await queue.put({"type": "step", "data": {"node": "validate", "status": "running"}})
        await asyncio.sleep(0.4)
        await queue.put({"type": "step", "data": {"node": "validate", "status": "completed", "detail": f"{len(inquiries)}건 검증 완료"}})

        # 채널 분기
        await queue.put({"type": "step", "data": {"node": "router", "status": "running"}})
        await asyncio.sleep(0.3)
        await queue.put({"type": "step", "data": {"node": "router", "status": "completed", "detail": f"{len(all_channels)}개 채널"}})

        # 채널별 전송
        for ch in all_channels:
            ch_count = sum(1 for inq in inquiries if ch in inq.get("channels", []))
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "running"}})
            await asyncio.sleep(0.3)
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "completed", "detail": f"{ch_count}건 전송"}})

        # 결과 기록
        await queue.put({"type": "step", "data": {"node": "log", "status": "running"}})
        await asyncio.sleep(0.3)
        await queue.put({"type": "step", "data": {"node": "log", "status": "completed", "detail": "이력 저장 완료"}})

        # 완료
        await queue.put({"type": "done", "data": {"total": len(inquiries), "channels": all_channels}})
    except Exception as e:
        st.logger.error("replay_steps error: %s", e)
        await queue.put({"type": "error", "data": str(e)})
    finally:
        await queue.put(None)


async def _simulate_workflow(job_id: str, inquiries: list, queue: asyncio.Queue):
    """n8n 미설정 시 시뮬레이션 (asyncio.sleep 기반)"""
    try:
        await queue.put({"type": "step", "data": {"node": "validate", "status": "running"}})
        await asyncio.sleep(0.5)
        await queue.put({"type": "step", "data": {"node": "validate", "status": "completed", "detail": f"{len(inquiries)}건 검증 완료"}})

        await queue.put({"type": "step", "data": {"node": "router", "status": "running"}})
        await asyncio.sleep(0.3)
        all_channels = sorted({ch for inq in inquiries for ch in inq.get("channels", [])})
        await queue.put({"type": "step", "data": {"node": "router", "status": "completed", "detail": f"{len(all_channels)}개 채널"}})

        for ch in all_channels:
            ch_count = sum(1 for inq in inquiries if ch in inq.get("channels", []))
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "running"}})
            await asyncio.sleep(0.5)
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "completed", "detail": f"{ch_count}건 전송 (시뮬레이션)"}})

        await queue.put({"type": "step", "data": {"node": "log", "status": "running"}})
        await asyncio.sleep(0.3)
        await queue.put({"type": "step", "data": {"node": "log", "status": "completed", "detail": "이력 저장 완료"}})

        await queue.put({"type": "done", "data": {"total": len(inquiries), "channels": all_channels}})
    except Exception as e:
        st.logger.error("simulate_workflow error: %s", e)
        await queue.put({"type": "error", "data": str(e)})
    finally:
        await queue.put(None)  # sentinel → stream 종료


@router.get("/cs/stream")
async def cs_stream(job_id: str, user: dict = Depends(verify_credentials)):
    """SSE 스트림 — job_id 기반 실시간 워크플로우 상태 전달"""
    queue = _cs_job_queues.get(job_id)
    if not queue:
        return JSONResponse({"status": "FAILED", "error": "유효하지 않은 job_id"}, status_code=404)

    async def gen():
        try:
            while True:
                evt = await asyncio.wait_for(queue.get(), timeout=120.0)
                if evt is None:
                    break
                yield f"data: {json.dumps(evt)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'data': 'timeout'})}\n\n"
        finally:
            _cs_job_queues.pop(job_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")



@router.post("/cs/callback")
async def cs_callback(request: Request):
    """n8n 콜백 수신 — n8n 워크플로우가 각 단계마다 호출"""
    body = await request.json()
    job_id = body.get("job_id", "")
    queue = _cs_job_queues.get(job_id)
    if not queue:
        return JSONResponse({"status": "FAILED", "error": "유효하지 않은 job_id"}, status_code=404)

    step = body.get("step", "")
    status = body.get("status", "")
    detail = body.get("detail", "")

    if step == "done":
        await queue.put({"type": "done", "data": body.get("data", {})})
        await queue.put(None)  # sentinel
    elif step == "error":
        await queue.put({"type": "error", "data": detail or body.get("data", "")})
        await queue.put(None)
    else:
        await queue.put({"type": "step", "data": {"node": step, "status": status, "detail": detail}})

    return {"status": "SUCCESS"}


# ============================================================
# 유저 분석 API
# ============================================================
@router.get("/sellers/autocomplete")
def sellers_autocomplete(q: str = "", limit: int = 8, user: dict = Depends(verify_credentials)):
    """셀러 자동완성 검색"""
    if st.SELLERS_DF is None:
        return {"status": "FAILED", "error": "셀러 데이터 없음"}

    q = q.strip().upper()
    if not q:
        return {"status": "SUCCESS", "users": []}

    df = st.SELLERS_DF
    mask = df["seller_id"].str.upper().str.contains(q, na=False)
    if "shop_name" in df.columns:
        mask |= df["shop_name"].str.upper().str.contains(q, na=False)

    matched = df[mask].head(limit)
    users = []
    for _, row in matched.iterrows():
        users.append({
            "id": row["seller_id"],
            "name": row["seller_id"],
        })
    return {"status": "SUCCESS", "users": users}


@router.get("/sellers/search")
def search_seller(q: str, days: int = 7, user: dict = Depends(verify_credentials)):
    """셀러 검색 (days: 활동 데이터 기간)"""
    if st.SELLERS_DF is None or st.SELLER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "셀러 데이터가 로드되지 않았습니다."}

    if days not in [7, 30, 90]:
        days = 7

    q = q.strip().upper()

    # seller_id로 검색
    seller_row = st.SELLERS_DF[st.SELLERS_DF["seller_id"] == q]
    if seller_row.empty:
        seller_row = st.SELLERS_DF[st.SELLERS_DF["seller_id"].str.contains(q, case=False, na=False)]

    if seller_row.empty:
        return {"status": "FAILED", "error": "셀러를 찾을 수 없습니다."}

    seller_data = seller_row.iloc[0].to_dict()
    seller_id = seller_data["seller_id"]

    # seller_analytics에서 추가 정보
    analytics_row = st.SELLER_ANALYTICS_DF[st.SELLER_ANALYTICS_DF["seller_id"] == seller_id]
    if not analytics_row.empty:
        analytics = analytics_row.iloc[0].to_dict()
        # analytics의 모든 필드 merge (ML 예측값, SHAP, 세그먼트 등 포함)
        for k, v in analytics.items():
            if k != "seller_id":
                seller_data[k] = v

    # 세그먼트 이름 - CSV의 segment_name 우선, 없으면 클러스터 번호
    seller_data["segment"] = seller_data.get("segment_name", f"세그먼트 {seller_data.get('cluster', 0)}")

    # 최근 활동
    activity = []
    period_stats = {"total_revenue": 0, "total_orders": 0, "active_days": 0,
                    "total_cs": 0, "total_refunds": 0}

    if st.SELLER_ACTIVITY_DF is not None:
        seller_activity = st.SELLER_ACTIVITY_DF[st.SELLER_ACTIVITY_DF["seller_id"] == seller_id]
        if not seller_activity.empty:
            seller_activity = seller_activity.tail(days)
            period_stats["active_days"] = len(seller_activity)

            for _, row in seller_activity.iterrows():
                revenue = int(row.get("revenue", row.get("daily_revenue", 0)))
                orders = int(row.get("orders_processed", row.get("daily_orders", 0)))
                cs = int(row.get("cs_handled", row.get("cs_tickets", 0)))
                products = int(row.get("products_updated", row.get("product_count", 0)))
                activity.append({
                    "date": row.get("date", ""),
                    "revenue": revenue,
                    "orders": orders,
                    "product_count": products,
                })
                period_stats["total_revenue"] += revenue
                period_stats["total_orders"] += orders
                period_stats["total_cs"] += cs

    # 셀러 스탯 (레이더 차트용) - 전체 셀러 대비 퍼센타일 기반
    def _percentile_score(value, col_name):
        """seller_analytics 전체에서 해당 값의 퍼센타일을 0-100 점수로"""
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

    # 주력 쇼핑몰 - 셀러 ID로 매칭
    top_shops = []
    if st.SHOPS_DF is not None:
        shop_id = seller_data["seller_id"].replace("SEL", "S")
        matched = st.SHOPS_DF[st.SHOPS_DF["shop_id"] == shop_id]
        if len(matched) > 0:
            top_shops = [s.get("shop_name", s.get("shop_id", "")) for _, s in matched.head(2).iterrows()]

    # ML 모델 예측 결과 수집
    model_predictions = {}

    # 1) 이탈 예측
    churn_prob = seller_data.get("churn_probability")
    churn_risk = seller_data.get("churn_risk")
    if churn_prob is not None:
        risk_label = "높음" if float(churn_prob) > 0.7 else "보통" if float(churn_prob) > 0.4 else "낮음"
        # SHAP 상위 요인 추출
        shap_factors = []
        shap_cols = [c for c in seller_data.keys() if str(c).startswith("shap_")]
        if shap_cols:
            shap_items = [(c.replace("shap_", ""), abs(float(seller_data.get(c, 0)))) for c in shap_cols]
            shap_items.sort(key=lambda x: x[1], reverse=True)
            factor_names = {
                "total_orders": "총 주문수", "total_revenue": "총 매출",
                "product_count": "상품 수", "cs_tickets": "CS 문의",
                "refund_rate": "환불률", "avg_response_time": "평균 응답시간",
                "days_since_last_login": "마지막 로그인", "days_since_register": "가입 경과일",
                "plan_tier_encoded": "플랜 등급",
            }
            for fname, fval in shap_items[:5]:
                shap_factors.append({"factor": factor_names.get(fname, fname), "importance": round(fval, 4)})

        model_predictions["churn"] = {
            "model": "셀러 이탈 예측 (RandomForest+SHAP)",
            "probability": round(float(churn_prob) * 100, 1),
            "risk_level": risk_label,
            "risk_code": int(churn_risk) if churn_risk is not None else (2 if float(churn_prob) > 0.7 else 1 if float(churn_prob) > 0.4 else 0),
            "factors": shap_factors,
        }

    # 2) 이상거래 탐지
    anomaly_score = seller_data.get("anomaly_score")
    is_anomaly = seller_data.get("is_anomaly")
    if anomaly_score is not None:
        model_predictions["fraud"] = {
            "model": "이상거래 탐지 (Isolation Forest)",
            "anomaly_score": round(float(anomaly_score), 4),
            "is_anomaly": bool(is_anomaly) if is_anomaly is not None else float(anomaly_score) > 0.7,
            "risk_level": "위험" if float(anomaly_score) > 0.7 else "주의" if float(anomaly_score) > 0.5 else "정상",
        }

    # 3) 셀러 세그먼트 - CSV segment_name 사용
    cluster = seller_data.get("cluster", 0)
    seg_name = seller_data.get("segment_name", f"세그먼트 {cluster}")
    model_predictions["segment"] = {
        "model": "셀러 세그먼트 (K-Means)",
        "cluster": int(cluster),
        "segment_name": seg_name,
    }

    # 4) CS 응답 품질 (seller_analytics.csv 사전계산 결과)
    cs_score_val = seller_data.get("cs_quality_score")
    cs_grade_val = seller_data.get("cs_quality_grade")
    refund_rate = float(seller_data.get("refund_rate", 0))
    avg_resp = float(seller_data.get("avg_response_time", 0))
    if cs_score_val is not None:
        model_predictions["cs_quality"] = {
            "model": "CS 응답 품질 (RandomForest)",
            "score": int(cs_score_val),
            "grade": str(cs_grade_val) if cs_grade_val else ("우수" if int(cs_score_val) >= 80 else "보통" if int(cs_score_val) >= 50 else "개선필요"),
            "refund_rate": round(refund_rate, 4),
            "avg_response_time": round(avg_resp, 1),
        }

    # 5) 매출 예측 (seller_analytics.csv 사전계산 결과)
    predicted_rev = seller_data.get("predicted_revenue")
    rev_growth = seller_data.get("revenue_growth_rate")
    if predicted_rev is not None and float(predicted_rev) > 0:
        model_predictions["revenue"] = {
            "model": "매출 예측 (LightGBM)",
            "predicted_next_month": int(float(predicted_rev)),
            "growth_rate": round(float(rev_growth), 1) if rev_growth is not None else 0.0,
            "confidence": round(r2_score_val * 100, 1) if (r2_score_val := _get_revenue_r2()) is not None else None,
        }

    seller_obj = {
        "id": seller_data["seller_id"],
        "segment": seller_data["segment"],
        "plan_tier": seller_data.get("plan_tier", "Standard"),
        "monthly_revenue": period_stats["total_revenue"],
        "total_revenue": period_stats["total_revenue"],
        "product_count": seller_data.get("product_count", 0),
        "order_count": period_stats["total_orders"],
        "shops_count": seller_data.get("product_count", 0),
        "region": seller_data.get("region", "서울"),
        "is_anomaly": seller_data.get("is_anomaly", False),
        "top_shops": top_shops,
        "stats": stats,
        "activity": activity,
        "model_predictions": model_predictions,
        "period_stats": {
            "days": days,
            "total_revenue": period_stats["total_revenue"],
            "total_orders": period_stats["total_orders"],
            "active_days": period_stats["active_days"],
            "avg_daily_revenue": round(period_stats["total_revenue"] / max(1, period_stats["active_days"]), 1),
            "total_cs": period_stats["total_cs"],
        },
    }
    result = {
        "status": "SUCCESS",
        "days": days,
        "user": seller_obj,
        "seller": seller_obj,
    }

    return json_sanitize(result)


@router.get("/sellers/analyze/{seller_id}")
def analyze_seller(seller_id: str, user: dict = Depends(verify_credentials)):
    """셀러 분석"""
    return tool_analyze_seller(seller_id)


@router.post("/sellers/segment")
def get_seller_segment(seller_features: dict, user: dict = Depends(verify_credentials)):
    """셀러 세그먼트 분류"""
    return tool_get_seller_segment(seller_features)


@router.post("/sellers/fraud")
def detect_seller_fraud(seller_features: dict, user: dict = Depends(verify_credentials)):
    """이상거래 탐지"""
    return tool_detect_fraud(seller_features=seller_features)


@router.get("/sellers/segments/statistics")
def get_segment_stats(user: dict = Depends(verify_credentials)):
    """세그먼트별 통계"""
    return tool_get_segment_statistics()


@router.get("/users/segments/{segment_name}/details")
def get_segment_details(segment_name: str, user: dict = Depends(verify_credentials)):
    """세그먼트 상세 정보 (대시보드 드릴다운)"""
    try:
        if st.SELLER_ANALYTICS_DF is None:
            return {"status": "FAILED", "error": "셀러 분석 데이터 없음"}

        df = st.SELLER_ANALYTICS_DF
        # segment_name 컬럼으로 직접 필터
        if "segment_name" in df.columns:
            seg = df[df["segment_name"] == segment_name]
        else:
            return {"status": "FAILED", "error": f"알 수 없는 세그먼트: {segment_name}"}
        total = len(df)
        count = len(seg)

        result = {
            "status": "SUCCESS",
            "segment": segment_name,
            "count": count,
            "percentage": round(count / max(total, 1) * 100, 1),
            "avg_monthly_revenue": int(seg["total_revenue"].mean()) if "total_revenue" in seg.columns else 0,
            "avg_product_count": int(seg["product_count"].mean()) if "product_count" in seg.columns else 0,
            "avg_order_count": int(seg["total_orders"].mean()) if "total_orders" in seg.columns else 0,
            "top_activities": [],
            "retention_rate": None,
        }
        return json_sanitize(result)
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/sellers/{seller_id}/activity")
def get_seller_activity(seller_id: str, days: int = 30, user: dict = Depends(verify_credentials)):
    """셀러 활동 리포트"""
    return tool_get_seller_activity_report(seller_id, days)


# ============================================================
# 주문/운영 통계 API
# ============================================================
@router.get("/orders/statistics")
def get_order_stats(
    event_type: Optional[str] = None,
    days: int = 30,
    user: dict = Depends(verify_credentials)
):
    """주문/운영 통계"""
    return tool_get_order_statistics(event_type=event_type, days=days)


# ============================================================
# 텍스트 분류 API
# ============================================================
@router.post("/classify/inquiry")
def classify_inquiry(req: TextClassifyRequest, user: dict = Depends(verify_credentials)):
    """문의 카테고리 분류"""
    return tool_classify_inquiry(req.text)


# ============================================================
# 대시보드 API
# ============================================================
@router.get("/dashboard/summary")
def get_dashboard_summary(user: dict = Depends(verify_credentials)):
    """대시보드 요약 정보"""
    return tool_get_dashboard_summary()


@router.get("/dashboard/insights")
def get_dashboard_insights(user: dict = Depends(verify_credentials)):
    """AI 인사이트 - 실제 데이터 기반 동적 생성"""
    insights = []

    try:
        # 1. 활성 쇼핑몰 트렌드 분석 (CSV: active_shops)
        activity_col = "active_shops" if st.DAILY_METRICS_DF is not None and "active_shops" in st.DAILY_METRICS_DF.columns else "dau"
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 14 and activity_col in st.DAILY_METRICS_DF.columns:
            df = st.DAILY_METRICS_DF
            recent_7 = df.tail(7)[activity_col].mean()
            prev_7 = df.tail(14).head(7)[activity_col].mean()
            dau_change = round((recent_7 - prev_7) / max(prev_7, 1) * 100, 1)

            if dau_change > 5:
                insights.append({
                    "type": "positive",
                    "icon": "arrow_up",
                    "title": "활성 쇼핑몰 상승 추세",
                    "description": f"최근 7일간 활성 쇼핑몰이 {dau_change}% 증가했습니다. 긍정적인 성장세입니다.",
                })
            elif dau_change < -5:
                insights.append({
                    "type": "warning",
                    "icon": "arrow_down",
                    "title": "활성 쇼핑몰 하락 주의",
                    "description": f"최근 7일간 활성 쇼핑몰이 {abs(dau_change)}% 감소했습니다. 원인 분석이 필요합니다.",
                })
            else:
                insights.append({
                    "type": "neutral",
                    "icon": "stable",
                    "title": "활성 쇼핑몰 안정적",
                    "description": f"최근 7일간 활성 쇼핑몰 변화가 {dau_change:+.1f}%로 안정적입니다.",
                })

        # 2. 리텐션 분석
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            cohort_df = st.COHORT_RETENTION_DF
            # 가장 최근 완전한 코호트의 week2 리텐션
            for _, row in cohort_df.iloc[::-1].iterrows():
                week2 = row.get("week2")
                if week2 is not None and not pd.isna(week2):
                    week2_val = float(week2)
                    if week2_val < 50:
                        insights.append({
                            "type": "warning",
                            "icon": "retention",
                            "title": "리텐션 개선 필요",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표(50%) 대비 낮습니다. 온보딩 개선을 권장합니다.",
                        })
                    elif week2_val >= 65:
                        insights.append({
                            "type": "positive",
                            "icon": "retention",
                            "title": "리텐션 우수",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 매우 우수합니다.",
                        })
                    else:
                        insights.append({
                            "type": "neutral",
                            "icon": "retention",
                            "title": "리텐션 양호",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표 수준입니다.",
                        })
                    break

        # 3. CS 만족도 분석 (CSV: satisfaction_score)
        quality_col = "satisfaction_score" if st.CS_STATS_DF is not None and "satisfaction_score" in st.CS_STATS_DF.columns else "avg_quality"
        category_col = "category" if st.CS_STATS_DF is not None and "category" in st.CS_STATS_DF.columns else "lang_name"
        if st.CS_STATS_DF is not None and len(st.CS_STATS_DF) > 0 and quality_col in st.CS_STATS_DF.columns:
            avg_quality = st.CS_STATS_DF[quality_col].mean()
            best_row = st.CS_STATS_DF.loc[st.CS_STATS_DF[quality_col].idxmax()]
            best_name = best_row.get(category_col, "일반")

            if avg_quality >= 90:
                insights.append({
                    "type": "positive",
                    "icon": "translation",
                    "title": "CS 만족도 우수",
                    "description": f"{best_name} 카테고리 만족도가 {best_row[quality_col]:.1f}점으로 목표치를 초과 달성했습니다.",
                })
            elif avg_quality < 80:
                insights.append({
                    "type": "warning",
                    "icon": "translation",
                    "title": "CS 만족도 개선 필요",
                    "description": f"평균 CS 만족도가 {avg_quality:.1f}점입니다. 개선이 필요합니다.",
                })
            else:
                insights.append({
                    "type": "neutral",
                    "icon": "translation",
                    "title": "CS 만족도 양호",
                    "description": f"평균 CS 만족도가 {avg_quality:.1f}점으로 양호합니다. {best_name} 카테고리가 {best_row[quality_col]:.1f}점으로 가장 높습니다.",
                })

        # 4. 이상 거래 셀러 분석
        if st.SELLER_ANALYTICS_DF is not None and "is_anomaly" in st.SELLER_ANALYTICS_DF.columns:
            anomaly_count = int(st.SELLER_ANALYTICS_DF["is_anomaly"].sum())
            total_users = len(st.SELLER_ANALYTICS_DF)
            anomaly_rate = round(anomaly_count / max(total_users, 1) * 100, 1)

            if anomaly_rate > 5:
                insights.append({
                    "type": "warning",
                    "icon": "anomaly",
                    "title": "이상 유저 주의",
                    "description": f"이상 행동 유저가 {anomaly_count}명({anomaly_rate}%)입니다. 모니터링 강화가 필요합니다.",
                })

        # 인사이트가 없으면 기본 메시지
        if not insights:
            insights.append({
                "type": "neutral",
                "icon": "info",
                "title": "데이터 분석 중",
                "description": "충분한 데이터가 수집되면 AI 인사이트가 제공됩니다.",
            })

        return json_sanitize({
            "status": "SUCCESS",
            "insights": insights[:3],  # 최대 3개
        })

    except Exception as e:
        st.logger.exception("인사이트 생성 실패")
        return {"status": "FAILED", "error": safe_str(e), "insights": []}


@router.get("/dashboard/alerts")
def get_dashboard_alerts(limit: int = 5, user: dict = Depends(verify_credentials)):
    """실시간 알림 - ANOMALY_DETAILS_DF 기반 이상 행동 알림"""
    from datetime import datetime, timedelta

    try:
        alerts = []
        anomaly_df = st.FRAUD_DETAILS_DF

        if anomaly_df is not None and len(anomaly_df) > 0:
            df = anomaly_df.copy()

            # detected_date 파싱 (CSV 컬럼: detected_date)
            date_col = "detected_date" if "detected_date" in df.columns else "detected_at"
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.sort_values(date_col, ascending=False)

            # 상위 N개만 가져오기
            for _, row in df.head(limit).iterrows():
                user_id = str(row.get("seller_id", row.get("user_id", "Unknown")))
                anomaly_type = str(row.get("anomaly_type", "이상 행동"))
                # severity: CSV에 없으면 anomaly_score 기반 추정
                if "severity" in row.index:
                    severity = str(row.get("severity", "medium")).lower()
                else:
                    score = float(row.get("anomaly_score", 0))
                    severity = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
                detail = str(row.get("details", row.get("detail", "")))
                detected_val = row.get(date_col)

                # 시간 경과 계산
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

                # severity에 따른 색상 타입
                color_type = "red" if severity == "high" else "orange" if severity == "medium" else "yellow"

                alerts.append({
                    "user_id": user_id,
                    "type": anomaly_type,
                    "severity": severity,
                    "color": color_type,
                    "detail": detail if detail else anomaly_type,
                    "time_ago": time_ago,
                })

        return json_sanitize({
            "status": "SUCCESS",
            "alerts": alerts,
            "total_count": len(anomaly_df) if anomaly_df is not None else 0,
        })

    except Exception as e:
        st.logger.exception("알림 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "alerts": []}


@router.get("/analysis/anomaly")
def get_anomaly_analysis(days: int = 7, user: dict = Depends(verify_credentials)):
    """이상탐지 분석 데이터 - ANOMALY_DETAILS_DF 실제 데이터 기반"""
    from datetime import datetime, timedelta

    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    try:
        df = st.SELLER_ANALYTICS_DF
        total_users = len(df)

        # ========================================
        # ANOMALY_DETAILS_DF 실제 데이터 활용
        # ========================================
        anomaly_df = st.FRAUD_DETAILS_DF
        today = datetime.now()

        if anomaly_df is not None and len(anomaly_df) > 0:
            # detected_date 기준으로 기간 필터링 (CSV: detected_date)
            anomaly_df = anomaly_df.copy()
            date_col = "detected_date" if "detected_date" in anomaly_df.columns else "detected_at"
            if date_col in anomaly_df.columns:
                anomaly_df[date_col] = pd.to_datetime(anomaly_df[date_col], errors="coerce")
                # 데이터의 최신 날짜를 기준으로 필터링 (데이터가 과거 날짜일 경우 대응)
                latest_date = anomaly_df[date_col].max()
                if pd.notna(latest_date):
                    reference_date = latest_date
                else:
                    reference_date = today
                cutoff_date = reference_date - timedelta(days=days)
                filtered_df = anomaly_df[anomaly_df[date_col] >= cutoff_date]
            else:
                filtered_df = anomaly_df

            anomaly_count = len(filtered_df)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0

            # severity별 실제 분포 (CSV에 severity 없으면 anomaly_score 기반 추정)
            if "severity" in filtered_df.columns:
                severity_counts = filtered_df["severity"].value_counts().to_dict()
                high_risk = int(severity_counts.get("high", 0))
                medium_risk = int(severity_counts.get("medium", 0))
                low_risk = int(severity_counts.get("low", 0))
            elif "anomaly_score" in filtered_df.columns:
                high_risk = int((filtered_df["anomaly_score"] > 0.8).sum())
                medium_risk = int(((filtered_df["anomaly_score"] > 0.5) & (filtered_df["anomaly_score"] <= 0.8)).sum())
                low_risk = max(0, anomaly_count - high_risk - medium_risk)
            else:
                high_risk = 0
                medium_risk = 0
                low_risk = anomaly_count

            # by_type: anomaly_type별 실제 집계
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
                    type_severity["severity"] = type_severity["avg_score"].apply(
                        lambda x: "high" if x > 0.8 else "medium" if x > 0.5 else "low"
                    )
                else:
                    type_severity.columns = ["type", "count"]
                    type_severity["severity"] = "medium"
                for _, row in type_severity.iterrows():
                    by_type.append({
                        "type": row["type"],
                        "count": int(row["count"]),
                        "severity": row["severity"]
                    })
                by_type.sort(key=lambda x: x["count"], reverse=True)

            # trend: 기간별로 다른 집계 방식 (reference_date 기준)
            # 7일: 일별, 30일: 5일 단위, 90일: 15일 단위
            trend = []
            if date_col in filtered_df.columns and len(filtered_df) > 0:
                if days == 7:
                    # 7일: 일별 집계
                    filtered_df["date_str"] = filtered_df[date_col].dt.strftime("%m/%d")
                    daily_counts = filtered_df.groupby("date_str").size().to_dict()
                    for i in range(7):
                        d = reference_date - timedelta(days=6 - i)
                        date_str = d.strftime("%m/%d")
                        trend.append({"date": date_str, "count": int(daily_counts.get(date_str, 0))})
                elif days == 30:
                    # 30일: 5일 단위 집계 (6개 포인트)
                    for i in range(6):
                        start_day = 30 - (i + 1) * 5
                        end_day = 30 - i * 5
                        start_date = reference_date - timedelta(days=end_day)
                        end_date = reference_date - timedelta(days=start_day)
                        period_df = filtered_df[(filtered_df[date_col] >= start_date) & (filtered_df[date_col] < end_date)]
                        label = (reference_date - timedelta(days=end_day - 2)).strftime("%m/%d")
                        trend.append({"date": label, "count": len(period_df)})
                else:
                    # 90일: 15일 단위 집계 (6개 포인트)
                    for i in range(6):
                        start_day = 90 - (i + 1) * 15
                        end_day = 90 - i * 15
                        start_date = reference_date - timedelta(days=end_day)
                        end_date = reference_date - timedelta(days=start_day)
                        period_df = filtered_df[(filtered_df[date_col] >= start_date) & (filtered_df[date_col] < end_date)]
                        label = (reference_date - timedelta(days=end_day - 7)).strftime("%m/%d")
                        trend.append({"date": label, "count": len(period_df)})
            else:
                # detected_at이 없으면 균등 분배
                points = {7: 7, 30: 6, 90: 6}.get(days, 7)
                for i in range(points):
                    d = today - timedelta(days=days - 1 - i * (days // points))
                    trend.append({
                        "date": d.strftime("%m/%d"),
                        "count": max(0, anomaly_count // points)
                    })

            # recent_alerts: 실제 데이터에서 최근 알림 생성
            recent_alerts = []
            alert_count = {7: 4, 30: 6, 90: 8}.get(days, 4)
            if date_col in filtered_df.columns:
                recent_df = filtered_df.nlargest(alert_count, date_col)
            else:
                recent_df = filtered_df.head(alert_count)

            for _, row in recent_df.iterrows():
                # 시간 차이 계산 (reference_date 기준)
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

                # severity 결정
                if "severity" in row.index:
                    sev = str(row.get("severity", "medium"))
                elif "anomaly_score" in row.index:
                    score = float(row.get("anomaly_score", 0))
                    sev = "high" if score > 0.8 else "medium" if score > 0.5 else "low"
                else:
                    sev = "medium"

                recent_alerts.append({
                    "id": str(row.get("seller_id", row.get("user_id", "M000000"))),
                    "type": str(row.get("anomaly_type", "알 수 없음")),
                    "severity": sev,
                    "detail": str(row.get("details", row.get("detail", "이상 패턴 감지"))),
                    "time": time_str,
                })
        else:
            # ANOMALY_DETAILS_DF가 없으면 USER_ANALYTICS_DF에서 기본 집계
            anomaly_users = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else df.iloc[:0]
            anomaly_count = len(anomaly_users)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0
            high_risk = 0
            medium_risk = 0
            low_risk = anomaly_count
            by_type = []
            trend = []
            recent_alerts = []

        return json_sanitize({
            "status": "SUCCESS",
            "data_source": "ANOMALY_DETAILS_DF" if (st.FRAUD_DETAILS_DF is not None and len(st.FRAUD_DETAILS_DF) > 0) else "USER_ANALYTICS_DF",
            "summary": {
                "total_users": total_users,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
            },
            "by_type": by_type,
            "trend": trend,
            "recent_alerts": recent_alerts,
        })
    except Exception as e:
        st.logger.error(f"이상탐지 분석 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/prediction/churn")
def get_churn_prediction(days: int = 7, user: dict = Depends(verify_credentials)):
    """이탈 예측 분석 (실제 ML 모델 + SHAP 기반)"""
    import numpy as np

    if days not in [7, 30, 90]:
        days = 7

    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.SELLER_ANALYTICS_DF.copy()
        total = len(df)
        risk_multiplier = 1.0  # 모델 결과 그대로 사용

        # 기본값
        model_accuracy = None
        top_factors = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        available_features = []
        feature_names_kr = {}

        # ========================================
        # 실제 모델 기반 이탈 예측
        # ========================================
        if st.SELLER_CHURN_MODEL is not None:
            config = st.CHURN_MODEL_CONFIG or {}
            features = config.get("features", [
                "total_orders", "total_revenue", "product_count",
                "cs_tickets", "refund_rate", "avg_response_time"
            ])
            feature_names_kr = config.get("feature_names_kr", {
                "total_orders": "총 주문 수",
                "total_revenue": "총 매출",
                "product_count": "등록 상품 수",
                "cs_tickets": "CS 문의 수",
                "refund_rate": "환불률",
                "avg_response_time": "평균 응답 시간",
            })
            model_accuracy = (config.get("model_accuracy") or 0) * 100

            available_features = [f for f in features if f in df.columns]
            if available_features:
                X = df[available_features].fillna(0)
                churn_proba = st.SELLER_CHURN_MODEL.predict_proba(X)[:, 1]
                df["churn_probability"] = churn_proba

                # 기간별 임계값
                high_threshold = {7: 0.7, 30: 0.6, 90: 0.5}.get(days, 0.7)
                medium_threshold = {7: 0.4, 30: 0.35, 90: 0.3}.get(days, 0.4)

                high_risk_count = int((churn_proba >= high_threshold).sum())
                medium_risk_count = int(((churn_proba >= medium_threshold) & (churn_proba < high_threshold)).sum())
                low_risk_count = total - high_risk_count - medium_risk_count

                # SHAP Feature Importance
                if st.SHAP_EXPLAINER_CHURN is not None:
                    try:
                        shap_values_raw = st.SHAP_EXPLAINER_CHURN.shap_values(X)

                        # SHAP 버전에 따른 처리
                        if hasattr(shap_values_raw, 'values'):
                            shap_values = shap_values_raw.values
                        elif isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                            shap_values = shap_values_raw[1]
                        elif isinstance(shap_values_raw, np.ndarray):
                            if shap_values_raw.ndim == 3:
                                shap_values = shap_values_raw[:, :, 1]
                            else:
                                shap_values = shap_values_raw
                        else:
                            shap_values = shap_values_raw

                        shap_values = np.array(shap_values)
                        shap_importance = np.abs(shap_values).mean(axis=0)
                        total_imp = shap_importance.sum()
                        if total_imp > 0:
                            shap_importance = shap_importance / total_imp

                        sorted_indices = np.argsort(shap_importance)[::-1]
                        for idx in sorted_indices[:5]:
                            feat = available_features[idx]
                            top_factors.append({
                                "factor": feature_names_kr.get(feat, feat),
                                "importance": round(float(shap_importance[idx]), 3),
                            })
                    except Exception as e:
                        st.logger.warning(f"SHAP 분석 실패: {e}")

                # Fallback: 모델 자체 importance
                if not top_factors and hasattr(st.SELLER_CHURN_MODEL, "feature_importances_"):
                    importances = st.SELLER_CHURN_MODEL.feature_importances_
                    sorted_indices = importances.argsort()[::-1]
                    for idx in sorted_indices[:5]:
                        feat = available_features[idx]
                        top_factors.append({
                            "factor": feature_names_kr.get(feat, feat),
                            "importance": round(float(importances[idx]), 3),
                        })

        # 모델이 없으면 빈 데이터 반환
        if not top_factors:
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = total
            top_factors = []

        # ========================================
        # 고위험 유저 목록 (SHAP 기반 요인 포함)
        # ========================================
        high_risk_users = []
        user_sample_count = min(3 + days // 30 * 2, 7)

        if "churn_probability" in df.columns:
            high_risk_df = df.nlargest(user_sample_count, "churn_probability")
            for _, row in high_risk_df.iterrows():
                user_id = row.get("seller_id", row.get("user_id", "M000000"))
                cluster = int(row.get("cluster", 0))
                prob = int(row["churn_probability"] * 100)

                # 유저별 SHAP 요인
                user_factors = []
                if st.SHAP_EXPLAINER_CHURN is not None and available_features:
                    try:
                        user_X = row[available_features].values.reshape(1, -1)
                        user_shap_raw = st.SHAP_EXPLAINER_CHURN.shap_values(user_X)

                        # SHAP 버전에 따른 처리
                        if hasattr(user_shap_raw, 'values'):
                            user_shap = user_shap_raw.values[0]
                        elif isinstance(user_shap_raw, list) and len(user_shap_raw) == 2:
                            user_shap = user_shap_raw[1][0]
                        elif isinstance(user_shap_raw, np.ndarray):
                            if user_shap_raw.ndim == 3:
                                user_shap = user_shap_raw[0, :, 1]
                            elif user_shap_raw.ndim == 2:
                                user_shap = user_shap_raw[0]
                            else:
                                user_shap = user_shap_raw
                        else:
                            user_shap = user_shap_raw[0] if hasattr(user_shap_raw, '__getitem__') else user_shap_raw

                        user_shap = np.array(user_shap).flatten()
                        sorted_idx = np.abs(user_shap).argsort()[::-1]
                        for idx in sorted_idx[:3]:
                            feat = available_features[idx]
                            shap_val = user_shap[idx]
                            user_factors.append({
                                "factor": feature_names_kr.get(feat, feat),
                                "direction": "위험" if shap_val > 0 else "양호",
                                "impact": round(abs(float(shap_val)), 3),
                            })
                    except Exception:
                        pass

                high_risk_users.append({
                    "id": user_id,
                    "name": user_id,
                    "segment": row.get("segment_name", f"세그먼트 {cluster}"),
                    "probability": prob,
                    "last_active": None,
                    "factors": user_factors if user_factors else None,
                })
        # churn_probability 컬럼이 없으면 빈 리스트

        # ========================================
        # 매출/참여도 예측 - 실제 데이터 기반
        # ========================================
        revenue_data = None
        engagement_data = None

        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            recent = st.DAILY_METRICS_DF.tail(days)
            gmv_col = "total_gmv" if "total_gmv" in recent.columns else None
            shops_col = "active_shops" if "active_shops" in recent.columns else None
            orders_col = "total_orders" if "total_orders" in recent.columns else None
            signups_col = "new_signups" if "new_signups" in recent.columns else None

            if gmv_col:
                current_gmv = float(recent[gmv_col].iloc[-1]) if len(recent) > 0 else 0
                prev_gmv = float(recent[gmv_col].iloc[0]) if len(recent) > 1 else current_gmv
                growth = round((current_gmv - prev_gmv) / max(1, prev_gmv) * 100, 1) if prev_gmv > 0 else 0
                active = int(recent[shops_col].mean()) if shops_col else total
                arpu = int(current_gmv / max(1, active))

                # 셀러 매출 분포에서 whale/dolphin/minnow 계산
                whale_count = 0
                dolphin_count = 0
                minnow_count = 0
                if st.SELLER_ANALYTICS_DF is not None and "total_revenue" in st.SELLER_ANALYTICS_DF.columns:
                    rev_col = st.SELLER_ANALYTICS_DF["total_revenue"].dropna()
                    if len(rev_col) > 0:
                        q90 = rev_col.quantile(0.90)
                        q70 = rev_col.quantile(0.70)
                        whale_count = int((rev_col >= q90).sum())
                        dolphin_count = int(((rev_col >= q70) & (rev_col < q90)).sum())
                        minnow_count = int((rev_col < q70).sum())
                revenue_data = {
                    "predicted_monthly": int(current_gmv * 30 / max(1, days)),
                    "growth_rate": growth,
                    "predicted_arpu": arpu,
                    "predicted_arppu": None,
                    "confidence": None,
                    "whale_count": whale_count,
                    "dolphin_count": dolphin_count,
                    "minnow_count": minnow_count,
                }

            if shops_col:
                active_avg = int(recent[shops_col].mean())
                # MAU = 기간 내 유니크 활성 셀러 수 (daily_metrics에서)
                mau = None
                stickiness = None
                if st.DAILY_METRICS_DF is not None and "active_shops" in st.DAILY_METRICS_DF.columns:
                    recent_30 = st.DAILY_METRICS_DF.tail(30)
                    if len(recent_30) > 0:
                        mau = int(recent_30["active_shops"].max())
                        stickiness = int(active_avg / max(1, mau) * 100) if mau else None
                # 세션 데이터 계산
                avg_session = None
                sessions_per_day = None
                if st.DAILY_METRICS_DF is not None:
                    dm = st.DAILY_METRICS_DF.tail(days)
                    if "avg_session_minutes" in dm.columns:
                        avg_session = round(float(dm["avg_session_minutes"].mean()), 1)
                    if "total_sessions" in dm.columns and "active_shops" in dm.columns:
                        avg_shops = dm["active_shops"].mean()
                        sessions_per_day = round(float(dm["total_sessions"].mean() / max(1, avg_shops)), 1)
                engagement_data = {
                    "predicted_dau": active_avg,
                    "predicted_mau": mau,
                    "stickiness": stickiness,
                    "avg_session": avg_session,
                    "sessions_per_day": sessions_per_day,
                }

        return json_sanitize({
            "status": "SUCCESS",
            "model_available": st.SELLER_CHURN_MODEL is not None,
            "shap_available": st.SHAP_EXPLAINER_CHURN is not None,
            "churn": {
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": low_risk_count,
                "predicted_churn_rate": round(high_risk_count / total * 100, 1) if total > 0 else 0,
                "model_accuracy": round(model_accuracy, 1),
                "top_factors": top_factors,
                "high_risk_users": high_risk_users,
            },
            "revenue": revenue_data,
            "engagement": engagement_data,
        })
    except Exception as e:
        st.logger.error(f"이탈 예측 API 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/prediction/churn/user/{user_id}")
def get_user_churn_prediction(user_id: str, user: dict = Depends(verify_credentials)):
    """개별 사용자 이탈 예측 + SHAP 분석"""
    import numpy as np

    if st.SELLER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.SELLER_ANALYTICS_DF.copy()

        # 사용자 찾기 (CSV: seller_id)
        id_col = "seller_id" if "seller_id" in df.columns else "user_id"
        user_row = df[df[id_col] == user_id]
        if user_row.empty:
            return {"status": "FAILED", "error": f"유저 {user_id}를 찾을 수 없습니다."}

        user_row = user_row.iloc[0]

        # 기본 설정
        config = st.CHURN_MODEL_CONFIG or {}
        features = config.get("features", [
            "total_orders", "total_revenue", "product_count",
            "cs_tickets", "refund_rate", "avg_response_time"
        ])
        feature_names_kr = config.get("feature_names_kr", {
            "total_orders": "총 주문 수",
            "total_revenue": "총 매출",
            "product_count": "등록 상품 수",
            "cs_tickets": "CS 문의 수",
            "refund_rate": "환불률",
            "avg_response_time": "평균 응답 시간",
        })

        available_features = [f for f in features if f in df.columns]

        # 모델 없으면 에러
        if st.SELLER_CHURN_MODEL is None:
            return {"status": "FAILED", "error": "이탈 예측 모델이 로드되지 않았습니다."}

        if not available_features:
            return {"status": "FAILED", "error": "필요한 feature가 데이터에 없습니다."}

        # 예측
        user_X = user_row[available_features].values.reshape(1, -1)
        churn_proba = st.SELLER_CHURN_MODEL.predict_proba(user_X)[0, 1]

        # 위험 등급 판정
        if churn_proba >= 0.7:
            risk_level = "high"
            risk_label = "고위험"
        elif churn_proba >= 0.4:
            risk_level = "medium"
            risk_label = "중위험"
        else:
            risk_level = "low"
            risk_label = "저위험"

        # SHAP 분석
        shap_factors = []
        if st.SHAP_EXPLAINER_CHURN is not None:
            try:
                user_shap_raw = st.SHAP_EXPLAINER_CHURN.shap_values(user_X)

                # SHAP 버전에 따른 처리 (train_models.py와 동일한 로직)
                if hasattr(user_shap_raw, 'values'):
                    # shap.Explanation 객체인 경우
                    user_shap = user_shap_raw.values[0]
                elif isinstance(user_shap_raw, list) and len(user_shap_raw) == 2:
                    # 이진 분류에서 [class_0_shap, class_1_shap] 리스트인 경우
                    user_shap = user_shap_raw[1][0]
                elif isinstance(user_shap_raw, np.ndarray):
                    # numpy array인 경우
                    if user_shap_raw.ndim == 3:
                        # (n_samples, n_features, n_classes) 형태
                        user_shap = user_shap_raw[0, :, 1]
                    elif user_shap_raw.ndim == 2:
                        user_shap = user_shap_raw[0]
                    else:
                        user_shap = user_shap_raw
                else:
                    user_shap = user_shap_raw[0] if hasattr(user_shap_raw, '__getitem__') else user_shap_raw

                user_shap = np.array(user_shap).flatten()

                # 모든 feature에 대한 SHAP 값
                for i, feat in enumerate(available_features):
                    shap_val = float(user_shap[i])
                    feature_val = float(user_row[feat])
                    shap_factors.append({
                        "feature": feat,
                        "feature_kr": feature_names_kr.get(feat, feat),
                        "shap_value": round(shap_val, 4),
                        "feature_value": round(feature_val, 2),
                        "direction": "위험" if shap_val > 0 else "양호",
                    })

                # 절대값 기준 정렬 (영향력 큰 순)
                shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                st.logger.info(f"SHAP 분석 완료: {len(shap_factors)}개 요인")
            except Exception as e:
                import traceback
                st.logger.warning(f"SHAP 분석 실패: {e}")
                st.logger.warning(traceback.format_exc())

        # 유저 정보
        cluster = int(user_row.get("cluster", 0))

        return json_sanitize({
            "status": "SUCCESS",
            "user_id": user_id,
            "user_name": user_id,
            "segment": user_row.get("segment_name", f"세그먼트 {cluster}"),
            "churn_probability": round(float(churn_proba) * 100, 1),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "shap_factors": shap_factors,
            "model_accuracy": round((config.get("model_accuracy") or 0) * 100, 1) if config.get("model_accuracy") else None,
            "shap_available": st.SHAP_EXPLAINER_CHURN is not None,
        })

    except Exception as e:
        st.logger.error(f"개별 유저 이탈 예측 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/cohort/retention")
def get_cohort_retention(days: int = 7, user: dict = Depends(verify_credentials)):
    """코호트 리텐션 분석 - 실제 데이터 기반"""
    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    # days를 주 수로 변환 (최소 1주, 최대 13주)
    weeks = max(1, min(13, days // 7))

    try:
        # COHORT_RETENTION_DF가 있으면 실제 데이터 사용
        # CSV 컬럼: cohort_month, week1, week2, week4, week8, week12
        # 프론트엔드 기대: cohort, week0(=100), week1, week2, week3, week4
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            raw_data = st.COHORT_RETENTION_DF.tail(weeks).to_dict("records")
            cohort_data = []
            for row in raw_data:
                cohort_data.append({
                    "cohort": row.get("cohort_month", row.get("cohort", "unknown")),
                    "week0": 100,
                    "week1": row.get("week1"),
                    "week2": row.get("week2"),
                    "week3": row.get("week4"),  # week4 → week3 슬롯에 매핑
                    "week4": row.get("week8"),   # week8 → week4 슬롯에 매핑
                })
        else:
            # 데이터 없음
            cohort_data = []

        # LTV 코호트 데이터 (DAILY_METRICS_DF 기반 계산, weeks 반영)
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            recent_df = st.DAILY_METRICS_DF.tail(days)
            # total_gmv를 셀러 수로 나눠 ARPU 근사
            gmv_col = "total_gmv" if "total_gmv" in recent_df.columns else "arpu"
            seller_count_col = "active_shops" if "active_shops" in recent_df.columns else None
            if gmv_col in recent_df.columns:
                total_gmv = recent_df[gmv_col].mean() if len(recent_df) > 0 else 500000
                active_cnt = recent_df[seller_count_col].mean() if seller_count_col and seller_count_col in recent_df.columns else 100
                avg_arpu = total_gmv / max(1, active_cnt)
            else:
                avg_arpu = 5000
            user_count = len(st.SELLERS_DF) if st.SELLERS_DF is not None else 1000

            # LTV: seller_analytics의 predicted_ltv로 코호트별 집계
            ltv_by_cohort = []
            if st.SELLER_ANALYTICS_DF is not None and "predicted_ltv" in st.SELLER_ANALYTICS_DF.columns:
                if "join_date" in st.SELLERS_DF.columns if st.SELLERS_DF is not None else False:
                    merged = st.SELLERS_DF[["seller_id", "join_date"]].merge(
                        st.SELLER_ANALYTICS_DF[["seller_id", "predicted_ltv"]], on="seller_id", how="inner"
                    )
                    merged["cohort_month"] = pd.to_datetime(merged["join_date"], errors="coerce").dt.to_period("M").astype(str)
                    cohort_grp = merged.groupby("cohort_month").agg(
                        ltv=("predicted_ltv", "mean"), users=("seller_id", "count")
                    ).reset_index().sort_values("cohort_month", ascending=False).head(6)
                    ltv_by_cohort = [
                        {"cohort": row["cohort_month"], "ltv": int(row["ltv"]), "users": int(row["users"])}
                        for _, row in cohort_grp.iterrows()
                    ]
        else:
            # 데이터 없음
            ltv_by_cohort = []

        # 전환 퍼널 데이터 - CONVERSION_FUNNEL_DF에서 읽기
        conversion = []
        if st.CONVERSION_FUNNEL_DF is not None and len(st.CONVERSION_FUNNEL_DF) > 0:
            conversion = st.CONVERSION_FUNNEL_DF.to_dict("records")

        return json_sanitize({
            "status": "SUCCESS",
            "retention": cohort_data,
            "ltv_by_cohort": ltv_by_cohort,
            "conversion": conversion,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/trend/kpis")
def get_trend_kpis(days: int = 7, user: dict = Depends(verify_credentials)):
    """트렌드 KPI 분석 - 실제 데이터 기반"""
    try:
        # days 파라미터 유효성 검사
        if days not in [7, 30, 90]:
            days = 7

        # DAILY_METRICS_DF가 있으면 실제 데이터 사용
        # CSV 컬럼: date, active_shops, total_gmv, new_signups, total_orders, avg_settlement_time, cs_tickets_open, cs_tickets_resolved, fraud_alerts
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            df = st.DAILY_METRICS_DF.copy()

            # 선택된 기간만큼 데이터 필터링
            recent_df = df.tail(min(days, len(df)))
            # 프론트엔드 차트 기대 키: dau, new_users, sessions
            daily_metrics = []
            for _, r in recent_df.iterrows():
                d_str = str(r.get("date", ""))
                daily_metrics.append({
                    "date": d_str[-5:].replace("-", "/") if len(d_str) >= 5 else d_str,
                    "dau": int(r.get("active_shops", 0)),
                    "new_users": int(r.get("new_signups", 0)),
                    "sessions": int(r.get("total_sessions", r.get("active_shops", 0) * 3)),
                    "active_shops": int(r.get("active_shops", 0)),
                    "total_gmv": int(r.get("total_gmv", 0)),
                    "total_orders": int(r.get("total_orders", 0)),
                })

            # 선택 기간 평균 vs 직전 동일 기간 평균 비교
            n = len(recent_df)
            prev_start = max(0, len(df) - n * 2)
            prev_end = len(df) - n
            prev_df = df.iloc[prev_start:prev_end] if prev_end > prev_start else recent_df

            def _avg(frame, col, default=0):
                if col in frame.columns:
                    return float(frame[col].mean())
                return default

            def _sum(frame, col, default=0):
                if col in frame.columns:
                    return float(frame[col].sum())
                return default

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

            # CS 해결률: 기간 내 총 해결 / 총 접수
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

            # 실제 데이터 기반 선형 추세 예측
            forecast = []
            if st.DAILY_METRICS_DF is not None and "active_shops" in st.DAILY_METRICS_DF.columns:
                recent = st.DAILY_METRICS_DF.tail(14)
                if len(recent) >= 3:
                    vals = recent["active_shops"].values
                    n = len(vals)
                    x = np.arange(n)
                    slope = (np.mean(x * vals) - np.mean(x) * np.mean(vals)) / max(1, np.var(x))
                    intercept = np.mean(vals) - slope * np.mean(x)
                    std_err = np.std(vals - (slope * x + intercept))
                    from datetime import timedelta
                    last_date = pd.to_datetime(recent["date"].iloc[-1], errors="coerce")
                    for i in range(1, 6):
                        pred_val = int(slope * (n + i) + intercept)
                        pred_date = (last_date + timedelta(days=i)).strftime("%m/%d") if last_date is not pd.NaT else f"D+{i}"
                        forecast.append({
                            "date": pred_date,
                            "predicted_dau": max(0, pred_val),
                            "lower": max(0, int(pred_val - 1.5 * std_err)),
                            "upper": int(pred_val + 1.5 * std_err),
                        })
        else:
            # 데이터 없음
            return {"status": "FAILED", "error": "일별 지표 데이터가 없습니다."}

        # 상관관계 - DAILY_METRICS_DF 수치 컬럼 간 피어슨 상관계수
        correlation = []
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 7:
            corr_cols = ["active_shops", "total_gmv", "total_orders", "new_signups"]
            corr_labels = {"active_shops": "활성 쇼핑몰", "total_gmv": "GMV", "total_orders": "주문수", "new_signups": "신규가입"}
            avail = [c for c in corr_cols if c in st.DAILY_METRICS_DF.columns]
            if len(avail) >= 2:
                corr_matrix = st.DAILY_METRICS_DF[avail].corr()
                for i in range(len(avail)):
                    for j in range(i + 1, len(avail)):
                        correlation.append({
                            "var1": corr_labels.get(avail[i], avail[i]),
                            "var2": corr_labels.get(avail[j], avail[j]),
                            "correlation": round(float(corr_matrix.iloc[i, j]), 3),
                        })

        return json_sanitize({
            "status": "SUCCESS",
            "kpis": kpis,
            "daily_metrics": daily_metrics,
            "correlation": correlation,
            "forecast": forecast,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/correlation")
def get_correlation_analysis(user: dict = Depends(verify_credentials)):
    """지표 상관관계 분석 - DAILY_METRICS_DF 기반"""
    correlation = []
    if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 7:
        corr_cols = ["active_shops", "total_gmv", "total_orders", "new_signups"]
        corr_labels = {"active_shops": "활성 쇼핑몰", "total_gmv": "GMV", "total_orders": "주문수", "new_signups": "신규가입"}
        avail = [c for c in corr_cols if c in st.DAILY_METRICS_DF.columns]
        if len(avail) >= 2:
            corr_matrix = st.DAILY_METRICS_DF[avail].corr()
            for i in range(len(avail)):
                for j in range(i + 1, len(avail)):
                    correlation.append({
                        "var1": corr_labels.get(avail[i], avail[i]),
                        "var2": corr_labels.get(avail[j], avail[j]),
                        "correlation": round(float(corr_matrix.iloc[i, j]), 3),
                    })
    return json_sanitize({
        "status": "SUCCESS",
        "correlation": correlation,
    })


@router.get("/stats/summary")
def get_summary_stats(days: int = 7, user: dict = Depends(verify_credentials)):
    """통계 요약 (분석 패널용) - CAFE24 데이터"""
    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    summary = {
        "status": "SUCCESS",
        "days": days,  # 선택된 기간 반환
        "shops_count": len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0,
        "categories_count": len(st.CATEGORIES_DF) if st.CATEGORIES_DF is not None else 0,
        "sellers_count": len(st.SELLERS_DF) if st.SELLERS_DF is not None else 0,
        "cs_stats_count": len(st.CS_STATS_DF) if st.CS_STATS_DF is not None else 0,
        "operation_logs_count": len(st.OPERATION_LOGS_DF) if st.OPERATION_LOGS_DF is not None else 0,
    }

    # 기간별 활성 유저 수 계산 (USER_ACTIVITY_DF 기반)
    if st.SELLER_ACTIVITY_DF is not None and len(st.SELLER_ACTIVITY_DF) > 0:
        try:
            activity_df = st.SELLER_ACTIVITY_DF.tail(100 * days)  # 최근 N일 활동
            active_users = activity_df["seller_id"].nunique()
            summary["active_users_in_period"] = active_users
            summary["active_user_ratio"] = round(active_users / summary["sellers_count"] * 100, 1) if summary["sellers_count"] > 0 else 0
        except Exception:
            pass

    # CS 품질 평균
    if st.CS_STATS_DF is not None and "avg_quality" in st.CS_STATS_DF.columns:
        summary["avg_cs_quality"] = round(float(st.CS_STATS_DF["avg_quality"].mean()), 1)

    # 플랜별 쇼핑몰 분포
    if st.SHOPS_DF is not None and "plan_tier" in st.SHOPS_DF.columns:
        summary["plan_tier_stats"] = st.SHOPS_DF["plan_tier"].value_counts().to_dict()

    # 유저 세그먼트 분포 - CSV segment_name 컬럼 사용
    if st.SELLER_ANALYTICS_DF is not None and "segment_name" in st.SELLER_ANALYTICS_DF.columns:
        raw_segments = st.SELLER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        seg_name_map = st.SELLER_ANALYTICS_DF.drop_duplicates("cluster").set_index("cluster")["segment_name"].to_dict()
        summary["user_segments"] = {
            seg_name_map.get(k, f"세그먼트 {k}"): v for k, v in raw_segments.items()
        }
    elif st.SELLER_ANALYTICS_DF is not None and "cluster" in st.SELLER_ANALYTICS_DF.columns:
        raw_segments = st.SELLER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        summary["user_segments"] = {f"세그먼트 {k}": v for k, v in raw_segments.items()}

        # USER_ACTIVITY_DF가 있으면 기간별 세그먼트 활동 지표 계산
        if st.SELLER_ACTIVITY_DF is not None and len(st.SELLER_ACTIVITY_DF) > 0:
            try:
                activity_df = st.SELLER_ACTIVITY_DF.copy()
                # 최근 N일 활동만 필터링
                activity_df = activity_df.tail(100 * days)  # 100명 유저 * N일

                # 유저별 활동 집계 (CSV: revenue, orders_processed, cs_handled)
                rev_col = "revenue" if "revenue" in activity_df.columns else "daily_revenue"
                ord_col = "orders_processed" if "orders_processed" in activity_df.columns else "daily_orders"
                cs_col = "cs_handled" if "cs_handled" in activity_df.columns else "cs_tickets"
                user_activity = activity_df.groupby("seller_id").agg({
                    rev_col: "sum",
                    ord_col: "sum",
                    cs_col: "sum",
                }).reset_index()
                user_activity.columns = ["seller_id", "daily_revenue", "daily_orders", "cs_tickets"]

                # 세그먼트 정보 조인
                user_activity = user_activity.merge(
                    st.SELLER_ANALYTICS_DF[["seller_id", "cluster"]],
                    on="seller_id",
                    how="left"
                )

                # 세그먼트별 지표 계산
                segment_metrics = {}
                seg_name_map = st.SELLER_ANALYTICS_DF.drop_duplicates("cluster").set_index("cluster")["segment_name"].to_dict() if "segment_name" in st.SELLER_ANALYTICS_DF.columns else {}
                analytics_df = st.SELLER_ANALYTICS_DF
                for cluster in raw_segments.keys():
                    name = seg_name_map.get(cluster, f"세그먼트 {cluster}")
                    seg_data = user_activity[user_activity["cluster"] == cluster]
                    seg_analytics = analytics_df[analytics_df["cluster"] == cluster]
                    cnt = int(raw_segments.get(cluster, 0))
                    # seller_analytics에서 직접 평균 계산 (더 정확)
                    avg_rev = int(seg_analytics["total_revenue"].mean()) if not seg_analytics.empty and "total_revenue" in seg_analytics.columns else 0
                    avg_prod = int(seg_analytics["product_count"].mean()) if not seg_analytics.empty and "product_count" in seg_analytics.columns else 0
                    avg_ord = int(seg_analytics["total_orders"].mean()) if not seg_analytics.empty and "total_orders" in seg_analytics.columns else 0
                    # 리텐션: 활성 셀러 비율
                    if not seg_analytics.empty and "churn_probability" in seg_analytics.columns:
                        active_ratio = (seg_analytics["churn_probability"] < 0.5).sum() / len(seg_analytics) * 100
                        retention = int(active_ratio)
                    else:
                        retention = 0
                    segment_metrics[name] = {
                        "count": cnt,
                        "avg_monthly_revenue": avg_rev,
                        "avg_product_count": avg_prod,
                        "avg_order_count": avg_ord,
                        "retention": retention,
                    }
                summary["segment_metrics"] = segment_metrics
            except Exception as e:
                st.logger.warning(f"세그먼트 지표 계산 실패: {e}")

    # 카테고리별 쇼핑몰 수
    if st.SHOPS_DF is not None and "category" in st.SHOPS_DF.columns:
        summary["category_shops"] = st.SHOPS_DF["category"].value_counts().to_dict()

    # CS 카테고리 분포 및 상세 통계
    # CSV 컬럼: category, total_tickets, avg_resolution_hours, satisfaction_score
    # 프론트엔드 기대: lang_name/category, total_count, avg_quality, pending_count
    if st.CS_STATS_DF is not None and len(st.CS_STATS_DF) > 0:
        stats_list = []
        for _, row in st.CS_STATS_DF.iterrows():
            stats_list.append({
                "category": str(row.get("category", "기타")),
                "lang_name": str(row.get("category", "기타")),
                "total_count": int(row.get("total_tickets", 0)),
                "avg_quality": float(row.get("satisfaction_score", 0)),
                "avg_resolution_hours": float(row.get("avg_resolution_hours", 0)),
                "pending_count": 0,
            })
        summary["cs_stats_detail"] = stats_list

    # 일별 활성 유저 트렌드 (최근 7일)
    date_col_logs = "event_date" if st.OPERATION_LOGS_DF is not None and "event_date" in st.OPERATION_LOGS_DF.columns else "timestamp"
    if st.OPERATION_LOGS_DF is not None and date_col_logs in st.OPERATION_LOGS_DF.columns:
        try:
            import pandas as pd
            df = st.OPERATION_LOGS_DF.copy()
            df["date"] = pd.to_datetime(df[date_col_logs], errors="coerce").dt.strftime("%m/%d")
            daily = df.groupby("date")["seller_id"].nunique().tail(7)
            # DAILY_METRICS_DF에서 new_signups 매핑 구축
            new_signups_map = {}
            if st.DAILY_METRICS_DF is not None and "new_signups" in st.DAILY_METRICS_DF.columns and "date" in st.DAILY_METRICS_DF.columns:
                dm = st.DAILY_METRICS_DF.copy()
                dm["_date_key"] = pd.to_datetime(dm["date"], errors="coerce").dt.strftime("%m/%d")
                new_signups_map = dict(zip(dm["_date_key"], dm["new_signups"].fillna(0).astype(int)))
            summary["daily_trend"] = [
                {"date": date, "active_users": int(count), "new_users": int(new_signups_map.get(date, 0))}
                for date, count in daily.items()
            ]
        except Exception:
            pass

    return json_sanitize(summary)


# ============================================================
# RAG (검색)
# ============================================================
@router.post("/rag/search")
def search_rag(req: RagRequest, user: dict = Depends(verify_credentials)):
    return tool_rag_search(req.query, top_k=req.top_k, api_key=req.api_key)


@router.post("/rag/search/hybrid")
def search_rag_hybrid(req: HybridSearchRequest, user: dict = Depends(verify_credentials)):
    """
    고급 RAG 검색 (Hybrid Search)
    - BM25 (키워드) + Vector (의미) 조합
    - Cross-Encoder Reranking (선택)
    - Knowledge Graph 보강 (선택)
    """
    return rag_search_hybrid(
        query=req.query,
        top_k=req.top_k,
        api_key=req.api_key,
        use_reranking=req.use_reranking,
        use_kg=req.use_kg
    )


@router.get("/rag/status")
def rag_status(user: dict = Depends(verify_credentials)):
    with st.RAG_LOCK:
        return {
            "status": "SUCCESS",
            "rag_ready": bool(st.RAG_STORE.get("ready")),
            "docs_dir": st.RAG_DOCS_DIR,
            "faiss_dir": st.RAG_FAISS_DIR,
            "embed_model": st.RAG_EMBED_MODEL,
            "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0),
            "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0),
            "hash": safe_str(st.RAG_STORE.get("hash", "")),
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or 0.0),
            "error": safe_str(st.RAG_STORE.get("error", "")),
            # Advanced RAG Features
            "bm25_available": BM25_AVAILABLE,
            "bm25_ready": bool(st.RAG_STORE.get("bm25_ready")),
            "reranker_available": False,  # 비활성화
            "kg_ready": False,  # 비활성화
            "kg_entities_count": 0,
            "kg_relations_count": 0,
        }


@router.post("/rag/reload")
def rag_reload(req: RagReloadRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    try:
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        if not k:
            return {"status": "FAILED", "error": "OpenAI API Key가 설정되지 않았습니다."}

        rag_build_or_load_index(api_key=k, force_rebuild=bool(req.force))

        with st.RAG_LOCK:
            ok = bool(st.RAG_STORE.get("ready"))
            err = safe_str(st.RAG_STORE.get("error", ""))
            return {
                "status": "SUCCESS" if ok else "FAILED",
                "rag_ready": ok,
                "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0),
                "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0),
                "hash": safe_str(st.RAG_STORE.get("hash", "")),
                "error": err if err else ("인덱스 빌드 실패" if not ok else ""),
                "embed_model": st.RAG_EMBED_MODEL,
            }
    except Exception as e:
        st.logger.exception("RAG 재빌드 실패")
        return {"status": "FAILED", "error": f"RAG 재빌드 실패: {safe_str(e)}"}


@router.post("/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    api_key: str = "",
    skip_reindex: bool = False,
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(verify_credentials),
):
    """
    RAG 문서 업로드.
    skip_reindex=True로 설정하면 인덱스 재빌드를 건너뜁니다 (배치 업로드용).
    """
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in st.RAG_ALLOWED_EXTS:
            return {"status": "FAILED", "error": f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(st.RAG_ALLOWED_EXTS)}"}

        MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "FAILED", "error": "파일 크기는 15MB를 초과할 수 없습니다."}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(st.RAG_DOCS_DIR, safe_filename)

        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)

        # 백그라운드에서 인덱스 재빌드 (skip_reindex가 False일 때만)
        if not skip_reindex:
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k and background_tasks:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)

        return {
            "status": "SUCCESS",
            "message": "파일이 업로드되었습니다." + ("" if skip_reindex else " 인덱스 재빌드 중..."),
            "filename": safe_filename,
            "original_filename": filename,
            "size": len(contents),
            "path": os.path.relpath(file_path, st.BASE_DIR),
            "reindex_skipped": skip_reindex,
        }
    except Exception as e:
        st.logger.exception("파일 업로드 실패")
        return {"status": "FAILED", "error": f"파일 업로드 실패: {safe_str(e)}"}


@router.get("/rag/files")
def list_rag_files(user: dict = Depends(verify_credentials)):
    try:
        files_info = []
        paths = _rag_list_files()

        for p in paths:
            try:
                stat = os.stat(p)
                rel_path = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
                files_info.append({
                    "filename": os.path.basename(p),
                    "path": rel_path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "ext": os.path.splitext(p)[1].lower(),
                })
            except Exception:
                continue

        return {"status": "SUCCESS", "files": files_info, "total": len(files_info)}
    except Exception as e:
        st.logger.exception("파일 목록 조회 실패")
        return {"status": "FAILED", "error": f"파일 목록 조회 실패: {safe_str(e)}"}


@router.post("/rag/delete")
def delete_rag_file(
    req: DeleteFileRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_credentials)
):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    try:
        filename = os.path.basename(req.filename)
        file_path = os.path.join(st.RAG_DOCS_DIR, filename)

        if not file_path.startswith(os.path.abspath(st.RAG_DOCS_DIR)):
            return {"status": "FAILED", "error": "잘못된 파일 경로입니다."}

        if not os.path.exists(file_path):
            return {"status": "FAILED", "error": "파일을 찾을 수 없습니다."}

        os.remove(file_path)

        # skip_reindex=True면 재빌드 건너뛰기 (다중 삭제 시 마지막에 한 번만 재빌드)
        if not req.skip_reindex:
            k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
            if k:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)
            return {"status": "SUCCESS", "message": "파일이 삭제되었습니다. 인덱스 재빌드 중...", "filename": filename}

        return {"status": "SUCCESS", "message": "파일이 삭제되었습니다.", "filename": filename}
    except Exception as e:
        st.logger.exception("파일 삭제 실패")
        return {"status": "FAILED", "error": f"파일 삭제 실패: {safe_str(e)}"}


# ============================================================
# LightRAG (듀얼 레벨 검색)
# ============================================================
@router.get("/lightrag/status")
def lightrag_status(user: dict = Depends(verify_credentials)):
    """LightRAG 상태 조회"""
    status = get_lightrag_status()
    return {"status": "SUCCESS", **status}


@router.post("/lightrag/build")
def lightrag_build(
    req: LightRagBuildRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_credentials)
):
    """
    LightRAG 지식 그래프 빌드

    기존 RAG 문서(rag_docs/)에서 경량 지식 그래프를 구축합니다.
    - 99% 토큰 절감 (vs Microsoft GraphRAG)
    - 엔티티 및 테마 추출
    - 듀얼 레벨 검색 지원 (local/global/hybrid)
    """
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    if not LIGHTRAG_AVAILABLE:
        return {
            "status": "FAILED",
            "error": "LightRAG가 설치되지 않았습니다. pip install lightrag-hku"
        }

    # 백그라운드에서 빌드
    background_tasks.add_task(build_lightrag_from_rag_docs, req.force_rebuild)

    return {
        "status": "SUCCESS",
        "message": "LightRAG 빌드가 시작되었습니다.",
    }


@router.post("/lightrag/search")
def lightrag_search(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    LightRAG 검색 (듀얼 레벨)

    검색 모드:
    - naive: 기본 검색 (컨텍스트만)
    - local: Low-level 검색 (엔티티 중심) - 구체적인 질문에 적합
    - global: High-level 검색 (테마 중심) - 추상적인 질문에 적합
    - hybrid: local + global 조합 (권장)
    """
    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    try:
        result = lightrag_search_sync(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        st.logger.exception("LightRAG 검색 실패")
        return {"status": "FAILED", "error": f"LightRAG 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/search-dual")
def lightrag_search_dual(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    LightRAG 듀얼 검색 (Low-level + High-level 결과 모두 반환)

    Low-level (local): 구체적인 엔티티 정보
    - 예: "S0001 쇼핑몰 정보는?" → 특정 쇼핑몰 엔티티 검색

    High-level (global): 추상적인 테마/개념
    - 예: "카페24 정산 정책이 뭐야?" → 플랫폼 테마 검색
    """
    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    try:
        result = lightrag_search_dual_sync(
            query=req.query,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        st.logger.exception("LightRAG 듀얼 검색 실패")
        return {"status": "FAILED", "error": f"LightRAG 듀얼 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/clear")
def lightrag_clear_endpoint(user: dict = Depends(verify_credentials)):
    """LightRAG 초기화"""
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    result = clear_lightrag()
    return result


# ============================================================
# K²RAG (KeyKnowledgeRAG - 2025)
# 논문: https://arxiv.org/abs/2507.07695
# 특징: KG + Hybrid Search + Summarization + Sub-question Pipeline
# ============================================================

@router.get("/k2rag/status")
def k2rag_status(user: dict = Depends(verify_credentials)):
    """
    K²RAG 상태 조회

    Returns:
        - initialized: 초기화 여부
        - chunks_count: 인덱싱된 청크 수
        - has_dense_store: Dense Vector Store 유무
        - has_sparse_store: Sparse Vector Store 유무
        - has_knowledge_graph: Knowledge Graph 유무
        - summarizer_available: Longformer 요약기 사용 가능 여부
        - config: 현재 설정
    """
    return k2rag_get_status()


@router.post("/k2rag/search")
def k2rag_search_endpoint(req: K2RagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    K²RAG 검색 (KeyKnowledgeRAG Pipeline)

    Pipeline:
    1. Knowledge Graph 검색 → 관련 토픽 추출
    2. KG 결과 요약 (Longformer)
    3. 서브 질문 생성 (KG 요약 청킹)
    4. 각 서브 질문에 Hybrid Search (80% Dense + 20% Sparse)
    5. 서브 답변 생성 및 요약
    6. 최종 답변 생성

    Args:
        query: 검색 쿼리
        top_k: 검색 결과 수 (기본값: 10)
        use_kg: Knowledge Graph 사용 여부 (기본값: True)
        use_summary: 요약 사용 여부 (기본값: True)

    Returns:
        - status: SUCCESS/FAILED
        - answer: 최종 답변
        - context: 사용된 컨텍스트
        - kg_results: KG 검색 결과
        - sub_answers: 서브 질문/답변 목록
        - elapsed_ms: 처리 시간 (ms)
    """
    try:
        result = k2rag_search_sync(
            query=req.query,
            top_k=req.top_k,
            use_kg=req.use_kg,
            use_summary=req.use_summary
        )
        return result
    except Exception as e:
        st.logger.exception("K2RAG 검색 실패")
        return {"status": "FAILED", "error": f"K2RAG 검색 실패: {safe_str(e)}"}


@router.post("/k2rag/config")
def k2rag_config_endpoint(req: K2RagConfigRequest, user: dict = Depends(verify_credentials)):
    """
    K²RAG 설정 업데이트

    Args:
        hybrid_lambda: Hybrid Search 가중치 (0.0-1.0, 기본값: 0.8 = 80% Dense)
        top_k: 검색 결과 수 (기본값: 10)
        use_summarization: 요약 사용 여부
        use_knowledge_graph: KG 사용 여부
        llm_model: LLM 모델명 (기본값: gpt-4o-mini)

    Returns:
        - status: SUCCESS
        - config: 업데이트된 설정
    """
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    config_updates = {}
    if req.hybrid_lambda is not None:
        config_updates["hybrid_lambda"] = req.hybrid_lambda
    if req.top_k is not None:
        config_updates["top_k"] = req.top_k
    if req.use_summarization is not None:
        config_updates["use_summarization"] = req.use_summarization
    if req.use_knowledge_graph is not None:
        config_updates["use_knowledge_graph"] = req.use_knowledge_graph
    if req.llm_model is not None:
        config_updates["llm_model"] = req.llm_model

    return k2rag_update_config(config_updates)


@router.post("/k2rag/load")
def k2rag_load_endpoint(user: dict = Depends(verify_credentials)):
    """
    기존 RAG 데이터를 K²RAG에 로드

    service.py의 FAISS, BM25, Knowledge Graph 데이터를 K²RAG에서 사용할 수 있도록 로드
    """
    try:
        success = k2rag_load_existing()
        if success:
            return {
                "status": "SUCCESS",
                "message": "기존 RAG 데이터가 K²RAG에 로드되었습니다.",
                "state": k2rag_get_status()
            }
        else:
            return {
                "status": "PARTIAL",
                "message": "일부 데이터만 로드되었습니다. 상태를 확인하세요.",
                "state": k2rag_get_status()
            }
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.post("/k2rag/summarize")
def k2rag_summarize_endpoint(
    text: str = Body(..., embed=True),
    max_length: int = Body(300, embed=True),
    user: dict = Depends(verify_credentials)
):
    """
    텍스트 요약 (Longformer LED 모델)

    K²RAG의 핵심 요약 기능을 단독으로 사용
    - 긴 텍스트 처리에 최적화 (최대 4096 토큰)
    - GPU 지원

    Args:
        text: 요약할 텍스트
        max_length: 최대 요약 길이 (기본값: 300)

    Returns:
        - status: SUCCESS
        - summary: 요약된 텍스트
        - original_length: 원본 길이
        - summary_length: 요약 길이
    """
    try:
        summary = k2rag_summarize(text, max_length=max_length)
        return {
            "status": "SUCCESS",
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "reduction_rate": round((1 - len(summary) / len(text)) * 100, 1) if text else 0
        }
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# OCR (이미지 → 텍스트 추출 → RAG 연동)
# ============================================================
OCR_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


@router.post("/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    api_key: str = "",
    save_to_rag: bool = True,
    user: dict = Depends(verify_credentials),
):
    """이미지에서 텍스트 추출 (EasyOCR) + RAG 연동"""
    global OCR_READER

    if not OCR_AVAILABLE:
        return {"status": "FAILED", "error": "OCR 라이브러리(easyocr)가 설치되지 않았습니다. pip install easyocr"}

    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in OCR_ALLOWED_EXTS:
            return {"status": "FAILED", "error": f"지원하지 않는 이미지 형식입니다. 허용된 형식: {', '.join(OCR_ALLOWED_EXTS)}"}

        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "FAILED", "error": "파일 크기는 20MB를 초과할 수 없습니다."}

        # EasyOCR Reader 초기화 (Lazy loading - 첫 호출시만)
        if OCR_READER is None:
            st.logger.info("OCR_INIT: EasyOCR Reader 초기화 중...")
            OCR_READER = easyocr.Reader(['ko', 'en'], gpu=False)
            st.logger.info("OCR_INIT: EasyOCR Reader 초기화 완료")

        # OCR 수행
        result_list = OCR_READER.readtext(contents)
        extracted_text = "\n".join([text for _, text, _ in result_list])
        extracted_text = extracted_text.strip()

        if not extracted_text:
            return {"status": "FAILED", "error": "이미지에서 텍스트를 추출할 수 없습니다."}

        result = {
            "status": "SUCCESS",
            "original_filename": filename,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
        }

        # RAG에 저장
        if save_to_rag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"{timestamp}_ocr_{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(st.RAG_DOCS_DIR, txt_filename)

            os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR 추출 문서]\n")
                f.write(f"원본 파일: {filename}\n")
                f.write(f"추출 일시: {datetime.now().isoformat()}\n")
                f.write(f"{'='*50}\n\n")
                f.write(extracted_text)

            # RAG 인덱스 재빌드
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k:
                rag_build_or_load_index(api_key=k, force_rebuild=True)

            result["saved_to_rag"] = True
            result["rag_filename"] = txt_filename
            result["message"] = "텍스트가 추출되어 RAG에 저장되었습니다."
        else:
            result["saved_to_rag"] = False
            result["message"] = "텍스트가 추출되었습니다."

        st.logger.info(f"OCR_EXTRACT file={filename} text_len={len(extracted_text)} saved_to_rag={save_to_rag}")
        return result

    except Exception as e:
        st.logger.exception("OCR 추출 실패")
        return {"status": "FAILED", "error": f"OCR 추출 실패: {safe_str(e)}"}


@router.get("/ocr/status")
def ocr_status(user: dict = Depends(verify_credentials)):
    """OCR 기능 상태 확인"""
    easyocr_version = None
    reader_initialized = False

    if OCR_AVAILABLE:
        try:
            easyocr_version = easyocr.__version__
            reader_initialized = OCR_READER is not None
        except Exception:
            pass

    return {
        "status": "SUCCESS",
        "ocr_available": OCR_AVAILABLE,
        "library": "EasyOCR",
        "version": easyocr_version,
        "reader_initialized": reader_initialized,
        "supported_formats": list(OCR_ALLOWED_EXTS),
        "supported_languages": ["ko", "en"],
    }


# ============================================================
# 에이전트 (동기/스트리밍)
# ============================================================
@router.post("/agent/chat")
def agent_chat(req: AgentRequest, user: dict = Depends(verify_credentials)):
    out = run_agent(req, username=user["username"])
    if isinstance(out, dict) and "status" not in out:
        out["status"] = "SUCCESS"
    return out


@router.post("/agent/memory/clear")
def clear_agent_memory(user: dict = Depends(verify_credentials)):
    clear_memory(user["username"])
    return {"status": "SUCCESS", "message": "메모리 초기화 완료"}


@router.post("/agent/stream")
async def agent_stream(req: AgentRequest, request: Request, user: dict = Depends(verify_credentials)):
    """
    LangGraph 기반 스트리밍 에이전트 (LangChain 1.x).
    - create_react_agent로 도구 호출
    - astream_events로 실시간 스트리밍
    """
    st.logger.info(
        "STREAM_REQ headers_auth=%s origin=%s ua=%s",
        request.headers.get("authorization"),
        request.headers.get("origin"),
        request.headers.get("user-agent"),
    )
    username = user["username"]

    async def gen():
        tool_calls_log = []
        final_buf = []

        try:
            from langgraph.prebuilt import create_react_agent
            from langchain_openai import ChatOpenAI
            from agent.tool_schemas import ALL_TOOLS
            from agent.router import classify_and_get_tools, IntentCategory

            user_text = safe_str(req.user_input)
            rag_mode = req.rag_mode or "auto"

            # ========== LLM Router: 의도 분류 → 도구 필터링 ==========
            api_key = pick_api_key(req.api_key)
            category, allowed_tool_names = classify_and_get_tools(
                user_text,
                api_key,
                use_llm_fallback=False,  # 스트리밍에서는 속도를 위해 키워드만 사용
            )

            st.logger.info(
                "STREAM_ROUTER category=%s allowed_tools=%s",
                category.value, allowed_tool_names,
            )

            # 카테고리에 해당하는 도구만 필터링
            if allowed_tool_names:
                tools = [t for t in ALL_TOOLS if t.name in allowed_tool_names]
                # 도구가 없으면 전체 사용 (fallback)
                if not tools:
                    tools = ALL_TOOLS
            elif category == IntentCategory.GENERAL:
                # 일반 대화: 도구 없음 (빠른 응답)
                tools = []
            else:
                tools = ALL_TOOLS

            # rag_mode에 따라 RAG 도구 추가 필터링 (플랫폼 카테고리인 경우만)
            if category == IntentCategory.PLATFORM:
                if rag_mode == "rag":
                    tools = [t for t in tools if t.name != "search_platform_lightrag"]
                elif rag_mode == "lightrag":
                    tools = [t for t in tools if t.name != "search_platform_docs"]
                elif rag_mode == "k2rag":
                    # K²RAG 모드: 기존 도구 제거 (K²RAG API 직접 사용)
                    tools = [t for t in tools if t.name not in ["search_platform_docs", "search_platform_lightrag"]]
                # auto: 둘 다 유지

            st.logger.info(
                "AGENT_TOOLS rag_mode=%s category=%s tools=%d (%s)",
                rag_mode, category.value, len(tools),
                [t.name for t in tools] if len(tools) <= 10 else f"{len(tools)} tools",
            )
            api_key = pick_api_key(req.api_key)

            if not api_key:
                msg = "처리 오류: OpenAI API Key가 없습니다. 환경변수 OPENAI_API_KEY 또는 요청의 api_key를 설정하세요."
                yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": []})
                return

            # ========== 1. RAG 검색 (PLATFORM/GENERAL만) ==========
            # PLATFORM: 미리 RAG 검색 → 컨텍스트 포함 → 도구 제거 (이중 검색 방지)
            # GENERAL: 미리 RAG 검색 → 컨텍스트 제공
            rag_context = ""
            simple_patterns = ["안녕", "고마워", "감사", "뭐해", "ㅎㅎ", "ㅋㅋ", "네", "응", "오케이", "bye", "hi", "hello", "thanks"]
            is_simple = any(p in user_text.lower() for p in simple_patterns) and len(user_text) < 20

            # PLATFORM/GENERAL/SHOP 미리 RAG 검색 (쇼핑몰 정보도 RAG에서 검색)
            skip_rag = category not in [IntentCategory.PLATFORM, IntentCategory.GENERAL, IntentCategory.SHOP]
            if skip_rag:
                st.logger.info("SKIP_RAG category=%s (not platform/general/shop)", category.value)

            if not is_simple and not skip_rag:
                try:
                    import time as _time
                    _rag_start = _time.time()

                    # rag_mode에 따라 적절한 RAG 검색 수행
                    if rag_mode == "lightrag":
                        # LightRAG 검색 - 설정은 state.LIGHTRAG_CONFIG에서 관리
                        rag_out = lightrag_search_sync(user_text, mode="hybrid")  # top_k는 state.py에서
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("LIGHTRAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            context_text = rag_out.get("context", "")
                            if context_text:
                                # 도구 결과에 context 일부 포함 (프론트엔드 표시용, 1000자 제한)
                                context_preview = context_text[:1000] + ("..." if len(context_text) > 1000 else "")
                                tool_calls_log.append({"tool": "lightrag_search", "args": {"query": user_text, "mode": "hybrid"}, "result": {"status": "SUCCESS", "context": context_preview, "context_len": len(context_text)}})
                                # 컨텍스트 크기 제한 (state.LIGHTRAG_CONFIG에서 관리)
                                max_chars = st.LIGHTRAG_CONFIG.get("context_max_chars", 1500)
                                rag_context = f"\n\n## 검색된 플랫폼 정보 (LightRAG):\n{context_text[:max_chars]}\n"
                                st.logger.info("LIGHTRAG_SEARCH ok=1 mode=hybrid ctx_len=%d truncated=%d", len(context_text), max_chars)
                                # 이미 검색했으므로 LLM에게 중복 검색 도구 제거 (이중 검색 방지)
                                tools = [t for t in tools if t.name != "search_platform_lightrag"]
                    elif rag_mode == "k2rag":
                        # K²RAG 검색 - 고정밀 검색 (KG 요약 → 서브질문 → 하이브리드 검색)
                        rag_out = k2rag_search_sync(user_text, top_k=10, use_kg=True, use_summary=True)
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("K2RAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            answer = rag_out.get("answer", "")
                            context = rag_out.get("context", "")
                            # answer 또는 context 중 하나라도 있으면 성공
                            if answer or context:
                                tool_calls_log.append({"tool": "k2rag_search", "args": {"query": user_text}, "result": {"status": "SUCCESS", "answer_len": len(answer), "context_len": len(context)}})
                                rag_context = f"\n\n## 검색된 플랫폼 정보 (K²RAG):\n{answer or context[:2000]}\n"
                                st.logger.info("K2RAG_SEARCH ok=1 answer_len=%d context_len=%d", len(answer), len(context))
                                # 이미 검색했으므로 중복 검색 도구 제거
                                tools = [t for t in tools if t.name not in ["search_platform_docs", "search_platform_lightrag"]]
                        else:
                            st.logger.warning("K2RAG_SEARCH_FAIL result=%s", rag_out)
                    else:
                        # 기본 RAG 검색 (rag 또는 auto 모드)
                        rag_out = tool_rag_search(user_text, top_k=st.RAG_DEFAULT_TOPK, api_key=api_key)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            results = rag_out.get("results") or []
                            if results:
                                tool_calls_log.append({"tool": "rag_search", "args": {"query": user_text}, "result": rag_out})
                                rag_context = "\n\n## 검색된 플랫폼 정보 (참고용):\n"
                                for r in results[:5]:
                                    content = r.get("content", "")[:500]
                                    rag_context += f"- {content}\n"
                                st.logger.info("RAG_SEARCH ok=1 results=%d", len(results))
                                # 이미 검색했으므로 LLM에게 중복 검색 도구 제거 (이중 검색 방지)
                                tools = [t for t in tools if t.name != "search_platform_docs"]
                except Exception as _e:
                    st.logger.warning("RAG_SEARCH_FAIL err=%s", safe_str(_e))

            # ========== 2. 시스템 프롬프트 구성 ==========
            base_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
            # RAG 모드에 따른 검색 도구 안내
            rag_tool_info = ""
            if rag_mode == "rag":
                rag_tool_info = "- `search_platform_docs`: 플랫폼 검색 (FAISS + BM25)"
            elif rag_mode == "lightrag":
                rag_tool_info = "- `search_platform_lightrag`: 플랫폼 검색 (LightRAG - 지식그래프 기반)"
            elif rag_mode == "k2rag":
                rag_tool_info = "- K²RAG 모드: 검색이 자동으로 수행됩니다 (KG 요약 → 서브질문 → 하이브리드)"
            else:  # auto
                rag_tool_info = """- `search_platform_docs`: 플랫폼 검색 (FAISS + BM25)
- `search_platform_lightrag`: 플랫폼 검색 (LightRAG - 관계 기반)"""

            system_prompt = base_prompt + f"""

## 도구 사용 규칙

당신은 카페24 이커머스 AI 어시스턴트입니다. 사용자 요청에 적합한 도구를 선택하여 호출하세요.

### 주요 도구:
- `get_shop_info`, `list_shops`: 쇼핑몰 정보 (DB에 있는 쇼핑몰만)
- `get_category_info`, `list_categories`: 카테고리 정보
- `auto_reply_cs`, `check_cs_quality`: CS 관련
- `analyze_seller`, `get_seller_segment`, `detect_fraud`: 셀러 분석
- `predict_seller_churn`: 셀러 이탈 예측 (ML 모델)
- `get_shop_performance`: 쇼핑몰 성과 분석 (ML 모델)
- `predict_shop_revenue`: 쇼핑몰 매출 예측 (ML 모델)
- `optimize_marketing`: 마케팅 예산 최적화 (P-PSO 알고리즘)
- `get_segment_statistics`: 세그먼트별 셀러 통계 (성장형/휴면/우수/파워/관리필요)
- `get_fraud_statistics`: 이상거래/이상 셀러/부정행위 전체 통계
- `get_order_statistics`: 운영 이벤트 통계 (event_type: order_received/payment_settled/refund_processed/cs_ticket/login/marketing_campaign/product_listed/product_updated)
- `get_dashboard_summary`: 대시보드 요약

### 플랫폼 카테고리 검색 도구 (현재 모드: {rag_mode}):
{rag_tool_info}

### 규칙:
1. **쇼핑몰, 정산, 정책 정보** 질문 → 먼저 **RAG 검색 도구**를 사용하세요. (DB에 없는 정보가 많습니다)
2. 사용자 요청에 맞는 도구를 **직접 선택**하세요.
3. 여러 정보가 필요하면 **여러 도구를 동시에 호출**하세요.
4. 도구 결과를 바탕으로 사용자에게 친절하게 답변하세요.
5. 간단한 인사나 대화에는 도구 호출 없이 바로 답변하세요.
6. 플랫폼 정책, 정산, 설정 관련 질문은 **검색 도구를 사용**하세요.
"""
            if rag_context:
                # RAG 컨텍스트가 있으면 엄격하게 참조하도록 지시 (할루시네이션 방지)
                system_prompt += f"""

## 🔍 검색된 플랫폼 정보 (공식 문서)
아래는 사용자 질문에 대해 검색된 **공식 플랫폼 정보**입니다.

{rag_context}

### ⚠️ 답변 규칙 (엄격히 준수)
1. **검색 결과에 명시된 정보만** 사용하세요.
2. 검색 결과를 **확장, 추론, 일반화하지 마세요**.
3. 검색 결과에 없는 내용은 "검색 결과에서 확인되지 않습니다"라고 답하세요.
4. 답변 시 "검색 결과에 따르면" 형식으로 출처를 명시하세요.

❌ 금지: "~일 것입니다", "~로 추정됩니다", "일반적으로~"
✅ 권장: "검색 결과에 따르면 [정확한 인용]입니다."
"""
                st.logger.info("PLATFORM_RAG_CONTEXT_INJECTED len=%d", len(rag_context))

            # ========== 3. LangGraph Agent 생성 ==========
            model_name = req.model or "gpt-4o-mini"

            # tool_choice는 auto (기본값) - ReAct 에이전트에서 required는 무한 루프 유발
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                streaming=True,
                max_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.7,
            )

            # ========== 직접 LLM 응답 모드 ==========
            # 1. GENERAL 카테고리: 도구 없이 대화
            # 2. PLATFORM + RAG 컨텍스트: 에이전트 없이 RAG 기반 답변 (가장 정확)
            use_direct_llm = not tools or (category == IntentCategory.PLATFORM and rag_context)

            if use_direct_llm:
                mode = "PLATFORM_RAG_DIRECT" if category == IntentCategory.PLATFORM else "GENERAL"
                st.logger.info("STREAM_%s_MODE direct LLM response (no agent)", mode)

                # PLATFORM용 LLM (RAG 컨텍스트 엄격 준수)
                if category == IntentCategory.PLATFORM:
                    llm = ChatOpenAI(
                        model=model_name,
                        api_key=api_key,
                        streaming=True,
                        max_tokens=req.max_tokens or 1500,
                        temperature=0.2,  # 낮은 창의성: 검색 결과에 충실하게 답변
                    )

                # 직접 스트리밍 응답 (TTFT 측정)
                _llm_start = _time.time()
                _first_token = True
                async for chunk in llm.astream([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]):
                    if await request.is_disconnected():
                        return
                    content = getattr(chunk, "content", "")
                    if content:
                        if _first_token:
                            _ttft = (_time.time() - _llm_start) * 1000
                            st.logger.info("LLM_TTFT elapsed=%.0fms", _ttft)
                            _first_token = False
                        final_buf.append(content)
                        yield sse_pack("delta", {"delta": content})  # 프론트엔드가 data.delta 기대

                _llm_total = (_time.time() - _llm_start) * 1000
                st.logger.info("LLM_COMPLETE elapsed=%.0fms tokens=%d", _llm_total, len(final_buf))
                full_response = "".join(final_buf)
                append_memory(username, user_text, full_response)
                yield sse_pack("done", {"ok": True, "final": full_response, "tool_calls": tool_calls_log})
                return

            # ========== 에이전트 모드 (PLATFORM 외 카테고리) ==========
            # LangGraph의 create_react_agent 사용 (Router 결과에 따라 필터링된 도구)
            agent = create_react_agent(llm, tools, prompt=system_prompt)

            # ========== 4. astream_events로 실시간 스트리밍 ==========
            current_tool = None

            async for event in agent.astream_events(
                {"messages": [("user", user_text)]},
                version="v2",
                config={"recursion_limit": 10},  # 무한 루프 방지 (기본 25 → 10)
            ):
                if await request.is_disconnected():
                    return

                kind = event.get("event", "")
                data = event.get("data", {})

                # 도구 실행 시작
                if kind == "on_tool_start":
                    tool_name = event.get("name", "도구")
                    tool_input = data.get("input", {})
                    current_tool = tool_name
                    st.logger.info("TOOL_START tool=%s", tool_name)
                    yield sse_pack("tool_start", {"tool": tool_name, "args": tool_input})

                # 도구 실행 완료
                elif kind == "on_tool_end":
                    tool_output = data.get("output", {})
                    # ToolMessage 객체인 경우 content만 추출
                    if hasattr(tool_output, "content"):
                        content = tool_output.content
                        # content가 JSON 문자열이면 파싱, 딕셔너리/리스트면 그대로
                        if isinstance(content, str):
                            try:
                                tool_output = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                tool_output = {"status": "SUCCESS", "data": content}
                        elif isinstance(content, (dict, list)):
                            tool_output = content
                        else:
                            tool_output = {"status": "SUCCESS", "data": safe_str(content)}
                    elif not isinstance(tool_output, (str, dict, list, int, float, bool, type(None))):
                        tool_output = {"status": "SUCCESS", "data": safe_str(tool_output)}
                    tool_calls_log.append({
                        "tool": current_tool or "unknown",
                        "result": tool_output,
                    })
                    st.logger.info("TOOL_END tool=%s", current_tool)
                    yield sse_pack("tool_end", {"tool": current_tool, "status": "SUCCESS"})
                    current_tool = None

                # LLM 토큰 스트리밍
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            final_buf.append(content)
                            yield sse_pack("delta", {"delta": content})

            final_text = "".join(final_buf).strip()
            if not final_text:
                final_text = "요청을 처리했습니다."

            append_memory(username, user_text, final_text)

            st.logger.info("STREAM_COMPLETE user=%s tools_used=%d", username, len(tool_calls_log))
            yield sse_pack("done", {"ok": True, "final": final_text, "tool_calls": tool_calls_log})
            return

        except Exception as e:
            msg = safe_str(e) or "스트리밍 오류"
            st.logger.exception("STREAM_ERROR err=%s", msg)
            try:
                yield sse_pack("error", {"message": msg})
            except Exception:
                pass
            yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": tool_calls_log})
            return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================================================
# MLflow 관련
# ============================================================
@router.get("/mlflow/experiments")
def get_mlflow_experiments(user: dict = Depends(verify_credentials)):
    """MLflow 실험 목록 조회"""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        experiments = client.search_experiments()
        result = []

        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )

            runs_data = []
            for run in runs:
                runs_data.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "params": dict(run.data.params),
                    "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
                    "tags": dict(run.data.tags),
                })

            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "runs": runs_data,
            })

        return {"status": "SUCCESS", "data": result}
    except ImportError:
        return {"status": "FAILED", "error": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "data": []}


@router.get("/mlflow/models")
def get_mlflow_registered_models(user: dict = Depends(verify_credentials)):
    """MLflow Model Registry에서 등록된 모델 목록 조회"""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        registered_models = client.search_registered_models()
        result = []

        for rm in registered_models:
            versions = []
            try:
                all_versions = client.search_model_versions(filter_string=f"name='{rm.name}'")
                for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
                    versions.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                    })
            except Exception:
                for v in rm.latest_versions:
                    versions.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                    })

            result.append({
                "name": rm.name,
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
                "description": rm.description or "",
                "versions": versions,
                "model_type": "registry",
            })

        return {"status": "SUCCESS", "data": result}
    except ImportError:
        return {"status": "FAILED", "error": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 모델 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "data": []}


class ModelSelectRequest(BaseModel):
    model_name: str
    version: str


@router.get("/mlflow/models/selected")
def get_selected_models(user: dict = Depends(verify_credentials)):
    """현재 서버에서 선택/로드된 모델 목록 조회"""
    # 저장된 선택 상태 로드
    st.load_selected_models()

    return {
        "status": "SUCCESS",
        "data": st.SELECTED_MODELS,
        "message": f"{len(st.SELECTED_MODELS)}개 모델이 선택되어 있습니다"
    }


@router.post("/mlflow/models/select")
def select_mlflow_model(req: ModelSelectRequest, user: dict = Depends(verify_credentials)):
    """MLflow 모델 선택/로드 - 실제로 모델을 state에 로드"""

    # 모델 이름 → state 변수 매핑 (한글 이름)
    MODEL_STATE_MAP = {
        "CS응답품질": "CS_QUALITY_MODEL",
        "문의분류": "INQUIRY_CLASSIFICATION_MODEL",
        "셀러세그먼트": "SELLER_SEGMENT_MODEL",
        "이상거래탐지": "FRAUD_DETECTION_MODEL",
        "셀러이탈예측": "SELLER_CHURN_MODEL",
        "매출예측": "REVENUE_PREDICTION_MODEL",
    }

    state_attr = MODEL_STATE_MAP.get(req.model_name)
    if not state_attr:
        return {
            "status": "FAILED",
            "error": f"알 수 없는 모델: {req.model_name}. 지원 모델: {list(MODEL_STATE_MAP.keys())}"
        }

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        try:
            # 모델 버전 정보 조회
            model_version = client.get_model_version(req.model_name, req.version)

            # 실제 모델 로드
            model_uri = f"models:/{req.model_name}/{req.version}"
            st.logger.info(f"모델 로드 시작: {model_uri}")

            loaded_model = mlflow.sklearn.load_model(model_uri)

            # state에 모델 할당
            setattr(st, state_attr, loaded_model)
            st.logger.info(f"모델 로드 완료: st.{state_attr} = {model_uri}")

            # 선택 상태 저장 (영구 보존)
            st.SELECTED_MODELS[req.model_name] = req.version
            st.save_selected_models()

            return {
                "status": "SUCCESS",
                "message": f"{req.model_name} v{req.version} 모델이 로드되었습니다",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                    "stage": model_version.current_stage,
                    "run_id": model_version.run_id,
                    "state_variable": f"st.{state_attr}",
                    "loaded": True,
                }
            }
        except Exception as e:
            st.logger.warning(f"MLflow 모델 로드 실패: {e}")
            return {
                "status": "FAILED",
                "error": f"모델 로드 실패: {safe_str(e)}",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                }
            }

    except ImportError:
        return {
            "status": "FAILED",
            "error": "MLflow가 설치되지 않았습니다. pip install mlflow 로 설치하세요.",
        }
    except Exception as e:
        st.logger.exception("MLflow 모델 선택 실패")
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# 사용자 관리
# ============================================================
@router.get("/users")
def get_users(user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    return {"status": "SUCCESS", "data": [{"아이디": k, "이름": v["name"], "권한": v["role"]} for k, v in st.USERS.items()]}


@router.post("/users")
def create_user(req: UserCreateRequest, user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    if req.user_id in st.USERS:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")
    st.USERS[req.user_id] = {"password": req.password, "role": req.role, "name": req.name}
    return {"status": "SUCCESS", "message": f"{req.name} 추가됨"}


# ============================================================
# 설정
# ============================================================
@router.get("/settings/default")
def get_default_settings(user: dict = Depends(verify_credentials)):
    return {
        "status": "SUCCESS",
        "data": {
            "selectedModel": "gpt-4o-mini",
            "maxTokens": 8000,
            "temperature": 0.1,  # 할루시네이션 방지를 위해 낮게 설정
            "topP": 1.0,
            "presencePenalty": 0.0,
            "frequencyPenalty": 0.0,
            "seed": "",
            "timeoutMs": 30000,
            "retries": 2,
            "stream": True,
            "systemPrompt": st.get_active_system_prompt(),  # 백엔드 중앙 관리 프롬프트
        },
    }


# ============================================================
# 시스템 프롬프트 관리 API (백엔드 중앙 관리)
# ============================================================
@router.get("/settings/prompt")
def get_system_prompt(user: dict = Depends(verify_credentials)):
    """현재 활성 시스템 프롬프트 조회"""
    return {
        "status": "SUCCESS",
        "data": {
            "systemPrompt": st.get_active_system_prompt(),
            "isCustom": st.CUSTOM_SYSTEM_PROMPT is not None and st.CUSTOM_SYSTEM_PROMPT.strip() != "",
            "defaultPrompt": DEFAULT_SYSTEM_PROMPT,
        },
    }


class SystemPromptRequest(BaseModel):
    system_prompt: str = Field(..., alias="systemPrompt")

    class Config:
        populate_by_name = True


class LLMSettingsRequest(BaseModel):
    """LLM 설정 요청 모델"""
    selected_model: str = Field("gpt-4o-mini", alias="selectedModel")
    custom_model: str = Field("", alias="customModel")
    temperature: float = Field(0.7, alias="temperature")
    top_p: float = Field(1.0, alias="topP")
    presence_penalty: float = Field(0.0, alias="presencePenalty")
    frequency_penalty: float = Field(0.0, alias="frequencyPenalty")
    max_tokens: int = Field(8000, alias="maxTokens")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: int = Field(30000, alias="timeoutMs")
    retries: int = Field(2, alias="retries")
    stream: bool = Field(True, alias="stream")

    class Config:
        populate_by_name = True


@router.post("/settings/prompt")
def save_system_prompt(req: SystemPromptRequest, user: dict = Depends(verify_credentials)):
    """시스템 프롬프트 저장 (백엔드에 영구 저장)"""
    # 관리자만 수정 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 수정할 수 있습니다.")

    prompt = req.system_prompt.strip() if req.system_prompt else ""

    if not prompt:
        raise HTTPException(status_code=400, detail="시스템 프롬프트가 비어있습니다.")

    success = st.save_system_prompt(prompt)

    if success:
        return {
            "status": "SUCCESS",
            "message": "시스템 프롬프트가 저장되었습니다.",
            "data": {
                "systemPrompt": st.get_active_system_prompt(),
                "isCustom": True,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="시스템 프롬프트 저장에 실패했습니다.")


@router.post("/settings/prompt/reset")
def reset_system_prompt(user: dict = Depends(verify_credentials)):
    """시스템 프롬프트를 기본값으로 초기화"""
    # 관리자만 초기화 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 초기화할 수 있습니다.")

    success = st.reset_system_prompt()

    if success:
        return {
            "status": "SUCCESS",
            "message": "시스템 프롬프트가 기본값으로 초기화되었습니다.",
            "data": {
                "systemPrompt": DEFAULT_SYSTEM_PROMPT,
                "isCustom": False,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="시스템 프롬프트 초기화에 실패했습니다.")


# ============================================================
# LLM 설정 관리 API (백엔드 중앙 관리)
# ============================================================
@router.get("/settings/llm")
def get_llm_settings(user: dict = Depends(verify_credentials)):
    """현재 LLM 설정 조회"""
    settings = st.get_active_llm_settings()
    is_custom = st.CUSTOM_LLM_SETTINGS is not None
    return {
        "status": "SUCCESS",
        "data": {
            **settings,
            "isCustom": is_custom,
        },
    }


@router.post("/settings/llm")
def save_llm_settings(req: LLMSettingsRequest, user: dict = Depends(verify_credentials)):
    """LLM 설정 저장 (백엔드에 영구 저장)"""
    # 관리자만 수정 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 수정할 수 있습니다.")

    settings_dict = {
        "selectedModel": req.selected_model,
        "customModel": req.custom_model,
        "temperature": req.temperature,
        "topP": req.top_p,
        "presencePenalty": req.presence_penalty,
        "frequencyPenalty": req.frequency_penalty,
        "maxTokens": req.max_tokens,
        "seed": req.seed,
        "timeoutMs": req.timeout_ms,
        "retries": req.retries,
        "stream": req.stream,
    }

    success = st.save_llm_settings(settings_dict)

    if success:
        return {
            "status": "SUCCESS",
            "message": "LLM 설정이 저장되었습니다.",
            "data": {
                **settings_dict,
                "isCustom": True,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="LLM 설정 저장에 실패했습니다.")


@router.post("/settings/llm/reset")
def reset_llm_settings(user: dict = Depends(verify_credentials)):
    """LLM 설정을 기본값으로 초기화"""
    # 관리자만 초기화 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 초기화할 수 있습니다.")

    success = st.reset_llm_settings()

    if success:
        return {
            "status": "SUCCESS",
            "message": "LLM 설정이 기본값으로 초기화되었습니다.",
            "data": {
                **st.DEFAULT_LLM_SETTINGS,
                "isCustom": False,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="LLM 설정 초기화에 실패했습니다.")


# ============================================================
# 내보내기
# ============================================================
@router.get("/export/csv")
def export_csv(user: dict = Depends(verify_credentials)):
    """CS 데이터 CSV 내보내기"""
    output = StringIO()
    export_df = st.OPERATION_LOGS_DF.copy() if st.OPERATION_LOGS_DF is not None else pd.DataFrame()
    export_df.to_csv(output, index=False, encoding="utf-8-sig")
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cafe24_data_{datetime.now().strftime('%Y%m%d')}.csv"},
    )


@router.get("/export/excel")
def export_excel(user: dict = Depends(verify_credentials)):
    """CS 데이터 Excel 내보내기"""
    output = BytesIO()
    export_df = st.OPERATION_LOGS_DF.copy() if st.OPERATION_LOGS_DF is not None else pd.DataFrame()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="OperationLogs")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=cafe24_data_{datetime.now().strftime('%Y%m%d')}.xlsx"},
    )


# ============================================================
# 도구 목록 (에이전트용)
# ============================================================
@router.get("/tools")
def get_available_tools(user: dict = Depends(verify_credentials)):
    """사용 가능한 도구 목록"""
    tools = []
    for name, func in AVAILABLE_TOOLS.items():
        tools.append({
            "name": name,
            "description": func.__doc__ or "",
        })
    return {"status": "SUCCESS", "tools": tools}


# ============================================================
# 마케팅 예산 최적화 (ML + P-PSO)
# ============================================================
@router.get("/marketing/seller/{seller_id}")
def get_marketing_seller_info(seller_id: str, user: dict = Depends(verify_credentials)):
    """마케팅 최적화용 셀러 정보 조회"""
    try:
        if st.SELLERS_DF is None:
            return {"status": "FAILED", "error": "셀러 데이터 없음"}

        sid = seller_id.strip().upper()
        row = st.SELLERS_DF[st.SELLERS_DF["seller_id"].str.upper() == sid]
        if row.empty:
            return {"status": "FAILED", "error": f"셀러 {seller_id}을(를) 찾을 수 없습니다"}

        seller = row.iloc[0]
        # 해당 셀러의 쇼핑몰 정보
        shops = []
        if st.SHOPS_DF is not None:
            seller_shops = st.SHOPS_DF[st.SHOPS_DF.get("seller_id", pd.Series()).str.upper() == sid] if "seller_id" in st.SHOPS_DF.columns else pd.DataFrame()
            if seller_shops.empty and st.SHOP_PERFORMANCE_DF is not None:
                # shop_performance에서 상위 5개
                shops = st.SHOP_PERFORMANCE_DF.head(5).to_dict("records")
            else:
                shops = seller_shops.head(5).to_dict("records")

        data = {
            "seller_id": seller.get("seller_id", sid),
            "total_revenue": float(seller.get("total_revenue", 0)),
            "total_orders": int(seller.get("total_orders", 0)),
            "product_count": int(seller.get("product_count", 0)),
            "shops": shops,
        }
        return json_sanitize({"status": "SUCCESS", "data": data})
    except Exception as e:
        st.logger.exception("마케팅 셀러 정보 조회 실패")
        return {"status": "FAILED", "error": safe_str(e)}


@router.post("/marketing/optimize")
def optimize_marketing_budget(req: MarketingOptimizeRequest, user: dict = Depends(verify_credentials)):
    """
    마케팅 예산 최적화 API

    ML 모델로 채널별 ROI를 예측하고 P-PSO로 최적 예산 배분을 탐색합니다.
    """
    try:
        # 예산 추출 (budget_constraints에서 total이 있으면 사용)
        total_budget = None
        if req.budget_constraints and "total" in req.budget_constraints:
            total_budget = float(req.budget_constraints["total"])

        result = tool_optimize_marketing(
            seller_id=req.seller_id or "SEL0001",
            goal="maximize_roas",
            total_budget=total_budget,
        )

        st.logger.info(
            f"MARKETING_OPTIMIZE user={user['username']} seller={req.seller_id}"
        )

        if result.get("status") == "FAILED":
            return {"status": "FAILED", "error": result.get("error", "최적화 실패")}

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except Exception as e:
        st.logger.exception("마케팅 최적화 실패")
        return {
            "status": "FAILED",
            "error": f"마케팅 최적화 중 오류: {safe_str(e)}",
        }


@router.get("/marketing/status")
def get_marketing_optimizer_status(user: dict = Depends(verify_credentials)):
    """마케팅 최적화 상태 조회"""
    return {
        "status": "SUCCESS",
        "data": {
            "optimizer_available": st.MARKETING_OPTIMIZER_AVAILABLE,
            "shops_loaded": st.SHOPS_DF is not None,
            "shops_count": len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0,
            "optimization_method": "P-PSO (Phasor Particle Swarm Optimization)",
        },
    }

@router.get("/sellers/performance")
async def get_sellers_performance(user: dict = Depends(verify_credentials)):
    """셀러 성과 목록 조회"""
    try:
        if st.SELLERS_DF is None or st.SELLERS_DF.empty:
            return {"status": "ERROR", "message": "셀러 데이터 없음"}

        sellers = []
        for _, row in st.SELLERS_DF.head(100).iterrows():
            sellers.append({
                "id": row.get("seller_id", ""),
                "name": row.get("seller_id", ""),
                "plan_tier": row.get("plan_tier", "Standard"),
                "segment": row.get("segment", "알 수 없음"),
            })

        return {"status": "SUCCESS", "sellers": sellers}
    except Exception as e:
        st.logger.error(f"셀러 목록 조회 오류: {e}")
        return {"status": "ERROR", "message": str(e)}


# ─────────────────────────────────────────────────────────
# Data Guardian Agent — AI 기반 데이터 보호 시스템
# ─────────────────────────────────────────────────────────

import sqlite3
import time as _time

_GUARDIAN_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "guardian.db")

def _guardian_conn():
    conn = sqlite3.connect(_GUARDIAN_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _guardian_init():
    """감사 로그 테이블 + 시드 데이터 초기화"""
    conn = _guardian_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_id TEXT NOT NULL,
        action TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_count INTEGER DEFAULT 0,
        affected_amount REAL DEFAULT 0,
        status TEXT DEFAULT 'executed',
        risk_level TEXT DEFAULT 'LOW',
        agent_reason TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_count INTEGER,
        was_mistake INTEGER DEFAULT 0,
        description TEXT
    )""")
    # 시드 데이터 (없을 때만)
    if c.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0:
        import random
        from datetime import timedelta
        _now = datetime.now()
        _users = ["kim", "park", "lee", "choi", "jung"]
        _tables = ["orders", "payments", "users", "products", "shipments", "logs", "temp_reports"]
        for _ in range(200):
            ts = (_now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 12))).isoformat()
            u = random.choice(_users)
            act = random.choice(["INSERT", "UPDATE", "DELETE"])
            tbl = random.choice(_tables)
            rc = random.randint(1, 8) if act == "DELETE" else random.randint(1, 20)
            amt = rc * random.randint(30000, 120000) if tbl in ("orders", "payments") else 0
            c.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                      (ts, u, act, tbl, rc, amt, "executed", "LOW"))
        # 과거 사건 (유사 사례 검색용)
        _incidents = [
            ("DELETE", "orders", 250, 1, "신입 직원이 WHERE 없이 DELETE 실행, 전체 복구"),
            ("DELETE", "orders", 180, 1, "테스트 DB와 혼동하여 프로덕션에서 삭제"),
            ("DELETE", "payments", 320, 1, "정산 데이터 삭제 실수, DBA가 백업에서 복구"),
            ("DELETE", "orders", 150, 1, "퇴근 전 급하게 작업하다 실수"),
            ("DELETE", "users", 500, 1, "탈퇴 처리 스크립트 오류로 활성 유저 삭제"),
            ("DELETE", "products", 200, 1, "카테고리 정리 중 실수로 전체 삭제"),
            ("DELETE", "orders", 400, 1, "연말 정산 중 데이터 혼동"),
            ("DELETE", "logs", 10000, 0, "정기 로그 정리 (스케줄 작업)"),
            ("DELETE", "temp_reports", 5000, 0, "임시 리포트 정리"),
            ("UPDATE", "orders", 300, 1, "금액 필드 일괄 0으로 업데이트 실수"),
            ("UPDATE", "products", 150, 1, "가격 일괄 변경 시 WHERE 조건 누락"),
            ("UPDATE", "users", 1000, 1, "권한 일괄 변경 실수"),
        ]
        for act, tbl, rc, mis, desc in _incidents:
            c.execute("INSERT INTO incidents (action,table_name,row_count,was_mistake,description) VALUES (?,?,?,?,?)",
                      (act, tbl, rc, mis, desc))
    conn.commit()
    conn.close()

# 서버 시작 시 초기화
try:
    _guardian_init()
    _guardian_train_model()
except Exception as _e:
    st.logger.warning("Guardian init failed: %s", _e)

# ── 룰엔진 ──
_CORE_TABLES = {"orders", "payments", "users", "products", "shipments"}

def _rule_engine_evaluate(action: str, table: str, row_count: int, hour: int = None):
    """룰 기반 1차 필터 (<1ms)"""
    start = _time.perf_counter()
    reasons = []
    level = "pass"
    if action in ("DROP", "TRUNCATE", "ALTER"):
        reasons.append(f"DDL 명령어 ({action}) 감지")
        level = "block"
    if action in ("DELETE", "UPDATE") and table in _CORE_TABLES and row_count > 100:
        reasons.append(f"핵심 테이블({table}) 대량 {action} ({row_count}건)")
        level = "block"
    elif action == "DELETE" and row_count > 1000:
        reasons.append(f"대량 삭제 ({row_count}건)")
        level = "block"
    elif action == "DELETE" and table in _CORE_TABLES and row_count > 10:
        reasons.append(f"핵심 테이블({table}) 삭제 {row_count}건")
        if level != "block":
            level = "warn"
    if hour is not None and (hour >= 22 or hour < 6):
        if table in _CORE_TABLES and action in ("DELETE", "UPDATE"):
            reasons.append(f"업무 시간 외 ({hour}시) 핵심 데이터 수정")
            if level == "pass":
                level = "warn"
            elif level == "warn":
                level = "block"
    elapsed = (_time.perf_counter() - start) * 1000
    if not reasons:
        reasons.append("정상 범위 쿼리")
    return {"level": level, "reasons": reasons, "elapsed_ms": round(elapsed, 3)}

# ── ML 이상탐지 (Isolation Forest) ──
import numpy as np

_GUARDIAN_ISO_MODEL = None
_GUARDIAN_SCALER = None

def _guardian_train_model():
    """pkl 파일 우선 로드, 없으면 감사 로그 기반 인라인 학습"""
    global _GUARDIAN_ISO_MODEL, _GUARDIAN_SCALER
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # (1) pkl 파일 로드 시도
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_guardian_anomaly.pkl")
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaler_guardian.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        _GUARDIAN_ISO_MODEL = joblib.load(model_path)
        _GUARDIAN_SCALER = joblib.load(scaler_path)
        st.logger.info("Guardian ML: pkl 파일에서 모델 로드 완료")
        return

    # (2) pkl 없으면 DB에서 인라인 학습
    conn = _guardian_conn()
    rows = conn.execute(
        "SELECT user_id, action, table_name, row_count, affected_amount, timestamp "
        "FROM audit_log WHERE status='executed'"
    ).fetchall()
    conn.close()

    if len(rows) < 20:
        st.logger.warning("Guardian ML: 학습 데이터 부족 (%d건) — pkl 파일도 없음", len(rows))
        return

    ACTION_MAP = {"INSERT": 0, "SELECT": 0, "UPDATE": 1, "DELETE": 2,
                  "ALTER": 3, "DROP": 4, "TRUNCATE": 4}
    features = []
    for r in rows:
        ts = r["timestamp"] or ""
        hour = int(ts[11:13]) if len(ts) > 13 else 12
        features.append([
            ACTION_MAP.get(r["action"], 0),
            1 if r["table_name"] in _CORE_TABLES else 0,
            r["row_count"],
            np.log1p(r["row_count"]),
            r["affected_amount"],
            hour,
            1 if (hour >= 22 or hour < 6) else 0,
        ])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    _GUARDIAN_ISO_MODEL = model
    _GUARDIAN_SCALER = scaler
    st.logger.info("Guardian ML: IsolationForest 인라인 학습 완료 (%d건, 7 features)", len(rows))


def _guardian_anomaly_score(user_id: str, action: str, table: str, row_count: int, hour: int = None):
    """단일 쿼리의 이상 점수 반환 (0~1, 높을수록 이상)"""
    global _GUARDIAN_ISO_MODEL, _GUARDIAN_SCALER
    h = hour if hour is not None else 12

    if _GUARDIAN_ISO_MODEL is None:
        # Lazy loading: pkl 파일이 나중에 생성됐을 수 있으므로 재시도
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_guardian_anomaly.pkl")
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scaler_guardian.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                _GUARDIAN_ISO_MODEL = joblib.load(model_path)
                _GUARDIAN_SCALER = joblib.load(scaler_path)
                st.logger.info("Guardian ML: lazy loading으로 pkl 모델 로드 완료")
            except Exception as e:
                st.logger.error("Guardian ML: lazy loading 실패 — %s", e)
                return None
        else:
            return None

    # ── ML 모델 기반 점수 ──
    ACTION_MAP = {"INSERT": 0, "SELECT": 0, "UPDATE": 1, "DELETE": 2,
                  "ALTER": 3, "DROP": 4, "TRUNCATE": 4}
    avg_amounts = {"orders": 67500, "payments": 67500, "products": 35000}
    amount = row_count * avg_amounts.get(table, 0)

    feat = np.array([[
        ACTION_MAP.get(action, 0),
        1 if table in _CORE_TABLES else 0,
        row_count,
        np.log1p(row_count),
        amount,
        h,
        1 if (h >= 22 or h < 6) else 0,
    ]])
    feat_scaled = _GUARDIAN_SCALER.transform(feat)
    raw_score = -_GUARDIAN_ISO_MODEL.decision_function(feat_scaled)[0]
    score = min(max((raw_score + 0.5) / 1.0, 0.0), 1.0)

    # 사용자별 이탈도
    conn = _guardian_conn()
    user_avg = conn.execute(
        "SELECT AVG(row_count) as avg_rc FROM audit_log "
        "WHERE user_id=? AND action=? AND status='executed'",
        (user_id, action)
    ).fetchone()
    conn.close()

    user_deviation = 0.0
    if user_avg and user_avg["avg_rc"] and user_avg["avg_rc"] > 0:
        user_deviation = min(row_count / user_avg["avg_rc"], 10.0) / 10.0

    combined = score * 0.6 + user_deviation * 0.4

    # SHAP 기반 위험 요인 분석
    FEATURE_LABELS = {
        0: ("작업 위험도", lambda v: {"0": "SELECT/INSERT", "1": "UPDATE", "2": "DELETE", "3": "ALTER", "4": "DROP/TRUNCATE"}.get(str(int(v)), action) + " 작업"),
        1: ("핵심 테이블", lambda v: f"{table}은 핵심 테이블" if v == 1 else "비핵심 테이블"),
        2: ("대상 행 수", lambda v: f"{int(v):,}건"),
        3: ("행 수(log)", None),  # skip, 행 수와 중복
        4: ("추정 금액", lambda v: f"₩{v:,.0f}"),
        5: ("시간대", lambda v: f"{int(v)}시"),
        6: ("야간 여부", lambda v: f"야간 작업 ({int(feat[0][5])}시)" if v == 1 else "주간 작업"),
    }
    risk_factors = []
    try:
        import shap
        explainer = shap.TreeExplainer(_GUARDIAN_ISO_MODEL)
        shap_values = explainer.shap_values(feat_scaled)
        sv = shap_values[0]
        # SHAP value가 음수일수록 이상 (IsolationForest 특성)
        for i in sorted(range(len(sv)), key=lambda x: sv[x]):
            if i == 3:  # log(row_count)는 건너뜀
                continue
            label, detail_fn = FEATURE_LABELS[i]
            if detail_fn is None:
                continue
            contribution = -sv[i]  # 부호 반전: 양수 = 이상에 기여
            if contribution > 0.01:
                severity = "high" if contribution > 0.05 else "medium"
                risk_factors.append({
                    "factor": label,
                    "detail": detail_fn(feat[0][i]),
                    "severity": severity,
                    "contribution": round(float(contribution), 4),
                })
    except Exception:
        # SHAP 실패 시 z-score 폴백
        z_scores = feat_scaled[0]
        for i, z in enumerate(z_scores):
            if i == 3 or abs(z) <= 0.8:
                continue
            label, detail_fn = FEATURE_LABELS[i]
            if detail_fn is None:
                continue
            risk_factors.append({
                "factor": label,
                "detail": detail_fn(feat[0][i]),
                "severity": "high" if abs(z) > 2 else "medium",
                "contribution": round(abs(float(z)) / 10, 4),
            })

    return {
        "anomaly_score": round(score, 4),
        "user_deviation": round(user_deviation, 4),
        "combined_score": round(combined, 4),
        "model": "IsolationForest",
        "features_used": 7,
        "risk_factors": risk_factors,
    }


# ── ML 결과 LLM 해석 ──

def _guardian_ml_interpret(ml_result: dict, action: str, table: str, row_count: int) -> str:
    """ML 이상탐지 결과를 LLM이 한 줄로 해석"""
    api_key = st.OPENAI_API_KEY
    if not api_key or not ml_result:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key, max_tokens=150)

        score = ml_result["combined_score"]
        risk_text = ""
        if ml_result.get("risk_factors"):
            factors = [f"{rf['factor']}({rf['detail']})" for rf in ml_result["risk_factors"][:3]]
            risk_text = f"주요 위험 요인: {', '.join(factors)}"

        prompt = (
            f"DB 감사 시스템의 ML 이상탐지 결과를 간결하게 1~2문장으로 해석해줘.\n"
            f"- 작업: {action} on {table} ({row_count:,}건)\n"
            f"- 종합 이상 점수: {score*100:.1f}/100\n"
            f"- 이상 점수(모델): {ml_result['anomaly_score']*100:.1f}%\n"
            f"- 사용자 이탈도: {ml_result['user_deviation']*100:.1f}%\n"
            f"{risk_text}\n"
            f"점수 기준: 0~40 정상, 40~70 주의, 70+ 위험. 한국어로 핵심만 답변."
        )
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        st.logger.warning("Guardian ML interpret error: %s", e)
        return None


# ── Guardian Agent (LangChain create_agent — v1.2+) ──

def _guardian_tool_defs(conn_cursor, outer_row_count: int):
    """Guardian Agent용 Tool 함수들을 정의하고 반환"""

    def analyze_impact(table_name: str, row_count: int) -> str:
        """삭제/수정 대상 데이터의 비즈니스 영향도를 분석한다. 금액, 연관 테이블, 복구 난이도를 반환."""
        avg_amounts = {"orders": 67500, "payments": 67500, "users": 0, "products": 35000, "shipments": 15000}
        related = {"orders": ["order_items", "payments", "shipments"], "payments": ["refunds", "settlements"],
                   "users": ["orders", "reviews", "addresses"], "products": ["order_items", "reviews", "inventory"],
                   "shipments": ["tracking_logs"]}
        amount = row_count * avg_amounts.get(table_name, 0)
        rel = related.get(table_name, [])
        diff = "높음" if table_name in ("orders", "payments") else "중간"
        return f"영향: {table_name} {row_count}건, 추정 금액: ₩{amount:,.0f}, 연쇄 테이블: {', '.join(rel) if rel else '없음'} ({len(rel)}개), 복구 난이도: {diff}"

    def get_user_pattern(user_id: str, current_row_count: int) -> str:
        """해당 사용자의 최근 30일 행동 패턴을 조회한다. 평소 작업량과 현재 편차를 반환."""
        row = conn_cursor.execute("SELECT AVG(row_count) as avg_c, COUNT(*) as cnt, MAX(row_count) as mx FROM audit_log WHERE user_id=? AND action IN ('DELETE','UPDATE') AND timestamp > datetime('now','-30 days')", (user_id,)).fetchone()
        if not row or not row["avg_c"]:
            return f"'{user_id}'의 최근 30일 이력 없음. 첫 작업."
        avg = row["avg_c"]
        dev = current_row_count / avg if avg > 0 else 999
        res = f"'{user_id}' 패턴: 평균 {avg:.0f}건/회, 최대 {row['mx']}건, 총 {row['cnt']}회. 현재 {current_row_count}건 = 평소의 {dev:.1f}배"
        if dev > 10:
            res += " ⚠️ 극단적 이탈"
        elif dev > 3:
            res += " ⚠️ 유의미한 이탈"
        return res

    def search_similar(action: str, table_name: str) -> str:
        """과거 유사 사건을 검색한다. 실수 비율과 상세 내역을 반환."""
        rows = conn_cursor.execute("SELECT * FROM incidents WHERE action=? AND row_count >= 50 ORDER BY ABS(row_count - ?) LIMIT 10", (action, outer_row_count)).fetchall()
        if not rows:
            return "유사 사례 없음"
        mistakes = sum(1 for r in rows if r["was_mistake"])
        total = len(rows)
        details = [f"  - [{'실수' if r['was_mistake'] else '정상'}] {r['table_name']} {r['row_count']}건: {r['description']}" for r in rows[:5]]
        return f"유사 {total}건 중 {mistakes}건 실수 ({mistakes/total*100:.0f}%)\n" + "\n".join(details)

    def execute_decision(decision: str, reason: str) -> str:
        """차단 또는 승인 결정을 실행한다. decision은 'block' 또는 'approve'."""
        return f"{'차단' if decision == 'block' else '승인'} 실행 완료. 사유: {reason}"

    return [analyze_impact, get_user_pattern, search_similar, execute_decision]


def _recovery_tool_defs(conn_cursor):
    """Recovery Agent용 Tool 함수들"""

    def search_audit_log(user_id: str = "", table_name: str = "", action: str = "DELETE") -> str:
        """감사 로그에서 최근 차단/삭제된 기록을 검색한다."""
        query = "SELECT * FROM audit_log WHERE status IN ('blocked','executed') AND action=?"
        params = [action]
        if user_id:
            query += " AND user_id=?"
            params.append(user_id)
        if table_name:
            query += " AND table_name=?"
            params.append(table_name)
        query += " ORDER BY id DESC LIMIT 5"
        rows = conn_cursor.execute(query, params).fetchall()
        if not rows:
            return "해당 조건의 감사 로그 없음"
        return "\n".join([f"  - [{r['id']}] {r['timestamp'][:16]} {r['user_id']}: {r['action']} {r['table_name']} {r['row_count']}건 (₩{r['affected_amount']:,.0f}) [{r['status']}]" for r in rows])

    def generate_restore_sql(table_name: str, row_count: int, description: str) -> str:
        """복구 SQL을 생성한다. 직접 실행하지 않고 DBA 승인 필요."""
        return f"""복구 SQL 생성 완료 (DBA 승인 필요):

-- 대상: {table_name} {row_count}건
-- 설명: {description}

INSERT INTO {table_name}
SELECT * FROM {table_name}_audit_backup
WHERE deleted_at >= NOW() - INTERVAL 1 HOUR;

-- 정합성 검증
SELECT COUNT(*) as restored FROM {table_name}
WHERE created_at >= NOW() - INTERVAL 1 HOUR;"""

    return [search_audit_log, generate_restore_sql]


def _extract_agent_steps(messages):
    """create_agent 결과 메시지에서 tool 호출 내역 추출"""
    steps = []
    from langchain_core.messages import AIMessage, ToolMessage
    tool_calls_map = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_map[tc["id"]] = {"tool": tc["name"], "input": str(tc["args"])}
        elif isinstance(msg, ToolMessage):
            tc_id = msg.tool_call_id
            if tc_id in tool_calls_map:
                tool_calls_map[tc_id]["output"] = msg.content
                steps.append(tool_calls_map[tc_id])
    return steps


def _run_guardian_agent(user_id: str, action: str, table: str, row_count: int, api_key: str):
    """LangChain create_agent로 위험도 상세 분석"""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    conn = _guardian_conn()
    tools = _guardian_tool_defs(conn.cursor(), row_count)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""당신은 Data Guardian Agent입니다. DB 변경 요청의 위험도를 분석합니다.

순서: 1) analyze_impact로 영향도 분석 2) get_user_pattern으로 사용자 패턴 확인 3) search_similar로 유사 사례 검색 4) execute_decision으로 최종 판단

최종 응답에 포함: 영향 범위(건수/금액), 위험 사유, 유사 사례 통계, 권고. 한국어로 응답.""",
    )

    result = graph.invoke({"messages": [{"role": "user", "content": f"사용자: {user_id}, 작업: {action}, 테이블: {table}, 대상: {row_count}건. 위험도 분석 후 차단 여부 판단해주세요."}]})
    conn.close()

    messages = result.get("messages", [])
    steps = _extract_agent_steps(messages)
    final_output = messages[-1].content if messages else "분석 완료"
    return {"output": final_output, "steps": steps}


def _run_recovery_agent(message: str, api_key: str):
    """복구 요청 처리 Agent"""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    conn = _guardian_conn()
    tools = _recovery_tool_defs(conn.cursor())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""당신은 Data Guardian 복구 Agent입니다.
사용자의 자연어 복구 요청을 받아서:
1) search_audit_log로 관련 기록 검색
2) generate_restore_sql로 복구 SQL 생성
복구 SQL은 직접 실행하지 않고 DBA 승인을 받아야 합니다. 한국어로 응답.""",
    )

    result = graph.invoke({"messages": [{"role": "user", "content": message}]})
    conn.close()

    messages = result.get("messages", [])
    steps = _extract_agent_steps(messages)
    final_output = messages[-1].content if messages else "복구 계획 수립 완료"
    return {"output": final_output, "steps": steps}


# ── Endpoints ──

@router.post("/guardian/analyze")
async def guardian_analyze(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: 쿼리 위험도 분석 (룰엔진 + ML 이상탐지 + AI Agent)"""
    body = await request.json()
    user_id = body.get("user_id", "unknown")
    action = body.get("action", "DELETE")
    table = body.get("table", "orders")
    row_count = int(body.get("row_count", 1))
    hour = body.get("hour")
    mode = body.get("mode", "rule+ml")  # "rule" | "ml" | "rule+ml"

    # ── 1) 룰엔진 ──
    rule = None
    if mode in ("rule", "rule+ml"):
        rule = _rule_engine_evaluate(action, table, row_count, hour)

    # ── 2) ML 이상탐지 ──
    ml = None
    if mode in ("ml", "rule+ml"):
        ml = _guardian_anomaly_score(user_id, action, table, row_count, hour)

    # ── ML 해석 (LLM) ──
    if ml:
        interpretation = _guardian_ml_interpret(ml, action, table, row_count)
        if interpretation:
            ml["interpretation"] = interpretation

    # ── 판단 로직 ──
    need_agent = False
    effective_level = "pass"

    if mode == "rule" and rule:
        effective_level = rule["level"]
        need_agent = (rule["level"] == "block")
    elif mode == "ml":
        if ml is None:
            return {"status": "SUCCESS",
                    "rule": None, "ml": None, "agent": None,
                    "ml_not_ready": True,
                    "message": "ML 모델이 아직 학습되지 않았습니다. 감사 로그가 20건 이상 쌓이면 자동으로 학습됩니다."}
        if ml["combined_score"] > 0.7:
            effective_level = "block"
            need_agent = True
        elif ml["combined_score"] > 0.4:
            effective_level = "warn"
        else:
            effective_level = "pass"
    elif mode == "rule+ml":
        rule_level = rule["level"] if rule else "pass"
        ml_level = "pass"
        if ml:
            if ml["combined_score"] > 0.7:
                ml_level = "block"
            elif ml["combined_score"] > 0.4:
                ml_level = "warn"
        # 둘 중 더 높은 쪽 채택
        level_order = {"pass": 0, "warn": 1, "block": 2}
        effective_level = rule_level if level_order.get(rule_level, 0) >= level_order.get(ml_level, 0) else ml_level
        need_agent = (effective_level == "block")

    # ── 통과/경고 → 로그만 기록 ──
    if effective_level == "pass":
        conn = _guardian_conn()
        conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(), user_id, action, table, row_count, 0, "executed", "LOW"))
        conn.commit(); conn.close()
        return {"status": "SUCCESS", "rule": rule, "ml": ml, "agent": None}

    if effective_level == "warn":
        conn = _guardian_conn()
        conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) VALUES (?,?,?,?,?,?,?,?)",
                     (datetime.now().isoformat(), user_id, action, table, row_count, 0, "warned", "MEDIUM"))
        conn.commit(); conn.close()
        return {"status": "SUCCESS", "rule": rule, "ml": ml, "agent": None}

    # ── 차단 → Agent 호출 ──
    api_key = st.OPENAI_API_KEY
    if not api_key:
        return {"status": "SUCCESS", "rule": rule, "ml": ml, "agent": {"output": "OpenAI API Key 미설정. Agent 분석 불가.", "steps": []}}

    try:
        agent_result = _run_guardian_agent(user_id, action, table, row_count, api_key)
    except Exception as e:
        st.logger.error("Guardian agent error: %s", e)
        agent_result = {"output": f"Agent 오류: {str(e)}", "steps": []}

    # 차단 로그 기록
    avg_amounts = {"orders": 67500, "payments": 67500, "products": 35000}
    amount = row_count * avg_amounts.get(table, 0)
    conn = _guardian_conn()
    conn.execute("INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level,agent_reason) VALUES (?,?,?,?,?,?,?,?,?)",
                 (datetime.now().isoformat(), user_id, action, table, row_count, amount, "blocked", "HIGH", agent_result["output"][:300]))
    conn.commit(); conn.close()

    return {"status": "SUCCESS", "rule": rule, "ml": ml, "agent": agent_result}


@router.post("/guardian/recover")
async def guardian_recover(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: 자연어 복구 요청"""
    body = await request.json()
    message = body.get("message", "")
    if not message:
        raise HTTPException(400, "message 필수")

    api_key = st.OPENAI_API_KEY
    if not api_key:
        return {"status": "ERROR", "message": "OpenAI API Key 미설정"}

    try:
        result = _run_recovery_agent(message, api_key)
        return {"status": "SUCCESS", **result}
    except Exception as e:
        st.logger.error("Guardian recovery error: %s", e)
        return {"status": "ERROR", "message": str(e)}


@router.get("/guardian/logs")
async def guardian_logs(user: dict = Depends(verify_credentials), limit: int = 30, status_filter: str = ""):
    """Data Guardian: 감사 로그 조회"""
    conn = _guardian_conn()
    if status_filter:
        rows = conn.execute("SELECT * FROM audit_log WHERE status=? ORDER BY id DESC LIMIT ?", (status_filter, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"status": "SUCCESS", "logs": [dict(r) for r in rows]}


@router.get("/guardian/stats")
async def guardian_stats(user: dict = Depends(verify_credentials)):
    """Data Guardian: 통계"""
    conn = _guardian_conn()
    total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    blocked = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='blocked'").fetchone()[0]
    warned = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='warned'").fetchone()[0]
    restored = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status='restored'").fetchone()[0]
    saved = conn.execute("SELECT COALESCE(SUM(affected_amount),0) FROM audit_log WHERE status='blocked'").fetchone()[0]
    # 최근 7일 일별 차단 수
    daily = conn.execute("SELECT date(timestamp) as d, COUNT(*) as cnt FROM audit_log WHERE status='blocked' AND timestamp > datetime('now','-7 days') GROUP BY d ORDER BY d").fetchall()
    conn.close()
    return {
        "status": "SUCCESS",
        "total": total, "blocked": blocked, "warned": warned, "restored": restored,
        "saved_amount": saved,
        "daily_blocked": [{"date": r["d"], "count": r["cnt"]} for r in daily],
    }


@router.post("/guardian/notify-dba")
async def guardian_notify_dba(request: Request, user: dict = Depends(verify_credentials)):
    """Data Guardian: DBA에게 Resend를 통해 이메일 알림 발송"""
    body = await request.json()
    dba_email = body.get("email", "")
    alert_data = body.get("alert", {})

    if not dba_email:
        raise HTTPException(400, "email 필수")

    resend_key = os.environ.get("RESEND_API_KEY", "")

    user_id = alert_data.get("user_id", "unknown")
    action = alert_data.get("action", "")
    table = alert_data.get("table", "")
    row_count = alert_data.get("row_count", 0)
    rule_reasons = alert_data.get("rule_reasons", [])
    agent_output = alert_data.get("agent_output", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reasons_html = "".join(f"<li>{r}</li>" for r in rule_reasons) if rule_reasons else "<li>고위험 쿼리 감지</li>"
    agent_html = (agent_output or "Agent 분석 없음").replace("\n", "<br>")

    html = f"""<div style="font-family:'Apple SD Gothic Neo','Malgun Gothic',sans-serif;max-width:640px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden">
  <div style="background:linear-gradient(135deg,#dc2626,#ef4444);padding:24px 32px">
    <h1 style="color:#fff;font-size:20px;margin:0;font-weight:700">Data Guardian Alert</h1>
    <p style="color:#fecaca;font-size:13px;margin:6px 0 0">고위험 쿼리 — DBA 승인 필요</p>
  </div>
  <div style="padding:28px 32px">
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:20px">
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;width:100px;border:1px solid #fecaca">시간</td><td style="padding:8px 12px;border:1px solid #fecaca">{ts}</td></tr>
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;border:1px solid #fecaca">사용자</td><td style="padding:8px 12px;border:1px solid #fecaca">{user_id}</td></tr>
      <tr><td style="padding:8px 12px;background:#fef2f2;font-weight:600;border:1px solid #fecaca">쿼리</td><td style="padding:8px 12px;border:1px solid #fecaca"><code>{action} FROM {table}</code> ({row_count}건)</td></tr>
    </table>
    <div style="margin-bottom:20px">
      <p style="color:#dc2626;font-size:13px;font-weight:600;margin:0 0 8px">차단 사유</p>
      <ul style="margin:0;padding-left:20px;font-size:13px;color:#374151;line-height:1.8">{reasons_html}</ul>
    </div>
    <div style="margin-bottom:20px">
      <p style="color:#4f46e5;font-size:13px;font-weight:600;margin:0 0 8px">AI Agent 분석</p>
      <div style="background:#f8fafc;padding:14px 18px;border-radius:8px;border-left:3px solid #4f46e5;font-size:13px;color:#374151;line-height:1.7">{agent_html}</div>
    </div>
    <div style="text-align:center;padding:16px 0">
      <p style="font-size:14px;color:#6b7280;margin:0 0 12px">이 쿼리를 승인하시겠습니까?</p>
      <a href="#" style="display:inline-block;background:#dc2626;color:#fff;padding:10px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;margin-right:8px">차단 유지</a>
      <a href="#" style="display:inline-block;background:#f59e0b;color:#fff;padding:10px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px">승인</a>
    </div>
  </div>
  <div style="background:#f9fafb;padding:16px 32px;border-top:1px solid #e5e7eb">
    <p style="margin:0;font-size:11px;color:#9ca3af">CAFE24 Data Guardian — 자동 발송 알림</p>
  </div>
</div>"""

    if resend_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                    json={
                        "from": "CAFE24 Guardian <onboarding@resend.dev>",
                        "to": [dba_email],
                        "subject": f"[Guardian Alert] {action} {table} {row_count}건 — DBA 승인 필요",
                        "html": html,
                    },
                )
            st.logger.info("[guardian-notify] resend status=%s", resp.status_code)
            if resp.status_code >= 400:
                return {"status": "ERROR", "message": f"Resend 오류: {resp.text[:100]}"}
            return {"status": "SUCCESS", "message": f"{dba_email}로 DBA 알림이 발송되었습니다."}
        except Exception as e:
            st.logger.error("[guardian-notify] resend error: %s", e)
            return {"status": "ERROR", "message": str(e)}
    else:
        st.logger.info("[guardian-notify] (RESEND_API_KEY 미설정 — 시뮬레이션) → %s", dba_email)
        return {"status": "SUCCESS", "message": f"{dba_email}로 DBA 알림이 발송되었습니다. (RESEND_API_KEY 미설정 — 시뮬레이션)"}
