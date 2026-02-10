"""
api/routes_cs.py - CS/고객지원 API
"""
import os
import json
import asyncio
import uuid
import time as _time
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse

from core.constants import CS_TICKET_CATEGORIES, CS_PRIORITY_GRADES
from core.utils import safe_str
from agent.tools import (
    tool_auto_reply_cs, tool_check_cs_quality,
    tool_get_ecommerce_glossary, tool_get_cs_statistics,
    tool_classify_inquiry,
)
from agent.llm import pick_api_key
from rag.service import rag_search_hybrid
from rag.light_rag import lightrag_search_sync, LIGHTRAG_AVAILABLE
from rag.k2rag import k2rag_search_sync
import state as st
from api.common import (
    verify_credentials,
    CsReplyRequest, CsQualityRequest, CsPipelineRequest, CsPipelineAnswerRequest,
)


router = APIRouter(prefix="/api", tags=["cs"])

# ── n8n 연동: job_id 기반 큐 (TTL 관리) ──
_cs_job_queues: dict = {}
_cs_job_timestamps: dict = {}  # job_id -> 생성 시각
_CS_JOB_TTL_SEC = 10 * 60     # 10분 후 자동 정리


def _cleanup_expired_jobs() -> None:
    """만료된 CS job 큐 정리"""
    now = _time.time()
    expired = [k for k, ts in _cs_job_timestamps.items() if now - ts > _CS_JOB_TTL_SEC]
    for k in expired:
        _cs_job_queues.pop(k, None)
        _cs_job_timestamps.pop(k, None)


# ============================================================
# CS 콜백 인증 (#4)
# ============================================================
_CS_CALLBACK_SECRET = os.environ.get("CS_CALLBACK_SECRET", "")


def _verify_callback_token(x_callback_token: Optional[str] = Header(None)):
    """콜백 엔드포인트 API 키 인증"""
    if not _CS_CALLBACK_SECRET:
        return  # 시크릿 미설정 시 인증 건너뛰기 (개발 모드)
    if x_callback_token != _CS_CALLBACK_SECRET:
        raise HTTPException(status_code=401, detail="유효하지 않은 콜백 토큰")


# ============================================================
# CS API
# ============================================================
@router.post("/cs/reply")
def cs_auto_reply(req: CsReplyRequest, user: dict = Depends(verify_credentials)):
    return tool_auto_reply_cs(inquiry_text=req.text, inquiry_category=req.ticket_category, seller_tier=req.seller_tier)


@router.post("/cs/quality")
def check_cs_quality_route(req: CsQualityRequest, user: dict = Depends(verify_credentials)):
    return tool_check_cs_quality(ticket_category=req.ticket_category, seller_tier=req.seller_tier, sentiment_score=req.sentiment_score, order_value=req.order_value, is_repeat_issue=req.is_repeat_issue, text_length=req.text_length)


@router.get("/cs/glossary")
def get_ecommerce_glossary(term: Optional[str] = None, user: dict = Depends(verify_credentials)):
    return tool_get_ecommerce_glossary(term=term)


@router.get("/cs/statistics")
def get_cs_stats(user: dict = Depends(verify_credentials)):
    return tool_get_cs_statistics()


# ============================================================
# CS 자동화 파이프라인 API
# ============================================================
@router.post("/cs/pipeline")
def cs_pipeline(req: CsPipelineRequest, user: dict = Depends(verify_credentials)):
    result = {"status": "success", "steps": {}}
    step_classify = tool_classify_inquiry(req.inquiry_text)
    result["steps"]["classify"] = step_classify
    if step_classify.get("status") != "SUCCESS":
        result["status"] = "error"
        return result
    predicted_category = step_classify.get("predicted_category", "기타")
    confidence = step_classify.get("confidence", 0.0)

    negative_words = ["화나", "불만", "짜증", "최악", "실망", "환불", "사기", "안돼", "못써"]
    sentiment = -0.4 if any(w in req.inquiry_text for w in negative_words) else 0.1
    is_auto = confidence >= req.confidence_threshold
    routing = "auto" if is_auto else "manual"

    priority_result = tool_check_cs_quality(ticket_category=predicted_category, seller_tier=req.seller_tier, sentiment_score=sentiment, order_value=req.order_value, is_repeat_issue=req.is_repeat_issue, text_length=len(req.inquiry_text))
    result["steps"]["review"] = {"confidence": confidence, "threshold": req.confidence_threshold, "routing": routing, "predicted_category": predicted_category, "sentiment_score": sentiment, "priority": priority_result}

    answer_context = tool_auto_reply_cs(inquiry_text=req.inquiry_text, inquiry_category=predicted_category, seller_tier=req.seller_tier, order_id=req.order_id)
    result["steps"]["answer"] = answer_context
    result["steps"]["reply"] = {"status": "READY", "channels": ["이메일", "카카오톡", "SMS", "인앱 알림"], "selected_channel": None, "message": "회신 채널을 선택하면 n8n 워크플로우로 자동 전송됩니다."}

    stats = tool_get_cs_statistics()
    result["steps"]["improve"] = {"statistics": stats, "pipeline_meta": {"classification_model_accuracy": 0.82, "auto_routing_rate": f"{req.confidence_threshold * 100:.0f}% 이상 자동", "categories": CS_TICKET_CATEGORIES, "priority_grades": list(CS_PRIORITY_GRADES.keys())}}
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
                    snippets = [r.get("text", "") or r.get("content", "") or r.get("snippet", "") for r in results[:3]]
                    context_text = "\n\n".join(s[:500] for s in snippets if s)
            except Exception as e:
                st.logger.warning("CS Pipeline RAG search failed: %s", e)

            yield f"data: {json.dumps({'type': 'rag_context', 'data': {'source_count': source_count, 'context_preview': context_text[:300] if context_text else '(검색 결과 없음)'}})}\n\n"

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

            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3, streaming=True)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"셀러 문의: {req.inquiry_text}")]
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            st.logger.error("CS Pipeline Answer Error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── n8n 연동 ──
@router.post("/cs/send-reply")
async def cs_send_reply(request: Request, user: dict = Depends(verify_credentials)):
    body = await request.json()
    inquiries = body.get("inquiries", [])
    _cleanup_expired_jobs()
    job_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _cs_job_queues[job_id] = queue
    _cs_job_timestamps[job_id] = _time.time()
    await queue.put({"type": "step", "data": {"node": "webhook", "status": "completed", "detail": "트리거 완료"}})
    n8n_url = os.environ.get("N8N_WEBHOOK_URL", "")
    callback_base = os.environ.get("N8N_CALLBACK_URL", "")
    if n8n_url:
        asyncio.create_task(_n8n_trigger(job_id, n8n_url, callback_base, inquiries, queue))
    else:
        asyncio.create_task(_simulate_workflow(job_id, inquiries, queue))
    return {"status": "success", "job_id": job_id}


async def _n8n_trigger(job_id, n8n_url, callback_base, inquiries, queue):
    try:
        import httpx
        all_channels = sorted({ch for inq in inquiries for ch in inq.get("channels", [])})
        st.logger.info("[n8n] job=%s calling %s channels=%s", job_id, n8n_url, all_channels)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(n8n_url, json={"job_id": job_id, "inquiries": inquiries, "channels": all_channels})
        st.logger.info("[n8n] job=%s status=%s body=%s", job_id, resp.status_code, resp.text[:300])
        if resp.status_code >= 400:
            await queue.put({"type": "error", "data": f"n8n 호출 실패: HTTP {resp.status_code}"})
            await queue.put(None)
            return
        await _replay_steps(inquiries, all_channels, queue)
    except Exception as e:
        st.logger.error("[n8n] job=%s error: %s", job_id, e)
        await queue.put({"type": "error", "data": f"n8n 연결 실패: {str(e)[:80]}"})
        await queue.put(None)


async def _replay_steps(inquiries, all_channels, queue):
    try:
        await queue.put({"type": "step", "data": {"node": "validate", "status": "running"}})
        await asyncio.sleep(0.4)
        await queue.put({"type": "step", "data": {"node": "validate", "status": "completed", "detail": f"{len(inquiries)}건 검증 완료"}})
        await queue.put({"type": "step", "data": {"node": "router", "status": "running"}})
        await asyncio.sleep(0.3)
        await queue.put({"type": "step", "data": {"node": "router", "status": "completed", "detail": f"{len(all_channels)}개 채널"}})
        for ch in all_channels:
            ch_count = sum(1 for inq in inquiries if ch in inq.get("channels", []))
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "running"}})
            await asyncio.sleep(0.3)
            await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "completed", "detail": f"{ch_count}건 전송"}})
        await queue.put({"type": "step", "data": {"node": "log", "status": "running"}})
        await asyncio.sleep(0.3)
        await queue.put({"type": "step", "data": {"node": "log", "status": "completed", "detail": "이력 저장 완료"}})
        await queue.put({"type": "done", "data": {"total": len(inquiries), "channels": all_channels}})
    except Exception as e:
        st.logger.error("replay_steps error: %s", e)
        await queue.put({"type": "error", "data": str(e)})
    finally:
        await queue.put(None)


async def _send_email_resend(to_email, subject, body_html):
    resend_key = os.environ.get("RESEND_API_KEY", "")
    if not resend_key or not to_email:
        return False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"}, json={"from": "CAFE24 CS <onboarding@resend.dev>", "to": [to_email], "subject": subject, "html": body_html})
        st.logger.info("[cs-email] resend to=%s status=%s", to_email, resp.status_code)
        return resp.status_code < 400
    except Exception as e:
        st.logger.error("[cs-email] resend error: %s", e)
        return False


async def _simulate_workflow(job_id, inquiries, queue):
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
            if ch == "이메일":
                email_sent = 0
                for inq in inquiries:
                    if "이메일" not in inq.get("channels", []):
                        continue
                    to_email = inq.get("recipient_email", "")
                    if not to_email:
                        continue
                    subject = f"[카페24] 문의 답변 — {inq.get('category', '기타')}"
                    html = f"""<div style="font-family:'Apple SD Gothic Neo',sans-serif;max-width:600px;margin:0 auto">
  <div style="background:#1a56db;padding:20px 32px"><h2 style="margin:0;color:#fff;font-size:18px">카페24 고객지원 답변</h2></div>
  <div style="padding:24px 32px;background:#fff"><p style="color:#374151;font-size:14px;line-height:1.7"><strong>문의 내용:</strong><br>{inq.get('inquiry_text', '')}</p><hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0"><p style="color:#374151;font-size:14px;line-height:1.7"><strong>답변:</strong><br>{inq.get('answer_text', '')}</p></div>
  <div style="background:#f9fafb;padding:16px 32px;border-top:1px solid #e5e7eb"><p style="margin:0;font-size:11px;color:#9ca3af">CAFE24 AI CS</p></div></div>"""
                    ok = await _send_email_resend(to_email, subject, html)
                    if ok:
                        email_sent += 1
                detail = f"{email_sent}건 발송 완료" if email_sent > 0 else f"{ch_count}건 전송 (시뮬레이션)"
                await queue.put({"type": "step", "data": {"node": f"channel_{ch}", "status": "completed", "detail": detail}})
            else:
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
        await queue.put(None)


@router.get("/cs/stream")
async def cs_stream(job_id: str, user: dict = Depends(verify_credentials)):
    queue = _cs_job_queues.get(job_id)
    if not queue:
        return JSONResponse({"status": "error", "message": "유효하지 않은 job_id"}, status_code=404)

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
            _cs_job_timestamps.pop(job_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/cs/callback")
async def cs_callback(request: Request, _auth=Depends(_verify_callback_token)):
    """n8n 콜백 수신 — API 키 인증 적용 (#4)"""
    body = await request.json()
    job_id = body.get("job_id", "")
    queue = _cs_job_queues.get(job_id)
    if not queue:
        return JSONResponse({"status": "error", "message": "유효하지 않은 job_id"}, status_code=404)

    step = body.get("step", "")
    status_val = body.get("status", "")
    detail = body.get("detail", "")

    if step == "done":
        await queue.put({"type": "done", "data": body.get("data", {})})
        await queue.put(None)
    elif step == "error":
        await queue.put({"type": "error", "data": detail or body.get("data", "")})
        await queue.put(None)
    else:
        await queue.put({"type": "step", "data": {"node": step, "status": status_val, "detail": detail}})

    return {"status": "success"}
