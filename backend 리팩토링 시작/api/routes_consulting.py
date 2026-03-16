"""
api/routes_consulting.py - 셀러 컨설팅 API
SSE 스트리밍 기반 4단계 인터랙티브 컨설팅 엔드포인트
"""
import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.utils import safe_str
from agent.llm import pick_api_key
from agent.consulting_agent import (
    run_consulting_stream,
    get_user_sessions,
    delete_session,
)
import state as st
from api.common import verify_credentials, sse_pack


router = APIRouter(prefix="/api/consulting", tags=["consulting"])


# ============================================================
# 요청 모델
# ============================================================
class ConsultingRequest(BaseModel):
    seller_id: str = Field(..., alias="sellerId")
    user_input: str = Field("", alias="userInput")
    session_id: str | None = Field(None, alias="sessionId")
    action: str = Field("message")              # message / confirm / rollback / reset
    strategy_choice: str | None = Field(None, alias="strategyChoice")
    rollback_target: str | None = Field(None, alias="rollbackTarget")
    model: str = Field("gpt-4o-mini")
    api_key: str = Field("", alias="apiKey")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


# ============================================================
# SSE 스트리밍 엔드포인트
# ============================================================
@router.post("/stream")
async def consulting_stream(req: ConsultingRequest, request: Request, user: dict = Depends(verify_credentials)):
    """셀러 컨설팅 SSE 스트리밍 엔드포인트"""
    username = user["username"]
    st.logger.info(
        "CONSULTING_REQ user=%s seller=%s action=%s",
        username, req.seller_id, req.action,
    )

    async def gen():
        queue = asyncio.Queue()

        async def sse_callback(event_type: str, data: dict):
            await queue.put((event_type, data))

        async def run_task():
            try:
                await run_consulting_stream(
                    seller_id=req.seller_id,
                    user_input=req.user_input,
                    session_id=req.session_id,
                    action=req.action,
                    strategy_choice=req.strategy_choice,
                    username=username,
                    sse_callback=sse_callback,
                    api_key=req.api_key,
                    model=req.model,
                )
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(run_task())

        while True:
            item = await queue.get()
            if item is None:
                break
            event_type, data = item
            yield sse_pack(event_type, data)

        await task
        return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================================================
# 세션 관리 엔드포인트
# ============================================================
@router.get("/sessions")
async def list_sessions(user: dict = Depends(verify_credentials)):
    """현재 사용자의 활성 컨설팅 세션 목록"""
    username = user["username"]
    sessions = get_user_sessions(username)
    return {"status": "success", "sessions": sessions}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str, user: dict = Depends(verify_credentials)):
    """컨설팅 세션 삭제"""
    username = user["username"]
    deleted = delete_session(username, session_id)
    if deleted:
        return {"status": "success", "message": f"세션 {session_id} 삭제 완료"}
    return {"status": "error", "message": f"세션 {session_id}을 찾을 수 없습니다."}
