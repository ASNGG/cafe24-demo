"""
api/routes_agent.py - 에이전트/채팅 API
모든 비단순 요청은 멀티에이전트 Supervisor(8종 워커)로 통합 처리
"""
import json
import time as _time
import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str
from core.memory import clear_memory, append_memory
from agent.llm import pick_api_key
from agent.runner import run_agent
from agent.multi_agent import run_multi_agent_stream
from agent.consulting_graph import run_consulting_graph_stream, has_active_consulting, reset_consulting_session
import state as st
from api.common import verify_credentials, AgentRequest, sse_pack


router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/agent/chat")
def agent_chat(req: AgentRequest, user: dict = Depends(verify_credentials)):
    out = run_agent(req, username=user["username"])
    if isinstance(out, dict) and "status" not in out:
        out["status"] = "success"
    return out


@router.post("/agent/memory/clear")
def clear_agent_memory(user: dict = Depends(verify_credentials)):
    clear_memory(user["username"])
    return {"status": "success", "message": "메모리 초기화 완료"}


@router.post("/agent/stream")
async def agent_stream(req: AgentRequest, request: Request, user: dict = Depends(verify_credentials)):
    """LangGraph 기반 스트리밍 에이전트 — 멀티에이전트 Supervisor 통합"""
    st.logger.info("STREAM_REQ headers_auth=%s origin=%s ua=%s", request.headers.get("authorization"), request.headers.get("origin"), request.headers.get("user-agent"))
    username = user["username"]

    async def gen():
        tool_calls_log = []
        final_buf = []

        try:
            from agent.router import classify_and_get_tools, IntentCategory

            user_text = safe_str(req.user_input)
            api_key = pick_api_key(req.api_key)
            category, _ = classify_and_get_tools(user_text, api_key, use_llm_fallback=True)

            st.logger.info("STREAM_ROUTER category=%s", category.value)

            if not api_key:
                yield sse_pack("done", {"ok": False, "final": "처리 오류: OpenAI API Key가 없습니다.", "tool_calls": []})
                return

            # 간단 인사 → 직접 LLM 응답 (Supervisor 불필요)
            simple_patterns = ["안녕", "고마워", "감사", "뭐해", "ㅎㅎ", "ㅋㅋ", "네", "응", "오케이", "bye", "hi", "hello", "thanks"]
            _analysis_kw = {"분석", "조회", "예측", "통계", "검색", "알려", "보여", "해줘", "셀러", "쇼핑몰", "매출", "이탈", "CS", "SEL", "GMV"}
            is_simple = (
                any(p in user_text.lower() for p in simple_patterns)
                and len(user_text) < 20
                and not any(kw in user_text for kw in _analysis_kw)
            )

            if is_simple:
                from langchain_openai import ChatOpenAI
                model_name = req.model or "gpt-4o-mini"
                base_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
                llm = ChatOpenAI(model=model_name, api_key=api_key, streaming=True, max_tokens=req.max_tokens or 1500, temperature=req.temperature or 0.7)
                st.logger.info("STREAM_SIMPLE_MODE direct LLM response")
                _llm_start = _time.time()
                _first_token = True
                async for chunk in llm.astream([{"role": "system", "content": base_prompt}, {"role": "user", "content": user_text}]):
                    if await request.is_disconnected():
                        return
                    content = getattr(chunk, "content", "")
                    if content:
                        if _first_token:
                            st.logger.info("LLM_TTFT elapsed=%.0fms", (_time.time() - _llm_start) * 1000)
                            _first_token = False
                        final_buf.append(content)
                        yield sse_pack("delta", {"delta": content})
                full_response = "".join(final_buf)
                append_memory(username, user_text, full_response)
                yield sse_pack("done", {"ok": True, "final": full_response, "tool_calls": tool_calls_log})
                return

            # ── 컨설팅 모드: CONSULTING 카테고리 또는 LangGraph 활성 세션 ──
            import re
            is_consulting = category == IntentCategory.CONSULTING or has_active_consulting(username)

            if is_consulting:
                seller_match = re.search(r'SEL\d{1,6}', user_text, re.IGNORECASE)
                seller_id = seller_match.group(0) if seller_match else ""

                st.logger.info("STREAM_CONSULTING_GRAPH seller=%s user=%s", seller_id, username)
                queue = asyncio.Queue()

                async def sse_callback(event_type: str, data: dict):
                    await queue.put((event_type, data))

                async def run_task():
                    try:
                        await run_consulting_graph_stream(
                            seller_id=seller_id,
                            user_input=user_text,
                            username=username,
                            sse_callback=sse_callback,
                            api_key=api_key,
                            model=req.model or "gpt-4o-mini",
                        )
                    finally:
                        await queue.put(None)

                task = asyncio.create_task(run_task())

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    event_type, data = item
                    yield sse_pack(event_type, data)

                await task
                return

            # ── 일반 멀티에이전트 Supervisor (8종 워커) ──
            st.logger.info("STREAM_MULTI_AGENT_MODE category=%s user=%s", category.value, username)
            queue = asyncio.Queue()

            async def sse_callback(event_type: str, data: dict):
                await queue.put((event_type, data))

            async def run_task():
                try:
                    await run_multi_agent_stream(req, username, sse_callback, category=category)
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

        except Exception as e:
            msg = safe_str(e) or "스트리밍 오류"
            st.logger.exception("STREAM_ERROR err=%s", msg)
            try:
                yield sse_pack("error", {"message": msg})
            except Exception:
                pass
            yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": tool_calls_log})
            return

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
