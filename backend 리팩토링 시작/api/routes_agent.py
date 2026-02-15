"""
api/routes_agent.py - 에이전트/채팅 API
(agent/stream 엔드포인트의 내부 로직은 그대로 유지)
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
from agent.multi_agent import run_sub_agent_stream
from rag.service import tool_rag_search
from rag.light_rag import lightrag_search_sync, LIGHTRAG_AVAILABLE
from rag.k2rag import k2rag_search_sync
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
    """LangGraph 기반 스트리밍 에이전트"""
    st.logger.info("STREAM_REQ headers_auth=%s origin=%s ua=%s", request.headers.get("authorization"), request.headers.get("origin"), request.headers.get("user-agent"))
    username = user["username"]

    async def gen():
        tool_calls_log = []
        final_buf = []

        try:
            from langgraph.prebuilt import create_react_agent
            from langchain_openai import ChatOpenAI
            from agent.tools import ALL_TOOLS
            from agent.router import classify_and_get_tools, IntentCategory

            user_text = safe_str(req.user_input)
            rag_mode = req.rag_mode or "auto"
            api_key = pick_api_key(req.api_key)
            category, allowed_tool_names = classify_and_get_tools(user_text, api_key, use_llm_fallback=False)

            st.logger.info("STREAM_ROUTER category=%s allowed_tools=%s", category.value, allowed_tool_names)

            # RETENTION 카테고리 → 서브에이전트 모드로 분기
            if category == IntentCategory.RETENTION:
                st.logger.info("STREAM_RETENTION_MODE sub_agent user=%s", username)
                queue = asyncio.Queue()

                async def sse_callback(event_type: str, data: dict):
                    await queue.put((event_type, data))

                async def run_task():
                    try:
                        await run_sub_agent_stream(req, username, sse_callback)
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

            if allowed_tool_names:
                tools = [t for t in ALL_TOOLS if t.name in allowed_tool_names]
                if not tools:
                    tools = ALL_TOOLS
            elif category == IntentCategory.GENERAL:
                tools = []
            else:
                tools = ALL_TOOLS

            if category == IntentCategory.PLATFORM:
                if rag_mode == "rag":
                    tools = [t for t in tools if t.name != "search_platform_lightrag"]
                elif rag_mode == "lightrag":
                    tools = [t for t in tools if t.name != "search_platform_docs"]
                elif rag_mode == "k2rag":
                    tools = [t for t in tools if t.name not in ["search_platform_docs", "search_platform_lightrag"]]

            st.logger.info("AGENT_TOOLS rag_mode=%s category=%s tools=%d (%s)", rag_mode, category.value, len(tools), [t.name for t in tools] if len(tools) <= 10 else f"{len(tools)} tools")

            if not api_key:
                yield sse_pack("done", {"ok": False, "final": "처리 오류: OpenAI API Key가 없습니다.", "tool_calls": []})
                return

            rag_context = ""
            simple_patterns = ["안녕", "고마워", "감사", "뭐해", "ㅎㅎ", "ㅋㅋ", "네", "응", "오케이", "bye", "hi", "hello", "thanks"]
            is_simple = any(p in user_text.lower() for p in simple_patterns) and len(user_text) < 20
            skip_rag = category not in [IntentCategory.PLATFORM, IntentCategory.GENERAL, IntentCategory.SHOP]
            if skip_rag:
                st.logger.info("SKIP_RAG category=%s", category.value)

            if not is_simple and not skip_rag:
                try:
                    _rag_start = _time.time()
                    if rag_mode == "lightrag":
                        rag_out = lightrag_search_sync(user_text, mode="hybrid")
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("LIGHTRAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "success":
                            context_text = rag_out.get("context", "")
                            if context_text:
                                context_preview = context_text[:1000] + ("..." if len(context_text) > 1000 else "")
                                tool_calls_log.append({"tool": "lightrag_search", "args": {"query": user_text, "mode": "hybrid"}, "result": {"status": "success", "context": context_preview, "context_len": len(context_text)}})
                                max_chars = st.LIGHTRAG_CONFIG.get("context_max_chars", 1500)
                                rag_context = f"\n\n## 검색된 플랫폼 정보 (LightRAG):\n{context_text[:max_chars]}\n"
                                tools = [t for t in tools if t.name != "search_platform_lightrag"]
                    elif rag_mode == "k2rag":
                        rag_out = k2rag_search_sync(user_text, top_k=10, use_kg=True, use_summary=True)
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("K2RAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "success":
                            answer = rag_out.get("answer", "")
                            context = rag_out.get("context", "")
                            if answer or context:
                                tool_calls_log.append({"tool": "k2rag_search", "args": {"query": user_text}, "result": {"status": "success", "answer_len": len(answer), "context_len": len(context)}})
                                rag_context = f"\n\n## 검색된 플랫폼 정보 (K2RAG):\n{answer or context[:2000]}\n"
                                tools = [t for t in tools if t.name not in ["search_platform_docs", "search_platform_lightrag"]]
                    else:
                        rag_out = tool_rag_search(user_text, top_k=st.RAG_DEFAULT_TOPK, api_key=api_key)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "success":
                            results = rag_out.get("results") or []
                            if results:
                                tool_calls_log.append({"tool": "rag_search", "args": {"query": user_text}, "result": rag_out})
                                rag_context = "\n\n## 검색된 플랫폼 정보 (참고용):\n"
                                for r in results[:5]:
                                    content = r.get("content", "")[:800]
                                    source = r.get("source", "")
                                    if source:
                                        rag_context += f"- [출처: {source}] {content}\n"
                                    else:
                                        rag_context += f"- {content}\n"
                                tools = [t for t in tools if t.name != "search_platform_docs"]
                except Exception as _e:
                    st.logger.warning("RAG_SEARCH_FAIL err=%s", safe_str(_e))

            base_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
            rag_tool_info = ""
            if rag_mode == "rag":
                rag_tool_info = "- `search_platform_docs`: 플랫폼 검색 (FAISS + BM25)"
            elif rag_mode == "lightrag":
                rag_tool_info = "- `search_platform_lightrag`: 플랫폼 검색 (LightRAG - 지식그래프 기반)"
            elif rag_mode == "k2rag":
                rag_tool_info = "- K2RAG 모드: 검색이 자동으로 수행됩니다"
            else:
                rag_tool_info = "- `search_platform_docs`: 플랫폼 검색 (FAISS + BM25)\n- `search_platform_lightrag`: 플랫폼 검색 (LightRAG - 관계 기반)"

            system_prompt = base_prompt + f"""

## 도구 사용 규칙

당신은 카페24 이커머스 AI 어시스턴트입니다. 사용자 요청에 적합한 도구를 선택하여 호출하세요.

### 주요 도구:
- `get_shop_info`, `list_shops`: 쇼핑몰 정보
- `get_category_info`, `list_categories`: 카테고리 정보
- `auto_reply_cs`, `check_cs_quality`: CS 관련
- `analyze_seller`, `get_seller_segment`, `detect_fraud`: 셀러 분석
- `predict_seller_churn`: 셀러 이탈 예측
- `get_shop_performance`: 쇼핑몰 성과 분석
- `predict_shop_revenue`: 쇼핑몰 매출 예측
- `optimize_marketing`: 마케팅 예산 최적화
- `get_segment_statistics`: 세그먼트별 셀러 통계
- `get_fraud_statistics`: 이상거래 통계
- `get_order_statistics`: 운영 이벤트 통계
- `get_dashboard_summary`: 대시보드 요약

### 플랫폼 카테고리 검색 도구 (현재 모드: {rag_mode}):
{rag_tool_info}

### 규칙:
1. 쇼핑몰, 정산, 정책 정보 질문 -> RAG 검색 도구 우선 사용
2. 사용자 요청에 맞는 도구를 직접 선택
3. 여러 정보가 필요하면 여러 도구를 동시에 호출
4. 도구 결과를 바탕으로 친절하게 답변
5. 간단한 인사나 대화에는 도구 호출 없이 바로 답변
6. 플랫폼 정책, 정산, 설정 관련 질문은 검색 도구 사용
"""
            if rag_context:
                system_prompt += f"""

## 검색된 플랫폼 정보 (공식 문서)
{rag_context}

### 답변 규칙
1. 검색 결과를 기반으로 사용자 질문에 친절하게 답변
2. 검색 결과가 짧은 안내 문구라도 해당 내용을 활용하여 안내
3. 검색 결과의 문서 제목(source)이 있으면 관련 가이드를 언급
4. 검색 결과에 전혀 관련 없는 내용만 있을 경우에만 "관련 정보를 찾지 못했습니다"라고 답변
5. 카페24 플랫폼 공식 가이드 문서 기반임을 자연스럽게 안내
"""

            model_name = req.model or "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name, api_key=api_key, streaming=True, max_tokens=req.max_tokens or 1500, temperature=req.temperature or 0.7)

            use_direct_llm = not tools or (category == IntentCategory.PLATFORM and rag_context)
            if use_direct_llm:
                mode = "PLATFORM_RAG_DIRECT" if category == IntentCategory.PLATFORM else "GENERAL"
                st.logger.info("STREAM_%s_MODE direct LLM response", mode)
                if category == IntentCategory.PLATFORM:
                    llm = ChatOpenAI(model=model_name, api_key=api_key, streaming=True, max_tokens=req.max_tokens or 1500, temperature=0.2)
                _llm_start = _time.time()
                _first_token = True
                async for chunk in llm.astream([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]):
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

            agent = create_react_agent(llm, tools, prompt=system_prompt)
            current_tool = None
            async for event in agent.astream_events({"messages": [("user", user_text)]}, version="v2", config={"recursion_limit": 10}):
                if await request.is_disconnected():
                    return
                kind = event.get("event", "")
                data = event.get("data", {})
                if kind == "on_tool_start":
                    tool_name = event.get("name", "도구")
                    tool_input = data.get("input", {})
                    current_tool = tool_name
                    yield sse_pack("tool_start", {"tool": tool_name, "args": tool_input})
                elif kind == "on_tool_end":
                    end_tool_name = event.get("name") or current_tool or "unknown"
                    tool_output = data.get("output", {})
                    if hasattr(tool_output, "content"):
                        content = tool_output.content
                        if isinstance(content, str):
                            try:
                                tool_output = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                tool_output = {"status": "success", "data": content}
                        elif isinstance(content, (dict, list)):
                            tool_output = content
                        else:
                            tool_output = {"status": "success", "data": safe_str(content)}
                    elif not isinstance(tool_output, (str, dict, list, int, float, bool, type(None))):
                        tool_output = {"status": "success", "data": safe_str(tool_output)}
                    tool_calls_log.append({"tool": end_tool_name, "result": tool_output})
                    yield sse_pack("tool_end", {"tool": end_tool_name, "status": "success"})
                    current_tool = None
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            final_buf.append(content)
                            yield sse_pack("delta", {"delta": content})

            final_text = "".join(final_buf).strip() or "요청을 처리했습니다."
            append_memory(username, user_text, final_text)
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

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
