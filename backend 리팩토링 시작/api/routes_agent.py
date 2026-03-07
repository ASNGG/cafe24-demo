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
from agent.multi_agent import run_multi_agent_stream, get_cached_supervisor, get_cached_worker, AGENT_DESCRIPTIONS, INTENT_AGENT_MAP
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
            from langchain_openai import ChatOpenAI
            from agent.tools import ALL_TOOLS
            from agent.router import classify_and_get_tools, IntentCategory

            user_text = safe_str(req.user_input)
            rag_mode = req.rag_mode or "auto"
            api_key = pick_api_key(req.api_key)
            category, allowed_tool_names = classify_and_get_tools(user_text, api_key, use_llm_fallback=True)

            st.logger.info("STREAM_ROUTER category=%s allowed_tools=%s", category.value, allowed_tool_names)

            # multi_agent 플래그 또는 RETENTION 카테고리면 멀티에이전트 모드
            if req.multi_agent or category == IntentCategory.RETENTION:
                st.logger.info("STREAM_MULTI_AGENT_MODE category=%s multi_agent=%s user=%s", category.value, req.multi_agent, username)
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

            if not is_simple:
                try:
                    _rag_start = _time.time()
                    if rag_mode == "lightrag":
                        yield sse_pack("tool_start", {"tool": "search_platform_lightrag", "args": {"query": user_text, "mode": "hybrid"}})
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
                        yield sse_pack("tool_end", {"tool": "search_platform_lightrag", "elapsed_ms": _rag_elapsed})
                    elif rag_mode == "k2rag":
                        yield sse_pack("tool_start", {"tool": "k2rag_search", "args": {"query": user_text}})
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
                        yield sse_pack("tool_end", {"tool": "k2rag_search", "elapsed_ms": _rag_elapsed})
                    else:
                        yield sse_pack("tool_start", {"tool": "search_platform_docs", "args": {"query": user_text}})
                        rag_out = tool_rag_search(user_text, top_k=st.RAG_DEFAULT_TOPK, api_key=api_key)
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        if isinstance(rag_out, dict) and rag_out.get("status") == "success":
                            results = rag_out.get("results") or []
                            if results:
                                tool_calls_log.append({"tool": "rag_search", "args": {"query": user_text}, "result": {
                                    "status": rag_out.get("status", "success"),
                                    "query": user_text,
                                    "results_count": len(results),
                                    "sources": [r.get("source", "") for r in results[:5]],
                                }})
                                rag_context = "\n\n## 검색된 플랫폼 정보 (참고용):\n"
                                for r in results[:5]:
                                    content = r.get("content", "")[:800]
                                    source = r.get("source", "")
                                    if source:
                                        rag_context += f"- [출처: {source}] {content}\n"
                                    else:
                                        rag_context += f"- {content}\n"
                                tools = [t for t in tools if t.name != "search_platform_docs"]
                        yield sse_pack("tool_end", {"tool": "search_platform_docs", "elapsed_ms": _rag_elapsed})
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

            # === 간단 인사 → 직접 LLM 응답 (Supervisor 불필요) ===
            if is_simple:
                st.logger.info("STREAM_SIMPLE_MODE direct LLM response")
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

            # === 멀티에이전트 실행 ===
            from core.utils import normalize_model_name
            from core.memory import memory_messages

            agent_llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                streaming=True,
                max_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.7,
            )
            model_key = normalize_model_name(model_name)

            # 입력 메시지 구성
            prev_messages = memory_messages(username)
            input_messages = []
            for msg in prev_messages:
                role, content = msg.get("role", ""), msg.get("content", "")
                if role == "user":
                    input_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    input_messages.append({"role": "assistant", "content": content})

            # RAG 컨텍스트가 있으면 system message로 포함
            if rag_context:
                input_messages.append({
                    "role": "system",
                    "content": f"다음은 사전 검색된 플랫폼 정보입니다. 참고하여 답변하세요:\n{rag_context}"
                })

            input_messages.append({"role": "user", "content": user_text})

            # 키워드 사전 라우팅: intent가 명확하면 supervisor 우회 → 워커 직접 호출
            target_agent_name = INTENT_AGENT_MAP.get(category.value)
            use_direct_worker = False
            if target_agent_name:
                worker_graph = get_cached_worker(agent_llm, model_key, target_agent_name)
                if worker_graph:
                    use_direct_worker = True
                    st.logger.info("DIRECT_WORKER category=%s agent=%s (supervisor bypass)", category.value, target_agent_name)

            if use_direct_worker:
                # === 워커 직접 호출 (supervisor 우회 — 3초 절감) ===
                yield sse_pack("agent_start", {
                    "agent": target_agent_name,
                    "description": AGENT_DESCRIPTIONS.get(target_agent_name, target_agent_name),
                })

                current_tool = None
                tool_fail_counts_w = {}  # 워커 도구 실패 횟수
                MAX_TOOL_RETRIES = 3
                worker_aborted = False
                async for event in worker_graph.astream_events(
                    {"messages": input_messages},
                    version="v2",
                    config={"recursion_limit": 25},
                ):
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
                        # RAG 도구 결과는 크기가 클 수 있으므로 요약만 저장
                        if isinstance(tool_output, dict) and "results" in tool_output and isinstance(tool_output.get("results"), list):
                            results_list = tool_output["results"]
                            tool_output = {
                                "status": tool_output.get("status", "success"),
                                "query": tool_output.get("query", ""),
                                "results_count": len(results_list),
                                "sources": [r.get("source", "") for r in results_list[:5]],
                            }
                        is_error = (isinstance(tool_output, dict) and tool_output.get("status") == "error")
                        if is_error:
                            tool_fail_counts_w[end_tool_name] = tool_fail_counts_w.get(end_tool_name, 0) + 1
                        tool_calls_log.append({"tool": end_tool_name, "result": tool_output})
                        yield sse_pack("tool_end", {"tool": end_tool_name, "status": "error" if is_error else "success"})
                        current_tool = None
                        # 동일 도구 3회 실패 시 루프 탈출
                        if tool_fail_counts_w.get(end_tool_name, 0) >= MAX_TOOL_RETRIES:
                            st.logger.warning("TOOL_RETRY_LIMIT tool=%s fails=%d", end_tool_name, tool_fail_counts_w[end_tool_name])
                            error_msg = f"도구 '{end_tool_name}'이(가) {MAX_TOOL_RETRIES}회 연속 실패하여 분석을 중단합니다."
                            final_buf.append(error_msg)
                            yield sse_pack("delta", {"delta": "\n\n" + error_msg})
                            worker_aborted = True
                            break

                    elif kind == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if not chunk or getattr(chunk, "tool_call_chunks", None):
                            continue
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            final_buf.append(content)
                            yield sse_pack("delta", {"delta": content})

                yield sse_pack("agent_end", {
                    "agent": target_agent_name,
                    "description": AGENT_DESCRIPTIONS.get(target_agent_name, target_agent_name),
                })

                final_text = "".join(final_buf).strip() or "요청을 처리했습니다."
                append_memory(username, user_text, final_text)
                st.logger.info("STREAM_DONE_WORKER tool_calls_count=%d tool_calls=%s", len(tool_calls_log), [t.get("tool") for t in tool_calls_log])
                yield sse_pack("done", {"ok": True, "final": final_text, "tool_calls": tool_calls_log})
                return

            # === Supervisor 멀티에이전트 (PLATFORM, GENERAL 등 라우팅 필요) ===
            supervisor_graph = get_cached_supervisor(agent_llm, model_key)

            current_tool = None
            active_worker = None
            worker_responded = False
            tool_fail_counts_s = {}  # supervisor 도구 실패 횟수
            supervisor_aborted = False
            async for event in supervisor_graph.astream_events(
                {"messages": input_messages},
                version="v2",
                config={"recursion_limit": 25},
            ):
                if await request.is_disconnected():
                    return
                kind = event.get("event", "")
                data = event.get("data", {})
                metadata = event.get("metadata", {})

                # langgraph_checkpoint_ns 에서 외부 노드 이름 추출
                # 형식: "supervisor:UUID|agent:UUID" → 첫 세그먼트 = "supervisor"
                checkpoint_ns = metadata.get("langgraph_checkpoint_ns", "")
                outer_node = checkpoint_ns.split(":")[0] if checkpoint_ns else ""

                # 워커 → supervisor 복귀 감지:
                # active_worker가 있는 상태에서 supervisor 모델 시작 = 워커 완료
                if (active_worker
                    and outer_node == "supervisor"
                    and kind == "on_chat_model_start"):
                    yield sse_pack("agent_end", {
                        "agent": active_worker,
                        "description": AGENT_DESCRIPTIONS.get(active_worker, active_worker),
                    })
                    active_worker = None

                if kind == "on_tool_start":
                    tool_name = event.get("name", "도구")
                    tool_input = data.get("input", {})
                    # handoff 도구는 agent_start로 변환
                    if tool_name.startswith("transfer_to_"):
                        agent_name = tool_name.replace("transfer_to_", "")
                        active_worker = agent_name
                        yield sse_pack("agent_start", {
                            "agent": agent_name,
                            "description": AGENT_DESCRIPTIONS.get(agent_name, agent_name),
                        })
                    else:
                        current_tool = tool_name
                        yield sse_pack("tool_start", {"tool": tool_name, "args": tool_input})

                elif kind == "on_tool_end":
                    end_tool_name = event.get("name") or current_tool or "unknown"
                    if end_tool_name.startswith("transfer_to_"):
                        pass  # handoff 종료는 스킵
                    else:
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
                        # RAG 도구 결과는 크기가 클 수 있으므로 요약만 저장
                        if isinstance(tool_output, dict) and "results" in tool_output and isinstance(tool_output.get("results"), list):
                            results_list = tool_output["results"]
                            tool_output = {
                                "status": tool_output.get("status", "success"),
                                "query": tool_output.get("query", ""),
                                "results_count": len(results_list),
                                "sources": [r.get("source", "") for r in results_list[:5]],
                            }
                        is_error = (isinstance(tool_output, dict) and tool_output.get("status") == "error")
                        if is_error:
                            tool_fail_counts_s[end_tool_name] = tool_fail_counts_s.get(end_tool_name, 0) + 1
                        tool_calls_log.append({"tool": end_tool_name, "result": tool_output})
                        yield sse_pack("tool_end", {"tool": end_tool_name, "status": "error" if is_error else "success"})
                        current_tool = None
                        if tool_fail_counts_s.get(end_tool_name, 0) >= 3:
                            st.logger.warning("TOOL_RETRY_LIMIT tool=%s fails=%d", end_tool_name, tool_fail_counts_s[end_tool_name])
                            error_msg = f"도구 '{end_tool_name}'이(가) 3회 연속 실패하여 분석을 중단합니다."
                            final_buf.append(error_msg)
                            yield sse_pack("delta", {"delta": "\n\n" + error_msg})
                            supervisor_aborted = True
                            break

                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if not chunk or getattr(chunk, "tool_call_chunks", None):
                        continue
                    content = getattr(chunk, "content", "")
                    if not isinstance(content, str) or not content:
                        continue

                    if outer_node == "supervisor" and not worker_responded:
                        # supervisor 직접 응답 (워커 미사용 시)
                        final_buf.append(content)
                        yield sse_pack("delta", {"delta": content})
                    elif outer_node != "supervisor" and active_worker:
                        # 워커 에이전트의 텍스트 응답 → 직접 스트리밍
                        worker_responded = True
                        final_buf.append(content)
                        yield sse_pack("delta", {"delta": content})

            # 루프 종료 후: 마지막 워커가 agent_end 없이 끝난 경우 처리
            if active_worker:
                yield sse_pack("agent_end", {
                    "agent": active_worker,
                    "description": AGENT_DESCRIPTIONS.get(active_worker, active_worker),
                })

            final_text = "".join(final_buf).strip() or "요청을 처리했습니다."
            append_memory(username, user_text, final_text)
            st.logger.info("STREAM_DONE tool_calls_count=%d tool_calls=%s", len(tool_calls_log), [t.get("tool") for t in tool_calls_log])
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
