"""
agent/runner.py - CAFE24 AI 운영 플랫폼 에이전트 실행 (Tool Calling 방식)
LLM이 직접 도구를 선택하고 호출합니다.
Rule-based 전처리로 필수 도구를 강제 호출합니다.

멀티 에이전트 모드 지원:
- agent_mode="single": 기존 단일 에이전트 + 다중 도구 (기본값)
- agent_mode="multi": LangGraph 기반 멀티 에이전트 시스템
"""
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Set

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str, format_openai_error, normalize_model_name, json_sanitize
from core.memory import append_memory, memory_messages
from agent.tool_schemas import ALL_TOOLS
from agent.llm import get_llm, pick_api_key
from agent.intent import detect_intent
from agent.router import classify_and_get_tools, IntentCategory
import state as st

# 멀티 에이전트 지원 확인
try:
    from agent.multi_agent import run_multi_agent, LANGGRAPH_AVAILABLE
except ImportError:
    LANGGRAPH_AVAILABLE = False
    run_multi_agent = None


# RAG 도구 이름 (분석 질문에서 제외)
RAG_TOOL_NAMES = {"search_platform", "search_platform_lightrag"}


MAX_TOOL_ITERATIONS = 10  # 무한 루프 방지

# 키워드-도구 매핑 (Rule-based 전처리용)
KEYWORD_TOOL_MAPPING = {
    "detect_fraud": ["부정행위 탐지", "비정상 셀러", "어뷰징", "사기 탐지", "허위 주문", "리뷰 조작"],
    "get_fraud_statistics": ["이상거래", "이상 거래", "이상 셀러", "부정행위 통계", "부정행위 현황", "사기 통계", "사기 현황", "fraud 통계", "부정 거래", "이상거래 탐지", "이상거래 현황"],
    "get_segment_statistics": ["세그먼트 통계", "셀러 세그먼트", "셀러 분포", "세그먼트 분석", "세그먼트 현황",
                               "성장형 셀러", "휴면 셀러", "우수 셀러", "파워 셀러", "관리 필요 셀러"],
    "get_order_statistics": ["운영 이벤트", "이벤트 통계", "주문 이벤트", "정산 이벤트", "이벤트 현황"],
    "get_cs_statistics": ["CS 통계", "상담 현황", "상담 품질", "CS 현황"],
    "classify_inquiry": ["카테고리 분류", "문의 분류", "분류해줘", "분류해 줘"],
    "get_dashboard_summary": ["대시보드", "전체 현황", "요약 통계", "셀러 활동", "활동 현황", "전체 셀러", "플랜별 분포", "등급별 분포", "티어별 분포", "전체 쇼핑몰", "쇼핑몰 수", "쇼핑몰 개수", "쇼핑몰 몇", "총 쇼핑몰"],
    # 쇼핑몰 목록/검색 도구
    "list_shops": ["쇼핑몰 목록", "쇼핑몰 리스트", "쇼핑몰 현황", "등급 쇼핑몰", "티어 쇼핑몰", "플랜 쇼핑몰"],
    "list_categories": ["카테고리 목록", "카테고리 전체", "카테고리 정보", "업종 목록", "업종 전체", "업종 정보", "업종 현황"],
    # ML 모델 예측 도구
    "predict_seller_churn": ["이탈 예측", "이탈 확률", "이탈 위험", "이탈률", "churn", "셀러 이탈"],
    "get_shop_performance": ["성과 분석", "쇼핑몰 성과", "쇼핑몰 매출", "성과 예측", "매출 분석"],
    "optimize_marketing": ["마케팅 추천", "마케팅 최적화", "광고 추천", "광고 전략", "마케팅 예산", "ROAS 최적화", "마케팅 예산 최적화", "광고 최적화", "마케팅 분석"],
    # 분석 도구
    "get_churn_prediction": ["이탈 분석", "이탈 현황", "이탈 통계", "고위험 셀러", "이탈 요인"],
    "get_cohort_analysis": ["코호트 분석", "리텐션 분석", "코호트 리텐션", "주간 리텐션", "잔존율"],
    "get_trend_analysis": ["트렌드 분석", "KPI 분석", "지표 분석", "DAU 분석", "상관관계", "활성 셀러", "가입 추이", "전환율", "신규 가입", "변화 분석", "추이 분석", "주문량 분석"],
    "get_gmv_prediction": ["매출 예측", "GMV 분석", "GMV 예측", "수익 분석", "ARPU", "ARPPU", "거래액"],
}


def extract_seller_id(text: str) -> str | None:
    """텍스트에서 셀러 ID 추출 (SEL0001 ~ SEL000001 형식, 1~6자리 지원)"""
    match = re.search(r'SEL\d{1,6}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_shop_id(text: str) -> str | None:
    """텍스트에서 쇼핑몰 ID 추출 (S0001 형식)"""
    match = re.search(r'S\d{4,6}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_order_id(text: str) -> str | None:
    """텍스트에서 주문 ID 추출 (O0001 형식)"""
    match = re.search(r'O\d{4,8}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_days(text: str) -> int | None:
    """텍스트에서 일수 추출 (최근 N일, N일간 등)"""
    patterns = [
        r'최근\s*(\d+)\s*일',
        r'(\d+)\s*일\s*(?:간|동안|기준)',
        r'지난\s*(\d+)\s*일',
        r'(\d+)days?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_date_range(text: str) -> tuple[str | None, str | None]:
    """텍스트에서 날짜 범위 추출 (YYYY-MM-DD 형식)"""
    # YYYY-MM-DD 패턴
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, text)
    if len(dates) >= 2:
        return dates[0], dates[1]
    elif len(dates) == 1:
        return dates[0], None
    return None, None


def extract_month(text: str) -> str | None:
    """텍스트에서 월 추출 (YYYY-MM 또는 N월 형식)"""
    # YYYY-MM 패턴
    match = re.search(r'(\d{4}-\d{2})', text)
    if match:
        return match.group(1)

    # N월 패턴 (현재 연도 기준)
    match = re.search(r'(\d{1,2})월', text)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            # 연도 추출 시도
            year_match = re.search(r'(\d{4})년', text)
            if year_match:
                year = int(year_match.group(1))
            else:
                year = datetime.now().year
            return f"{year}-{month:02d}"
    return None


def extract_risk_level(text: str) -> str | None:
    """텍스트에서 위험 등급 추출 (high/medium/low)"""
    text_lower = text.lower()
    if '고위험' in text or 'high' in text_lower:
        return 'high'
    elif '중위험' in text or 'medium' in text_lower:
        return 'medium'
    elif '저위험' in text or 'low' in text_lower:
        return 'low'
    return None


def extract_cohort(text: str) -> str | None:
    """텍스트에서 코호트명 추출 (YYYY-MM WN 형식)"""
    match = re.search(r'(\d{4}-\d{2}\s*W\d)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper().replace(' ', ' ')
    return None


def detect_required_tools(text: str) -> Set[str]:
    """텍스트에서 필수 도구 감지"""
    required = set()
    text_lower = text.lower()

    for tool_name, keywords in KEYWORD_TOOL_MAPPING.items():
        for keyword in keywords:
            if keyword in text_lower or keyword in text:
                required.add(tool_name)
                break

    return required


def execute_tool_by_name(tool_name: str, args: dict) -> dict:
    """도구 이름으로 실행"""
    for t in ALL_TOOLS:
        if t.name == tool_name:
            try:
                return t.invoke(args)
            except Exception as e:
                return {"status": "FAILED", "error": safe_str(e)}
    return {"status": "FAILED", "error": f"도구 '{tool_name}'을 찾을 수 없습니다."}


def run_agent(req, username: str) -> dict:
    """
    에이전트 실행 (모드 선택 가능).

    Args:
        req: 요청 객체 (user_input, model, agent_mode 등)
        username: 사용자 이름

    agent_mode:
        - "single" (기본값): 단일 에이전트 + 다중 도구
        - "multi": LangGraph 기반 멀티 에이전트 (Coordinator → 전문 에이전트)
    """
    # 멀티 에이전트 모드 체크
    agent_mode = getattr(req, "agent_mode", "single")

    if agent_mode == "multi":
        if not LANGGRAPH_AVAILABLE or run_multi_agent is None:
            return {
                "status": "FAILED",
                "response": "멀티 에이전트 모드를 사용하려면 langgraph를 설치하세요: pip install langgraph",
                "tool_calls": [],
                "log_file": st.LOG_FILE,
            }
        return run_multi_agent(req, username)

    # 기존 단일 에이전트 모드
    user_text = safe_str(req.user_input)

    st.logger.info(
        "AGENT_START user=%s model=%s input_len=%s mode=tool_calling",
        username, normalize_model_name(req.model), len(user_text),
    )

    try:
        # ========== LLM Router 패턴: 의도 분류 → 도구 필터링 ==========
        api_key = pick_api_key(req.api_key)
        category, allowed_tool_names = classify_and_get_tools(
            user_text,
            api_key,
            use_llm_fallback=True,  # 키워드 분류 실패 시 LLM 사용
        )

        st.logger.info(
            "ROUTER_RESULT user=%s category=%s tools=%s",
            username, category.value, allowed_tool_names,
        )

        # 카테고리에 해당하는 도구만 필터링
        if allowed_tool_names:
            filtered_tools = [t for t in ALL_TOOLS if t.name in allowed_tool_names]

            # 도구가 없으면 전체 도구 사용 (fallback)
            if not filtered_tools:
                st.logger.warning(
                    "ROUTER_NO_TOOLS category=%s, fallback to ALL_TOOLS",
                    category.value,
                )
                filtered_tools = ALL_TOOLS
            else:
                st.logger.info(
                    "TOOL_FILTER category=%s, tools=%d→%d",
                    category.value, len(ALL_TOOLS), len(filtered_tools),
                )
        else:
            # GENERAL 카테고리: 도구 없이 대화만
            if category == IntentCategory.GENERAL:
                filtered_tools = []
                st.logger.info("ROUTER_GENERAL_MODE no tools bound")
            else:
                filtered_tools = ALL_TOOLS

        # API 키 확인
        if not api_key:
            msg = "처리 오류: OpenAI API Key가 없습니다."
            append_memory(username, user_text, msg)
            return {"status": "FAILED", "response": msg, "tool_calls": [], "log_file": st.LOG_FILE}

        # LLM 생성 및 도구 바인딩
        # 사용자 설정 temperature 사용 (기본값: 0.3)
        user_temperature = req.temperature if req.temperature is not None else 0.3
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=user_temperature, top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        # 도구가 있으면 바인딩, 없으면 일반 LLM 사용
        # tool_choice는 auto (기본값) - required는 ReAct 에이전트에서 무한 루프 유발
        if filtered_tools:
            llm_with_tools = llm.bind_tools(filtered_tools)
        else:
            llm_with_tools = llm  # GENERAL 카테고리: 도구 없이 대화

        # 시스템 프롬프트 구성 — constants.py의 DEFAULT_SYSTEM_PROMPT를 단일 소스로 사용
        system_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT

        # 메시지 구성 (이전 대화 기록 포함)
        messages: List = [
            SystemMessage(content=system_prompt),
        ]

        # 이전 대화 기록 추가 (맥락 유지)
        prev_messages = memory_messages(username)
        for msg in prev_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 현재 사용자 입력 추가
        messages.append(HumanMessage(content=user_text))

        tool_calls_log: List[Dict[str, Any]] = []
        iteration = 0

        # ========== Rule-based 전처리: 필수 도구 강제 호출 ==========
        required_tools = detect_required_tools(user_text)
        seller_id = extract_seller_id(user_text)

        if required_tools:
            st.logger.info(
                "RULE_BASED_PREPROCESS user=%s required_tools=%s seller_id=%s",
                username, required_tools, seller_id,
            )

            # 필수 도구 강제 실행
            forced_results = []
            shop_id = extract_shop_id(user_text)

            # 분석 도구용 파라미터 추출
            days = extract_days(user_text)
            start_date, end_date = extract_date_range(user_text)
            month = extract_month(user_text)
            risk_level = extract_risk_level(user_text)
            cohort = extract_cohort(user_text)

            for tool_name in required_tools:
                # 도구별 인자 설정
                if tool_name == "detect_fraud" and seller_id:
                    args = {"seller_id": seller_id}
                elif tool_name == "predict_seller_churn" and seller_id:
                    args = {"seller_id": seller_id}
                elif tool_name == "optimize_marketing" and seller_id:
                    args = {"seller_id": seller_id}
                elif tool_name == "get_shop_performance" and shop_id:
                    args = {"shop_id": shop_id}
                elif tool_name == "get_churn_prediction":
                    args = {}
                    if risk_level:
                        args["risk_level"] = risk_level
                elif tool_name == "get_cohort_analysis":
                    args = {}
                    if month:
                        args["month"] = month
                    elif cohort:
                        # cohort에서 월 형식 추출 (예: "2024-11 W1" → "2024-11")
                        import re as _re
                        m = _re.search(r'(\d{4}-\d{2})', cohort)
                        if m:
                            args["month"] = m.group(1)
                        else:
                            args["month"] = cohort
                elif tool_name == "get_trend_analysis":
                    args = {}
                    if days:
                        args["days"] = days
                    elif start_date:
                        args["start_date"] = start_date
                        if end_date:
                            args["end_date"] = end_date
                elif tool_name == "get_gmv_prediction":
                    args = {}
                    if days:
                        args["days"] = days
                    elif start_date:
                        args["start_date"] = start_date
                        if end_date:
                            args["end_date"] = end_date
                elif tool_name == "get_order_statistics":
                    args = {}
                    if days:
                        args["days"] = days
                    # 텍스트에서 이벤트 타입 추출
                    _et_map = {
                        "주문": "order_received", "정산": "payment_settled",
                        "환불": "refund_processed", "CS": "cs_ticket",
                        "마케팅": "marketing_campaign", "상품 등록": "product_listed",
                    }
                    for kw, et in _et_map.items():
                        if kw in user_text:
                            args["event_type"] = et
                            break
                elif tool_name in ["get_segment_statistics", "get_cs_statistics", "get_dashboard_summary", "get_fraud_statistics"]:
                    args = {}
                elif tool_name == "list_shops":
                    args = {}
                    # 카테고리 필터 추출
                    for cat in ["패션", "뷰티", "식품", "가전", "리빙", "디지털"]:
                        if cat in user_text:
                            args["category"] = cat
                            break
                    # 티어 필터 추출
                    tier_map = {
                        "프리미엄": "프리미엄", "premium": "프리미엄",
                        "스탠다드": "스탠다드", "standard": "스탠다드",
                        "베이직": "베이직", "basic": "베이직",
                        "엔터프라이즈": "엔터프라이즈", "enterprise": "엔터프라이즈",
                    }
                    text_lower_for_tier = user_text.lower()
                    for kw, tier_val in tier_map.items():
                        if kw in text_lower_for_tier:
                            args["tier"] = tier_val
                            break
                elif tool_name == "list_categories":
                    args = {}
                elif tool_name == "classify_inquiry":
                    # 따옴표로 감싼 텍스트 추출, 없으면 전체 텍스트 사용
                    quoted = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', user_text)
                    if not quoted:
                        quoted = re.findall(r"['\u2018\u2019](.*?)['\u2018\u2019]", user_text)
                    args = {"text": quoted[0] if quoted else user_text}
                else:
                    args = {}

                # 도구 실행
                result = execute_tool_by_name(tool_name, args)

                st.logger.info(
                    "FORCED_TOOL_CALL user=%s tool=%s args=%s status=%s",
                    username, tool_name, args, result.get("status", "UNKNOWN"),
                )

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "forced": True,  # 강제 호출 표시
                })

                forced_results.append({
                    "tool": tool_name,
                    "result": result,
                })

            # 강제 호출 결과를 메시지에 추가
            forced_context = "다음은 사용자 요청에 따라 자동으로 실행된 도구 결과입니다:\n\n"
            for fr in forced_results:
                try:
                    result_str = json.dumps(json_sanitize(fr["result"]), ensure_ascii=False, indent=2)
                except Exception:
                    result_str = safe_str(fr["result"])
                forced_context += f"### {fr['tool']} 결과:\n```json\n{result_str}\n```\n\n"

            forced_context += "위 결과를 종합하여 사용자에게 답변을 작성하세요. 단순히 숫자를 나열하지 말고, 데이터에서 발견되는 추세·이상값·패턴을 분석하고 실행 가능한 인사이트와 구체적인 제안을 반드시 포함하세요."

            # 메시지에 도구 결과 컨텍스트 추가
            messages.append(HumanMessage(content=forced_context))

        # ========== Tool Calling 루프 ==========
        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            # LLM 호출
            response = llm_with_tools.invoke(messages)

            # 도구 호출이 없으면 최종 응답
            if not response.tool_calls:
                final_text = safe_str(response.content).strip()
                if not final_text:
                    final_text = "요청을 처리했습니다."

                append_memory(username, user_text, final_text)

                st.logger.info(
                    "AGENT_COMPLETE user=%s iterations=%s tools_used=%s",
                    username, iteration, len(tool_calls_log),
                )

                return {
                    "status": "SUCCESS",
                    "response": final_text,
                    "tool_calls": tool_calls_log,
                    "log_file": st.LOG_FILE,
                    "mode": "tool_calling",
                    "iterations": iteration,
                }

            # 도구 실행
            messages.append(response)  # AI 메시지 추가

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                st.logger.info(
                    "TOOL_CALL user=%s tool=%s args=%s",
                    username, tool_name, json.dumps(tool_args, ensure_ascii=False),
                )

                # 도구 찾기 및 실행
                tool_result = {"status": "FAILED", "error": f"도구 '{tool_name}'을 찾을 수 없습니다."}
                for t in ALL_TOOLS:
                    if t.name == tool_name:
                        try:
                            tool_result = t.invoke(tool_args)
                        except Exception as e:
                            tool_result = {"status": "FAILED", "error": safe_str(e)}
                            st.logger.exception("TOOL_EXEC_FAIL tool=%s err=%s", tool_name, e)
                        break

                # 결과 로깅
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                # ToolMessage 추가
                try:
                    result_str = json.dumps(json_sanitize(tool_result), ensure_ascii=False)
                except Exception:
                    result_str = safe_str(tool_result)

                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tool_id,
                ))

        # 최대 반복 도달
        st.logger.warning("AGENT_MAX_ITERATIONS user=%s", username)
        final_text = "요청 처리 중 최대 반복 횟수에 도달했습니다."
        append_memory(username, user_text, final_text)

        return {
            "status": "SUCCESS",
            "response": final_text,
            "tool_calls": tool_calls_log,
            "log_file": st.LOG_FILE,
            "mode": "tool_calling",
            "iterations": iteration,
            "max_iterations_reached": True,
        }

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("AGENT_FAIL err=%s", err)

        msg = f"처리 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        append_memory(username, user_text, msg)

        if req.debug:
            return {
                "status": "FAILED",
                "response": msg,
                "tool_calls": [],
                "debug_error": err,
                "log_file": st.LOG_FILE,
            }

        return {
            "status": "FAILED",
            "response": "처리 오류가 발생했습니다.",
            "tool_calls": [],
            "log_file": st.LOG_FILE,
        }
