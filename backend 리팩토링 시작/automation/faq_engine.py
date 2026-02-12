"""
automation/faq_engine.py - CS FAQ 자동 생성 엔진
================================================
CS 문의 패턴 분석 -> LLM FAQ 자동 생성 -> 승인 관리
카페24 PRO CS 패턴: 문의 패턴 분석 -> FAQ 자동 생성
"""
import json
import uuid
import time
from typing import Dict, List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str, safe_int, safe_float
from core.constants import CS_TICKET_CATEGORIES
from agent.llm import get_llm, invoke_with_retry, pick_api_key
from automation.action_logger import (
    save_faq,
    get_faq,
    get_all_faqs,
    delete_faq,
    update_faq_status,
    log_action,
    create_pipeline_run,
    update_pipeline_step,
    complete_pipeline_run,
)
import state as st


# ── FAQ 생성용 시스템 프롬프트 ──
_FAQ_SYSTEM_PROMPT = (
    "당신은 카페24 이커머스 플랫폼 CS 전문가입니다.\n"
    "고객 문의 패턴을 분석하여 FAQ를 생성합니다.\n"
    "각 FAQ는 question, answer, category, tags 형식으로 JSON 배열로 반환하세요.\n"
    "반드시 유효한 JSON 배열만 출력하세요. 다른 텍스트는 포함하지 마세요.\n\n"
    "중요 규칙:\n"
    "- 특정 카테고리가 지정된 경우, 모든 FAQ의 category 값은 반드시 해당 카테고리와 동일해야 합니다.\n"
    "- 지정된 카테고리 외의 FAQ는 절대 생성하지 마세요.\n"
    "- 허용 카테고리: 배송, 환불, 결제, 상품, 계정, 정산, 기술지원, 마케팅, 기타"
)


def analyze_cs_patterns(top_n: int = 10) -> Dict[str, Any]:
    """CS 문의 패턴을 분석하여 빈출 유형을 추출합니다."""
    if st.CS_STATS_DF is None:
        st.logger.warning("CS_STATS_DF is None - CS 통계 데이터 없음")
        return {
            "total_inquiries": 0,
            "categories": [],
            "top_patterns": [],
            "warning": "CS 통계 데이터가 로드되지 않았습니다.",
        }

    df = st.CS_STATS_DF
    cat_col = "category" if "category" in df.columns else "ticket_category"

    if cat_col not in df.columns or "total_tickets" not in df.columns:
        st.logger.warning("CS_STATS_DF 필수 컬럼 누락: %s", list(df.columns))
        return {
            "total_inquiries": 0,
            "categories": [],
            "top_patterns": [],
            "warning": "CS 데이터에 필수 컬럼이 없습니다.",
        }

    total = safe_int(df["total_tickets"].sum())

    categories = []
    for _, row in df.iterrows():
        count = safe_int(row["total_tickets"])
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        categories.append({
            "category": safe_str(row[cat_col]),
            "count": count,
            "percentage": pct,
        })

    # 건수 내림차순 정렬
    categories.sort(key=lambda x: x["count"], reverse=True)
    top_patterns = categories[:top_n]

    return {
        "total_inquiries": total,
        "categories": categories,
        "top_patterns": top_patterns,
    }


def generate_faq_items(
    category: Optional[str] = None,
    count: int = 5,
    api_key: str = "",
) -> Dict[str, Any]:
    """CS 패턴 분석 결과 기반으로 LLM을 사용하여 FAQ를 자동 생성합니다."""
    run_id = None
    try:
        # 카테고리 유효성 검증
        if category and category not in CS_TICKET_CATEGORIES:
            return {
                "generated_count": 0, "faqs": [],
                "error": f"유효하지 않은 카테고리입니다: {category}. "
                         f"사용 가능: {', '.join(CS_TICKET_CATEGORIES)}",
            }

        api_key = pick_api_key(api_key)
        if not api_key:
            return {"generated_count": 0, "faqs": [], "error": "API 키가 설정되지 않았습니다."}

        run_id = create_pipeline_run("faq", ["analyze", "select", "generate", "review", "approve"])
        update_pipeline_step(run_id, "analyze", "processing")

        # CS 패턴 분석
        patterns = analyze_cs_patterns()
        update_pipeline_step(run_id, "analyze", "complete", {"patterns": len(patterns.get("top_patterns", []))})
        update_pipeline_step(run_id, "select", "complete", {"category": category or "all"})

        if not patterns.get("top_patterns"):
            return {
                "generated_count": 0,
                "faqs": [],
                "warning": "CS 문의 패턴 데이터가 없어 기본 카테고리로 생성합니다.",
            }

        update_pipeline_step(run_id, "generate", "processing")

        # 카테고리 필터링
        if category:
            target_patterns = [p for p in patterns["top_patterns"] if p["category"] == category]
            if not target_patterns:
                target_patterns = [{"category": category, "count": 0, "percentage": 0}]
        else:
            target_patterns = patterns["top_patterns"]

        # LLM 프롬프트 구성
        pattern_text = "\n".join(
            f"- {p['category']}: {p['count']}건 ({p['percentage']}%)"
            for p in target_patterns[:5]
        )

        # 카테고리 제약 조건
        if category:
            category_constraint = (
                f"\n\n[필수 조건]\n"
                f"반드시 '{category}' 카테고리에 해당하는 FAQ만 생성하세요.\n"
                f"모든 FAQ의 category 값은 반드시 '{category}'이어야 합니다.\n"
                f"다른 카테고리의 FAQ는 생성하지 마세요."
            )
        else:
            category_constraint = ""

        user_prompt = (
            f"아래 카페24 CS 문의 패턴을 분석하여 FAQ {count}개를 생성하세요.\n\n"
            f"[문의 패턴]\n{pattern_text}"
            f"{category_constraint}\n\n"
            f"각 FAQ는 다음 형식의 JSON 배열로 반환하세요:\n"
            f'[{{"question": "...", "answer": "...", "category": "...", "tags": ["태그1", "태그2"]}}]\n\n'
            f"답변은 카페24 이커머스 플랫폼 맥락에 맞게 구체적으로 작성하세요."
        )

        # LLM 설정
        settings = st.get_active_llm_settings()
        llm = get_llm(
            model=settings.get("selectedModel", "gpt-4o-mini"),
            api_key=api_key,
            max_tokens=settings.get("maxTokens", 4000),
            streaming=False,
            temperature=0.7,
        )

        messages = [
            SystemMessage(content=_FAQ_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        raw = invoke_with_retry(llm, messages)

        # JSON 파싱
        faq_list = _parse_faq_json(raw)
        if not faq_list:
            st.logger.error("FAQ JSON 파싱 실패: %s", raw[:200])
            return {"generated_count": 0, "faqs": [], "error": "LLM 응답 파싱 실패"}

        # FAQ 저장 (카테고리 후처리 강제)
        saved_faqs = []
        for item in faq_list[:count]:
            faq_id = str(uuid.uuid4())[:8]
            faq_data = {
                "id": faq_id,
                "question": safe_str(item.get("question")),
                "answer": safe_str(item.get("answer")),
                "category": category if category else safe_str(item.get("category")),
                "tags": item.get("tags", []),
                "status": "draft",
                "created_at": time.time(),
            }
            save_faq(faq_id, faq_data)
            saved_faqs.append(faq_data)

        update_pipeline_step(run_id, "generate", "complete", {"count": len(saved_faqs)})

        log_action(
            "faq_generate",
            "system",
            {"count": len(saved_faqs), "category": category or "all"},
        )

        return {"generated_count": len(saved_faqs), "faqs": saved_faqs, "pipeline_run_id": run_id}

    except Exception as e:
        st.logger.error("FAQ 생성 실패: %s", str(e))
        if run_id:
            update_pipeline_step(run_id, "generate", "error", {"error": str(e)})
        return {"generated_count": 0, "faqs": [], "error": str(e)}


def approve_faq(faq_id: str) -> Dict[str, Any]:
    """FAQ를 승인 상태로 변경합니다."""
    ok = update_faq_status(faq_id, "approved")
    if not ok:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}

    log_action("faq_approve", faq_id, {"faq_id": faq_id})
    return {"status": "success", "faq_id": faq_id}


def update_faq(
    faq_id: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> Dict[str, Any]:
    """FAQ 내용을 수정합니다."""
    existing = get_faq(faq_id)
    if not existing:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}

    updated_fields = []
    if question is not None:
        existing["question"] = question
        updated_fields.append("question")
    if answer is not None:
        existing["answer"] = answer
        updated_fields.append("answer")

    if not updated_fields:
        return {"status": "error", "message": "수정할 필드가 없습니다."}

    existing["updated_at"] = time.time()
    save_faq(faq_id, existing)

    log_action("faq_update", faq_id, {"updated_fields": updated_fields})
    return {"status": "success", "faq_id": faq_id, "updated_fields": updated_fields}


def delete_faq_item(faq_id: str) -> Dict[str, Any]:
    """FAQ를 삭제합니다."""
    ok = delete_faq(faq_id)
    if not ok:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}

    log_action("faq_delete", faq_id, {"faq_id": faq_id})
    return {"status": "success", "faq_id": faq_id}


def list_faqs(status: Optional[str] = None) -> Dict[str, Any]:
    """FAQ 목록을 조회합니다. status로 필터링 가능 (draft/approved/all)."""
    all_faqs = get_all_faqs()

    if status and status != "all":
        all_faqs = [f for f in all_faqs if f.get("status") == status]

    return {"total": len(all_faqs), "faqs": all_faqs}


# ── 내부 유틸리티 ──
def _parse_faq_json(raw: str) -> List[Dict[str, Any]]:
    """LLM 응답에서 FAQ JSON 배열을 파싱합니다."""
    # 직접 파싱 시도
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # ```json ... ``` 블록 추출
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # [ ... ] 블록 추출
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    return []
