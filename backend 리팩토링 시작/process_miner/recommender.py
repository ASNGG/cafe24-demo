"""
process_miner/recommender.py
LLM 기반 자동화 추천 + SOP 생성 엔진
프로세스 발견 결과와 병목 분석 결과를 종합하여 자동화 추천안을 생성한다.
"""

import json
import re
from typing import Any

import state as st

# ---------------------------------------------------------------------------
# LLM 프롬프트 템플릿
# ---------------------------------------------------------------------------
_RECOMMEND_PROMPT = """\
당신은 e-커머스 운영 자동화 전문가입니다.
아래 프로세스 마이닝 결과를 분석하여 자동화 추천과 SOP 문서를 JSON으로 작성하세요.

## 프로세스 유형
{process_type}

## 프로세스 패턴 (상위 빈도순)
{patterns_text}

## 병목 구간 (소요시간 상위순)
{bottleneck_text}

## 응답 형식 (반드시 아래 JSON 구조로만 응답)
```json
{{
  "recommendations": [
    {{
      "target_step": "단계A → 단계B",
      "current_avg_minutes": 숫자,
      "automation_type": "rule_based | ml_based | llm_based",
      "description": "구체적 자동화 방안 설명",
      "expected_improvement": "개선 예상 수치",
      "priority": "high | medium | low",
      "implementation_effort": "low | medium | high"
    }}
  ],
  "sop_document": "마크다운 형식 SOP 문서 (## 제목으로 시작)",
  "summary": "전체 분석 요약 1~2문장",
  "estimated_time_saving_percent": 숫자
}}
```

규칙:
- recommendations는 최대 5개까지만
- priority가 높은 순으로 정렬
- 한국어로 작성
- JSON만 출력 (```json 블록 없이 순수 JSON)
"""

# ---------------------------------------------------------------------------
# 프로세스 유형 한글 매핑
# ---------------------------------------------------------------------------
_PROCESS_TYPE_KR = {
    "order": "주문 처리",
    "cs": "CS 문의 처리",
    "settlement": "정산 처리",
}


def _format_patterns(discovery_result: dict, top_n: int = 5) -> str:
    """프로세스 발견 결과에서 패턴 텍스트를 생성한다."""
    patterns = discovery_result.get("top_patterns", [])[:top_n]
    if not patterns:
        return "(패턴 데이터 없음)"
    lines = []
    for i, p in enumerate(patterns, 1):
        path = " → ".join(p.get("sequence", []))
        count = p.get("count", 0)
        ratio = p.get("ratio", 0)
        lines.append(f"{i}. {path}  (빈도: {count}, 비율: {ratio * 100:.1f}%)")
    return "\n".join(lines)


def _format_bottlenecks(bottleneck_result: dict, top_n: int = 5) -> str:
    """병목 분석 결과에서 병목 텍스트를 생성한다."""
    bottlenecks = bottleneck_result.get("bottlenecks", [])[:top_n]
    if not bottlenecks:
        return "(병목 데이터 없음)"
    lines = []
    for i, b in enumerate(bottlenecks, 1):
        transition = f"{b.get('from_step', '')} → {b.get('to_step', '')}"
        avg_min = b.get("avg_duration_minutes", 0)
        p95_min = b.get("p95_duration_minutes", 0)
        lines.append(f"{i}. {transition}  (평균: {avg_min:.1f}분, P95: {p95_min:.1f}분)")
    return "\n".join(lines)


def _parse_llm_json(text: str) -> dict | None:
    """LLM 응답에서 JSON을 추출한다."""
    # ```json ... ``` 블록 추출 시도
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 첫 번째 { ... 마지막 } 사이 추출
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _fallback_recommend(
    discovery_result: dict,
    bottleneck_result: dict,
    process_type: str,
) -> dict:
    """API 키가 없거나 LLM 호출 실패 시 규칙 기반 추천을 반환한다."""
    type_kr = _PROCESS_TYPE_KR.get(process_type, process_type)
    bottlenecks = bottleneck_result.get("bottlenecks", [])[:5]
    patterns = discovery_result.get("top_patterns", [])[:5]

    recommendations = []
    for b in bottlenecks:
        transition = f"{b.get('from_step', '?')} → {b.get('to_step', '?')}"
        avg_min = float(b.get("avg_duration_minutes", 0))
        if avg_min > 60:
            priority = "high"
            auto_type = "ml_based"
            desc = f"평균 {avg_min:.0f}분 소요되는 병목 구간. ML 기반 자동화로 대기시간 단축 권장."
        elif avg_min > 20:
            priority = "medium"
            auto_type = "rule_based"
            desc = f"평균 {avg_min:.0f}분 소요. 규칙 기반 자동 라우팅으로 개선 가능."
        else:
            priority = "low"
            auto_type = "rule_based"
            desc = f"평균 {avg_min:.0f}분 소요. 간단한 규칙 자동화로 소폭 개선 가능."

        recommendations.append({
            "target_step": transition,
            "current_avg_minutes": round(avg_min, 1),
            "automation_type": auto_type,
            "description": desc,
            "expected_improvement": f"약 {min(int(avg_min * 0.5), 80)}% 시간 절감 예상",
            "priority": priority,
            "implementation_effort": "medium",
        })

    # 반복 패턴이 많은 구간 추가 추천
    for p in patterns[:2]:
        seq = p.get("sequence", [])
        count = p.get("count", 0)
        if count > 50 and len(seq) >= 2:
            step = f"{seq[0]} → {seq[1]}"
            if not any(r["target_step"] == step for r in recommendations):
                recommendations.append({
                    "target_step": step,
                    "current_avg_minutes": 0,
                    "automation_type": "rule_based",
                    "description": f"빈도 {count}회의 반복 패턴. 규칙 기반 자동화 추천.",
                    "expected_improvement": "처리 일관성 향상",
                    "priority": "medium",
                    "implementation_effort": "low",
                })

    # priority 정렬
    order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: order.get(r["priority"], 9))

    total_avg = sum(b.get("avg_duration_minutes", 0) for b in bottlenecks)
    saving_pct = min(40, int(total_avg * 0.3)) if total_avg else 10

    return {
        "recommendations": recommendations[:5],
        "sop_document": _generate_fallback_sop(type_kr, patterns),
        "summary": f"{type_kr} 프로세스에서 {len(recommendations)}개 구간의 자동화를 통해 약 {saving_pct}% 효율 개선이 가능합니다. (규칙 기반 분석)",
        "estimated_time_saving_percent": saving_pct,
    }


def _generate_fallback_sop(type_kr: str, patterns: list[dict]) -> str:
    """규칙 기반 SOP 문서를 생성한다."""
    lines = [f"## {type_kr} 표준 운영 절차 (SOP)", ""]
    if not patterns:
        lines.append("프로세스 패턴 데이터가 부족하여 SOP를 생성할 수 없습니다.")
        return "\n".join(lines)

    # 가장 빈도 높은 패턴을 기준으로 SOP 작성
    top = patterns[0]
    path = top.get("sequence", [])
    for i, step in enumerate(path, 1):
        lines.append(f"### {i}단계: {step}")
        lines.append(f"- 담당: 해당 부서")
        lines.append(f"- 완료 조건: {step} 처리 완료 확인")
        lines.append("")

    return "\n".join(lines)


async def recommend_automation(
    discovery_result: dict,
    bottleneck_result: dict,
    process_type: str,
) -> dict:
    """
    프로세스 발견 + 병목 분석 결과를 LLM에 전달하여
    자동화 추천 + SOP 생성.

    API 키가 없으면 규칙 기반 폴백을 반환한다.
    """
    api_key = st.OPENAI_API_KEY
    if not api_key:
        st.logger.info("PM_RECOMMEND fallback (no API key)")
        return _fallback_recommend(discovery_result, bottleneck_result, process_type)

    type_kr = _PROCESS_TYPE_KR.get(process_type, process_type)
    patterns_text = _format_patterns(discovery_result)
    bottleneck_text = _format_bottlenecks(bottleneck_result)

    prompt = _RECOMMEND_PROMPT.format(
        process_type=type_kr,
        patterns_text=patterns_text,
        bottleneck_text=bottleneck_text,
    )

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            max_tokens=2000,
        )
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        parsed = _parse_llm_json(content)
        if parsed and "recommendations" in parsed:
            st.logger.info("PM_RECOMMEND llm_success recs=%d", len(parsed["recommendations"]))
            return {
                "recommendations": parsed.get("recommendations", [])[:5],
                "sop_document": parsed.get("sop_document", ""),
                "summary": parsed.get("summary", ""),
                "estimated_time_saving_percent": int(parsed.get("estimated_time_saving_percent", 0)),
            }

        st.logger.warning("PM_RECOMMEND llm_parse_fail, using fallback")
        return _fallback_recommend(discovery_result, bottleneck_result, process_type)

    except Exception as e:
        st.logger.warning("PM_RECOMMEND llm_error=%s, using fallback", e)
        return _fallback_recommend(discovery_result, bottleneck_result, process_type)
