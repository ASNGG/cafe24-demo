"""
병목 분석 로직.

이벤트 로그에서 전이별 소요시간 분포를 분석하고,
병목 지점과 이상치 케이스를 식별한다.
"""

from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Optional


# M34: 타임스탬프 파싱 캐싱 (중복 파싱 방지)
@lru_cache(maxsize=4096)
def _parse_timestamp(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _get_time_slot(dt: datetime) -> str:
    """시간대를 morning/afternoon/evening/night 으로 분류한다."""
    hour = dt.hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:
        return "night"


def _percentile(sorted_values: list[float], pct: float) -> float:
    """정렬된 리스트에서 백분위수를 계산한다."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = pct / 100.0 * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _classify_severity(avg_minutes: float, p95_minutes: float) -> str:
    """병목 심각도를 분류한다."""
    # p95가 24시간(1440분) 이상이면 high
    if p95_minutes >= 1440:
        return "high"
    # p95가 4시간(240분) 이상이면 medium
    if p95_minutes >= 240:
        return "medium"
    return "low"


def analyze_bottlenecks(events: list[dict]) -> dict:
    """
    이벤트 로그에서 병목 지점을 분석한다.

    Args:
        events: 이벤트 레코드 딕셔너리 리스트.

    Returns:
        병목 분석 결과 딕셔너리:
        - bottlenecks: 평균 소요시간이 긴 전이 Top 5
        - time_distribution: 시간대별 병목 패턴
        - total_avg_process_hours: 전체 평균 프로세스 소요시간(시간)
        - efficiency_score: 효율성 점수 (0~1)
    """
    if not events:
        return {
            "bottlenecks": [],
            "time_analysis": {},
            "total_avg_process_hours": 0.0,
            "efficiency_score": 0.0,
        }

    # ── case별 이벤트 그룹화 ──
    cases: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        cases[event["case_id"]].append(event)

    for case_id in cases:
        cases[case_id].sort(key=lambda e: e["timestamp"])

    # ── 전이별 소요시간 수집 ──
    # (from_act, to_act) -> [(duration_minutes, case_id, timestamp), ...]
    transition_data: dict[tuple[str, str], list[tuple[float, str, datetime]]] = defaultdict(list)

    # ── 시간대별 소요시간 수집 ──
    time_slot_data: dict[str, list[float]] = defaultdict(list)

    # ── 전체 케이스 소요시간 ──
    case_durations_hours: list[float] = []

    for case_id, case_events in cases.items():
        if len(case_events) < 2:
            continue

        # 케이스 전체 소요시간
        t_start = _parse_timestamp(case_events[0]["timestamp"])
        t_end = _parse_timestamp(case_events[-1]["timestamp"])
        case_dur_h = (t_end - t_start).total_seconds() / 3600.0
        case_durations_hours.append(case_dur_h)

        for i in range(len(case_events) - 1):
            from_act = case_events[i]["activity"]
            to_act = case_events[i + 1]["activity"]
            t1 = _parse_timestamp(case_events[i]["timestamp"])
            t2 = _parse_timestamp(case_events[i + 1]["timestamp"])
            dur_min = (t2 - t1).total_seconds() / 60.0

            transition_data[(from_act, to_act)].append((dur_min, case_id, t2))

            # 시간대 분류 (전이가 완료되는 시점 기준)
            slot = _get_time_slot(t2)
            time_slot_data[slot].append(dur_min)

    # ── 병목 분석 (전이별) ──
    bottleneck_list: list[dict] = []

    for (from_act, to_act), records in transition_data.items():
        durations = [r[0] for r in records]
        durations_sorted = sorted(durations)
        n = len(durations_sorted)

        avg_min = sum(durations) / n
        median_min = _percentile(durations_sorted, 50)
        p95_min = _percentile(durations_sorted, 95)

        # IQR 기반 이상치 탐지
        q1 = _percentile(durations_sorted, 25)
        q3 = _percentile(durations_sorted, 75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr

        outlier_cases: list[dict] = []
        seen_outliers: set[str] = set()
        for dur, cid, _ in records:
            if dur > upper_fence and cid not in seen_outliers:
                seen_outliers.add(cid)
                outlier_cases.append({
                    "case_id": cid,
                    "duration_minutes": round(dur, 1),
                })

        severity = _classify_severity(avg_min, p95_min)

        bottleneck_list.append({
            "from_step": from_act,
            "to_step": to_act,
            "avg_duration_minutes": round(avg_min, 1),
            "median_duration_minutes": round(median_min, 1),
            "p95_duration_minutes": round(p95_min, 1),
            "case_count": n,
            "severity": severity,
            "outlier_count": len(outlier_cases),
            "outlier_cases": outlier_cases,
        })

    # 평균 소요시간 기준 내림차순 정렬, Top 5
    bottleneck_list.sort(key=lambda b: b["avg_duration_minutes"], reverse=True)
    top_bottlenecks = bottleneck_list[:5]

    # ── 시간대별 분포 (간소화: 시간대 -> 평균 소요시간) ──
    time_analysis: dict[str, float] = {}
    for slot in ["morning", "afternoon", "evening", "night"]:
        durations = time_slot_data.get(slot, [])
        if durations:
            time_analysis[slot] = round(sum(durations) / len(durations), 1)
        else:
            time_analysis[slot] = 0.0

    # ── 전체 평균 프로세스 소요시간 ──
    total_avg_hours = (
        round(sum(case_durations_hours) / len(case_durations_hours), 1)
        if case_durations_hours else 0.0
    )

    # ── 효율성 점수 (0~100 스케일) ──
    efficiency_score = _compute_efficiency_score(bottleneck_list, case_durations_hours)

    return {
        "bottlenecks": top_bottlenecks,
        "time_analysis": time_analysis,
        "total_avg_process_hours": total_avg_hours,
        "efficiency_score": efficiency_score,
    }


def _compute_efficiency_score(
    bottlenecks: list[dict],
    case_durations_hours: list[float],
) -> float:
    """
    효율성 점수를 0~100 범위로 계산한다.
    - 이상치 케이스가 적을수록 점수 높음
    - 중앙값/평균 비율이 1에 가까울수록 점수 높음 (편차가 적음)
    """
    if not bottlenecks or not case_durations_hours:
        return 0.0

    # 요소 1: 이상치 비율 (전체 bottleneck에서 이상치가 차지하는 비율)
    total_cases_in_bottlenecks = sum(b["case_count"] for b in bottlenecks)
    total_outliers = sum(b["outlier_count"] for b in bottlenecks)
    outlier_ratio = total_outliers / total_cases_in_bottlenecks if total_cases_in_bottlenecks else 0
    outlier_score = max(0.0, 1.0 - outlier_ratio * 3)  # 33% 이상이면 0점

    # 요소 2: 중앙값/평균 비율 (1에 가까울수록 좋음, 편차 적음)
    median_avg_ratios: list[float] = []
    for b in bottlenecks:
        if b["avg_duration_minutes"] > 0:
            ratio = b["median_duration_minutes"] / b["avg_duration_minutes"]
            median_avg_ratios.append(ratio)

    if median_avg_ratios:
        avg_ratio = sum(median_avg_ratios) / len(median_avg_ratios)
        skew_score = min(1.0, avg_ratio)  # 중앙값 < 평균이면 양적 편향 (비효율)
    else:
        skew_score = 0.5

    # 요소 3: high severity 비율
    high_count = sum(1 for b in bottlenecks if b["severity"] == "high")
    severity_score = max(0.0, 1.0 - high_count / len(bottlenecks))

    # 가중 평균 (0~1 -> 0~100 스케일)
    efficiency = 0.4 * outlier_score + 0.3 * skew_score + 0.3 * severity_score
    return round(min(100.0, max(0.0, efficiency * 100)), 1)
