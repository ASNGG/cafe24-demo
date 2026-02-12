"""
프로세스 마이닝 코어 로직.

이벤트 로그로부터 프로세스 패턴을 발견하고,
전이 행렬 및 Mermaid 다이어그램을 생성한다.
"""

from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache


# M34: 타임스탬프 파싱 캐싱 (중복 파싱 방지)
@lru_cache(maxsize=4096)
def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _group_by_case(events: list[dict]) -> dict[str, list[dict]]:
    """이벤트를 case_id별로 그룹화하고 timestamp 순 정렬한다."""
    cases: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        cases[event["case_id"]].append(event)

    for case_id in cases:
        cases[case_id].sort(key=lambda e: e["timestamp"])

    return dict(cases)


def _extract_sequences(cases: dict[str, list[dict]]) -> list[tuple[str, ...]]:
    """각 케이스에서 activity 시퀀스를 추출한다."""
    sequences: list[tuple[str, ...]] = []
    for case_id, events in cases.items():
        seq = tuple(e["activity"] for e in events)
        sequences.append(seq)
    return sequences


def _compute_transitions(cases: dict[str, list[dict]]) -> list[dict]:
    """
    전이 행렬을 계산한다.
    A -> B 전이별 횟수, 확률, 평균 소요시간(분)을 반환.
    """
    transition_counts: Counter = Counter()
    transition_durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    outgoing_counts: Counter = Counter()

    for case_id, events in cases.items():
        for i in range(len(events) - 1):
            from_act = events[i]["activity"]
            to_act = events[i + 1]["activity"]
            pair = (from_act, to_act)

            transition_counts[pair] += 1
            outgoing_counts[from_act] += 1

            # 소요시간 계산
            t1 = _parse_ts(events[i]["timestamp"])
            t2 = _parse_ts(events[i + 1]["timestamp"])
            duration_minutes = (t2 - t1).total_seconds() / 60.0
            transition_durations[pair].append(duration_minutes)

    transitions: list[dict] = []
    for (from_act, to_act), count in transition_counts.most_common():
        total_out = outgoing_counts[from_act]
        durations = transition_durations[(from_act, to_act)]
        avg_minutes = sum(durations) / len(durations) if durations else 0

        transitions.append({
            "from": from_act,
            "to": to_act,
            "count": count,
            "probability": round(count / total_out, 4) if total_out else 0,
            "avg_minutes": round(avg_minutes, 1),
        })

    return transitions


def _compute_case_duration_hours(events: list[dict]) -> float:
    """단일 케이스의 전체 소요시간(시간)을 계산한다."""
    if len(events) < 2:
        return 0.0
    t_start = _parse_ts(events[0]["timestamp"])
    t_end = _parse_ts(events[-1]["timestamp"])
    return (t_end - t_start).total_seconds() / 3600.0


def _build_mermaid_diagram(transitions: list[dict], activities: list[str]) -> str:
    """
    전이 행렬 기반으로 Mermaid 프로세스 플로우 다이어그램 문자열을 생성한다.
    노드: 각 activity, 엣지: 전이 (라벨에 빈도/확률).
    """
    lines: list[str] = ["graph LR"]

    # 노드 ID 매핑 (한국어 activity를 안전한 ID로)
    node_ids: dict[str, str] = {}
    for i, act in enumerate(activities):
        node_id = f"A{i}"
        node_ids[act] = node_id
        lines.append(f"    {node_id}[\"{act}\"]")

    # 엣지 (상위 전이만 표시, 너무 많으면 다이어그램이 복잡해짐)
    for t in transitions:
        from_id = node_ids.get(t["from"])
        to_id = node_ids.get(t["to"])
        if from_id and to_id:
            prob_pct = round(t["probability"] * 100, 1)
            label = f"{t['count']}건<br/>{prob_pct}%"
            lines.append(f"    {from_id} -->|{label}| {to_id}")

    return "\n".join(lines)


def discover_process(events: list[dict]) -> dict:
    """
    이벤트 로그로부터 프로세스를 발견한다.

    Args:
        events: 이벤트 레코드 딕셔너리 리스트.

    Returns:
        프로세스 발견 결과 딕셔너리:
        - total_cases: 전체 케이스 수
        - unique_patterns: 고유 패턴 수
        - top_patterns: Top 10 패턴 (시퀀스, 횟수, 비율, 평균 소요시간)
        - transitions: 전이 행렬
        - activities: 전체 활동 목록
        - mermaid_diagram: Mermaid 다이어그램 문자열
    """
    if not events:
        return {
            "total_cases": 0,
            "unique_patterns": 0,
            "avg_duration_minutes": 0.0,
            "top_patterns": [],
            "transitions": [],
            "activities": [],
            "mermaid_diagram": "",
        }

    cases = _group_by_case(events)
    total_cases = len(cases)

    # ── 시퀀스 패턴 분석 ──
    sequences = _extract_sequences(cases)
    pattern_counter = Counter(sequences)
    unique_patterns = len(pattern_counter)

    # 패턴별 평균 소요시간 계산
    pattern_durations: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for case_id, case_events in cases.items():
        seq = tuple(e["activity"] for e in case_events)
        duration_h = _compute_case_duration_hours(case_events)
        pattern_durations[seq].append(duration_h)

    top_patterns: list[dict] = []
    for seq, count in pattern_counter.most_common(10):
        durations = pattern_durations[seq]
        avg_dur_h = sum(durations) / len(durations) if durations else 0
        top_patterns.append({
            "sequence": list(seq),
            "count": count,
            "ratio": round(count / total_cases, 4),
            "avg_duration_minutes": round(avg_dur_h * 60, 1),
        })

    # ── 전체 평균 소요시간 (분) ──
    all_durations_h = [_compute_case_duration_hours(ce) for ce in cases.values()]
    avg_duration_minutes = round(
        sum(all_durations_h) / len(all_durations_h) * 60, 1
    ) if all_durations_h else 0.0

    # ── 전이 행렬 ──
    transitions = _compute_transitions(cases)

    # ── 전체 활동 목록 (등장 순서 유지) ──
    seen: set[str] = set()
    activities: list[str] = []
    for event in events:
        act = event["activity"]
        if act not in seen:
            seen.add(act)
            activities.append(act)

    # ── Mermaid 다이어그램 ──
    mermaid_diagram = _build_mermaid_diagram(transitions, activities)

    return {
        "total_cases": total_cases,
        "unique_patterns": unique_patterns,
        "avg_duration_minutes": avg_duration_minutes,
        "top_patterns": top_patterns,
        "transitions": transitions,
        "activities": activities,
        "mermaid_diagram": mermaid_diagram,
    }
