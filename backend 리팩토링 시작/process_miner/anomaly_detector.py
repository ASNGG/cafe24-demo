"""
프로세스 경로 기반 이상 탐지 모듈.

IsolationForest를 사용하여 비정상적인 활동 시퀀스를 가진
케이스를 탐지한다.  bottleneck.py가 *시간 기반* 이상치를 찾는 반면,
이 모듈은 *경로 기반* 이상 프로세스를 식별한다.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import IsolationForest

import state as st
from .helpers import parse_timestamp as _parse_timestamp, group_events_by_case

# ── 프로세스별 정상 활동 집합 ──────────────────────────────────
NORMAL_ACTIVITIES: dict[str, set[str]] = {
    "order": {"주문생성", "결제확인", "재고차감", "배송요청", "배송중", "배송완료"},
    "cs": {"CS접수", "자동분류", "담당자배정", "1차응대", "해결완료"},
    "settlement": {"정산요청", "데이터검증", "금액계산", "승인", "지급완료"},
}


# ── 단일 케이스 인코딩 ────────────────────────────────────────
def _encode_case(
    case_events: list[dict],
    all_activities: list[str],
    normal_activities: set[str],
) -> list[float]:
    """단일 케이스의 이벤트 리스트를 피처 벡터로 변환한다."""
    activities = [e["activity"] for e in case_events]

    # sequence_length
    sequence_length = float(len(activities))

    # total_duration_minutes
    if len(case_events) >= 2:
        t_start = _parse_timestamp(case_events[0]["timestamp"])
        t_end = _parse_timestamp(case_events[-1]["timestamp"])
        total_duration_minutes = (t_end - t_start).total_seconds() / 60.0
    else:
        total_duration_minutes = 0.0

    # unique_activity_count
    unique_activity_count = float(len(set(activities)))

    # has_loop (같은 활동이 두 번 이상 등장)
    has_loop = 1.0 if len(activities) != len(set(activities)) else 0.0

    # exception_step_count (정상 경로에 없는 활동 수)
    exception_step_count = float(
        sum(1 for a in activities if a not in normal_activities)
    )

    # activity별 등장 횟수 (one-hot count)
    activity_counts: dict[str, int] = defaultdict(int)
    for a in activities:
        activity_counts[a] += 1

    activity_vector = [float(activity_counts.get(a, 0)) for a in all_activities]

    return [
        sequence_length,
        total_duration_minutes,
        unique_activity_count,
        has_loop,
        exception_step_count,
        *activity_vector,
    ]


# ── 전체 이벤트 → 케이스별 피처 행렬 ─────────────────────────
def _prepare_case_features(
    events: list[dict],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    전체 이벤트에서 케이스별 피처 행렬을 생성한다.

    Returns:
        (features_matrix, case_ids, all_activities)
    """
    # case별 이벤트 그룹화
    cases = group_events_by_case(events)

    # 모든 활동 종류 수집 (정렬해서 일관된 순서 보장)
    all_activity_set: set[str] = set()
    for event in events:
        all_activity_set.add(event["activity"])
    all_activities = sorted(all_activity_set)

    # 해당 이벤트들의 프로세스 타입에 맞는 정상 활동 집합 결정
    process_types = {e["process_type"] for e in events}
    normal_acts: set[str] = set()
    for pt in process_types:
        normal_acts |= NORMAL_ACTIVITIES.get(pt, set())

    # 케이스별 피처 벡터 생성
    case_ids: list[str] = []
    feature_rows: list[list[float]] = []

    for cid in sorted(cases.keys()):
        case_events = cases[cid]
        row = _encode_case(case_events, all_activities, normal_acts)
        feature_rows.append(row)
        case_ids.append(cid)

    features_matrix = np.array(feature_rows, dtype=np.float64)
    return features_matrix, case_ids, all_activities


# ── 이상 판정 근거 생성 ──────────────────────────────────────
def _generate_anomaly_reason(case_features: dict) -> str:
    """이상 판정 근거를 자연어로 생성한다."""
    parts: list[str] = []

    # 소요시간 비교
    if case_features.get("duration_ratio", 1.0) > 1.5:
        ratio = case_features["duration_ratio"]
        parts.append(f"평균보다 {ratio:.1f}배 긴 소요시간")

    # 시퀀스 길이
    if case_features.get("length_ratio", 1.0) > 1.5:
        ratio = case_features["length_ratio"]
        parts.append(f"평균보다 {ratio:.1f}배 많은 활동 단계({case_features['sequence_length']}개)")

    # 루프
    if case_features.get("has_loop"):
        parts.append("활동 반복(루프) 존재")

    # 예외 경로
    exception_steps = case_features.get("exception_steps", [])
    if exception_steps:
        steps_str = ", ".join(exception_steps)
        parts.append(f"{len(exception_steps)}개의 예외 활동({steps_str})이 포함됨")

    if not parts:
        parts.append("복합적 피처 조합에 의한 이상 판정")

    return " / ".join(parts)


# ── 메인: 이상 프로세스 탐지 ──────────────────────────────────
def detect_anomalies(
    events: list[dict],
    contamination: float = 0.15,
) -> dict:
    """
    이벤트 로그에서 IsolationForest 기반 이상 프로세스를 탐지한다.

    Args:
        events: 이벤트 레코드 딕셔너리 리스트.
        contamination: 이상 비율 (기본 0.15).

    Returns:
        탐지 결과 딕셔너리.
    """
    if not events:
        return {
            "total_cases": 0,
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "anomalies": [],
            "feature_importance": {},
            "normal_pattern_summary": "이벤트 데이터 없음",
        }

    st.logger.info("ANOMALY_DETECT start events=%d contamination=%.2f", len(events), contamination)

    features_matrix, case_ids, all_activities = _prepare_case_features(events)
    total_cases = len(case_ids)

    # IsolationForest 학습 & 예측
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(features_matrix)       # 1=normal, -1=anomaly
    scores = model.decision_function(features_matrix)  # 음수일수록 이상

    # 피처 이름 목록
    base_feature_names = [
        "sequence_length",
        "total_duration_minutes",
        "unique_activity_count",
        "has_loop",
        "exception_step_count",
    ]
    feature_names = base_feature_names + [f"act_{a}" for a in all_activities]

    # 피처 중요도 (feature_importances_ from tree depth)
    importances = np.zeros(features_matrix.shape[1])
    for tree in model.estimators_:
        fi = tree.feature_importances_
        importances += fi
    importances /= len(model.estimators_)
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, importances)
    }

    # 전체 평균 산출 (이상 근거 생성용)
    avg_duration = float(np.mean(features_matrix[:, 1]))
    avg_length = float(np.mean(features_matrix[:, 0]))

    # 프로세스 타입별 정상 활동 집합
    process_types = {e["process_type"] for e in events}
    normal_acts: set[str] = set()
    for pt in process_types:
        normal_acts |= NORMAL_ACTIVITIES.get(pt, set())

    # 케이스별 이벤트 맵 (이상 케이스 상세 조회용)
    cases_map = group_events_by_case(events)

    # 이상 케이스 상세
    anomalies: list[dict] = []
    for idx, (cid, label, score) in enumerate(zip(case_ids, labels, scores)):
        if label != -1:
            continue

        case_events = cases_map[cid]
        activities = [e["activity"] for e in case_events]
        exception_steps = [a for a in activities if a not in normal_acts]

        seq_len = float(features_matrix[idx, 0])
        dur = float(features_matrix[idx, 1])
        has_loop = bool(features_matrix[idx, 3] > 0.5)

        reason_features = {
            "sequence_length": int(seq_len),
            "duration_ratio": dur / avg_duration if avg_duration > 0 else 1.0,
            "length_ratio": seq_len / avg_length if avg_length > 0 else 1.0,
            "has_loop": has_loop,
            "exception_steps": exception_steps,
        }

        anomalies.append({
            "case_id": cid,
            "anomaly_score": round(float(score), 4),
            "sequence": activities,
            "sequence_length": int(seq_len),
            "total_duration_minutes": round(dur, 1),
            "has_loop": has_loop,
            "exception_steps": list(dict.fromkeys(exception_steps)),  # 중복 제거, 순서 유지
            "reason": _generate_anomaly_reason(reason_features),
        })

    # anomaly_score 오름차순 정렬 (가장 이상한 것부터 = 가장 음수)
    anomalies.sort(key=lambda a: a["anomaly_score"])

    anomaly_count = len(anomalies)
    anomaly_ratio = anomaly_count / total_cases if total_cases > 0 else 0.0

    # 정상 패턴 요약
    normal_count = total_cases - anomaly_count
    process_types_str = ", ".join(sorted(process_types))
    normal_pattern_summary = (
        f"전체 {total_cases}건 중 {normal_count}건({(1 - anomaly_ratio) * 100:.1f}%)이 "
        f"정상 패턴으로 분류됨. 프로세스 타입: {process_types_str}. "
        f"평균 활동 수 {avg_length:.1f}개, 평균 소요시간 {avg_duration:.0f}분."
    )

    st.logger.info(
        "ANOMALY_DETECT done total=%d anomalies=%d ratio=%.2f",
        total_cases, anomaly_count, anomaly_ratio,
    )

    return {
        "total_cases": total_cases,
        "anomaly_count": anomaly_count,
        "anomaly_ratio": round(anomaly_ratio, 4),
        "anomalies": anomalies,
        "feature_importance": feature_importance,
        "normal_pattern_summary": normal_pattern_summary,
    }
