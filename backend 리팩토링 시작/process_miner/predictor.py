"""
다음 활동 예측 모듈.

이벤트 로그에서 RandomForest 기반으로 다음 활동(activity)을 예측한다.
매 호출마다 모델을 학습하는 데모용 구현.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import state as st

# ── 모델 캐시 (재학습 방지) ──
_MODEL_CACHE = {
    "model": None,
    "encoders": None,
    "accuracy": 0.0,
    "feature_importance": {},
    "class_labels": [],
    "data_hash": "",  # 학습 데이터 해시 (변경 감지)
}

# ── 프로세스 타입 인코딩 ──
PROCESS_TYPE_MAP = {"order": 0, "cs": 1, "settlement": 2}

# ── 메타데이터 인코딩 매핑 ──
PAYMENT_METHOD_MAP = {"card": 0, "bank_transfer": 1, "virtual_account": 2, "phone": 3, "kakao_pay": 4}
CHANNEL_MAP = {"chat": 0, "phone": 1, "email": 2, "kakao": 3}
INQUIRY_TYPE_MAP = {"배송문의": 0, "반품/교환": 1, "결제문의": 2, "상품문의": 3, "기타": 4}
PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2}
SETTLEMENT_TYPE_MAP = {"일반정산": 0, "긴급정산": 1, "수동정산": 2}
SELLER_TIER_MAP = {"silver": 0, "gold": 1, "platinum": 2, "diamond": 3}


def _build_features(event: dict, step_index: int, encoders: dict) -> list:
    """
    단일 이벤트에서 피처 벡터를 추출한다.

    Features:
        0: process_type (인코딩)
        1: current_activity (인코딩)
        2: step_index
        3: hour (0~23)
        4: weekday (0~6)
        5: metadata_0 (payment_method / channel / settlement_type)
        6: metadata_1 (product_category(unused=0) / inquiry_type / seller_tier)
    """
    process_type = event.get("process_type", "order")
    activity = event.get("activity", "")
    timestamp_str = event.get("timestamp", "")
    metadata = event.get("metadata", {})

    # process_type 인코딩
    pt_encoded = PROCESS_TYPE_MAP.get(process_type, 0)

    # activity 인코딩
    activity_enc = encoders["activity"]
    if activity in activity_enc.classes_:
        act_encoded = int(activity_enc.transform([activity])[0])
    else:
        act_encoded = 0

    # timestamp 파싱
    try:
        ts = datetime.fromisoformat(timestamp_str)
        hour = ts.hour
        weekday = ts.weekday()
    except (ValueError, TypeError):
        hour = 12
        weekday = 0

    # 메타데이터 인코딩 (프로세스별)
    if process_type == "order":
        meta_0 = PAYMENT_METHOD_MAP.get(metadata.get("payment_method", ""), 0)
        meta_1 = 0  # product_category는 피처 중요도가 낮으므로 단순화
    elif process_type == "cs":
        meta_0 = CHANNEL_MAP.get(metadata.get("channel", ""), 0)
        meta_1 = INQUIRY_TYPE_MAP.get(metadata.get("inquiry_type", ""), 0)
    elif process_type == "settlement":
        meta_0 = SETTLEMENT_TYPE_MAP.get(metadata.get("settlement_type", ""), 0)
        meta_1 = SELLER_TIER_MAP.get(metadata.get("seller_tier", ""), 0)
    else:
        meta_0 = 0
        meta_1 = 0

    return [pt_encoded, act_encoded, step_index, hour, weekday, meta_0, meta_1]


def _compute_events_hash(events: list[dict]) -> str:
    """이벤트 리스트의 해시값 계산 (변경 감지용)"""
    import hashlib
    # case_id + activity + timestamp 조합으로 해시
    parts = []
    for e in events:
        parts.append(f"{e.get('case_id', '')}|{e.get('activity', '')}|{e.get('timestamp', '')}")
    content = "\n".join(sorted(parts))
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _prepare_training_data(events: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    이벤트 로그에서 학습 데이터를 추출한다.

    각 케이스의 이벤트 시퀀스에서 (현재 상태 피처, 다음 activity) 쌍을 생성한다.
    마지막 이벤트는 타겟이 없으므로 제외한다.

    Returns:
        (X, y, encoders)
        - X: 피처 행렬 (n_samples, n_features)
        - y: 타겟 배열 (인코딩된 next_activity)
        - encoders: {"activity": LabelEncoder, "target": LabelEncoder}
    """
    # case별 이벤트 그룹화
    cases: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        cases[event["case_id"]].append(event)

    for case_id in cases:
        cases[case_id].sort(key=lambda e: e["timestamp"])

    # 전체 activity 수집 (인코더 학습용)
    all_activities = list({e["activity"] for e in events})
    all_activities.sort()

    activity_enc = LabelEncoder()
    activity_enc.fit(all_activities)

    target_enc = LabelEncoder()
    target_enc.fit(all_activities)

    encoders = {"activity": activity_enc, "target": target_enc}

    # 학습 데이터 생성
    X_list: list[list] = []
    y_list: list[int] = []

    for case_id, case_events in cases.items():
        for i in range(len(case_events) - 1):
            current_event = case_events[i]
            next_activity = case_events[i + 1]["activity"]

            features = _build_features(current_event, step_index=i, encoders=encoders)
            X_list.append(features)
            y_list.append(int(target_enc.transform([next_activity])[0]))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    return X, y, encoders


def train_predictor(events: list[dict], force_retrain: bool = False) -> dict:
    """
    이벤트 로그로 RandomForest 모델을 학습한다.
    캐싱: 동일 데이터면 기존 모델 재사용 (매 호출마다 재학습 방지).

    Returns:
        {
            "model": trained RandomForestClassifier,
            "encoders": encoders dict,
            "accuracy": float (학습 정확도),
            "feature_importance": {feature_name: importance},
            "class_labels": [activity_name, ...]
        }
    """
    global _MODEL_CACHE

    # 데이터 변경 감지
    data_hash = _compute_events_hash(events)

    # 캐시 히트: 동일 데이터 + 모델 존재 → 재사용
    if (
        not force_retrain
        and _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["data_hash"] == data_hash
    ):
        st.logger.info("PREDICTOR cache_hit hash=%s", data_hash)
        return {
            "model": _MODEL_CACHE["model"],
            "encoders": _MODEL_CACHE["encoders"],
            "accuracy": _MODEL_CACHE["accuracy"],
            "feature_importance": _MODEL_CACHE["feature_importance"],
            "class_labels": _MODEL_CACHE["class_labels"],
        }

    X, y, encoders = _prepare_training_data(events)

    if len(X) == 0:
        raise ValueError("학습 데이터가 충분하지 않습니다.")

    n_classes = len(np.unique(y))
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)

    accuracy = float(model.score(X, y))

    feature_names = [
        "process_type", "current_activity", "step_index",
        "hour", "weekday", "metadata_0", "metadata_1",
    ]
    importance = dict(zip(feature_names, [round(float(v), 4) for v in model.feature_importances_]))

    class_labels = list(encoders["target"].classes_)

    # 캐시 업데이트
    _MODEL_CACHE.update({
        "model": model,
        "encoders": encoders,
        "accuracy": accuracy,
        "feature_importance": importance,
        "class_labels": class_labels,
        "data_hash": data_hash,
    })

    st.logger.info(
        "PREDICTOR train done samples=%d classes=%d accuracy=%.3f hash=%s",
        len(X), n_classes, accuracy, data_hash,
    )

    return {
        "model": model,
        "encoders": encoders,
        "accuracy": accuracy,
        "feature_importance": importance,
        "class_labels": class_labels,
    }


def predict_next_activity(
    events: list[dict],
    case_id: str,
    process_type: str = "order",
    n_cases: int = 200,
) -> dict:
    """
    이벤트 로그로 모델을 학습한 후, 특정 case_id의 다음 활동을 예측한다.

    Args:
        events: 이벤트 레코드 리스트.
        case_id: 예측 대상 케이스 ID.
        process_type: 프로세스 타입 (로깅용).
        n_cases: 케이스 수 (로깅용).

    Returns:
        {
            "case_id": str,
            "current_activity": str,
            "predictions": [{"activity": str, "probability": float}, ...],  # top 3
            "model_accuracy": float,
            "feature_importance": {feature_name: importance_value, ...}
        }
    """
    st.logger.info(
        "PREDICTOR predict start case=%s type=%s n_cases=%d",
        case_id, process_type, n_cases,
    )

    # 모델 학습
    result = train_predictor(events)
    model = result["model"]
    encoders = result["encoders"]

    # 해당 case의 이벤트 추출
    case_events = [e for e in events if e["case_id"] == case_id]
    if not case_events:
        raise ValueError(f"케이스 '{case_id}'를 찾을 수 없습니다.")

    case_events.sort(key=lambda e: e["timestamp"])
    last_event = case_events[-1]
    step_index = len(case_events) - 1

    # 피처 추출 및 예측
    features = _build_features(last_event, step_index=step_index, encoders=encoders)
    X_pred = np.array([features], dtype=np.float64)

    probas = model.predict_proba(X_pred)[0]
    class_labels = result["class_labels"]

    # 확률 기준 상위 3개
    top_indices = np.argsort(probas)[::-1][:3]
    predictions = []
    for idx in top_indices:
        predictions.append({
            "activity": class_labels[idx],
            "probability": round(float(probas[idx]), 4),
        })

    st.logger.info(
        "PREDICTOR predict done case=%s current=%s top1=%s (%.2f%%)",
        case_id,
        last_event["activity"],
        predictions[0]["activity"] if predictions else "N/A",
        predictions[0]["probability"] * 100 if predictions else 0,
    )

    return {
        "case_id": case_id,
        "current_activity": last_event["activity"],
        "predictions": predictions,
        "model_accuracy": result["accuracy"],
        "feature_importance": result["feature_importance"],
    }
