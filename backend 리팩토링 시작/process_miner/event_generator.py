"""
카페24 운영 시나리오 기반 데모 이벤트 로그 데이터 생성기.

3가지 프로세스 타입(order, cs, settlement)에 대해
정상/예외 경로를 포함한 현실적인 이벤트 로그를 생성한다.
"""

import random
from datetime import datetime, timedelta
from typing import Optional

# ── 프로세스 정의 ──────────────────────────────────────────────

# 정상 경로
NORMAL_PATHS: dict[str, list[str]] = {
    "order": ["주문생성", "결제확인", "재고차감", "배송요청", "배송중", "배송완료"],
    "cs": ["CS접수", "자동분류", "담당자배정", "1차응대", "해결완료"],
    "settlement": ["정산요청", "데이터검증", "금액계산", "승인", "지급완료"],
}

# 예외 경로: (분기 지점 activity, 예외 시퀀스, 가중치)
# 가중치가 높을수록 해당 예외가 더 자주 발생
EXCEPTION_PATHS: dict[str, list[tuple[str, list[str], int]]] = {
    "order": [
        ("결제확인", ["결제실패"], 3),
        ("배송요청", ["배송지연", "배송중", "배송완료"], 4),
        ("배송완료", ["반품요청"], 2),
        ("결제확인", ["CS접수", "1차응대", "해결완료"], 1),
    ],
    "cs": [
        ("1차응대", ["에스컬레이션", "담당자배정", "1차응대", "해결완료"], 5),
        ("해결완료", ["재문의", "담당자배정", "1차응대", "해결완료"], 3),
    ],
    "settlement": [
        ("데이터검증", ["검증실패", "데이터검증", "금액계산", "승인", "지급완료"], 4),
        ("금액계산", ["재계산", "금액계산", "승인", "지급완료"], 3),
        ("승인", ["보류"], 2),
    ],
}

# 각 activity 전이에 걸리는 시간 범위 (분 단위: min, max)
DURATION_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "order": {
        "주문생성": (0, 0),          # 시작점
        "결제확인": (3, 30),         # 3분~30분
        "결제실패": (1, 10),
        "재고차감": (1, 5),
        "배송요청": (10, 60),        # 10분~1시간
        "배송지연": (120, 1440),     # 2시간~1일
        "배송중": (60, 2880),        # 1시간~2일
        "배송완료": (720, 4320),     # 12시간~3일
        "반품요청": (1440, 10080),   # 1일~7일
        "CS접수": (5, 60),
        "1차응대": (30, 240),
        "해결완료": (60, 480),
    },
    "cs": {
        "CS접수": (0, 0),
        "자동분류": (1, 3),          # 1~3분 (시스템 자동)
        "담당자배정": (5, 30),       # 5~30분
        "1차응대": (10, 120),        # 10분~2시간
        "해결완료": (15, 360),       # 15분~6시간
        "에스컬레이션": (5, 30),
        "재문의": (60, 1440),        # 1시간~1일
    },
    "settlement": {
        "정산요청": (0, 0),
        "데이터검증": (60, 480),     # 1~8시간
        "검증실패": (30, 120),
        "금액계산": (30, 240),       # 30분~4시간
        "재계산": (60, 360),
        "승인": (120, 1440),         # 2시간~1일
        "보류": (60, 2880),
        "지급완료": (60, 480),       # 1~8시간
    },
}

# 액터 풀
ACTORS: dict[str, list[str]] = {
    "system": ["system"],
    "operator": [f"operator_{i:02d}" for i in range(1, 6)],
    "manager": ["manager_01", "manager_02"],
    "finance": ["finance_01", "finance_02"],
}

# activity별 담당 액터 타입
ACTIVITY_ACTORS: dict[str, str] = {
    "주문생성": "system",
    "결제확인": "system",
    "결제실패": "system",
    "재고차감": "system",
    "배송요청": "system",
    "배송지연": "system",
    "배송중": "operator",
    "배송완료": "operator",
    "반품요청": "operator",
    "CS접수": "system",
    "자동분류": "system",
    "담당자배정": "system",
    "1차응대": "operator",
    "해결완료": "operator",
    "에스컬레이션": "manager",
    "재문의": "operator",
    "정산요청": "system",
    "데이터검증": "system",
    "검증실패": "system",
    "금액계산": "finance",
    "재계산": "finance",
    "승인": "manager",
    "보류": "manager",
    "지급완료": "finance",
}

# 메타데이터 템플릿
METADATA_TEMPLATES: dict[str, dict] = {
    "order": {
        "payment_method": ["card", "bank_transfer", "virtual_account", "phone", "kakao_pay"],
        "product_category": ["의류", "식품", "전자기기", "화장품", "생활용품"],
        "order_amount": (10000, 500000),
    },
    "cs": {
        "channel": ["chat", "phone", "email", "kakao"],
        "inquiry_type": ["배송문의", "반품/교환", "결제문의", "상품문의", "기타"],
        "priority": ["low", "medium", "high"],
    },
    "settlement": {
        "settlement_type": ["일반정산", "긴급정산", "수동정산"],
        "amount_range": (100000, 50000000),
        "seller_tier": ["silver", "gold", "platinum", "diamond"],
    },
}

# ── 케이스 ID 프리픽스 ──────────────────────────────────────

CASE_PREFIX: dict[str, str] = {
    "order": "ORD",
    "cs": "CS",
    "settlement": "STL",
}


def _generate_metadata(process_type: str, rng: random.Random) -> dict:
    """프로세스 타입에 맞는 메타데이터를 생성한다."""
    template = METADATA_TEMPLATES[process_type]
    metadata: dict = {}

    for key, value in template.items():
        if isinstance(value, list):
            metadata[key] = rng.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            metadata[key] = rng.randint(value[0], value[1])

    return metadata


def _build_case_sequence(process_type: str, rng: random.Random) -> list[str]:
    """
    정상/예외 경로를 확률적으로 선택하여 하나의 케이스 시퀀스를 생성한다.
    정상 경로 70~80%, 예외 경로 20~30%.
    """
    normal = NORMAL_PATHS[process_type]
    is_normal = rng.random() < rng.uniform(0.70, 0.80)

    if is_normal:
        return list(normal)

    # 예외 경로 선택
    exceptions = EXCEPTION_PATHS[process_type]
    weights = [e[2] for e in exceptions]
    chosen = rng.choices(exceptions, weights=weights, k=1)[0]
    branch_point, exception_seq, _ = chosen

    # 정상 경로에서 분기 지점까지 진행 후 예외 시퀀스 붙이기
    sequence: list[str] = []
    for activity in normal:
        sequence.append(activity)
        if activity == branch_point:
            sequence.extend(exception_seq)
            break

    return sequence


def _pick_actor(activity: str, rng: random.Random) -> str:
    """activity에 맞는 액터를 선택한다."""
    actor_type = ACTIVITY_ACTORS.get(activity, "operator")
    pool = ACTORS.get(actor_type, ACTORS["operator"])
    return rng.choice(pool)


def _get_duration(process_type: str, activity: str, rng: random.Random) -> int:
    """activity 전이에 걸리는 시간(분)을 반환한다."""
    ranges = DURATION_RANGES.get(process_type, {})
    dur_range = ranges.get(activity, (5, 60))
    return rng.randint(dur_range[0], dur_range[1])


def generate_event_logs(
    process_type: str = "all",
    n_cases: int = 200,
    seed: int = 42,
    base_date: Optional[str] = None,
) -> list[dict]:
    """
    카페24 운영 시나리오 기반 데모 이벤트 로그를 생성한다.

    Args:
        process_type: "order", "cs", "settlement", 또는 "all" (전체).
        n_cases: 생성할 케이스 수. "all"인 경우 프로세스별로 n_cases개씩 생성.
        seed: 랜덤 시드 (재현 가능).
        base_date: 시작 기준일 (ISO 형식). None이면 "2025-02-01".

    Returns:
        이벤트 레코드 딕셔너리의 리스트.
    """
    rng = random.Random(seed)
    start_date = datetime.fromisoformat(base_date) if base_date else datetime(2025, 2, 1)

    if process_type == "all":
        types = list(NORMAL_PATHS.keys())
    else:
        if process_type not in NORMAL_PATHS:
            raise ValueError(f"Unknown process_type: {process_type}. "
                             f"Expected one of {list(NORMAL_PATHS.keys())} or 'all'.")
        types = [process_type]

    events: list[dict] = []

    for ptype in types:
        prefix = CASE_PREFIX[ptype]

        for case_idx in range(1, n_cases + 1):
            case_id = f"{prefix}-{start_date.strftime('%Y%m%d')}-{case_idx:04d}"

            # 케이스 시작 시간: 기준일로부터 랜덤 오프셋
            case_start = start_date + timedelta(
                hours=rng.randint(0, 720),   # 최대 30일
                minutes=rng.randint(0, 59),
            )

            sequence = _build_case_sequence(ptype, rng)
            metadata = _generate_metadata(ptype, rng)
            current_time = case_start

            for i, activity in enumerate(sequence):
                if i == 0:
                    duration = 0
                else:
                    duration = _get_duration(ptype, activity, rng)
                    current_time += timedelta(minutes=duration)

                events.append({
                    "case_id": case_id,
                    "process_type": ptype,
                    "activity": activity,
                    "timestamp": current_time.isoformat(),
                    "actor": _pick_actor(activity, rng),
                    "duration_minutes": duration,
                    "metadata": metadata,
                })

    # 전체 이벤트를 timestamp 기준 정렬
    events.sort(key=lambda e: e["timestamp"])

    return events
