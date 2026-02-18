"""
process_miner/helpers.py - 프로세스 마이너 공통 헬퍼

타임스탬프 파싱, 케이스 그룹화 등 여러 모듈에서 중복되던 로직을 통합.
"""

from collections import defaultdict
from datetime import datetime
from functools import lru_cache


@lru_cache(maxsize=4096)
def parse_timestamp(ts: str) -> datetime:
    """타임스탬프 문자열을 datetime으로 파싱 (LRU 캐시 적용)"""
    return datetime.fromisoformat(ts)


def group_events_by_case(events: list[dict]) -> dict[str, list[dict]]:
    """이벤트를 case_id별로 그룹화하고 timestamp 순 정렬한다."""
    cases: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        cases[event["case_id"]].append(event)

    for case_id in cases:
        cases[case_id].sort(key=lambda e: e["timestamp"])

    return dict(cases)
