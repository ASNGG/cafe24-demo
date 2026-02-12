"""
automation/action_logger.py - 자동화 조치 로깅
==============================================
모든 자동 조치(이탈 방지, FAQ 생성, 리포트 등)의 실행 기록을 관리합니다.
"""
import time
import uuid
from typing import Dict, List, Any, Optional
from threading import Lock

import state as st


# 전역 액션 로그 저장소
_ACTION_LOG: List[Dict[str, Any]] = []
_ACTION_LOCK = Lock()

# FAQ 저장소
_FAQ_STORE: Dict[str, Dict[str, Any]] = {}
_FAQ_LOCK = Lock()

# 리포트 히스토리
_REPORT_HISTORY: List[Dict[str, Any]] = []
_REPORT_LOCK = Lock()

# 리텐션 메시지 히스토리
_RETENTION_HISTORY: List[Dict[str, Any]] = []
_RETENTION_LOCK = Lock()


def log_action(
    action_type: str,
    target_id: str,
    detail: Dict[str, Any],
    status: str = "success",
) -> Dict[str, Any]:
    """자동화 조치를 로깅합니다."""
    entry = {
        "id": str(uuid.uuid4())[:8],
        "action_type": action_type,
        "target_id": target_id,
        "detail": detail,
        "status": status,
        "timestamp": time.time(),
    }
    with _ACTION_LOCK:
        _ACTION_LOG.append(entry)
    st.logger.info("ACTION_LOG type=%s target=%s status=%s", action_type, target_id, status)
    return entry


def get_action_log(
    action_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """액션 로그를 조회합니다."""
    with _ACTION_LOCK:
        logs = list(_ACTION_LOG)
    if action_type:
        logs = [l for l in logs if l["action_type"] == action_type]
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return logs[:limit]


def get_action_stats() -> Dict[str, Any]:
    """액션 통계를 반환합니다."""
    with _ACTION_LOCK:
        logs = list(_ACTION_LOG)
    total = len(logs)
    by_type: Dict[str, int] = {}
    for l in logs:
        t = l["action_type"]
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "total_actions": total,
        "by_type": by_type,
    }


# ── FAQ 저장소 ──
def save_faq(faq_id: str, faq_data: Dict[str, Any]) -> None:
    with _FAQ_LOCK:
        _FAQ_STORE[faq_id] = faq_data


def get_faq(faq_id: str) -> Optional[Dict[str, Any]]:
    with _FAQ_LOCK:
        return _FAQ_STORE.get(faq_id)


def get_all_faqs() -> List[Dict[str, Any]]:
    with _FAQ_LOCK:
        return list(_FAQ_STORE.values())


def delete_faq(faq_id: str) -> bool:
    with _FAQ_LOCK:
        if faq_id in _FAQ_STORE:
            del _FAQ_STORE[faq_id]
            return True
        return False


def update_faq_status(faq_id: str, status: str) -> bool:
    with _FAQ_LOCK:
        if faq_id in _FAQ_STORE:
            _FAQ_STORE[faq_id]["status"] = status
            _FAQ_STORE[faq_id]["updated_at"] = time.time()
            return True
        return False


# ── 리포트 히스토리 ──
def save_report(report: Dict[str, Any]) -> None:
    with _REPORT_LOCK:
        _REPORT_HISTORY.append(report)


def get_report_history(limit: int = 20) -> List[Dict[str, Any]]:
    with _REPORT_LOCK:
        return sorted(_REPORT_HISTORY, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]


# ── 리텐션 히스토리 ──
def save_retention_action(action: Dict[str, Any]) -> None:
    with _RETENTION_LOCK:
        _RETENTION_HISTORY.append(action)


def get_retention_history(limit: int = 50) -> List[Dict[str, Any]]:
    with _RETENTION_LOCK:
        return sorted(_RETENTION_HISTORY, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]
