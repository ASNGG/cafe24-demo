"""
CAFE24 AI 운영 플랫폼 - 전역 상태 관리
================================
카페24 AI 기반 내부 시스템 개발 프로젝트

모든 공유 가변 상태를 한 곳에서 관리합니다.
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from threading import Lock

import pandas as pd

# ============================================================
# 경로
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = Path(BASE_DIR)  # Path 객체 (routes.py에서 / 연산자 사용용)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "backend.log")

# ============================================================
# 로깅
# ============================================================
_logging_initialized = False

def setup_logging() -> logging.Logger:
    """로깅 초기화 (싱글톤 — 중복 호출 시 기존 로거 반환)"""
    global _logging_initialized
    lg = logging.getLogger("cafe24-ai")
    if _logging_initialized:
        return lg
    _logging_initialized = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8", delay=True),
        ],
        force=True,
    )
    lg.setLevel(logging.INFO)
    lg.propagate = True
    for uvn in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        ul = logging.getLogger(uvn)
        ul.setLevel(logging.INFO)
        ul.propagate = True
    # pdfminer 경고 숨기기
    for pdflogger in ("pdfminer", "pdfminer.pdffont", "pdfminer.pdfinterp", "pdfminer.pdfpage"):
        logging.getLogger(pdflogger).setLevel(logging.ERROR)
    lg.info("LOGGER_READY log_file=%s", LOG_FILE)
    return lg

logger = setup_logging()

# ============================================================
# OpenAI 설정
# ============================================================
def _load_api_key() -> str:
    """API 키 로드 (우선순위: 환경변수 > 파일)"""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    key_file = os.path.join(BASE_DIR, "openai_api_key.txt")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        except Exception:
            pass

    return ""

OPENAI_API_KEY: str = _load_api_key()

# ============================================================
# 사용자 DB (메모리)
# ============================================================
USERS: Dict[str, Dict[str, str]] = {
    "admin": {"password": "admin123", "role": "관리자", "name": "관리자"},
    "user": {"password": "user123", "role": "사용자", "name": "사용자"},
    "operator": {"password": "oper123", "role": "운영자", "name": "운영자"},
    "analyst": {"password": "analyst123", "role": "분석가", "name": "분석가"},
}

# ============================================================
# 카페24 이커머스 데이터프레임
# ============================================================
# 쇼핑몰 데이터
SHOPS_DF: Optional[pd.DataFrame] = None

# 상품 카테고리 데이터
CATEGORIES_DF: Optional[pd.DataFrame] = None

# 플랫폼 서비스 데이터
SERVICES_DF: Optional[pd.DataFrame] = None

# 상품 리스팅 데이터
PRODUCTS_DF: Optional[pd.DataFrame] = None

# 셀러 데이터
SELLERS_DF: Optional[pd.DataFrame] = None

# 운영 로그 데이터
OPERATION_LOGS_DF: Optional[pd.DataFrame] = None

# 셀러 분석 데이터
SELLER_ANALYTICS_DF: Optional[pd.DataFrame] = None

# 플랫폼 문서 데이터
PLATFORM_DOCS_DF: Optional[pd.DataFrame] = None

# 이커머스 용어집
ECOMMERCE_GLOSSARY_DF: Optional[pd.DataFrame] = None

# ============================================================
# 분석용 추가 데이터프레임
# ============================================================
# 쇼핑몰별 성과 KPI
SHOP_PERFORMANCE_DF: Optional[pd.DataFrame] = None

# 일별 플랫폼 지표 (GMV, 활성셀러, 주문수 등)
DAILY_METRICS_DF: Optional[pd.DataFrame] = None

# CS 문의 통계
CS_STATS_DF: Optional[pd.DataFrame] = None

# 이상거래 상세 데이터
FRAUD_DETAILS_DF: Optional[pd.DataFrame] = None

# 셀러 코호트 리텐션 데이터
COHORT_RETENTION_DF: Optional[pd.DataFrame] = None

# 전환 퍼널 데이터
CONVERSION_FUNNEL_DF: Optional[pd.DataFrame] = None

# 셀러 일별 활동 데이터
SELLER_ACTIVITY_DF: Optional[pd.DataFrame] = None

# ============================================================
# ML 모델
# ============================================================
SELECTED_MODELS_FILE = os.path.join(BASE_DIR, "selected_models.json")
SELECTED_MODELS: Dict[str, str] = {}

def save_selected_models() -> bool:
    """선택된 모델 상태를 JSON 파일에 저장"""
    try:
        with open(SELECTED_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(SELECTED_MODELS, f, ensure_ascii=False, indent=2)
        logger.info(f"선택된 모델 상태 저장 완료: {SELECTED_MODELS}")
        return True
    except Exception as e:
        logger.error(f"선택된 모델 상태 저장 실패: {e}")
        return False

_selected_models_loaded = False

def load_selected_models() -> Dict[str, str]:
    """저장된 모델 선택 상태 로드 (1회 캐시 — 이미 로드된 경우 디스크 I/O 스킵)"""
    global SELECTED_MODELS, _selected_models_loaded
    if _selected_models_loaded and SELECTED_MODELS:
        return SELECTED_MODELS
    try:
        if os.path.exists(SELECTED_MODELS_FILE):
            with open(SELECTED_MODELS_FILE, "r", encoding="utf-8") as f:
                SELECTED_MODELS = json.load(f)
                _selected_models_loaded = True
                logger.info(f"선택된 모델 상태 로드 완료: {SELECTED_MODELS}")
                return SELECTED_MODELS
    except Exception as e:
        logger.warning(f"선택된 모델 상태 로드 실패: {e}")
    _selected_models_loaded = True
    return {}

# ── 핵심 6개 모델 (리팩토링) ──
# CS 응답 품질 예측 모델
CS_QUALITY_MODEL: Optional[Any] = None

# 문의 자동 분류 모델 (TF-IDF + RF)
INQUIRY_CLASSIFICATION_MODEL: Optional[Any] = None

# 셀러 세그먼트 모델 (K-Means)
SELLER_SEGMENT_MODEL: Optional[Any] = None

# 이상거래 탐지 모델 (Isolation Forest)
FRAUD_DETECTION_MODEL: Optional[Any] = None

# 셀러 이탈 예측 모델 (RandomForest + SHAP)
SELLER_CHURN_MODEL: Optional[Any] = None

# SHAP Explainer (이탈 예측용)
SHAP_EXPLAINER_CHURN: Optional[Any] = None

# 이탈 예측 모델 설정
CHURN_MODEL_CONFIG: Optional[Dict[str, Any]] = None

# ── 신규 4개 모델 ──
# 매출 예측 모델 (LightGBM)
REVENUE_PREDICTION_MODEL: Optional[Any] = None

# 고객 LTV 예측 모델 (GradientBoosting)
CUSTOMER_LTV_MODEL: Optional[Any] = None

# 리뷰 감성 분석 모델 (TF-IDF + LogisticRegression)
REVIEW_SENTIMENT_MODEL: Optional[Any] = None

# 상품 수요 예측 모델 (XGBoost)
DEMAND_FORECAST_MODEL: Optional[Any] = None

# 정산 이상 탐지 모델 (DBSCAN)
SETTLEMENT_ANOMALY_MODEL: Optional[Any] = None

# ── 공용 모델 도구 ──
# TF-IDF 벡터라이저 (문의 분류용)
TFIDF_VECTORIZER: Optional[Any] = None

# TF-IDF 벡터라이저 (리뷰 감성 분석용)
TFIDF_VECTORIZER_SENTIMENT: Optional[Any] = None

# 스케일러 (셀러 세그먼트용)
SCALER_CLUSTER: Optional[Any] = None

# 마케팅 최적화 모듈 사용 가능 여부
MARKETING_OPTIMIZER_AVAILABLE: bool = False

# ============================================================
# 라벨 인코더
# ============================================================
LE_TICKET_CATEGORY: Optional[Any] = None   # CS 문의 카테고리
LE_SELLER_TIER: Optional[Any] = None       # 셀러 등급
LE_CS_PRIORITY: Optional[Any] = None       # CS 우선순위
LE_INQUIRY_CATEGORY: Optional[Any] = None  # 문의 분류 카테고리

# ============================================================
# 캐시
# ============================================================
# 쇼핑몰별 서비스 매핑
SHOP_SERVICE_MAP: Dict[str, Dict[str, Any]] = {}

# 셀러별 분석 데이터 캐시
SELLER_CACHE: Dict[str, Dict[str, Any]] = {}

# ============================================================
# 최근 컨텍스트 저장 (요약 재활용, 크기 제한 + TTL)
# ============================================================
LAST_CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}
LAST_CONTEXT_LOCK = Lock()
LAST_CONTEXT_TTL_SEC = 600
LAST_CONTEXT_MAX_ENTRIES = 200


def set_last_context(key: str, value: Dict[str, Any]) -> None:
    """컨텍스트 저장 (TTL 타임스탬프 자동 부여, 크기 제한 적용)"""
    import time as _time
    with LAST_CONTEXT_LOCK:
        value["_ts"] = _time.time()
        LAST_CONTEXT_STORE[key] = value
        _evict_context_if_needed()


def get_last_context(key: str) -> Optional[Dict[str, Any]]:
    """컨텍스트 조회 (TTL 만료 시 None 반환)"""
    import time as _time
    with LAST_CONTEXT_LOCK:
        entry = LAST_CONTEXT_STORE.get(key)
        if entry is None:
            return None
        if _time.time() - entry.get("_ts", 0) > LAST_CONTEXT_TTL_SEC:
            LAST_CONTEXT_STORE.pop(key, None)
            return None
        return entry


def _evict_context_if_needed() -> None:
    """컨텍스트 스토어가 최대 크기 초과 시 가장 오래된 항목부터 제거 (lock 내부에서 호출)"""
    if len(LAST_CONTEXT_STORE) <= LAST_CONTEXT_MAX_ENTRIES:
        return
    sorted_keys = sorted(
        LAST_CONTEXT_STORE.keys(),
        key=lambda k: LAST_CONTEXT_STORE[k].get("_ts", 0),
    )
    to_remove = len(LAST_CONTEXT_STORE) - LAST_CONTEXT_MAX_ENTRIES
    for k in sorted_keys[:to_remove]:
        LAST_CONTEXT_STORE.pop(k, None)

# ============================================================
# RAG 설정/상태
# ============================================================
RAG_DOCS_DIR = os.path.join(BASE_DIR, "rag_docs")
RAG_FAISS_DIR = os.path.join(BASE_DIR, "rag_faiss")
RAG_STATE_FILE = os.path.join(RAG_FAISS_DIR, "rag_state.json")
RAG_EMBED_MODEL = "text-embedding-3-small"
RAG_ALLOWED_EXTS = {".txt", ".md", ".json", ".csv", ".log", ".pdf"}
RAG_MAX_DOC_CHARS = 200000
RAG_SNIPPET_CHARS = 1200
RAG_DEFAULT_TOPK = 5
RAG_MAX_TOPK = 20

RAG_LOCK = Lock()
RAG_STORE: Dict[str, Any] = {
    "ready": False,
    "hash": "",
    "docs_count": 0,
    "last_build_ts": 0.0,
    "error": "",
    "index": None,
}

# ============================================================
# LightRAG 설정 (중앙 관리)
# ============================================================
LIGHTRAG_CONFIG = {
    "top_k": 10,
    "top_k_dual": 1,
    "context_max_chars": 6000,
}

# ============================================================
# 멀티 에이전트 상태
# ============================================================
AGENT_TASKS: Dict[str, Dict[str, Any]] = {}
AGENT_LOCK = Lock()

# ============================================================
# CS 작업 큐
# ============================================================
CS_QUEUE: List[Dict[str, Any]] = []
CS_LOCK = Lock()

# ============================================================
# 시스템 상태
# ============================================================
SYSTEM_STATUS = {
    "initialized": False,
    "data_loaded": False,
    "models_loaded": False,
    "rag_ready": False,
    "startup_time": 0.0,
    "last_error": "",
}

# ============================================================
# 시스템 프롬프트 설정 (백엔드 중앙 관리)
# ============================================================
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.json")
CUSTOM_SYSTEM_PROMPT: Optional[str] = None

def save_system_prompt(prompt: str) -> bool:
    """시스템 프롬프트를 파일에 저장"""
    global CUSTOM_SYSTEM_PROMPT, _system_prompt_loaded
    try:
        CUSTOM_SYSTEM_PROMPT = prompt
        _system_prompt_loaded = True  # 메모리 캐시 갱신
        with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"system_prompt": prompt}, f, ensure_ascii=False, indent=2)
        logger.info("시스템 프롬프트 저장 완료")
        return True
    except Exception as e:
        logger.error(f"시스템 프롬프트 저장 실패: {e}")
        return False

_system_prompt_loaded = False

def load_system_prompt() -> Optional[str]:
    """저장된 시스템 프롬프트 로드 (1회 캐시)"""
    global CUSTOM_SYSTEM_PROMPT, _system_prompt_loaded
    if _system_prompt_loaded:
        return CUSTOM_SYSTEM_PROMPT
    _system_prompt_loaded = True
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                CUSTOM_SYSTEM_PROMPT = data.get("system_prompt")
                if CUSTOM_SYSTEM_PROMPT:
                    logger.info("시스템 프롬프트 로드 완료")
                return CUSTOM_SYSTEM_PROMPT
    except Exception as e:
        logger.warning(f"시스템 프롬프트 로드 실패: {e}")
    return None

def get_active_system_prompt() -> str:
    """현재 활성 시스템 프롬프트 반환 (커스텀 > 기본값)"""
    from core.constants import DEFAULT_SYSTEM_PROMPT
    if CUSTOM_SYSTEM_PROMPT and CUSTOM_SYSTEM_PROMPT.strip():
        return CUSTOM_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT

def reset_system_prompt() -> bool:
    """시스템 프롬프트를 기본값으로 초기화"""
    global CUSTOM_SYSTEM_PROMPT
    try:
        CUSTOM_SYSTEM_PROMPT = None
        if os.path.exists(SYSTEM_PROMPT_FILE):
            os.remove(SYSTEM_PROMPT_FILE)
        logger.info("시스템 프롬프트 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"시스템 프롬프트 초기화 실패: {e}")
        return False

# ============================================================
# LLM 설정 (백엔드 중앙 관리)
# ============================================================
LLM_SETTINGS_FILE = os.path.join(BASE_DIR, "llm_settings.json")

DEFAULT_LLM_SETTINGS: Dict[str, Any] = {
    "selectedModel": "gpt-4o-mini",
    "customModel": "",
    "temperature": 0.3,
    "topP": 1.0,
    "presencePenalty": 0.0,
    "frequencyPenalty": 0.0,
    "maxTokens": 8000,
    "seed": None,
    "timeoutMs": 30000,
    "retries": 2,
    "stream": True,
}

CUSTOM_LLM_SETTINGS: Optional[Dict[str, Any]] = None

def save_llm_settings(settings: Dict[str, Any]) -> bool:
    """LLM 설정을 파일에 저장"""
    global CUSTOM_LLM_SETTINGS, _llm_settings_loaded
    try:
        merged = {**DEFAULT_LLM_SETTINGS, **settings}
        CUSTOM_LLM_SETTINGS = merged
        _llm_settings_loaded = True  # 메모리 캐시 갱신
        with open(LLM_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM 설정 저장 완료: model={merged.get('selectedModel')}")
        return True
    except Exception as e:
        logger.error(f"LLM 설정 저장 실패: {e}")
        return False

_llm_settings_loaded = False

def load_llm_settings() -> Dict[str, Any]:
    """저장된 LLM 설정 로드 (1회 캐시)"""
    global CUSTOM_LLM_SETTINGS, _llm_settings_loaded
    if _llm_settings_loaded:
        return CUSTOM_LLM_SETTINGS if CUSTOM_LLM_SETTINGS else DEFAULT_LLM_SETTINGS.copy()
    _llm_settings_loaded = True
    try:
        if os.path.exists(LLM_SETTINGS_FILE):
            with open(LLM_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                CUSTOM_LLM_SETTINGS = {**DEFAULT_LLM_SETTINGS, **data}
                logger.info(f"LLM 설정 로드 완료: model={CUSTOM_LLM_SETTINGS.get('selectedModel')}")
                return CUSTOM_LLM_SETTINGS
    except Exception as e:
        logger.warning(f"LLM 설정 로드 실패: {e}")
    return DEFAULT_LLM_SETTINGS.copy()

def get_active_llm_settings() -> Dict[str, Any]:
    """현재 활성 LLM 설정 반환 (커스텀 > 기본값)"""
    if CUSTOM_LLM_SETTINGS:
        return CUSTOM_LLM_SETTINGS.copy()
    return DEFAULT_LLM_SETTINGS.copy()

def reset_llm_settings() -> bool:
    """LLM 설정을 기본값으로 초기화"""
    global CUSTOM_LLM_SETTINGS
    try:
        CUSTOM_LLM_SETTINGS = None
        if os.path.exists(LLM_SETTINGS_FILE):
            os.remove(LLM_SETTINGS_FILE)
        logger.info("LLM 설정 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"LLM 설정 초기화 실패: {e}")
        return False
