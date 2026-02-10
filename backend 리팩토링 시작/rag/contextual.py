"""
rag/contextual.py - Contextual Retrieval (Anthropic 2024)

각 청크에 LLM으로 문서 맥락 요약을 추가하여 검색 정확도 향상
"""
import os
import json
import time
import hashlib
import threading
from typing import Dict, Optional

from core.utils import safe_str
import state as st

# ============================================================
# Contextual Retrieval 설정
# ============================================================
OPENAI_CLIENT = None
CONTEXTUAL_CACHE: Dict[str, str] = {}
CONTEXTUAL_CACHE_FILE = None
CONTEXTUAL_RETRIEVAL_ENABLED = True
CONTEXTUAL_MODEL = "gpt-4o-mini"

CONTEXTUAL_MAX_WORKERS = 5
CONTEXTUAL_MAX_RETRIES = 3
CONTEXTUAL_CACHE_LOCK = threading.Lock()


def _sha1_text(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


def _get_openai_client():
    """Lazy initialization of OpenAI client"""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    try:
        from openai import OpenAI
        api_key = getattr(st, "OPENAI_API_KEY", None)
        if not api_key:
            st.logger.warning("OPENAI_API_KEY_NOT_SET in state.py")
            return None
        OPENAI_CLIENT = OpenAI(api_key=api_key)
        st.logger.info("OPENAI_CLIENT_INITIALIZED for Contextual Retrieval")
        return OPENAI_CLIENT
    except ImportError as e:
        st.logger.warning("OPENAI_IMPORT_FAIL err=%s (pip install openai)", safe_str(e))
        return None
    except Exception as e:
        st.logger.warning("OPENAI_CLIENT_INIT_FAIL err=%s", safe_str(e))
        return None


def _load_contextual_cache() -> Dict[str, str]:
    """캐시 파일에서 Contextual Prefix 로드"""
    global CONTEXTUAL_CACHE, CONTEXTUAL_CACHE_FILE

    if CONTEXTUAL_CACHE_FILE is None:
        CONTEXTUAL_CACHE_FILE = os.path.join(st.RAG_FAISS_DIR, "contextual_cache.json")

    try:
        if os.path.exists(CONTEXTUAL_CACHE_FILE):
            with open(CONTEXTUAL_CACHE_FILE, "r", encoding="utf-8") as f:
                CONTEXTUAL_CACHE = json.load(f)
                st.logger.info("CONTEXTUAL_CACHE_LOADED count=%d", len(CONTEXTUAL_CACHE))
    except Exception as e:
        st.logger.warning("CONTEXTUAL_CACHE_LOAD_FAIL err=%s", safe_str(e))
        CONTEXTUAL_CACHE = {}

    return CONTEXTUAL_CACHE


def _save_contextual_cache() -> None:
    """캐시를 파일로 저장"""
    global CONTEXTUAL_CACHE, CONTEXTUAL_CACHE_FILE

    if CONTEXTUAL_CACHE_FILE is None:
        CONTEXTUAL_CACHE_FILE = os.path.join(st.RAG_FAISS_DIR, "contextual_cache.json")

    try:
        os.makedirs(os.path.dirname(CONTEXTUAL_CACHE_FILE), exist_ok=True)
        with open(CONTEXTUAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(CONTEXTUAL_CACHE, f, ensure_ascii=False, indent=2)
        st.logger.info("CONTEXTUAL_CACHE_SAVED count=%d", len(CONTEXTUAL_CACHE))
    except Exception as e:
        st.logger.warning("CONTEXTUAL_CACHE_SAVE_FAIL err=%s", safe_str(e))


def _generate_contextual_prefix(
    doc_content: str,
    chunk_content: str,
    source: str,
    section_title: str = ""
) -> str:
    """
    Contextual Retrieval: LLM으로 청크의 문서 내 맥락 생성

    Args:
        doc_content: 전체 문서 내용 (앞부분만 사용)
        chunk_content: 청크 내용
        source: 문서 출처
        section_title: 섹션 제목

    Returns:
        맥락 설명 문자열
    """
    global CONTEXTUAL_CACHE

    if not CONTEXTUAL_RETRIEVAL_ENABLED:
        return ""

    client = _get_openai_client()
    if client is None:
        return ""

    cache_key = _sha1_text(chunk_content)[:32]

    with CONTEXTUAL_CACHE_LOCK:
        if cache_key in CONTEXTUAL_CACHE:
            return CONTEXTUAL_CACHE[cache_key]

    doc_preview = doc_content[:6000] if len(doc_content) > 6000 else doc_content

    prompt = f"""<document>
{doc_preview}
</document>

위 문서에서 추출한 청크입니다:

<chunk>
{chunk_content[:1000]}
</chunk>

{f'섹션: {section_title}' if section_title else ''}
문서 출처: {source}

이 청크가 전체 문서에서 어떤 맥락에 있는지 간결하게 설명해주세요.
검색 정확도 향상을 위해 핵심 키워드(인물명, 주제, 카테고리)를 포함해주세요.
1-2문장으로 답변하세요.
"""

    for attempt in range(CONTEXTUAL_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=CONTEXTUAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )

            contextual_prefix = response.choices[0].message.content.strip()

            with CONTEXTUAL_CACHE_LOCK:
                CONTEXTUAL_CACHE[cache_key] = contextual_prefix

            return contextual_prefix

        except Exception as e:
            err_str = safe_str(e)
            if "429" in err_str or "500" in err_str or "502" in err_str or "503" in err_str:
                wait_time = (2 ** attempt) + 0.5
                st.logger.warning("CONTEXTUAL_RATE_LIMIT attempt=%d/%d wait=%.1fs err=%s",
                                 attempt + 1, CONTEXTUAL_MAX_RETRIES, wait_time, err_str[:50])
                time.sleep(wait_time)
            else:
                st.logger.warning("CONTEXTUAL_PREFIX_FAIL err=%s", err_str)
                return ""

    st.logger.warning("CONTEXTUAL_PREFIX_MAX_RETRIES_EXCEEDED source=%s", source[:30])
    return ""
