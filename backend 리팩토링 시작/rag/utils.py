"""
rag/utils.py - RAG 공통 유틸리티

중복 제거를 위한 공유 함수:
- _sha1_text: 텍스트 해시 (chunking, contextual, k2rag)
- _clean_text: 텍스트 정제 (chunking, k2rag)
- _tokenize_korean: 한국어 토큰화 (search, k2rag)
- _extract_text_from_pdf: PDF 텍스트 추출 (chunking, light_rag)
- get_openai_client: OpenAI 클라이언트 팩토리 (contextual, k2rag, light_rag, search)
"""
import os
import re
import hashlib
import threading
from typing import List, Optional
from functools import lru_cache

import state as st
from core.utils import safe_str


# ============================================================
# M21: _sha1_text 통합 (chunking, contextual, k2rag 3중 중복)
# ============================================================
def sha1_text(s: str) -> str:
    """텍스트 SHA1 해시 생성"""
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


# ============================================================
# M22: _clean_text 통합 (chunking._clean_text_for_rag, k2rag._clean_text)
# ============================================================
def clean_text(txt: str) -> str:
    """텍스트 정제 (제어문자 제거, 공백 정리)"""
    if not txt:
        return ""
    txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# ============================================================
# M23: _tokenize_korean 통합 (search, k2rag 2중 중복)
# ============================================================
_KOREAN_CHAR_RE = re.compile(r'[가-힣]')
_SUFFIX_PATTERN = re.compile(
    r'(은|는|이|가|을|를|에|에서|으로|로|와|과|의|도|만|까지|부터|에게|한테|께)$'
)


def tokenize_korean(text: str) -> List[str]:
    """
    한국어/영어 토큰화 (whitespace + 한글 바이그램 + 조사 제거)

    search.py와 k2rag.py의 토큰화를 통합:
    - 공백 기준 분리
    - 한글 조사/어미 제거
    - 2-gram 생성 (부분 매칭 지원)
    """
    if not text:
        return []

    tokens = text.lower().split()
    result = []

    for tok in tokens:
        result.append(tok)

        if _KOREAN_CHAR_RE.search(tok):
            # 조사 제거 버전 추가
            stripped = _SUFFIX_PATTERN.sub('', tok)
            if stripped and stripped != tok and len(stripped) >= 2:
                result.append(stripped)

            # 2-gram 생성
            if len(tok) >= 2:
                for i in range(len(tok) - 1):
                    bigram = tok[i:i + 2]
                    if _KOREAN_CHAR_RE.search(bigram):
                        result.append(bigram)

    return result


# ============================================================
# M24: _extract_text_from_pdf 통합 (chunking, light_rag 2중 중복)
# ============================================================
def extract_text_from_pdf(path: str) -> str:
    """PDF에서 텍스트 추출 (PyMuPDF > pypdf fallback)"""
    raw_text = ""

    try:
        import fitz
        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
        raw_text = "\n".join(text_parts).strip()
        if raw_text:
            st.logger.info("PDF_EXTRACTED_PYMUPDF path=%s chars=%d",
                          os.path.basename(path), len(raw_text))
    except ImportError:
        st.logger.debug("pymupdf not available, falling back to pypdf")
    except Exception as e:
        st.logger.warning("PYMUPDF_FAIL path=%s err=%s", path, safe_str(e))

    if not raw_text:
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return ""
            reader = PdfReader(path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            raw_text = "\n".join(text_parts).strip()
            if raw_text:
                st.logger.info("PDF_EXTRACTED_PYPDF path=%s chars=%d",
                              os.path.basename(path), len(raw_text))
        except Exception as e:
            st.logger.warning("PYPDF_FAIL path=%s err=%s", path, safe_str(e))
            return ""

    return raw_text


# ============================================================
# M25 / cross-1: OpenAI 클라이언트 팩토리
# (contextual, k2rag, light_rag, search 4중 중복 제거)
# ============================================================
_openai_sync_client = None
_openai_async_client = None
_openai_client_lock = threading.Lock()


def get_openai_client():
    """동기 OpenAI 클라이언트 싱글톤 (lazy init)"""
    global _openai_sync_client

    with _openai_client_lock:
        if _openai_sync_client is not None:
            return _openai_sync_client

        try:
            from openai import OpenAI
            api_key = getattr(st, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                st.logger.warning("OPENAI_API_KEY not set")
                return None
            _openai_sync_client = OpenAI(api_key=api_key)
            st.logger.info("OPENAI_SYNC_CLIENT_INITIALIZED (shared)")
            return _openai_sync_client
        except ImportError as e:
            st.logger.warning("OPENAI_IMPORT_FAIL err=%s", safe_str(e))
            return None
        except Exception as e:
            st.logger.warning("OPENAI_CLIENT_INIT_FAIL err=%s", safe_str(e))
            return None


def get_openai_async_client():
    """비동기 OpenAI 클라이언트 싱글톤 (lazy init)"""
    global _openai_async_client

    with _openai_client_lock:
        if _openai_async_client is not None:
            return _openai_async_client

        try:
            from openai import AsyncOpenAI
            api_key = getattr(st, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                st.logger.warning("OPENAI_API_KEY not set")
                return None
            _openai_async_client = AsyncOpenAI(api_key=api_key)
            st.logger.info("OPENAI_ASYNC_CLIENT_INITIALIZED (shared)")
            return _openai_async_client
        except ImportError as e:
            st.logger.warning("OPENAI_ASYNC_IMPORT_FAIL err=%s", safe_str(e))
            return None
        except Exception as e:
            st.logger.warning("OPENAI_ASYNC_CLIENT_INIT_FAIL err=%s", safe_str(e))
            return None
