"""
rag/search.py - RAG 검색 로직

BM25 키워드 검색, Vector(FAISS) 검색, Hybrid Search, Reranking,
RAG-Fusion, 쿼리 확장 등 검색 관련 기능
"""
import re
import time
import threading
from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import safe_str
from rag.utils import tokenize_korean as _shared_tokenize_korean
import state as st

# ============================================================
# Optional imports
# ============================================================
BM25Okapi = None
BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    pass

CrossEncoder = None
RERANKER_AVAILABLE = False
RERANKER_MODEL = None
_RERANKER_LOAD_FAILED = False  # M28: 실패 기억 플래그
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# BM25 인덱스 상태
# ============================================================
BM25_INDEX: Optional[Any] = None
BM25_CORPUS: List[str] = []
BM25_DOC_MAP: List[Dict] = []

# ============================================================
# 스레드 안전성 (BM25 인덱스 접근 보호)
# ============================================================
_BM25_LOCK = threading.Lock()

# ============================================================
# 동일 쿼리 TTL 캐시 (하이브리드 검색 결과 캐싱)
# ============================================================
_SEARCH_CACHE: OrderedDict = OrderedDict()
_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE_MAX_SIZE = 200
_SEARCH_CACHE_TTL = 300  # 5분


def _get_search_cache(key: str) -> Optional[Any]:
    """TTL 검색 캐시 조회 (thread-safe, LRU)"""
    with _SEARCH_CACHE_LOCK:
        if key not in _SEARCH_CACHE:
            return None
        result, ts = _SEARCH_CACHE[key]
        if time.time() - ts > _SEARCH_CACHE_TTL:
            del _SEARCH_CACHE[key]
            return None
        _SEARCH_CACHE.move_to_end(key)
        return result


def _set_search_cache(key: str, value: Any) -> None:
    """TTL 검색 캐시 저장 (thread-safe, LRU eviction)"""
    with _SEARCH_CACHE_LOCK:
        if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX_SIZE:
            _SEARCH_CACHE.popitem(last=False)
        _SEARCH_CACHE[key] = (value, time.time())


# ============================================================
# Korean Tokenizer (M23: 공통 유틸로 위임)
# ============================================================
_tokenize_korean = _shared_tokenize_korean


# ============================================================
# BM25 인덱스 관리
# ============================================================
def _build_bm25_index(chunks: List[Any]) -> bool:
    """BM25 인덱스 빌드 (thread-safe)"""
    global BM25_INDEX, BM25_CORPUS, BM25_DOC_MAP

    if not BM25_AVAILABLE or BM25Okapi is None:
        return False

    try:
        corpus = []
        doc_map = []

        for chunk in chunks:
            try:
                content = safe_str(getattr(chunk, "page_content", ""))
                metadata = getattr(chunk, "metadata", {})
                if content:
                    corpus.append(content)
                    doc_map.append({
                        "content": content,
                        "source": metadata.get("source", ""),
                        "parent_id": metadata.get("parent_id", ""),
                    })
            except (AttributeError, TypeError):
                continue

        if not corpus:
            return False

        tokenized_corpus = [_tokenize_korean(doc) for doc in corpus]
        new_index = BM25Okapi(tokenized_corpus)

        with _BM25_LOCK:
            BM25_CORPUS = corpus
            BM25_DOC_MAP = doc_map
            BM25_INDEX = new_index

        st.logger.info("BM25_INDEX_BUILT docs=%d", len(corpus))
        return True
    except (ValueError, RuntimeError) as e:
        st.logger.warning("BM25_BUILD_FAIL err=%s", safe_str(e))
        return False


def _bm25_search(query: str, top_k: int = 5, parent_content_func=None) -> List[Tuple[Dict, float]]:
    """BM25 검색 (키워드 기반, thread-safe, TTL 캐시)"""
    # TTL 캐시 확인
    cache_key = f"bm25:{query}:{top_k}"
    cached = _get_search_cache(cache_key)
    if cached is not None:
        return cached

    with _BM25_LOCK:
        if BM25_INDEX is None or not BM25_DOC_MAP:
            return []
        local_index = BM25_INDEX
        local_doc_map = list(BM25_DOC_MAP)

    try:
        tokenized_query = _tokenize_korean(query)
        scores = local_index.get_scores(tokenized_query)

        scored_docs = list(zip(range(len(scores)), scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_parents = set()

        for idx, score in scored_docs:
            if score <= 0:
                continue
            if len(results) >= top_k:
                break

            doc = local_doc_map[idx]
            child_content = doc.get("content", "")
            source = doc.get("source", "")
            parent_id = doc.get("parent_id", "")

            if parent_id and parent_id not in seen_parents:
                if parent_content_func:
                    parent_content = parent_content_func(parent_id, child_content)
                else:
                    parent_content = child_content
                seen_parents.add(parent_id)
                content = parent_content
            elif not parent_id:
                content = child_content
            else:
                continue

            results.append((
                {
                    "content": content,
                    "source": source,
                    "parent_id": parent_id,
                    "matched_child": child_content[:100],
                },
                score
            ))

        _set_search_cache(cache_key, results)
        return results
    except (ValueError, IndexError, RuntimeError) as e:
        st.logger.warning("BM25_SEARCH_FAIL err=%s", safe_str(e))
        return []


# ============================================================
# Cross-Encoder Reranking
# ============================================================
def _get_reranker():
    """Reranker 모델 로드 (Lazy Loading, M28: 실패 기억)"""
    global RERANKER_MODEL, _RERANKER_LOAD_FAILED

    if not RERANKER_AVAILABLE or CrossEncoder is None:
        return None

    if RERANKER_MODEL is not None:
        return RERANKER_MODEL

    # M28: 이전에 모든 모델 로드 실패했으면 재시도 하지 않음
    if _RERANKER_LOAD_FAILED:
        return None

    models_to_try = [
        ('BAAI/bge-reranker-v2-m3', 1024),
        ('BAAI/bge-reranker-large', 512),
        ('cross-encoder/ms-marco-MiniLM-L-6-v2', 512),
    ]

    for model_name, max_len in models_to_try:
        try:
            RERANKER_MODEL = CrossEncoder(model_name, max_length=max_len)
            st.logger.info("RERANKER_LOADED model=%s max_length=%d", model_name, max_len)
            return RERANKER_MODEL
        except Exception as e:
            st.logger.warning("RERANKER_TRY_FAIL model=%s err=%s", model_name, safe_str(e))
            continue

    st.logger.warning("RERANKER_ALL_MODELS_FAILED - will not retry")
    _RERANKER_LOAD_FAILED = True
    return None


def _rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Cross-Encoder로 결과 재정렬"""
    reranker = _get_reranker()
    if reranker is None or not results:
        return results[:top_k]

    try:
        pairs = [(query, r.get("content", "")[:800]) for r in results]
        scores = reranker.predict(pairs)

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for r, score in scored_results[:top_k]:
            r["rerank_score"] = round(float(score), 4)
            reranked.append(r)

        st.logger.info("RERANK_DONE query=%s input=%d output=%d top_score=%.3f",
                       query[:30], len(results), len(reranked),
                       reranked[0]["rerank_score"] if reranked else 0)

        return reranked
    except Exception as e:
        st.logger.warning("RERANK_FAIL err=%s", safe_str(e))
        return results[:top_k]


# ============================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================
def _reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict, float]],
    vector_results: List[Tuple[Dict, float]],
    k: int = 60,
    bm25_weight: float = 1.5,
    vector_weight: float = 1.0
) -> List[Dict]:
    """BM25와 Vector 결과 병합 (RRF)"""
    fusion_scores: Dict[str, Dict] = {}

    for rank, (doc, score) in enumerate(bm25_results):
        key = doc.get("content", "")[:100]
        if key not in fusion_scores:
            fusion_scores[key] = {
                "doc": doc,
                "bm25_score": score,
                "vector_score": 0.0,
                "rrf_score": 0.0,
            }
        fusion_scores[key]["bm25_score"] = score
        fusion_scores[key]["rrf_score"] += bm25_weight / (k + rank + 1)

    for rank, (doc, dist) in enumerate(vector_results):
        key = doc.get("content", "")[:100]
        if key not in fusion_scores:
            fusion_scores[key] = {
                "doc": doc,
                "bm25_score": 0.0,
                "vector_score": 1.0 / (1.0 + dist),
                "rrf_score": 0.0,
            }
        fusion_scores[key]["vector_score"] = 1.0 / (1.0 + dist)
        fusion_scores[key]["rrf_score"] += vector_weight / (k + rank + 1)

    sorted_results = sorted(
        fusion_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return [
        {
            **item["doc"],
            "bm25_score": round(item["bm25_score"], 4),
            "vector_score": round(item["vector_score"], 4),
            "fusion_score": round(item["rrf_score"], 4),
        }
        for item in sorted_results
    ]


# ============================================================
# RAG-Fusion: Multi-Query
# ============================================================
RAG_FUSION_ENABLED = True
RAG_FUSION_NUM_QUERIES = 2

# L11: FIFO → LRU 캐시로 변경
_FUSION_QUERY_CACHE: OrderedDict = OrderedDict()
_FUSION_CACHE_MAX_SIZE = 200


def _is_simple_query(query: str) -> bool:
    """단순 쿼리 판별 - Fusion 스킵 대상"""
    q = query.strip()

    if len(q) <= 10:
        return True
    if ' ' not in q:
        return True

    simple_patterns = [
        r'^[가-힣A-Za-z]+ ?쿠키$',
        r'^[가-힣A-Za-z]+ ?왕국$',
        r'^[가-힣A-Za-z]+[은는이가] ?뭐',
    ]

    for pattern in simple_patterns:
        if re.match(pattern, q):
            return True

    return False


def _get_cached_fusion_queries(query: str) -> Optional[List[str]]:
    """캐시에서 Fusion 쿼리 조회 (L11: LRU - 접근 시 순서 갱신)"""
    if query in _FUSION_QUERY_CACHE:
        _FUSION_QUERY_CACHE.move_to_end(query)
        return _FUSION_QUERY_CACHE[query]
    return None


def _cache_fusion_queries(query: str, queries: List[str]) -> None:
    """Fusion 쿼리를 캐시에 저장 (L11: LRU eviction)"""
    if len(_FUSION_QUERY_CACHE) >= _FUSION_CACHE_MAX_SIZE:
        _FUSION_QUERY_CACHE.popitem(last=False)  # LRU: 가장 오래된 항목 제거

    _FUSION_QUERY_CACHE[query] = queries


def _generate_fusion_queries(original_query: str, api_key: str = "", num_queries: int = 4) -> List[str]:
    """RAG-Fusion: LLM으로 다양한 쿼리 변형 생성"""
    queries = [original_query]

    if not original_query or num_queries <= 1:
        return queries

    if _is_simple_query(original_query):
        st.logger.info("RAG_FUSION_SKIP: Simple query '%s'", original_query[:30])
        return queries

    cached = _get_cached_fusion_queries(original_query)
    if cached:
        st.logger.info("RAG_FUSION_CACHE_HIT: '%s' -> %d queries", original_query[:30], len(cached))
        return cached

    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY
    if not effective_key:
        st.logger.warning("RAG_FUSION: No API key, skipping query generation")
        return queries

    try:
        from rag.utils import get_openai_client
        client = get_openai_client()
        if client is None:
            return queries

        prompt = f"""당신은 검색 쿼리 전문가입니다. 주어진 쿼리를 다양한 관점에서 {num_queries - 1}개의 변형 쿼리로 재작성하세요.

원본 쿼리: "{original_query}"

규칙:
1. 원본 의도를 유지하되 다른 표현/키워드 사용
2. 동의어, 유사어, 관련 용어 활용
3. 질문 형식과 서술 형식 혼합
4. 이커머스 플랫폼 용어가 있으면 관련 용어도 포함
5. 각 쿼리는 한 줄로, 번호 없이

{num_queries - 1}개의 변형 쿼리만 출력하세요 (번호/설명 없이):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )

        generated = response.choices[0].message.content.strip()
        new_queries = [q.strip() for q in generated.split("\n") if q.strip()]

        for q in new_queries[:num_queries - 1]:
            if q and q != original_query and q not in queries:
                queries.append(q)

        st.logger.info("RAG_FUSION: Generated %d queries from '%s'", len(queries), original_query[:30])

        if len(queries) > 1:
            _cache_fusion_queries(original_query, queries)

    except Exception as e:
        st.logger.warning("RAG_FUSION: Query generation failed: %s", safe_str(e)[:100])

    return queries


def _multi_query_rrf(
    all_results: List[List[Tuple[Dict, float]]],
    k: int = 60
) -> List[Tuple[Dict, float]]:
    """Multi-Query RRF: 여러 쿼리 결과를 RRF로 병합"""
    fusion_scores: Dict[str, Dict] = {}

    for query_idx, results in enumerate(all_results):
        for rank, (doc, score) in enumerate(results):
            key = doc.get("content", "")[:150]
            if not key:
                continue

            if key not in fusion_scores:
                fusion_scores[key] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "query_hits": 0,
                    "best_rank": rank,
                }

            fusion_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
            fusion_scores[key]["query_hits"] += 1
            fusion_scores[key]["best_rank"] = min(fusion_scores[key]["best_rank"], rank)

    for key, data in fusion_scores.items():
        if data["query_hits"] > 1:
            data["rrf_score"] *= (1.0 + 0.1 * data["query_hits"])

    sorted_results = sorted(
        fusion_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return [(item["doc"], item["rrf_score"]) for item in sorted_results]


# ============================================================
# Query Expansion (카페24 플랫폼 도메인)
# ============================================================
QUERY_EXPANSION_MAP: Dict[str, List[str]] = {
    # 이커머스 관련
    "정산": ["PG", "결제", "수수료", "환불"],
    "배송": ["택배", "물류", "출고", "반품"],
    "결제": ["PG", "이니시스", "토스페이먼츠", "카카오페이"],
    "회원": ["고객", "사용자", "계정"],
    "SEO": ["검색엔진", "메타태그", "사이트맵"],
    "API": ["연동", "REST", "웹훅", "엔드포인트"],
}

QUERY_INJECTION_RULES: List[Tuple[str, str]] = []


@lru_cache(maxsize=256)
def _expand_query(query: str) -> str:
    """쿼리를 도메인 표현에 맞게 확장 (L13: lru_cache 적용)"""
    if not query:
        return query

    expanded_terms = set()
    query_lower = query.lower()

    for keyword, expansions in QUERY_EXPANSION_MAP.items():
        if keyword in query or keyword in query_lower:
            expanded_terms.update(expansions)

    for trigger, injection in QUERY_INJECTION_RULES:
        if trigger in query or trigger in query_lower:
            expanded_terms.add(injection)

    if expanded_terms:
        new_terms = [t for t in expanded_terms if t not in query]
        if new_terms:
            expanded_query = f"{query} {' '.join(new_terms[:5])}"
            st.logger.info("QUERY_EXPANSION original='%s' expanded='%s'", query[:50], expanded_query[:80])
            return expanded_query

    return query


# ============================================================
# 글로서리(사전) 검색
# ============================================================
@lru_cache(maxsize=256)
def rag_search_glossary(query: str, top_k: int = 3) -> List[dict]:
    """글로서리 검색 (L13: lru_cache 적용)"""
    from core.constants import RAG_DOCUMENTS
    query_lower = (query or "").lower()
    scores = []

    for _, doc in RAG_DOCUMENTS.items():
        score = 0
        for kw in doc.get("keywords", []):
            kw_lower = (kw or "").lower().strip()
            if kw_lower and kw_lower in query_lower:
                score += 2
        title_lower = (doc.get("title") or "").lower().strip()
        if title_lower and title_lower in query_lower:
            score += 3
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        {"title": doc["title"], "content": doc["content"], "source": "glossary", "score": float(sc)}
        for sc, doc in scores[:top_k]
        if sc > 0
    ]
