"""
rag/search.py - RAG 검색 로직

BM25 키워드 검색, Vector(FAISS) 검색, Hybrid Search, Reranking,
RAG-Fusion, 쿼리 확장 등 검색 관련 기능
"""
import re
import time
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import safe_str
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
# Korean Tokenizer
# ============================================================
def _tokenize_korean(text: str) -> List[str]:
    """한국어/영어 토큰화 (whitespace + 한글 바이그램)"""
    if not text:
        return []
    tokens = text.lower().split()
    result = []
    for tok in tokens:
        result.append(tok)
        if re.search(r'[가-힣]', tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                result.append(tok[i:i + 2])
    return result


# ============================================================
# BM25 인덱스 관리
# ============================================================
def _build_bm25_index(chunks: List[Any]) -> bool:
    """BM25 인덱스 빌드"""
    global BM25_INDEX, BM25_CORPUS, BM25_DOC_MAP

    if not BM25_AVAILABLE or BM25Okapi is None:
        return False

    try:
        BM25_CORPUS = []
        BM25_DOC_MAP = []

        for chunk in chunks:
            try:
                content = safe_str(getattr(chunk, "page_content", ""))
                metadata = getattr(chunk, "metadata", {})
                if content:
                    BM25_CORPUS.append(content)
                    BM25_DOC_MAP.append({
                        "content": content,
                        "source": metadata.get("source", ""),
                        "parent_id": metadata.get("parent_id", ""),
                    })
            except Exception:
                continue

        if not BM25_CORPUS:
            return False

        tokenized_corpus = [_tokenize_korean(doc) for doc in BM25_CORPUS]
        BM25_INDEX = BM25Okapi(tokenized_corpus)
        st.logger.info("BM25_INDEX_BUILT docs=%d", len(BM25_CORPUS))
        return True
    except Exception as e:
        st.logger.warning("BM25_BUILD_FAIL err=%s", safe_str(e))
        return False


def _bm25_search(query: str, top_k: int = 5, parent_content_func=None) -> List[Tuple[Dict, float]]:
    """BM25 검색 (키워드 기반)"""
    global BM25_INDEX, BM25_DOC_MAP

    if BM25_INDEX is None or not BM25_DOC_MAP:
        return []

    try:
        tokenized_query = _tokenize_korean(query)
        scores = BM25_INDEX.get_scores(tokenized_query)

        scored_docs = list(zip(range(len(scores)), scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_parents = set()

        for idx, score in scored_docs:
            if score <= 0:
                continue
            if len(results) >= top_k:
                break

            doc = BM25_DOC_MAP[idx]
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

        return results
    except Exception as e:
        st.logger.warning("BM25_SEARCH_FAIL err=%s", safe_str(e))
        return []


# ============================================================
# Cross-Encoder Reranking
# ============================================================
def _get_reranker():
    """Reranker 모델 로드 (Lazy Loading)"""
    global RERANKER_MODEL

    if not RERANKER_AVAILABLE or CrossEncoder is None:
        return None

    if RERANKER_MODEL is not None:
        return RERANKER_MODEL

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

    st.logger.warning("RERANKER_ALL_MODELS_FAILED")
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

_FUSION_QUERY_CACHE: Dict[str, List[str]] = {}
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
    """캐시에서 Fusion 쿼리 조회"""
    return _FUSION_QUERY_CACHE.get(query)


def _cache_fusion_queries(query: str, queries: List[str]) -> None:
    """Fusion 쿼리를 캐시에 저장"""
    global _FUSION_QUERY_CACHE

    if len(_FUSION_QUERY_CACHE) >= _FUSION_CACHE_MAX_SIZE:
        oldest = next(iter(_FUSION_QUERY_CACHE))
        del _FUSION_QUERY_CACHE[oldest]

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
        import openai
        client = openai.OpenAI(api_key=effective_key)

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


def _expand_query(query: str) -> str:
    """쿼리를 도메인 표현에 맞게 확장"""
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
def rag_search_glossary(query: str, top_k: int = 3) -> List[dict]:
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
