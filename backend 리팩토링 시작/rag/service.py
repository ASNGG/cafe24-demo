"""
rag/service.py - RAG (Retrieval-Augmented Generation) 서비스 파사드

기능별 모듈을 통합하는 진입점:
- rag/chunking.py: 텍스트 청킹 (Parent-Child, 섹션 분할, 불릿 분리)
- rag/search.py: 검색 (BM25, Vector, Hybrid, Reranking, RAG-Fusion)
- rag/kg.py: Knowledge Graph (Entity-Relation 추출)
- rag/contextual.py: Contextual Retrieval (LLM 맥락 생성)
"""
import os
import re
import json
import time
import tempfile
import shutil
import logging
from typing import List, Any, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# pdfminer 경고 숨기기
for _pdflogger in ("pdfminer", "pdfminer.pdffont", "pdfminer.pdfinterp", "pdfminer.pdfpage"):
    logging.getLogger(_pdflogger).setLevel(logging.ERROR)

from core.utils import safe_str
import state as st

# ============================================================
# 모듈 임포트 (기능별 분리)
# ============================================================
from rag.chunking import (
    _sha1_text, _clean_text_for_rag, _is_garbage_text,
    _extract_key_nouns,
    BULLET_REGEX, SECTION_TITLE_PATTERN,
    _is_bullet_line, _extract_bullet_blocks, _create_bullet_chunks,
    _split_by_sections, _extract_text_from_pdf, _rag_read_file,
    _create_parent_child_chunks,
)
import rag.search as _search_mod
import rag.kg as _kg_mod
from rag.search import (
    BM25_AVAILABLE, RERANKER_AVAILABLE,
    _tokenize_korean, _build_bm25_index, _bm25_search,
    _get_reranker, _rerank_results,
    _reciprocal_rank_fusion, _multi_query_rrf,
    RAG_FUSION_ENABLED, RAG_FUSION_NUM_QUERIES,
    _is_simple_query, _generate_fusion_queries,
    _expand_query,
    QUERY_EXPANSION_MAP, QUERY_INJECTION_RULES,
    rag_search_glossary,
)
from rag.kg import (
    build_knowledge_graph,
    search_knowledge_graph,
)
from rag.contextual import (
    CONTEXTUAL_RETRIEVAL_ENABLED, CONTEXTUAL_MAX_WORKERS,
    _get_openai_client, _load_contextual_cache, _save_contextual_cache,
    _generate_contextual_prefix,
)

# ============================================================
# 뮤터블 글로벌 프록시 (k2rag.py 등 service.BM25_INDEX 접근 호환)
# ============================================================
def __getattr__(name):
    """service.BM25_INDEX, service.KNOWLEDGE_GRAPH 등 모듈 레벨 접근 프록시"""
    if name in ("BM25_INDEX", "BM25_CORPUS", "BM25_DOC_MAP"):
        return getattr(_search_mod, name)
    if name == "KNOWLEDGE_GRAPH":
        return getattr(_kg_mod, name)
    raise AttributeError(f"module 'rag.service' has no attribute {name!r}")


# ============================================================
# 선택적 import (없으면 RAG 비활성화)
# ============================================================
FAISS = None
OpenAIEmbeddings = None
Document = None
RecursiveCharacterTextSplitter = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    pass

# ============================================================
# Parent-Child Chunking 상태
# ============================================================
PARENT_CHUNKS_STORE: Dict[str, Any] = {}
CHILD_TO_PARENT_MAP: Dict[str, str] = {}


# ============================================================
# 파일 관리
# ============================================================
def _rag_list_files() -> List[str]:
    files: List[str] = []
    try:
        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)
        for root, _, names in os.walk(st.RAG_DOCS_DIR):
            for n in names:
                ext = os.path.splitext(n)[1].lower()
                if ext in st.RAG_ALLOWED_EXTS:
                    files.append(os.path.join(root, n))
    except Exception:
        return []
    return sorted(list(set(files)))


def _rag_files_fingerprint(paths: List[str]) -> str:
    parts: List[str] = []
    for p in paths:
        try:
            s = os.stat(p)
            parts.append(f"{os.path.relpath(p, st.RAG_DOCS_DIR)}|{s.st_size}|{int(s.st_mtime)}")
        except Exception:
            parts.append(f"{os.path.relpath(p, st.RAG_DOCS_DIR)}|ERR")
    return _sha1_text("\n".join(parts))


# ============================================================
# Embeddings
# ============================================================
def _make_embeddings(api_key: str):
    if OpenAIEmbeddings is None:
        return None
    k = (api_key or "").strip()
    try:
        return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL, openai_api_key=k)
    except TypeError:
        try:
            return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL, api_key=k)
        except TypeError:
            try:
                return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL)
            except Exception:
                return None
    except Exception:
        return None


# ============================================================
# State 파일 관리
# ============================================================
def _rag_load_state_file() -> dict:
    try:
        if not os.path.exists(st.RAG_STATE_FILE):
            return {}
        with open(st.RAG_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _rag_save_state_file(payload: dict) -> None:
    try:
        os.makedirs(st.RAG_FAISS_DIR, exist_ok=True)
        with open(st.RAG_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# Parent Content 조회
# ============================================================
def _get_parent_content(parent_id: str, fallback_content: str = "") -> str:
    """Parent ID로 Parent 청크의 content를 조회"""
    global PARENT_CHUNKS_STORE

    if not parent_id or not PARENT_CHUNKS_STORE:
        return fallback_content

    parent = PARENT_CHUNKS_STORE.get(parent_id)
    if parent is None:
        return fallback_content

    try:
        parent_content = safe_str(getattr(parent, "page_content", ""))
        return parent_content if parent_content else fallback_content
    except Exception:
        return fallback_content


# ============================================================
# FAISS 한글 경로 대응 (Windows C++ I/O 우회)
# ============================================================
def _safe_faiss_save(idx, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    try:
        idx.save_local(target_dir)
        return
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        idx.save_local(tmp)
        for fname in os.listdir(tmp):
            shutil.copy2(os.path.join(tmp, fname), os.path.join(target_dir, fname))


def _safe_faiss_load(target_dir: str, emb):
    try:
        try:
            return FAISS.load_local(target_dir, emb, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(target_dir, emb)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        for fname in os.listdir(target_dir):
            src = os.path.join(target_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, fname))
        try:
            return FAISS.load_local(tmp, emb, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(tmp, emb)


# ============================================================
# 인덱스 빌드/로드
# ============================================================
def rag_build_or_load_index(api_key: str, force_rebuild: bool = False) -> None:
    global PARENT_CHUNKS_STORE, CHILD_TO_PARENT_MAP

    with st.RAG_LOCK:
        st.RAG_STORE["error"] = ""

    if (FAISS is None) or (OpenAIEmbeddings is None) or (Document is None):
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None,
                "error": "RAG 비활성화: langchain_community/FAISS 또는 OpenAIEmbeddings 또는 Document import 실패",
            })
        return

    k = (api_key or "").strip()
    if not k:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None,
                "error": "RAG 비활성화: OpenAI API Key가 없습니다.(환경변수 OPENAI_API_KEY 또는 요청 apiKey 필요)",
            })
        return

    paths = _rag_list_files()
    fp = _rag_files_fingerprint(paths)

    # 기존 인덱스 로드 (파일 해시 동일)
    if (not force_rebuild) and os.path.exists(st.RAG_FAISS_DIR):
        saved = _rag_load_state_file()
        if isinstance(saved, dict) and saved.get("hash") == fp:
            try:
                emb = _make_embeddings(k)
                if emb is None:
                    raise RuntimeError("embeddings_init_failed")
                idx = _safe_faiss_load(st.RAG_FAISS_DIR, emb)

                bm25_built = False

                load_docs: List[Any] = []
                for p in paths:
                    txt = _rag_read_file(p)
                    if not txt:
                        continue
                    rel = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
                    try:
                        load_docs.append(Document(page_content=txt, metadata={"source": rel}))
                    except Exception:
                        continue

                if load_docs:
                    child_chunks, parent_store, child_to_parent = _create_parent_child_chunks(
                        load_docs,
                        parent_size=3000,
                        parent_overlap=500,
                        child_size=500,
                        child_overlap=100,
                        enable_contextual=False
                    )
                    PARENT_CHUNKS_STORE = parent_store
                    CHILD_TO_PARENT_MAP = child_to_parent

                    bm25_built = _build_bm25_index(child_chunks)

                    kg_result = build_knowledge_graph(child_chunks)
                    kg_built = bool(kg_result and kg_result.get("entities"))
                    st.logger.info("BM25_KG_REBUILT_ON_LOAD docs=%d bm25=%s kg=%s", len(load_docs), bm25_built, kg_built)

                with st.RAG_LOCK:
                    st.RAG_STORE.update({
                        "ready": True, "hash": fp,
                        "files_count": int(saved.get("files_count") or saved.get("docs_count") or 0),
                        "chunks_count": int(saved.get("chunks_count") or saved.get("docs_count") or 0),
                        "last_build_ts": float(saved.get("last_build_ts") or time.time()),
                        "error": "", "index": idx,
                        "bm25_ready": bm25_built,
                        "kg_ready": kg_built,
                    })
                st.logger.info("RAG_READY(load) files=%s chunks=%s bm25=%s hash=%s",
                              st.RAG_STORE.get("files_count"), st.RAG_STORE.get("chunks_count"), bm25_built, safe_str(fp)[:10])
                return
            except Exception as e:
                st.logger.warning("RAG_LOAD_FAIL err=%s", safe_str(e))

    # 새로 빌드
    docs: List[Any] = []
    for p in paths:
        txt = _rag_read_file(p)
        if not txt:
            continue
        rel = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
        try:
            docs.append(Document(page_content=txt, metadata={"source": rel}))
        except Exception:
            continue

    if not docs:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None, "hash": fp,
                "files_count": 0, "chunks_count": 0,
                "last_build_ts": time.time(),
                "error": "rag_docs 폴더에 인덱싱할 문서가 없습니다.",
            })
        _rag_save_state_file({
            "hash": fp, "files_count": 0, "chunks_count": 0,
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or time.time()),
            "error": safe_str(st.RAG_STORE.get("error", "")),
            "embed_model": st.RAG_EMBED_MODEL,
        })
        st.logger.info("RAG_EMPTY docs_dir=%s", st.RAG_DOCS_DIR)
        return

    files_count = len(docs)
    child_chunks, parent_store, child_to_parent = _create_parent_child_chunks(
        docs,
        parent_size=3000,
        parent_overlap=500,
        child_size=500,
        child_overlap=100,
        enable_contextual=True,
        contextual_prefix_func=_generate_contextual_prefix,
        contextual_cache_load_func=_load_contextual_cache,
        contextual_cache_save_func=_save_contextual_cache,
        contextual_client_func=_get_openai_client,
        contextual_max_workers=CONTEXTUAL_MAX_WORKERS,
    )

    PARENT_CHUNKS_STORE = parent_store
    CHILD_TO_PARENT_MAP = child_to_parent

    chunks = child_chunks
    chunks_count = len(chunks)
    parent_count = len(parent_store)
    st.logger.info("PARENT_CHILD_READY children=%d parents=%d files=%d",
                   chunks_count, parent_count, files_count)

    try:
        emb = _make_embeddings(k)
        if emb is None:
            raise RuntimeError("embeddings_init_failed")

        idx = FAISS.from_documents(chunks, emb)
        _safe_faiss_save(idx, st.RAG_FAISS_DIR)

        bm25_built = _build_bm25_index(chunks)

        kg_result = build_knowledge_graph(chunks)
        kg_built = bool(kg_result and kg_result.get("entities"))

        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": True, "hash": fp,
                "files_count": files_count,
                "chunks_count": chunks_count,
                "last_build_ts": time.time(),
                "error": "", "index": idx,
                "bm25_ready": bm25_built,
                "kg_ready": kg_built,
            })

        _rag_save_state_file({
            "hash": fp, "files_count": files_count, "chunks_count": chunks_count,
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or time.time()),
            "error": "", "embed_model": st.RAG_EMBED_MODEL,
            "bm25_ready": bm25_built, "kg_ready": kg_built,
        })
        st.logger.info("RAG_READY(build) files=%s chunks=%s bm25=%s hash=%s",
                       files_count, chunks_count, bm25_built, safe_str(fp)[:10])
    except Exception as e:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None, "hash": fp,
                "files_count": 0, "chunks_count": 0,
                "error": f"RAG 인덱싱 실패: {safe_str(e)}",
            })
        st.logger.exception("RAG_BUILD_FAIL err=%s", safe_str(e))


# ============================================================
# FAISS 벡터 검색 (기본)
# ============================================================
def rag_search_local(query: str, top_k: int = st.RAG_DEFAULT_TOPK, api_key: str = "") -> List[dict]:
    original_q = safe_str(query).strip()
    if not original_q:
        return []

    q = _expand_query(original_q)

    k = max(1, min(int(top_k), st.RAG_MAX_TOPK))

    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")
        err = safe_str(st.RAG_STORE.get("error", ""))

    if (not ready) or (idx is None):
        rag_build_or_load_index(api_key=api_key, force_rebuild=False)
        with st.RAG_LOCK:
            ready = bool(st.RAG_STORE.get("ready"))
            idx = st.RAG_STORE.get("index")
            err = safe_str(st.RAG_STORE.get("error", ""))
        if (not ready) or (idx is None):
            return [{"title": "RAG_ERROR", "source": "", "score": 0.0, "content": err}] if err else []

    try:
        pairs = idx.similarity_search_with_score(q, k=k * 3)

        max_dist = float(getattr(st, "RAG_MAX_DISTANCE", 1.6))

        out: List[dict] = []
        seen_parents = set()

        for doc, score in pairs:
            if len(out) >= k:
                break
            if score is None:
                continue
            try:
                dist = float(score)
            except Exception:
                continue
            if dist > max_dist:
                continue

            try:
                metadata = getattr(doc, "metadata", {})
                src = safe_str(metadata.get("source", ""))
                parent_id = metadata.get("parent_id", "")
            except Exception:
                src = ""
                parent_id = ""

            try:
                child_txt = safe_str(getattr(doc, "page_content", ""))
            except Exception:
                child_txt = ""

            if parent_id and parent_id not in seen_parents:
                txt = _get_parent_content(parent_id, child_txt)
                seen_parents.add(parent_id)
            elif not parent_id:
                txt = child_txt
            else:
                continue

            if not txt:
                continue

            out.append({
                "title": src or "doc",
                "source": src,
                "score": round(dist, 6),
                "content": txt[:st.RAG_SNIPPET_CHARS],
            })
        return out
    except Exception as e:
        return [{"title": "RAG_ERROR", "source": "", "score": 0.0, "content": f"RAG 검색 실패: {safe_str(e)}"}]


# ============================================================
# Single Query Search (RAG-Fusion 내부 헬퍼)
# ============================================================
def _search_single_query(
    q: str,
    retrieval_k: int,
    effective_key: str
) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
    """단일 쿼리로 Vector + BM25 검색 실행"""

    vector_results = []
    bm25_results = []

    # Vector Search (FAISS)
    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")

    if ready and idx is not None:
        try:
            pairs = idx.similarity_search_with_score(q, k=retrieval_k)
            seen_parents = set()

            for doc, dist in pairs:
                try:
                    child_content = safe_str(getattr(doc, "page_content", ""))
                    metadata = getattr(doc, "metadata", {})
                    source = safe_str(metadata.get("source", ""))
                    parent_id = metadata.get("parent_id", "")

                    if parent_id and parent_id not in seen_parents:
                        parent_content = _get_parent_content(parent_id, child_content)
                        seen_parents.add(parent_id)
                        content = parent_content
                    elif not parent_id:
                        content = child_content
                    else:
                        continue

                    if content:
                        vector_results.append((
                            {
                                "content": content,
                                "source": source,
                                "title": source or "doc",
                                "parent_id": parent_id,
                                "matched_child": child_content[:100],
                            },
                            float(dist)
                        ))
                except Exception:
                    continue
        except Exception as e:
            st.logger.warning("SEARCH_VECTOR_FAIL q=%s err=%s", q[:20], safe_str(e)[:50])

    # BM25 Search
    with st.RAG_LOCK:
        bm25_ready = bool(st.RAG_STORE.get("bm25_ready"))

    if bm25_ready and _search_mod.BM25_INDEX is not None:
        bm25_results = _bm25_search(q, top_k=retrieval_k, parent_content_func=_get_parent_content)

    return vector_results, bm25_results


# ============================================================
# Hybrid Search (BM25 + Vector + Reranking + RAG-Fusion)
# ============================================================
def rag_search_hybrid(
    query: str,
    top_k: int = st.RAG_DEFAULT_TOPK,
    api_key: str = "",
    use_reranking: bool = True,
    use_kg: bool = True,
    use_fusion: bool = True
) -> dict:
    """
    고급 RAG 검색:
    - KG 검색, RAG-Fusion, Hybrid Search, Reranking
    """

    original_query = safe_str(query).strip()
    if not original_query:
        return {"status": "error", "message": "Empty query", "results": []}

    k = max(1, min(int(top_k), st.RAG_MAX_TOPK))
    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY

    # RAG 인덱스 준비 확인
    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")
        kg_ready = bool(st.RAG_STORE.get("kg_ready"))

    if (not ready) or (idx is None):
        rag_build_or_load_index(api_key=effective_key, force_rebuild=False)

    # KG 검색
    kg_entities = []
    kg_location_answer = None
    kg_matched_entities = set()

    if use_kg and kg_ready and _kg_mod.KNOWLEDGE_GRAPH:
        kg_entities = search_knowledge_graph(original_query, top_k=5)

        for kg_ent in kg_entities:
            if kg_ent.get("location_answer"):
                kg_location_answer = kg_ent.get("location_answer")
                st.logger.info("KG_LOCATION_ANSWER query=%s answer=%s",
                              original_query[:30], kg_location_answer)

            kg_matched_entities.add(kg_ent.get("entity", "").lower())
            for rel in kg_ent.get("relations", []):
                kg_matched_entities.add(rel.get("source", "").lower())
                kg_matched_entities.add(rel.get("target", "").lower())

        st.logger.info("KG_SEARCH query=%s entities=%d location=%s",
                      original_query[:30], len(kg_entities),
                      kg_location_answer or "none")

    retrieval_k = k

    # RAG-Fusion
    if use_fusion and RAG_FUSION_ENABLED:
        fusion_queries = _generate_fusion_queries(
            original_query,
            api_key=effective_key,
            num_queries=RAG_FUSION_NUM_QUERIES
        )

        def _search_and_fuse(fq: str) -> List[Tuple[Dict, float]]:
            expanded_q = _expand_query(fq)
            vec_res, bm25_res = _search_single_query(expanded_q, retrieval_k, effective_key)

            if bm25_res and vec_res:
                single_fused = _reciprocal_rank_fusion(bm25_res, vec_res)
                return [(r, r.get("fusion_score", 0.5)) for r in single_fused]
            elif vec_res:
                return [(doc, 1.0 / (1.0 + dist)) for doc, dist in vec_res]
            elif bm25_res:
                return [(doc, score / 100.0) for doc, score in bm25_res]
            return []

        all_combined_results: List[List[Tuple[Dict, float]]] = []
        with ThreadPoolExecutor(max_workers=len(fusion_queries)) as executor:
            futures = [executor.submit(_search_and_fuse, fq) for fq in fusion_queries]
            for future in as_completed(futures):
                try:
                    combined = future.result()
                    if combined:
                        all_combined_results.append(combined)
                except Exception as e:
                    st.logger.warning("RAG_FUSION_PARALLEL_FAIL: %s", safe_str(e)[:50])

        if all_combined_results:
            multi_fused = _multi_query_rrf(all_combined_results)
            fused_results = [
                {**doc, "fusion_score": round(score, 4), "multi_query_hits": doc.get("query_hits", 1)}
                for doc, score in multi_fused
            ]
            search_method = f"rag_fusion_{len(fusion_queries)}q"
            st.logger.info("RAG_FUSION: %d queries -> %d results", len(fusion_queries), len(fused_results))
        else:
            return {"status": "error", "message": "No search results from RAG-Fusion", "results": []}

    else:
        q = _expand_query(original_query)
        vector_results, bm25_results = _search_single_query(q, retrieval_k, effective_key)

        if bm25_results and vector_results:
            fused_results = _reciprocal_rank_fusion(bm25_results, vector_results)
            search_method = "hybrid"
        elif vector_results:
            fused_results = [
                {**doc, "vector_score": round(1.0 / (1.0 + dist), 4), "fusion_score": round(1.0 / (1.0 + dist), 4)}
                for doc, dist in vector_results
            ]
            search_method = "vector"
        elif bm25_results:
            fused_results = [
                {**doc, "bm25_score": round(score, 4), "fusion_score": round(score / 100.0, 4)}
                for doc, score in bm25_results
            ]
            search_method = "bm25"
        else:
            return {"status": "error", "message": "No search results", "results": []}

    if not fused_results:
        return {"status": "error", "message": "No search results", "results": []}

    # 쿼리-섹션 제목 매칭 보너스
    query_keywords = set(re.findall(r'[가-힣]{2,}', original_query))
    if query_keywords:
        for r in fused_results:
            content = r.get("content", "") + r.get("matched_child", "")
            section_matches = re.findall(r'\[섹션[^:]*:\s*([^\]]+)\]', content)
            title_matches = re.findall(r'\[제목:\s*([^\]]+)\]', content)
            topic_matches = re.findall(r'\[주제:\s*([^\]]+)\]', content)
            all_titles = section_matches + title_matches + topic_matches

            section_match_bonus = 0.0
            for title in all_titles:
                for kw in query_keywords:
                    if kw in title:
                        section_match_bonus = max(section_match_bonus, 0.5)
                        pure_title = re.sub(r'^\d+\.[\d.]*\s*', '', title).strip()
                        if kw == pure_title:
                            section_match_bonus = max(section_match_bonus, 1.0)

            if section_match_bonus > 0:
                original_score = r.get("fusion_score", 0.0)
                r["fusion_score"] = round(original_score + section_match_bonus, 4)
                r["section_match_bonus"] = round(section_match_bonus, 4)

        fused_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

    # KG 매칭 보너스
    if kg_matched_entities and fused_results:
        for r in fused_results:
            content_lower = r.get("content", "").lower()
            kg_match_bonus = 0.0

            for kg_entity in kg_matched_entities:
                if kg_entity and len(kg_entity) >= 2 and kg_entity in content_lower:
                    kg_match_bonus = max(kg_match_bonus, 0.3)

                    if kg_location_answer and kg_location_answer.lower() in content_lower:
                        kg_match_bonus = max(kg_match_bonus, 0.8)

            if kg_match_bonus > 0:
                original_score = r.get("fusion_score", 0.0)
                r["fusion_score"] = round(original_score + kg_match_bonus, 4)
                r["kg_match_bonus"] = round(kg_match_bonus, 4)

        fused_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

    # Reranking
    reranked = False
    if use_reranking and RERANKER_AVAILABLE and len(fused_results) > 1:
        fused_results = _rerank_results(original_query, fused_results, top_k=k)
        reranked = True

    final_results = []
    for r in fused_results[:k]:
        r["content"] = r.get("content", "")[:st.RAG_SNIPPET_CHARS]
        final_results.append(r)

    return {
        "status": "success",
        "query": original_query,
        "search_method": search_method,
        "fusion_enabled": use_fusion and RAG_FUSION_ENABLED,
        "top_k": k,
        "reranked": reranked,
        "bm25_available": BM25_AVAILABLE,
        "reranker_available": RERANKER_AVAILABLE,
        "kg_available": bool(_kg_mod.KNOWLEDGE_GRAPH),
        "kg_used": use_kg and kg_ready,
        "kg_location_answer": kg_location_answer,
        "results": final_results,
        "kg_entities": kg_entities,
    }


# ============================================================
# 통합 RAG 검색 (Hybrid Search + 글로서리)
# ============================================================
def tool_rag_search(query: str, top_k: int = st.RAG_DEFAULT_TOPK, api_key: str = "") -> dict:
    """통합 RAG 검색 - Hybrid Search (BM25 + Vector) 사용"""
    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY

    try:
        k = int(max(1, min(int(top_k), st.RAG_MAX_TOPK)))
    except (ValueError, TypeError):
        k = st.RAG_DEFAULT_TOPK

    gloss = rag_search_glossary(query, top_k=k)

    hybrid_result = rag_search_hybrid(query, top_k=k, api_key=effective_key, use_reranking=False)
    hybrid_results = hybrid_result.get("results", []) if hybrid_result.get("status") == "success" else []

    merged: List[dict] = []
    seen = set()

    if isinstance(gloss, list):
        for r in gloss:
            key = safe_str(r.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append({
                    "title": r.get("title"),
                    "source": "glossary",
                    "score": 0.0,
                    "priority": 1000.0 + float(r.get("score") or 0.0),
                    "content": safe_str(r.get("content") or "")[:st.RAG_SNIPPET_CHARS],
                })

    remain = max(0, k - len(merged))
    if remain > 0 and isinstance(hybrid_results, list):
        for r in hybrid_results:
            key = safe_str(r.get("source") or r.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                fusion_score = float(r.get("rerank_score") or r.get("fusion_score") or 0.0)
                merged.append({
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "score": round(fusion_score, 6),
                    "priority": 100.0 * fusion_score,
                    "content": safe_str(r.get("content") or "")[:st.RAG_SNIPPET_CHARS],
                    "matched_child": safe_str(r.get("matched_child") or "")[:200],
                })
                remain -= 1
                if remain <= 0:
                    break

    merged.sort(key=lambda x: float(x.get("priority") or 0.0), reverse=True)
    for m in merged:
        m.pop("priority", None)

    with st.RAG_LOCK:
        rag_ready = bool(st.RAG_STORE.get("ready"))
        rag_err = safe_str(st.RAG_STORE.get("error", ""))
        rag_docs = int(st.RAG_STORE.get("docs_count") or 0)

    guardrail = ""

    return {
        "status": "success" if merged else "error",
        "query": safe_str(query),
        "top_k": k,
        "rag_ready": rag_ready,
        "rag_docs_count": rag_docs,
        "rag_error": rag_err,
        "search_method": hybrid_result.get("search_method", "unknown"),
        "reranked": hybrid_result.get("reranked", False),
        "results": merged,
        "guardrail": guardrail,
    }
