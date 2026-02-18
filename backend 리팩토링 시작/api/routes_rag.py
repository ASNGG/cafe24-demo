"""
api/routes_rag.py - RAG/LightRAG/K2RAG/OCR API
"""
import asyncio
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Body

from core.utils import safe_str
from rag.service import rag_build_or_load_index, tool_rag_search, _rag_list_files, rag_search_hybrid, BM25_AVAILABLE
from rag.light_rag import (
    lightrag_search_sync, lightrag_search_dual_sync,
    build_lightrag_from_rag_docs, get_lightrag_status, clear_lightrag,
    LIGHTRAG_AVAILABLE,
)
from rag.k2rag import (
    k2rag_search_sync, get_status as k2rag_get_status,
    update_config as k2rag_update_config,
    load_from_existing_rag as k2rag_load_existing, summarize_text as k2rag_summarize,
)
import state as st
from api.common import (
    verify_credentials,
    RagRequest, RagReloadRequest, DeleteFileRequest,
    HybridSearchRequest, LightRagSearchRequest, LightRagBuildRequest,
    K2RagSearchRequest, K2RagConfigRequest,
)

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_READER = None
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None


router = APIRouter(prefix="/api", tags=["rag"])

OCR_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


# ============================================================
# RAG (검색)
# ============================================================
@router.post("/rag/search")
def search_rag(req: RagRequest, user: dict = Depends(verify_credentials)):
    return tool_rag_search(req.query, top_k=req.top_k, api_key=req.api_key)


@router.post("/rag/search/hybrid")
def search_rag_hybrid(req: HybridSearchRequest, user: dict = Depends(verify_credentials)):
    return rag_search_hybrid(query=req.query, top_k=req.top_k, api_key=req.api_key, use_reranking=req.use_reranking, use_kg=req.use_kg)


@router.get("/rag/status")
def rag_status(user: dict = Depends(verify_credentials)):
    with st.RAG_LOCK:
        return {
            "status": "success",
            "rag_ready": bool(st.RAG_STORE.get("ready")),
            "docs_dir": st.RAG_DOCS_DIR,
            "faiss_dir": st.RAG_FAISS_DIR,
            "embed_model": st.RAG_EMBED_MODEL,
            "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0),
            "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0),
            "hash": safe_str(st.RAG_STORE.get("hash", "")),
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or 0.0),
            "error": safe_str(st.RAG_STORE.get("error", "")),
            "bm25_available": BM25_AVAILABLE,
            "bm25_ready": bool(st.RAG_STORE.get("bm25_ready")),
            "reranker_available": False,
            "kg_ready": False,
            "kg_entities_count": 0,
            "kg_relations_count": 0,
        }


@router.post("/rag/reload")
def rag_reload(req: RagReloadRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    try:
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        if not k:
            return {"status": "error", "message": "OpenAI API Key가 설정되지 않았습니다."}
        rag_build_or_load_index(api_key=k, force_rebuild=bool(req.force))
        with st.RAG_LOCK:
            ok = bool(st.RAG_STORE.get("ready"))
            err = safe_str(st.RAG_STORE.get("error", ""))
            return {"status": "success" if ok else "error", "rag_ready": ok, "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0), "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0), "hash": safe_str(st.RAG_STORE.get("hash", "")), "error": err if err else ("인덱스 빌드 실패" if not ok else ""), "embed_model": st.RAG_EMBED_MODEL}
    except Exception as e:
        st.logger.exception("RAG 재빌드 실패")
        return {"status": "error", "message": f"RAG 재빌드 실패: {safe_str(e)}"}


@router.post("/rag/upload")
async def upload_rag_document(file: UploadFile = File(...), api_key: str = "", skip_reindex: bool = False, background_tasks: BackgroundTasks = None, user: dict = Depends(verify_credentials)):
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in st.RAG_ALLOWED_EXTS:
            return {"status": "error", "message": f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(st.RAG_ALLOWED_EXTS)}"}
        MAX_FILE_SIZE = 15 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "error", "message": "파일 크기는 15MB를 초과할 수 없습니다."}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(st.RAG_DOCS_DIR, safe_filename)
        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(contents)
        if not skip_reindex:
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k and background_tasks:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)
        return {"status": "success", "message": "파일이 업로드되었습니다." + ("" if skip_reindex else " 인덱스 재빌드 중..."), "filename": safe_filename, "original_filename": filename, "size": len(contents), "path": os.path.relpath(file_path, st.BASE_DIR), "reindex_skipped": skip_reindex}
    except Exception as e:
        st.logger.exception("파일 업로드 실패")
        return {"status": "error", "message": f"파일 업로드 실패: {safe_str(e)}"}


@router.get("/rag/files")
def list_rag_files(user: dict = Depends(verify_credentials)):
    try:
        files_info = []
        paths = _rag_list_files()
        for p in paths:
            try:
                stat = os.stat(p)
                rel_path = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
                files_info.append({"filename": os.path.basename(p), "path": rel_path, "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(), "ext": os.path.splitext(p)[1].lower()})
            except Exception:
                continue
        return {"status": "success", "files": files_info, "total": len(files_info)}
    except Exception as e:
        st.logger.exception("파일 목록 조회 실패")
        return {"status": "error", "message": f"파일 목록 조회 실패: {safe_str(e)}"}


@router.post("/rag/delete")
def delete_rag_file(req: DeleteFileRequest, background_tasks: BackgroundTasks, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    try:
        filename = os.path.basename(req.filename)
        file_path = os.path.join(st.RAG_DOCS_DIR, filename)
        if not file_path.startswith(os.path.abspath(st.RAG_DOCS_DIR)):
            return {"status": "error", "message": "잘못된 파일 경로입니다."}
        if not os.path.exists(file_path):
            return {"status": "error", "message": "파일을 찾을 수 없습니다."}
        os.remove(file_path)
        if not req.skip_reindex:
            k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
            if k:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)
            return {"status": "success", "message": "파일이 삭제되었습니다. 인덱스 재빌드 중...", "filename": filename}
        return {"status": "success", "message": "파일이 삭제되었습니다.", "filename": filename}
    except Exception as e:
        st.logger.exception("파일 삭제 실패")
        return {"status": "error", "message": f"파일 삭제 실패: {safe_str(e)}"}


# ============================================================
# LightRAG
# ============================================================
@router.get("/lightrag/status")
def lightrag_status(user: dict = Depends(verify_credentials)):
    status = get_lightrag_status()
    return {"status": "success", **status}


@router.post("/lightrag/build")
def lightrag_build(req: LightRagBuildRequest, background_tasks: BackgroundTasks, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    if not LIGHTRAG_AVAILABLE:
        return {"status": "error", "message": "LightRAG가 설치되지 않았습니다. pip install lightrag-hku"}
    background_tasks.add_task(build_lightrag_from_rag_docs, req.force_rebuild)
    return {"status": "success", "message": "LightRAG 빌드가 시작되었습니다."}


@router.post("/lightrag/search")
def lightrag_search(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    if not LIGHTRAG_AVAILABLE:
        return {"status": "error", "message": "LightRAG not available"}
    try:
        return lightrag_search_sync(query=req.query, mode=req.mode, top_k=req.top_k)
    except Exception as e:
        st.logger.exception("LightRAG 검색 실패")
        return {"status": "error", "message": f"LightRAG 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/search-dual")
def lightrag_search_dual(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    if not LIGHTRAG_AVAILABLE:
        return {"status": "error", "message": "LightRAG not available"}
    try:
        return lightrag_search_dual_sync(query=req.query, top_k=req.top_k)
    except Exception as e:
        st.logger.exception("LightRAG 듀얼 검색 실패")
        return {"status": "error", "message": f"LightRAG 듀얼 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/clear")
def lightrag_clear_endpoint(user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    return clear_lightrag()


# ============================================================
# K2RAG
# ============================================================
@router.get("/k2rag/status")
def k2rag_status(user: dict = Depends(verify_credentials)):
    return k2rag_get_status()


@router.post("/k2rag/search")
def k2rag_search_endpoint(req: K2RagSearchRequest, user: dict = Depends(verify_credentials)):
    try:
        return k2rag_search_sync(query=req.query, top_k=req.top_k, use_kg=req.use_kg, use_summary=req.use_summary)
    except Exception as e:
        st.logger.exception("K2RAG 검색 실패")
        return {"status": "error", "message": f"K2RAG 검색 실패: {safe_str(e)}"}


@router.post("/k2rag/config")
def k2rag_config_endpoint(req: K2RagConfigRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    config_updates = {}
    if req.hybrid_lambda is not None: config_updates["hybrid_lambda"] = req.hybrid_lambda
    if req.top_k is not None: config_updates["top_k"] = req.top_k
    if req.use_summarization is not None: config_updates["use_summarization"] = req.use_summarization
    if req.use_knowledge_graph is not None: config_updates["use_knowledge_graph"] = req.use_knowledge_graph
    if req.llm_model is not None: config_updates["llm_model"] = req.llm_model
    return k2rag_update_config(config_updates)


@router.post("/k2rag/load")
def k2rag_load_endpoint(user: dict = Depends(verify_credentials)):
    try:
        success = k2rag_load_existing()
        if success:
            return {"status": "success", "message": "기존 RAG 데이터가 K2RAG에 로드되었습니다.", "state": k2rag_get_status()}
        return {"status": "success", "message": "일부 데이터만 로드되었습니다.", "state": k2rag_get_status()}
    except Exception as e:
        return {"status": "error", "message": safe_str(e)}


@router.post("/k2rag/summarize")
def k2rag_summarize_endpoint(text: str = Body(..., embed=True), max_length: int = Body(300, embed=True), user: dict = Depends(verify_credentials)):
    try:
        summary = k2rag_summarize(text, max_length=max_length)
        return {"status": "success", "summary": summary, "original_length": len(text), "summary_length": len(summary), "reduction_rate": round((1 - len(summary) / len(text)) * 100, 1) if text else 0}
    except Exception as e:
        return {"status": "error", "message": safe_str(e)}


# ============================================================
# OCR
# ============================================================
def _ocr_init_reader():
    """EasyOCR Reader 초기화 (CPU 바운드, 스레드에서 실행)"""
    global OCR_READER
    if OCR_READER is None:
        st.logger.info("OCR_INIT: EasyOCR Reader 초기화 중...")
        OCR_READER = easyocr.Reader(['ko', 'en'], gpu=False)
        st.logger.info("OCR_INIT: EasyOCR Reader 초기화 완료")
    return OCR_READER


def _ocr_readtext(reader, contents: bytes) -> list:
    """OCR 텍스트 추출 (CPU 바운드, 스레드에서 실행)"""
    return reader.readtext(contents)


@router.post("/ocr/extract")
async def ocr_extract(file: UploadFile = File(...), api_key: str = "", save_to_rag: bool = True, user: dict = Depends(verify_credentials)):
    if not OCR_AVAILABLE:
        return {"status": "error", "message": "OCR 라이브러리(easyocr)가 설치되지 않았습니다."}
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in OCR_ALLOWED_EXTS:
            return {"status": "error", "message": f"지원하지 않는 이미지 형식입니다. 허용된 형식: {', '.join(OCR_ALLOWED_EXTS)}"}
        MAX_FILE_SIZE = 20 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "error", "message": "파일 크기는 20MB를 초과할 수 없습니다."}

        # OCR 초기화 + 텍스트 추출을 스레드에서 비동기 실행 (이벤트 루프 블로킹 방지)
        reader = await asyncio.to_thread(_ocr_init_reader)
        result_list = await asyncio.to_thread(_ocr_readtext, reader, contents)

        extracted_text = "\n".join([text for _, text, _ in result_list]).strip()
        if not extracted_text:
            return {"status": "error", "message": "이미지에서 텍스트를 추출할 수 없습니다."}
        result = {"status": "success", "original_filename": filename, "extracted_text": extracted_text, "text_length": len(extracted_text)}
        if save_to_rag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"{timestamp}_ocr_{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(st.RAG_DOCS_DIR, txt_filename)
            os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR 추출 문서]\n원본 파일: {filename}\n추출 일시: {datetime.now().isoformat()}\n{'='*50}\n\n{extracted_text}")
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k:
                # RAG 재빌드도 스레드에서 비동기 실행
                await asyncio.to_thread(rag_build_or_load_index, api_key=k, force_rebuild=True)
            result["saved_to_rag"] = True
            result["rag_filename"] = txt_filename
            result["message"] = "텍스트가 추출되어 RAG에 저장되었습니다."
        else:
            result["saved_to_rag"] = False
            result["message"] = "텍스트가 추출되었습니다."
        return result
    except Exception as e:
        st.logger.exception("OCR 추출 실패")
        return {"status": "error", "message": f"OCR 추출 실패: {safe_str(e)}"}


@router.get("/ocr/status")
def ocr_status(user: dict = Depends(verify_credentials)):
    easyocr_version = None
    reader_initialized = False
    if OCR_AVAILABLE:
        try:
            easyocr_version = easyocr.__version__
            reader_initialized = OCR_READER is not None
        except Exception:
            pass
    return {"status": "success", "ocr_available": OCR_AVAILABLE, "library": "EasyOCR", "version": easyocr_version, "reader_initialized": reader_initialized, "supported_formats": list(OCR_ALLOWED_EXTS), "supported_languages": ["ko", "en"]}
