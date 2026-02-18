"""
main.py - 애플리케이션 진입점
FastAPI 앱 생성, 미들웨어, lifespan, 라우터 등록
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# OpenMP 충돌 방지 (EasyOCR + numpy/sklearn 등)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

# numpy 호환성 패치
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # noqa: N816

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import state as st
from api.routes import router as api_router
from process_miner.routes import pm_router
from data.loader import init_data_models
from rag.service import rag_build_or_load_index
from rag.light_rag import (
    LIGHTRAG_AVAILABLE,
    LIGHTRAG_STORE,
    run_in_lightrag_loop,
    get_lightrag_instance_async,
)


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    st.logger.info("APP_STARTUP")
    st.logger.info("BASE_DIR=%s", st.BASE_DIR)
    st.logger.info("LOG_FILE=%s", st.LOG_FILE)
    st.logger.info("PID=%s", os.getpid())
    try:
        st.load_system_prompt()
        st.load_llm_settings()

        init_data_models()

        # RAG 인덱스 비동기 백그라운드 로딩 (서버 시작 블로킹 방지)
        _skip_rag = os.environ.get("SKIP_RAG_STARTUP", "").strip() in ("1", "true", "yes")
        _k = st.OPENAI_API_KEY
        if _skip_rag:
            st.logger.info("RAG_SKIP_STARTUP env SKIP_RAG_STARTUP=1")
        elif _k:
            import asyncio
            asyncio.get_running_loop().run_in_executor(
                None, lambda: rag_build_or_load_index(api_key=_k, force_rebuild=False)
            )
            st.logger.info("RAG_INDEX 백그라운드 로딩 시작")
        else:
            st.logger.info("RAG_SKIP_STARTUP no_env_api_key docs_dir=%s", st.RAG_DOCS_DIR)

        _skip_lightrag = os.environ.get("SKIP_LIGHTRAG", "").strip() in ("1", "true", "yes")
        if _skip_lightrag:
            st.logger.info("LIGHTRAG_STARTUP_SKIP env SKIP_LIGHTRAG=1")
        elif LIGHTRAG_AVAILABLE and LIGHTRAG_STORE.get("ready"):
            st.logger.info("LIGHTRAG_STARTUP_INIT starting...")
            try:
                rag_instance = run_in_lightrag_loop(get_lightrag_instance_async(force_new=False))
                if rag_instance:
                    st.logger.info("LIGHTRAG_STARTUP_INIT instance loaded")
                    st.logger.info("LIGHTRAG_STARTUP_WARMUP skipped (rate limit prevention)")
                else:
                    st.logger.warning("LIGHTRAG_STARTUP_INIT instance is None")
            except Exception as e:
                st.logger.warning("LIGHTRAG_STARTUP_INIT failed: %s", e)
        else:
            st.logger.info("LIGHTRAG_STARTUP_SKIP available=%s ready=%s",
                          LIGHTRAG_AVAILABLE, LIGHTRAG_STORE.get("ready", False))
    except Exception as e:
        st.logger.exception("BOOTSTRAP_FAIL: %s", e)
        raise

    yield  # 앱 실행 중

    # ── shutdown ──
    st.logger.info("APP_SHUTDOWN")


# ============================================================
# 앱 생성
# ============================================================
app = FastAPI(title="CAFE24 AI 운영 플랫폼", version="2.0.0", lifespan=lifespan)

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 요청/응답 로깅 미들웨어
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        st.logger.info("REQ %s %s", request.method, request.url.path)
        resp = await call_next(request)
        st.logger.info("RES %s %s %s", request.method, request.url.path, resp.status_code)
        return resp
    except Exception:
        st.logger.exception("UNHANDLED %s %s", request.method, request.url.path)
        raise

# ============================================================
# 전역 예외 핸들러 (#3: 스택 트레이스 노출 제거)
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    st.logger.exception("EXCEPTION %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.",
        },
    )

# ============================================================
# 라우터 등록
# ============================================================
app.include_router(api_router)
app.include_router(pm_router)

# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_config=None,
        access_log=True,
    )
