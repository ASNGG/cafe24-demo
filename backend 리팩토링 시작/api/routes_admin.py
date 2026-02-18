"""
api/routes_admin.py - 관리/설정/사용자/헬스체크/내보내기
"""
import os
from datetime import datetime
from io import StringIO, BytesIO

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str
from core.memory import clear_memory
from agent.tools import ALL_TOOLS
import state as st
from api.common import (
    verify_credentials, security,
    UserCreateRequest, SystemPromptRequest, LLMSettingsRequest,
)
from fastapi.security import HTTPBasicCredentials

router = APIRouter(prefix="/api", tags=["admin"])


# ============================================================
# 헬스 체크
# ============================================================
@router.get("/health")
def health():
    st.logger.info("HEALTH_CHECK")
    return {
        "status": "success",
        "message": "ok",
        "pid": os.getpid(),
        "platform": "CAFE24 AI Platform",
        "models_ready": bool(
            st.CS_QUALITY_MODEL is not None and
            st.SELLER_SEGMENT_MODEL is not None and
            st.FRAUD_DETECTION_MODEL is not None
        ),
        "data_ready": {
            "shops": st.SHOPS_DF is not None and len(st.SHOPS_DF) > 0,
            "categories": st.CATEGORIES_DF is not None and len(st.CATEGORIES_DF) > 0,
            "sellers": st.SELLERS_DF is not None and len(st.SELLERS_DF) > 0,
            "operation_logs": st.OPERATION_LOGS_DF is not None and len(st.OPERATION_LOGS_DF) > 0,
        },
    }


# ============================================================
# 로그인
# ============================================================
@router.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(status_code=401, detail="인증 실패")
    user = st.USERS[username]
    clear_memory(username)
    return {"status": "success", "username": username, "user_name": user["name"], "user_role": user["role"]}


# ============================================================
# 사용자 관리
# ============================================================
@router.get("/users")
def get_users(user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    return {"status": "success", "data": [{"아이디": k, "이름": v["name"], "권한": v["role"]} for k, v in st.USERS.items()]}


@router.post("/users")
def create_user(req: UserCreateRequest, user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    if req.user_id in st.USERS:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")
    st.USERS[req.user_id] = {"password": req.password, "role": req.role, "name": req.name}
    return {"status": "success", "message": f"{req.name} 추가됨"}


# ============================================================
# 설정
# ============================================================
@router.get("/settings/default")
def get_default_settings(user: dict = Depends(verify_credentials)):
    return {
        "status": "success",
        "data": {
            "selectedModel": "gpt-4o-mini",
            "maxTokens": 8000,
            "temperature": 0.1,
            "topP": 1.0,
            "presencePenalty": 0.0,
            "frequencyPenalty": 0.0,
            "seed": "",
            "timeoutMs": 30000,
            "retries": 2,
            "stream": True,
            "systemPrompt": st.get_active_system_prompt(),
        },
    }


# ============================================================
# 시스템 프롬프트 관리
# ============================================================
@router.get("/settings/prompt")
def get_system_prompt(user: dict = Depends(verify_credentials)):
    return {
        "status": "success",
        "data": {
            "systemPrompt": st.get_active_system_prompt(),
            "isCustom": st.CUSTOM_SYSTEM_PROMPT is not None and st.CUSTOM_SYSTEM_PROMPT.strip() != "",
            "defaultPrompt": DEFAULT_SYSTEM_PROMPT,
        },
    }


@router.post("/settings/prompt")
def save_system_prompt(req: SystemPromptRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 수정할 수 있습니다.")
    prompt = req.system_prompt.strip() if req.system_prompt else ""
    if not prompt:
        raise HTTPException(status_code=400, detail="시스템 프롬프트가 비어있습니다.")
    success = st.save_system_prompt(prompt)
    if success:
        return {
            "status": "success",
            "message": "시스템 프롬프트가 저장되었습니다.",
            "data": {
                "systemPrompt": st.get_active_system_prompt(),
                "isCustom": True,
            },
        }
    raise HTTPException(status_code=500, detail="시스템 프롬프트 저장에 실패했습니다.")


@router.post("/settings/prompt/reset")
def reset_system_prompt(user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 초기화할 수 있습니다.")
    success = st.reset_system_prompt()
    if success:
        return {
            "status": "success",
            "message": "시스템 프롬프트가 기본값으로 초기화되었습니다.",
            "data": {
                "systemPrompt": DEFAULT_SYSTEM_PROMPT,
                "isCustom": False,
            },
        }
    raise HTTPException(status_code=500, detail="시스템 프롬프트 초기화에 실패했습니다.")


# ============================================================
# LLM 설정 관리
# ============================================================
@router.get("/settings/llm")
def get_llm_settings(user: dict = Depends(verify_credentials)):
    settings = st.get_active_llm_settings()
    is_custom = st.CUSTOM_LLM_SETTINGS is not None
    return {
        "status": "success",
        "data": {
            **settings,
            "isCustom": is_custom,
        },
    }


@router.post("/settings/llm")
def save_llm_settings(req: LLMSettingsRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 수정할 수 있습니다.")
    settings_dict = {
        "selectedModel": req.selected_model,
        "customModel": req.custom_model,
        "temperature": req.temperature,
        "topP": req.top_p,
        "presencePenalty": req.presence_penalty,
        "frequencyPenalty": req.frequency_penalty,
        "maxTokens": req.max_tokens,
        "seed": req.seed,
        "timeoutMs": req.timeout_ms,
        "retries": req.retries,
        "stream": req.stream,
    }
    success = st.save_llm_settings(settings_dict)
    if success:
        return {
            "status": "success",
            "message": "LLM 설정이 저장되었습니다.",
            "data": {**settings_dict, "isCustom": True},
        }
    raise HTTPException(status_code=500, detail="LLM 설정 저장에 실패했습니다.")


@router.post("/settings/llm/reset")
def reset_llm_settings(user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 초기화할 수 있습니다.")
    success = st.reset_llm_settings()
    if success:
        return {
            "status": "success",
            "message": "LLM 설정이 기본값으로 초기화되었습니다.",
            "data": {**st.DEFAULT_LLM_SETTINGS, "isCustom": False},
        }
    raise HTTPException(status_code=500, detail="LLM 설정 초기화에 실패했습니다.")


# ============================================================
# 내보내기
# ============================================================
@router.get("/export/csv")
def export_csv(user: dict = Depends(verify_credentials)):
    output = StringIO()
    export_df = st.OPERATION_LOGS_DF if st.OPERATION_LOGS_DF is not None else pd.DataFrame()
    export_df.to_csv(output, index=False, encoding="utf-8-sig")
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cafe24_data_{datetime.now().strftime('%Y%m%d')}.csv"},
    )


@router.get("/export/excel")
def export_excel(user: dict = Depends(verify_credentials)):
    output = BytesIO()
    export_df = st.OPERATION_LOGS_DF if st.OPERATION_LOGS_DF is not None else pd.DataFrame()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="OperationLogs")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=cafe24_data_{datetime.now().strftime('%Y%m%d')}.xlsx"},
    )


# ============================================================
# 도구 목록 (에이전트용)
# ============================================================
@router.get("/tools")
def get_available_tools(user: dict = Depends(verify_credentials)):
    tools = []
    for t in ALL_TOOLS:
        tools.append({
            "name": t.name,
            "description": t.description or "",
        })
    return {"status": "success", "tools": tools}
