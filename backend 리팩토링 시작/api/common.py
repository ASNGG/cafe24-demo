"""
api/common.py - 라우트 공통 모듈
Pydantic 모델, 인증, 유틸 등 모든 라우트 파일에서 공유하는 요소
"""
import json
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import json_sanitize
import state as st

# ============================================================
# 인증
# ============================================================
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 실패",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": username, "role": st.USERS[username]["role"], "name": st.USERS[username]["name"]}


# ============================================================
# SSE 유틸
# ============================================================
def sse_pack(event: str, data: dict) -> str:
    """SSE 이벤트 포맷으로 직렬화 (LangChain 객체도 안전하게 처리)"""
    safe_data = json_sanitize(data)
    return f"event: {event}\ndata: {json.dumps(safe_data, ensure_ascii=False)}\n\n"


# ============================================================
# Pydantic 모델
# ============================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class ShopRequest(BaseModel):
    shop_id: str


class CategoryRequest(BaseModel):
    category_id: str


class UserRequest(BaseModel):
    user_id: str


class CsReplyRequest(BaseModel):
    text: str
    ticket_category: str = Field("일반", description="문의 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")


class CsQualityRequest(BaseModel):
    ticket_category: str = Field("일반", description="CS 티켓 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")
    sentiment_score: float = Field(0.0, description="감성 점수 (-1.0 ~ 1.0)")
    order_value: float = Field(50000, description="주문 금액")
    is_repeat_issue: bool = Field(False, description="반복 문의 여부")
    text_length: int = Field(100, description="문의 텍스트 길이")


class CsPipelineRequest(BaseModel):
    """CS 자동화 파이프라인 요청"""
    inquiry_text: str = Field(..., description="고객 문의 텍스트")
    seller_tier: str = Field("Standard", description="셀러 등급")
    order_id: Optional[str] = Field(None, description="주문 ID")
    order_value: float = Field(50000, description="주문 금액")
    is_repeat_issue: bool = Field(False, description="반복 문의 여부")
    confidence_threshold: float = Field(0.75, description="자동 처리 신뢰도 임계값 (0.0~1.0)")


class CsPipelineAnswerRequest(BaseModel):
    """CS 파이프라인 답변 생성 요청"""
    inquiry_text: str = Field(..., description="고객 문의 텍스트")
    inquiry_category: str = Field("기타", description="문의 카테고리")
    seller_tier: str = Field("Standard", description="셀러 등급")
    order_id: Optional[str] = Field(None, description="주문 ID")
    rag_mode: str = Field("rag", description="RAG 모드: rag | lightrag | k2rag")
    api_key: str = Field("", alias="apiKey")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class TextClassifyRequest(BaseModel):
    text: str


class RagRequest(BaseModel):
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(st.RAG_DEFAULT_TOPK, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class AgentRequest(BaseModel):
    user_input: str = Field(..., alias="user_input")
    api_key: str = Field("", alias="apiKey")
    model: str = Field("gpt-4o-mini", alias="model")
    max_tokens: int = Field(8000, alias="maxTokens")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="systemPrompt")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="topP")
    presence_penalty: Optional[float] = Field(None, alias="presencePenalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequencyPenalty")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: Optional[int] = Field(None, alias="timeoutMs")
    retries: Optional[int] = Field(None, alias="retries")
    stream: Optional[bool] = Field(None, alias="stream")
    rag_mode: str = Field("rag", alias="ragMode")
    agent_mode: str = Field("single", alias="agentMode")
    debug: bool = Field(True, alias="debug")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class UserCreateRequest(BaseModel):
    user_id: str
    name: str
    password: str
    role: str


class RagReloadRequest(BaseModel):
    api_key: str = Field("", alias="apiKey")
    force: bool = Field(True, alias="force")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class DeleteFileRequest(BaseModel):
    filename: str
    api_key: str = Field("", alias="apiKey")
    skip_reindex: bool = Field(False, alias="skipReindex")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class HybridSearchRequest(BaseModel):
    """Hybrid Search 요청 모델"""
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(5, alias="topK")
    use_reranking: bool = Field(True, alias="useReranking")
    use_kg: bool = Field(False, alias="useKg")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagSearchRequest(BaseModel):
    """LightRAG 검색 요청 모델"""
    query: str
    mode: str = Field("hybrid", description="검색 모드: naive, local, global, hybrid")
    top_k: int = Field(5, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagBuildRequest(BaseModel):
    """LightRAG 빌드 요청 모델"""
    force_rebuild: bool = Field(False, alias="forceRebuild")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagSearchRequest(BaseModel):
    """K2RAG 검색 요청 모델"""
    query: str
    top_k: int = Field(10, alias="topK")
    use_kg: bool = Field(True, alias="useKg", description="Knowledge Graph 사용")
    use_summary: bool = Field(True, alias="useSummary", description="요약 사용")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagConfigRequest(BaseModel):
    """K2RAG 설정 요청 모델"""
    hybrid_lambda: Optional[float] = Field(None, alias="hybridLambda", description="Hybrid 가중치 (0.0-1.0)")
    top_k: Optional[int] = Field(None, alias="topK")
    use_summarization: Optional[bool] = Field(None, alias="useSummarization")
    use_knowledge_graph: Optional[bool] = Field(None, alias="useKnowledgeGraph")
    llm_model: Optional[str] = Field(None, alias="llmModel")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class MarketingOptimizeRequest(BaseModel):
    """마케팅 예산 최적화 요청 모델"""
    seller_id: Optional[str] = Field(None, description="셀러 ID (프론트엔드 호환)")
    top_n: int = Field(10, description="상위 N개 결과")
    target_shops: Optional[List[str]] = Field(None, alias="targetShops", description="대상 쇼핑몰 ID 리스트")
    budget_constraints: Optional[dict] = Field(None, alias="budgetConstraints", description="예산 제약 (예: {'total': 1000000})")
    max_iterations: int = Field(10, alias="maxIterations", description="PSO 최대 반복 횟수")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class ModelSelectRequest(BaseModel):
    model_name: str
    version: str


class SystemPromptRequest(BaseModel):
    system_prompt: str = Field(..., alias="systemPrompt")
    class Config:
        populate_by_name = True


class LLMSettingsRequest(BaseModel):
    """LLM 설정 요청 모델"""
    selected_model: str = Field("gpt-4o-mini", alias="selectedModel")
    custom_model: str = Field("", alias="customModel")
    temperature: float = Field(0.7, alias="temperature")
    top_p: float = Field(1.0, alias="topP")
    presence_penalty: float = Field(0.0, alias="presencePenalty")
    frequency_penalty: float = Field(0.0, alias="frequencyPenalty")
    max_tokens: int = Field(8000, alias="maxTokens")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: int = Field(30000, alias="timeoutMs")
    retries: int = Field(2, alias="retries")
    stream: bool = Field(True, alias="stream")
    class Config:
        populate_by_name = True
