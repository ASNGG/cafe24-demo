"""
agent/semantic_router.py - Semantic Router (임베딩 기반 의도 분류)
============================================================
키워드 기반 분류의 한계를 극복하기 위한 시맨틱 라우터

특징:
- LLM 호출 없이 빠른 분류 (임베딩 유사도만 계산)
- 유사한 의미의 다양한 표현 인식 ("매출" = "수익" = "돈 벌이")
- 카테고리별 예시 쿼리로 학습

References:
- https://github.com/aurelio-labs/semantic-router
- https://www.patronus.ai/ai-agent-development/ai-agent-routing
"""
import time
import numpy as np
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

from openai import OpenAI

from agent.router import IntentCategory
from core.utils import safe_str
import state as st


# ============================================================
# 카테고리별 예시 쿼리 (Route Utterances)
# ============================================================
ROUTE_EXAMPLES: Dict[IntentCategory, List[str]] = {
    IntentCategory.ANALYSIS: [
        # 매출/GMV
        "GMV 성장률 알려줘",
        "이번 달 매출 어때?",
        "월간 GMV 분석해줘",
        "매출 추이 분석",
        "revenue 분석",
        "AOV 얼마야?",
        "전환율 분석해줘",
        "상위 쇼핑몰 매출",
        "카테고리별 매출",
        # 이탈/Churn
        "셀러 이탈 예측 분석",
        "이탈률 알려줘",
        "고위험 셀러 현황",
        "이탈 요인이 뭐야?",
        "churn prediction",
        "이탈할 것 같은 셀러",
        # 코호트/리텐션
        "코호트 분석 보여줘",
        "리텐션 현황",
        "Month1 리텐션 얼마야?",
        "잔존율 분석",
        "cohort retention",
        # KPI/트렌드
        "활성 셀러 수 얼마야?",
        "오늘 주문 수",
        "트렌드 분석",
        "KPI 변화율",
        "신규 셀러 가입 수",
        "정산 현황",
    ],

    IntentCategory.WORLDVIEW: [
        # 플랫폼/정책
        "카페24 정산 정책 알려줘",
        "정산 주기가 어떻게 돼?",
        "카페24 플랫폼 소개해줘",
        "카페24 역사",
        "platform 정책",
        # 용어/개념
        "GMV가 뭐야?",
        "CVR이란?",
        "AOV가 뭔가요?",
        "ROAS란?",
        "풀필먼트란?",
        "SKU meaning",
        # 서비스/기능
        "카페24 결제 서비스",
        "카페24 배송 서비스",
        "카페24 마케팅 기능",
        "카페24 호스팅 서비스",
        "카페24 API 가이드",
        # 정책/가이드
        "반품 정책",
        "카페24 서비스 지역",
        "셀러 등급 기준",
        "수수료 정책",
        "CS 가이드라인",
    ],

    IntentCategory.COOKIE: [
        # 쇼핑몰 정보
        "S0001 쇼핑몰 정보",
        "이 쇼핑몰 알려줘",
        "쇼핑몰 성과가 어때?",
        "쇼핑몰 KPI",
        "shop performance",
        # 서비스 정보
        "이 쇼핑몰 이용 서비스",
        "어떤 서비스 사용 중?",
        "서비스 목록",
        # 매출/성과
        "쇼핑몰 매출 알려줘",
        "월간 주문 수",
        "이 쇼핑몰 잘 되나?",
        "매출 순위",
        "revenue ranking",
        # 마케팅 최적화
        "마케팅 예산 추천",
        "광고 ROI 최적화",
        "마케팅 전략 추천",
        "예산 배분 추천",
        # 목록
        "프리미엄 플랜 쇼핑몰",
        "패션 카테고리 쇼핑몰",
        "전체 쇼핑몰 보여줘",
    ],

    IntentCategory.USER: [
        # 특정 셀러 분석
        "SEL0001 셀러 분석해줘",
        "이 셀러 정보 알려줘",
        "SEL0100 세그먼트가 뭐야?",
        "셀러 행동 패턴",
        "seller analysis",
        # 세그먼트
        "세그먼트별 통계",
        "셀러 군집 분석",
        "세그먼트 분류",
        "파워 셀러 몇 명?",
        # 이상 탐지
        "이상거래 탐지",
        "사기 의심 거래",
        "비정상 정산 패턴",
        "비정상 주문 셀러",
        # 개별 셀러 예측
        "이 셀러 이탈할까?",
        "셀러 이탈 확률",
    ],

    IntentCategory.TRANSLATE: [
        # CS 응답
        "이 문의에 답변해줘",
        "CS 자동 응답 생성",
        "고객 문의 답변",
        "auto reply CS",
        "CS 응답 부탁해",
        # CS 품질
        "CS 응답 품질 검사",
        "이 응답 괜찮아?",
        "CS quality check",
        # 용어집
        "이커머스 용어집",
        "플랫폼 용어 알려줘",
        "커머스 용어 정리",
    ],

    IntentCategory.DASHBOARD: [
        # 대시보드/현황
        "대시보드 보여줘",
        "전체 현황",
        "요약 통계",
        "dashboard summary",
        "전체 요약",
        "현황판",
        # 통계 요약
        "오늘 통계",
        "전체 통계 요약",
        "시스템 현황",
    ],

    IntentCategory.GENERAL: [
        # 인사
        "안녕",
        "안녕하세요",
        "하이",
        "헬로",
        "hi",
        "hello",
        # 감사
        "고마워",
        "감사합니다",
        "thanks",
        "thank you",
        # 일반 대화
        "뭐해?",
        "누구야?",
        "자기소개 해줘",
        "넌 뭘 할 수 있어?",
        "도움말",
        "help",
    ],
}


# ============================================================
# 임베딩 캐시 (싱글톤)
# ============================================================
class SemanticRouterCache:
    """카테고리별 예시 쿼리 임베딩 캐시"""

    def __init__(self):
        self.route_embeddings: Dict[IntentCategory, np.ndarray] = {}
        self.route_texts: Dict[IntentCategory, List[str]] = {}
        self.is_initialized: bool = False
        self.embed_model: str = "text-embedding-3-small"
        self.embed_dim: int = 1536
        self._client: Optional[OpenAI] = None

    def _get_client(self, api_key: str) -> OpenAI:
        """OpenAI 클라이언트 싱글톤"""
        if self._client is None:
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _embed_texts(self, texts: List[str], api_key: str) -> np.ndarray:
        """텍스트 목록을 임베딩으로 변환"""
        client = self._get_client(api_key)

        try:
            response = client.embeddings.create(
                model=self.embed_model,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            st.logger.error("SEMANTIC_ROUTER_EMBED_FAIL err=%s", safe_str(e))
            raise

    def initialize(self, api_key: str) -> bool:
        """
        카테고리별 예시 쿼리 임베딩 사전 계산

        서버 시작 시 1회 호출 권장
        """
        if self.is_initialized:
            st.logger.info("SEMANTIC_ROUTER already initialized")
            return True

        if not api_key:
            st.logger.warning("SEMANTIC_ROUTER_INIT_SKIP no api_key")
            return False

        start_time = time.time()
        st.logger.info("SEMANTIC_ROUTER_INIT_START")

        try:
            for category, examples in ROUTE_EXAMPLES.items():
                if not examples:
                    continue

                # 임베딩 계산
                embeddings = self._embed_texts(examples, api_key)
                self.route_embeddings[category] = embeddings
                self.route_texts[category] = examples

                st.logger.info(
                    "SEMANTIC_ROUTER_EMBED category=%s examples=%d",
                    category.value, len(examples),
                )

            self.is_initialized = True
            elapsed = time.time() - start_time

            st.logger.info(
                "SEMANTIC_ROUTER_INIT_DONE categories=%d elapsed=%.2fs",
                len(self.route_embeddings), elapsed,
            )
            return True

        except Exception as e:
            st.logger.error("SEMANTIC_ROUTER_INIT_FAIL err=%s", safe_str(e))
            return False

    def classify(
        self,
        query: str,
        api_key: str,
        threshold: float = 0.5,
    ) -> Tuple[Optional[IntentCategory], float]:
        """
        쿼리를 가장 유사한 카테고리로 분류

        Args:
            query: 사용자 쿼리
            api_key: OpenAI API 키
            threshold: 최소 유사도 임계값 (기본 0.5)

        Returns:
            (카테고리, 유사도) 또는 (None, 0.0) if 임계값 미달
        """
        if not self.is_initialized:
            if not self.initialize(api_key):
                return (None, 0.0)

        try:
            # 쿼리 임베딩
            query_embedding = self._embed_texts([query], api_key)[0]

            best_category: Optional[IntentCategory] = None
            best_score: float = -1.0

            # 각 카테고리별 최대 유사도 계산
            for category, route_embeds in self.route_embeddings.items():
                # 코사인 유사도 계산 (정규화된 내적)
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                route_norms = route_embeds / np.linalg.norm(route_embeds, axis=1, keepdims=True)

                similarities = np.dot(route_norms, query_norm)
                max_sim = float(np.max(similarities))

                if max_sim > best_score:
                    best_score = max_sim
                    best_category = category

            # 임계값 확인
            if best_score < threshold:
                st.logger.info(
                    "SEMANTIC_ROUTER_LOW_CONFIDENCE query=%s best=%s score=%.3f threshold=%.2f",
                    query[:30], best_category.value if best_category else "none", best_score, threshold,
                )
                return (None, best_score)

            st.logger.info(
                "SEMANTIC_ROUTER_CLASSIFY query=%s category=%s score=%.3f",
                query[:30], best_category.value, best_score,
            )

            return (best_category, best_score)

        except Exception as e:
            st.logger.error("SEMANTIC_ROUTER_CLASSIFY_FAIL err=%s", safe_str(e))
            return (None, 0.0)


# 싱글톤 인스턴스
_router_cache = SemanticRouterCache()


def init_semantic_router(api_key: str) -> bool:
    """
    Semantic Router 초기화 (서버 시작 시 호출)

    Returns:
        성공 여부
    """
    return _router_cache.initialize(api_key)


def semantic_classify(
    query: str,
    api_key: str,
    threshold: float = 0.5,
) -> Tuple[Optional[IntentCategory], float]:
    """
    Semantic Router로 쿼리 분류

    Args:
        query: 사용자 쿼리
        api_key: OpenAI API 키
        threshold: 최소 유사도 (기본 0.5)

    Returns:
        (카테고리, 유사도) 또는 (None, 0.0)
    """
    return _router_cache.classify(query, api_key, threshold)


def get_semantic_router_status() -> dict:
    """Semantic Router 상태 조회"""
    return {
        "initialized": _router_cache.is_initialized,
        "categories": list(_router_cache.route_embeddings.keys()) if _router_cache.is_initialized else [],
        "total_examples": sum(len(v) for v in _router_cache.route_texts.values()) if _router_cache.is_initialized else 0,
        "embed_model": _router_cache.embed_model,
    }
