"""
api/routes.py - 라우터 통합 모듈
=================================
각 도메인별 라우터를 하나의 APIRouter로 통합합니다.
main.py에서는 이 router 하나만 include하면 됩니다.

도메인 분리:
  - routes_admin.py     : 인증, 사용자, 설정, 내보내기, 헬스체크
  - routes_shop.py      : 쇼핑몰, 카테고리, 대시보드, 분석, 통계
  - routes_seller.py    : 셀러 검색/분석/세그먼트/성과
  - routes_cs.py        : CS 자동응답, 파이프라인, n8n 콜백
  - routes_rag.py       : RAG/LightRAG/K2RAG 검색, OCR
  - routes_ml.py        : MLflow, 마케팅 최적화
  - routes_agent.py     : 에이전트/채팅, 스트리밍
  - routes_guardian.py   : Data Guardian 보안 감시
"""
from fastapi import APIRouter

from api.routes_admin import router as admin_router
from api.routes_shop import router as shop_router
from api.routes_seller import router as seller_router
from api.routes_cs import router as cs_router
from api.routes_rag import router as rag_router
from api.routes_ml import router as ml_router
from api.routes_agent import router as agent_router
from api.routes_guardian import router as guardian_router

router = APIRouter()

router.include_router(admin_router)
router.include_router(shop_router)
router.include_router(seller_router)
router.include_router(cs_router)
router.include_router(rag_router)
router.include_router(ml_router)
router.include_router(agent_router)
router.include_router(guardian_router)
