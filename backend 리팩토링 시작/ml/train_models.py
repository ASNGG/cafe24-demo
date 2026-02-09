"""
카페24 이커머스 AI 플랫폼 - 데이터 생성 및 모델 학습
===================================================
카페24 운영 AI 프로젝트

구조:
  PART 1: 설정 및 환경
  PART 2: 데이터 생성 (18개 CSV)
  PART 3: 모델 학습 (10개 ML 모델)
  PART 4: 저장 및 테스트
"""

# ============================================================================
# PART 1: 설정 및 환경
# ============================================================================
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    silhouette_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.cluster import KMeans, DBSCAN
import joblib
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

np.random.seed(42)
rng = np.random.default_rng(42)

# MLflow 설정 (선택적)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
    MLFLOW_TRACKING_URI = "file:./mlruns"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    EXPERIMENT_NAME = "cafe24-ops-ai"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow Experiment: {EXPERIMENT_NAME}")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow 미설치 - 실험 추적을 건너뜁니다")

# LightGBM (매출 예측용)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM 미설치 - GradientBoosting으로 대체합니다")

# XGBoost (수요 예측용)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost 미설치 - GradientBoosting으로 대체합니다")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP 미설치 - SHAP 분석을 건너뜁니다")

# 저장 경로
try:
    BACKEND_DIR = Path(__file__).parent.parent
except NameError:
    BACKEND_DIR = Path.cwd()
    if BACKEND_DIR.name == "ml":
        BACKEND_DIR = BACKEND_DIR.parent
    elif "backend" in str(BACKEND_DIR).lower() or "리팩토링" in str(BACKEND_DIR):
        pass
    else:
        BACKEND_DIR = Path.cwd()

BACKEND_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PART 1: 설정 완료")
print(f"  BACKEND_DIR: {BACKEND_DIR}")
print("=" * 70)


# ============================================================================
# PART 2: 데이터 생성 (18개 CSV)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: 데이터 생성 (카페24 이커머스)")
print("=" * 70)

reference_date = pd.to_datetime("2025-01-15")

# --------------------------------------------------------------------------
# 카페24 관련 상수 데이터
# --------------------------------------------------------------------------
PLAN_TIERS = ["Basic", "Standard", "Premium", "Enterprise"]
PLAN_WEIGHTS = [0.35, 0.30, 0.25, 0.10]
PLAN_TIER_ENCODE = {"Basic": 0, "Standard": 1, "Premium": 2, "Enterprise": 3}

CATEGORIES_KO = ["패션", "뷰티", "식품", "전자기기", "생활용품", "IT서비스", "교육", "스포츠"]
CATEGORIES_EN = ["Fashion", "Beauty", "Food", "Electronics", "Household", "IT Services", "Education", "Sports"]

REGIONS = ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "제주"]
REGION_WEIGHTS = [0.30, 0.25, 0.12, 0.10, 0.07, 0.06, 0.05, 0.05]


FASHION_PRODUCTS = [
    "캐시미어 니트", "오버핏 맨투맨", "와이드 슬랙스", "플리스 자켓", "데님 팬츠",
    "롱 패딩", "가죽 백팩", "울 코트", "스트라이프 셔츠", "테일러드 블레이저",
    "크로스백", "미니 스커트", "카고 팬츠", "린넨 원피스", "트렌치 코트",
]
BEAUTY_PRODUCTS = [
    "수분 에센스", "선크림 SPF50+", "클렌징 오일", "립틴트", "쿠션 파운데이션",
    "아이크림", "토너패드", "세럼 앰플", "마스크팩 세트", "헤어 에센스",
    "비타민C 세럼", "각질제거 젤", "미스트 토너", "나이트 크림", "볼 블러셔",
]
FOOD_PRODUCTS = [
    "프리미엄 한우 세트", "유기농 과일 박스", "수제 잼 세트", "건강 견과류", "전통 된장",
    "올리브유 선물세트", "프로바이오틱스", "녹차 선물세트", "수제 초콜릿", "건과일 믹스",
    "홍삼 추출액", "제주 감귤 박스", "참치캔 세트", "꿀 선물세트", "특등급 쌀",
]
ELECTRONICS_PRODUCTS = [
    "블루투스 이어폰", "무선 충전기", "보조배터리", "스마트 워치", "USB 허브",
    "LED 모니터", "기계식 키보드", "웹캠 HD", "마우스패드", "노트북 거치대",
    "태블릿 케이스", "HDMI 케이블", "외장 SSD", "스피커 미니", "멀티탭",
]
HOUSEHOLD_PRODUCTS = [
    "공기청정기 필터", "무선 청소기", "아로마 디퓨저", "텀블러 세트", "수건 선물세트",
    "주방세제 세트", "쿠션 커버", "수납 박스", "LED 스탠드", "매트리스 토퍼",
    "실리콘 주방도구", "빨래 건조대", "미니 가습기", "유리 밀폐용기", "방향제 세트",
]

PRODUCT_MAP = {
    "패션": FASHION_PRODUCTS,
    "뷰티": BEAUTY_PRODUCTS,
    "식품": FOOD_PRODUCTS,
    "전자기기": ELECTRONICS_PRODUCTS,
    "생활용품": HOUSEHOLD_PRODUCTS,
}

SHOP_NAME_PREFIXES = {
    "패션": ["스타일", "모드", "트렌드", "패션", "옷장", "룩", "데일리"],
    "뷰티": ["글로우", "뷰티", "스킨", "코스", "피부", "에스테"],
    "식품": ["맛있는", "신선한", "건강한", "프레시", "자연", "오가닉"],
    "전자기기": ["테크", "디지", "스마트", "기가", "넥스트", "이노"],
    "생활용품": ["홈", "리빙", "편리한", "깔끔한", "모던", "데코"],
    "IT서비스": ["클라우드", "코드", "데이터", "AI", "넷", "웹"],
    "교육": ["에듀", "러닝", "스쿨", "지식", "클래스", "배움"],
    "스포츠": ["피트", "스포", "액티브", "런닝", "헬스", "짐"],
}
SHOP_NAME_SUFFIXES = ["마켓", "스토어", "몰", "샵", "하우스", "플러스", "랩", "존"]

CS_INQUIRY_TEMPLATES = {
    "배송": [
        "택배사 연동 설정은 어디서 하나요? CJ대한통운 추가하고 싶습니다.",
        "배송 추적 API 연동이 안 됩니다. 송장번호가 자동 입력되지 않아요.",
        "해외배송 서비스 활성화 방법 알려주세요.",
        "배송비 정책 설정에서 제주/도서산간 추가 배송비를 어떻게 설정하나요?",
        "묶음배송 설정이 제대로 작동하지 않습니다.",
        "배송 대행 서비스(풀필먼트) 신청 방법이 궁금합니다.",
        "택배사별 운송장 출력이 안 됩니다. 프린터 연동 오류가 발생해요.",
        "당일배송/새벽배송 옵션을 추가하고 싶은데 가능한가요?",
        "배송비 조건부 무료 설정(5만원 이상 무료) 방법을 모르겠습니다.",
        "해외 주문 건 통관 정보 입력은 어디서 하나요?",
    ],
    "환불": [
        "고객 환불 처리 시 PG 수수료도 환불되나요?",
        "부분 환불 처리 방법을 알려주세요. 관리자 페이지에서 어떻게 하나요?",
        "반품 접수된 주문 건 자동 환불 설정이 가능한가요?",
        "환불 처리했는데 고객에게 카드 취소가 며칠 걸리나요?",
        "교환/반품 사유별 배송비 부담 설정은 어디서 하나요?",
        "환불 정책을 쇼핑몰에 자동 표시하는 방법이 있나요?",
        "PG사 정산 후 환불 건 처리 프로세스가 궁금합니다.",
        "환불 대기 중인 주문 목록을 일괄 조회하고 싶습니다.",
        "고객이 반품 요청했는데 관리자에서 승인 처리가 안 됩니다.",
        "환불 완료된 건인데 고객이 입금 안 됐다고 합니다. 확인 부탁드립니다.",
    ],
    "결제": [
        "PG사 연동 설정 중 오류가 발생합니다. 이니시스 키 등록이 안 돼요.",
        "간편결제(카카오페이, 네이버페이) 추가 연동 방법을 알려주세요.",
        "결제 테스트 모드에서 실제 결제로 전환하려면 어떻게 하나요?",
        "해외 결제 수단(PayPal 등) 연동이 가능한가요?",
        "결제 수수료율 확인 및 변경은 어디서 하나요?",
        "가상계좌(무통장입금) 입금 확인이 자동으로 안 됩니다.",
        "결제 오류가 자주 발생한다는 고객 문의가 많습니다. 원인이 뭔가요?",
        "할부 결제 설정 방법과 PG사별 차이점이 궁금합니다.",
        "정기결제(구독) 기능을 구현하고 싶은데 지원되나요?",
        "세금계산서 자동 발행 설정 방법을 알려주세요.",
    ],
    "상품": [
        "상품 대량 등록(엑셀 업로드)에서 오류가 발생합니다.",
        "상품 옵션 조합 설정이 너무 복잡한데 쉬운 방법이 있나요?",
        "카테고리별 상품 진열 순서를 변경하고 싶습니다.",
        "상품 상세페이지 에디터에서 이미지가 깨집니다.",
        "품절 상품 자동 숨김 처리 설정이 가능한가요?",
        "상품 리뷰 관리에서 악성 리뷰를 일괄 삭제하는 방법이 있나요?",
        "네이버 쇼핑 EP 연동 시 상품 정보가 누락됩니다.",
        "상품 재고 관리 시 옵션별 재고 설정은 어디서 하나요?",
        "타 플랫폼(쿠팡, 11번가)에서 상품을 가져오기 할 수 있나요?",
        "상품 할인가 설정과 회원 등급별 할인 중복 적용이 가능한가요?",
    ],
    "계정": [
        "쇼핑몰 관리자 비밀번호를 분실했습니다. 재설정 방법 알려주세요.",
        "부운영자 계정을 추가하고 권한을 제한하고 싶습니다.",
        "쇼핑몰 플랜(요금제) 업그레이드 방법과 차이점이 궁금합니다.",
        "사업자 정보 변경(상호명, 대표자) 신청은 어떻게 하나요?",
        "쇼핑몰 도메인을 변경하고 싶습니다. 기존 URL은 어떻게 되나요?",
        "관리자 페이지 2단계 인증(OTP) 설정 방법을 알려주세요.",
        "쇼핑몰 임시 폐쇄/휴면 처리 방법이 궁금합니다.",
        "멀티쇼핑몰 기능으로 추가 쇼핑몰을 개설하고 싶습니다.",
        "API 키 발급 방법과 사용 한도가 어떻게 되나요?",
        "카페24 파트너 프로그램 가입 조건이 궁금합니다.",
    ],
    "정산": [
        "이번 달 정산 금액이 예상보다 적습니다. 수수료 내역 확인 부탁드립니다.",
        "정산 주기를 월 2회에서 주간 정산으로 변경할 수 있나요?",
        "정산 보류된 건이 있는데 해제 방법을 알려주세요.",
        "세금계산서와 정산 금액이 일치하지 않습니다.",
        "해외 판매 건 정산 시 환율 적용 기준이 궁금합니다.",
        "PG 수수료 외에 추가로 발생하는 비용이 있나요?",
        "정산 내역 엑셀 다운로드가 안 됩니다.",
        "카드 매출 정산과 계좌이체 정산 일정이 다른 이유가 뭔가요?",
        "정산 계좌를 변경하고 싶습니다. 처리 기간이 어떻게 되나요?",
        "부가세 신고용 자료를 어디서 다운로드할 수 있나요?",
    ],
    "기술지원": [
        "카페24 API로 주문 목록을 가져오는데 인증 오류가 발생합니다.",
        "쇼핑몰 디자인 스킨 수정 방법을 알려주세요. HTML/CSS를 직접 편집하고 싶습니다.",
        "외부 스크립트(채널톡, GA4)를 쇼핑몰에 삽입하는 방법이 궁금합니다.",
        "모바일 페이지 로딩 속도가 너무 느립니다. 최적화 방법이 있나요?",
        "쇼핑몰에 커스텀 기능을 추가 개발 의뢰하고 싶습니다.",
        "카페24 앱스토어에서 설치한 앱이 정상 작동하지 않습니다.",
        "SSL 인증서 갱신이 자동으로 안 됩니다. 수동 갱신 방법 알려주세요.",
        "REST API 호출 시 rate limit이 초과됐다는 오류가 나옵니다.",
        "카페24 웹훅 설정 방법과 지원하는 이벤트 목록이 궁금합니다.",
        "쇼핑몰 백업/복원 기능이 있나요? 데이터 이전이 필요합니다.",
    ],
    "마케팅": [
        "네이버 쇼핑 광고 연동 설정 방법을 알려주세요.",
        "카카오 픽셀 설치와 전환 추적 설정이 안 됩니다.",
        "쿠폰 발급 시 특정 상품/카테고리만 적용하려면 어떻게 하나요?",
        "회원 등급별 자동 할인 설정 방법이 궁금합니다.",
        "Google Analytics 4(GA4) 연동은 어디서 하나요?",
        "메타(페이스북) 픽셀 설치 방법을 알려주세요.",
        "이메일 마케팅/뉴스레터 발송 기능이 있나요?",
        "SEO 설정(메타 태그, 구조화 데이터) 방법이 궁금합니다.",
        "타임세일/기획전 페이지를 만들고 싶은데 방법을 모르겠습니다.",
        "적립금/포인트 정책을 변경하고 싶습니다. 기존 적립금에도 영향이 있나요?",
    ],
    "기타": [
        "카페24 교육 프로그램이나 웨비나 일정이 궁금합니다.",
        "쇼핑몰 운영 컨설팅 서비스를 받을 수 있나요?",
        "카페24 서비스 장애 공지는 어디서 확인하나요?",
        "타 플랫폼(고도몰, 메이크샵)에서 카페24로 이전하려면 어떻게 하나요?",
        "카페24 파트너사/디자인 에이전시 추천 부탁드립니다.",
    ],
}

REVIEW_TEMPLATES_POSITIVE = [
    "배송도 빠르고 상품 품질도 아주 좋습니다! 재구매 의사 있어요.",
    "가격 대비 품질이 정말 훌륭합니다. 추천해요!",
    "포장이 꼼꼼하고 상품 상태도 완벽했습니다.",
    "사진과 동일하게 왔어요. 매우 만족합니다.",
    "생각보다 훨씬 좋아서 놀랐습니다. 감사합니다!",
    "색상도 예쁘고 사이즈도 딱 맞아요. 최고!",
    "이 가격에 이 품질이면 정말 가성비 좋아요.",
    "친구한테 선물했는데 너무 좋아했습니다.",
    "배송 엄청 빨라요! 다음날 바로 도착했습니다.",
    "재구매합니다. 이번에도 역시 좋네요.",
    "두번째 구매인데 변함없는 품질에 만족합니다.",
    "상세페이지 그대로 왔어요. 신뢰가 갑니다.",
    "가족 모두 만족하고 있습니다. 감사해요!",
    "소재가 정말 좋아요. 고급스러운 느낌입니다.",
    "고객센터 응대도 친절하고 상품도 좋아요!",
]

REVIEW_TEMPLATES_NEGATIVE = [
    "기대했는데 실망이에요. 사진과 너무 다릅니다.",
    "배송이 너무 늦었습니다. 개선이 필요해요.",
    "상품에 하자가 있어서 반품 진행 중입니다.",
    "가격에 비해 품질이 떨어집니다.",
    "포장이 엉성해서 상품이 파손되었습니다.",
    "사이즈가 표기와 전혀 다릅니다.",
    "냄새가 심해서 사용하기 어렵습니다.",
    "색상이 사진과 완전히 다릅니다. 환불 원합니다.",
    "한번 세탁했더니 형태가 변형되었습니다.",
    "마감 처리가 매우 조잡합니다.",
    "주문 후 일주일이 지났는데 배송 시작도 안 했어요.",
    "고객센터 연락이 전혀 안 됩니다.",
    "제품 설명에 없는 결함이 있었습니다.",
    "가격 대비 매우 실망스러운 품질입니다.",
    "두번째 교환인데도 불량이 왔어요.",
]

REVIEW_TEMPLATES_NEUTRAL = [
    "보통이에요. 나쁘지는 않지만 특별하지도 않네요.",
    "가격 생각하면 적당한 것 같습니다.",
    "기대한 것과 비슷하게 왔습니다.",
    "무난한 상품입니다. 평범해요.",
    "배송은 빨랐는데 상품은 그저 그래요.",
    "사용해봐야 알 것 같아요. 우선 보통입니다.",
    "나쁘지 않은데 굳이 재구매는 안 할 것 같아요.",
    "가격만큼의 가치는 있습니다.",
    "포장은 좋았는데 상품은 보통이네요.",
    "기능은 괜찮은데 디자인이 아쉬워요.",
]

PLATFORM_DOCS_DATA = [
    {"doc_id": "DOC001", "title": "카페24 쇼핑몰 개설 가이드", "category": "시작하기",
     "content_ko": "카페24에서 쇼핑몰을 개설하려면 사업자등록번호와 통신판매업 신고가 필요합니다. 기본 플랜부터 시작하여 매출 성장에 따라 업그레이드할 수 있습니다."},
    {"doc_id": "DOC002", "title": "결제 시스템 연동 방법", "category": "결제",
     "content_ko": "카페24는 KG이니시스, 토스페이먼츠, NHN KCP 등 주요 PG사와 연동되어 있습니다. 관리자 페이지에서 결제 수단을 설정할 수 있습니다."},
    {"doc_id": "DOC003", "title": "배송 관리 정책", "category": "배송",
     "content_ko": "주문 접수 후 2영업일 이내에 발송해야 합니다. 배송 추적번호는 자동으로 고객에게 발송됩니다."},
    {"doc_id": "DOC004", "title": "환불 및 교환 처리 가이드", "category": "CS",
     "content_ko": "고객 환불 요청 시 7일 이내 처리가 원칙입니다. 상품 하자의 경우 판매자 부담으로 반품 배송비를 처리합니다."},
    {"doc_id": "DOC005", "title": "정산 시스템 안내", "category": "정산",
     "content_ko": "정산은 매주 수요일에 진행됩니다. 결제 수수료는 플랜에 따라 2.5%~3.5%가 적용됩니다."},
    {"doc_id": "DOC006", "title": "마케팅 도구 활용 가이드", "category": "마케팅",
     "content_ko": "카페24의 마케팅 도구를 활용하여 SEO 최적화, 이메일 마케팅, SNS 연동 등을 설정할 수 있습니다."},
    {"doc_id": "DOC007", "title": "상품 등록 및 관리", "category": "상품",
     "content_ko": "상품 등록 시 카테고리, 가격, 재고, 옵션, 상세 설명을 입력합니다. 대량 등록은 엑셀 업로드로 가능합니다."},
    {"doc_id": "DOC008", "title": "쇼핑몰 디자인 커스터마이징", "category": "디자인",
     "content_ko": "카페24는 다양한 무료/유료 디자인 템플릿을 제공합니다. HTML/CSS 수정으로 세부 디자인을 조정할 수 있습니다."},
    {"doc_id": "DOC009", "title": "고객 데이터 분석 가이드", "category": "분석",
     "content_ko": "고객 행동 분석, 매출 리포트, 상품별 판매 현황 등을 대시보드에서 확인할 수 있습니다."},
    {"doc_id": "DOC010", "title": "API 연동 개발자 가이드", "category": "개발",
     "content_ko": "카페24 Open API를 통해 주문, 상품, 고객 데이터에 접근할 수 있습니다. OAuth 2.0 인증을 사용합니다."},
    {"doc_id": "DOC011", "title": "플랜별 기능 비교", "category": "시작하기",
     "content_ko": "Basic 플랜은 월 500건 주문 처리, Standard는 2000건, Premium은 무제한입니다. Enterprise는 전담 매니저를 제공합니다."},
    {"doc_id": "DOC012", "title": "해외 판매(글로벌 쇼핑몰) 가이드", "category": "글로벌",
     "content_ko": "카페24 글로벌 서비스를 통해 다국어 쇼핑몰을 운영할 수 있습니다. 해외 배송 및 환율 자동 적용을 지원합니다."},
]

ECOMMERCE_GLOSSARY = [
    {"term_ko": "GMV", "term_en": "Gross Merchandise Value", "definition": "총 거래액. 플랫폼에서 발생한 전체 상품 판매 금액", "category": "매출"},
    {"term_ko": "전환율", "term_en": "Conversion Rate", "definition": "방문자 중 실제 구매로 이어진 비율", "category": "마케팅"},
    {"term_ko": "객단가", "term_en": "Average Order Value", "definition": "주문 1건당 평균 결제 금액", "category": "매출"},
    {"term_ko": "재구매율", "term_en": "Repeat Purchase Rate", "definition": "기존 고객이 다시 구매하는 비율", "category": "고객"},
    {"term_ko": "이탈률", "term_en": "Churn Rate", "definition": "일정 기간 내 서비스를 떠난 고객 또는 셀러의 비율", "category": "고객"},
    {"term_ko": "LTV", "term_en": "Customer Lifetime Value", "definition": "고객 한 명이 전체 이용 기간 동안 발생시키는 총 매출", "category": "고객"},
    {"term_ko": "CAC", "term_en": "Customer Acquisition Cost", "definition": "신규 고객 한 명을 획득하는 데 드는 비용", "category": "마케팅"},
    {"term_ko": "정산", "term_en": "Settlement", "definition": "판매 대금에서 수수료를 차감한 후 셀러에게 지급하는 과정", "category": "결제"},
    {"term_ko": "SKU", "term_en": "Stock Keeping Unit", "definition": "재고 관리 단위. 개별 상품을 식별하는 코드", "category": "상품"},
    {"term_ko": "풀필먼트", "term_en": "Fulfillment", "definition": "주문 접수부터 배송 완료까지의 전체 물류 처리 과정", "category": "물류"},
    {"term_ko": "PG", "term_en": "Payment Gateway", "definition": "온라인 결제를 중개하는 서비스 (이니시스, 토스페이먼츠 등)", "category": "결제"},
    {"term_ko": "SEO", "term_en": "Search Engine Optimization", "definition": "검색 엔진 최적화. 검색 결과 상위 노출을 위한 전략", "category": "마케팅"},
    {"term_ko": "RFM", "term_en": "Recency Frequency Monetary", "definition": "고객 세분화 기법. 최근 구매일, 구매 빈도, 구매 금액 기반", "category": "고객"},
    {"term_ko": "ROAS", "term_en": "Return on Ad Spend", "definition": "광고비 대비 매출 비율", "category": "마케팅"},
    {"term_ko": "크로스셀링", "term_en": "Cross-selling", "definition": "고객이 구매한 상품과 관련된 다른 상품을 추천하는 전략", "category": "마케팅"},
]


# --------------------------------------------------------------------------
# 2.1 shops.csv (300개 쇼핑몰)
# --------------------------------------------------------------------------
print("\n[2.1] 쇼핑몰 데이터 생성")


def generate_shop_name(category, idx):
    prefixes = SHOP_NAME_PREFIXES.get(category, ["마이"])
    prefix = rng.choice(prefixes)
    suffix = rng.choice(SHOP_NAME_SUFFIXES)
    return f"{prefix}{suffix}"


shops_data = []
for i in range(300):
    shop_id = f"S{i+1:04d}"
    category = rng.choice(CATEGORIES_KO, p=[0.20, 0.15, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08])
    region = rng.choice(REGIONS, p=REGION_WEIGHTS)
    plan = rng.choice(PLAN_TIERS, p=PLAN_WEIGHTS)
    open_date = reference_date - timedelta(days=int(rng.integers(30, 900)))

    status_prob = rng.random()
    if status_prob < 0.70:
        status = "active"
    elif status_prob < 0.88:
        status = "dormant"
    else:
        status = "churned"

    shop_name = generate_shop_name(category, i)
    desc = f"{category} 전문 온라인 쇼핑몰. {region} 기반으로 다양한 {category} 상품을 판매합니다."
    shops_data.append({
        "shop_id": shop_id, "shop_name": shop_name, "plan_tier": plan,
        "category": category, "region": region,
        "open_date": open_date.strftime("%Y-%m-%d"), "description": desc, "status": status,
    })

shops_df = pd.DataFrame(shops_data)
print(f"  - 쇼핑몰: {len(shops_df)}개 (active: {(shops_df['status']=='active').sum()}, "
      f"dormant: {(shops_df['status']=='dormant').sum()}, churned: {(shops_df['status']=='churned').sum()})")


# --------------------------------------------------------------------------
# 2.2 categories.csv (8개 카테고리)
# --------------------------------------------------------------------------
print("\n[2.2] 카테고리 데이터 생성")

categories_data = []
for idx, (ko, en) in enumerate(zip(CATEGORIES_KO, CATEGORIES_EN)):
    categories_data.append({
        "cat_id": f"CAT{idx+1:03d}", "name_ko": ko, "name_en": en,
        "parent_cat": "ROOT",
        "description_ko": f"{ko} 관련 상품을 판매하는 카테고리입니다.",
        "description_en": f"Category for {en} related products.",
    })
categories_df = pd.DataFrame(categories_data)
print(f"  - 카테고리: {len(categories_df)}개")


# --------------------------------------------------------------------------
# 2.3 services.csv (쇼핑몰별 서비스)
# --------------------------------------------------------------------------
print("\n[2.3] 서비스 데이터 생성")

SERVICE_TYPES = ["hosting", "payment", "shipping", "marketing"]
SERVICE_NAMES = {
    "hosting": ["웹호스팅 기본", "클라우드 서버", "CDN 가속", "SSL 인증서"],
    "payment": ["카드결제", "간편결제(카카오페이)", "간편결제(네이버페이)", "가상계좌"],
    "shipping": ["CJ대한통운", "한진택배", "롯데택배", "우체국택배"],
    "marketing": ["SEO 최적화", "이메일 마케팅", "SNS 연동", "키워드 광고"],
}

services_data = []
svc_idx = 0
for _, shop in shops_df.iterrows():
    n_services = rng.integers(2, 7)
    for _ in range(n_services):
        svc_type = rng.choice(SERVICE_TYPES)
        svc_name = rng.choice(SERVICE_NAMES[svc_type])
        svc_status = "active" if shop["status"] == "active" and rng.random() < 0.85 else "inactive"
        svc_idx += 1
        services_data.append({
            "service_id": f"SVC{svc_idx:05d}", "shop_id": shop["shop_id"],
            "service_name": svc_name, "service_type": svc_type,
            "status": svc_status,
            "description": f"{shop['shop_name']}의 {svc_name} 서비스",
        })

services_df = pd.DataFrame(services_data)
print(f"  - 서비스: {len(services_df)}건")


# --------------------------------------------------------------------------
# 2.4 products.csv (2000+ 상품)
# --------------------------------------------------------------------------
print("\n[2.4] 상품 데이터 생성")

products_data = []
prod_idx = 0
for _, shop in shops_df.iterrows():
    category = shop["category"]
    n_products = rng.integers(10, 40)
    product_names_pool = PRODUCT_MAP.get(category, HOUSEHOLD_PRODUCTS)
    for j in range(n_products):
        prod_idx += 1
        pname = rng.choice(product_names_pool) + f" {rng.choice(['A', 'B', 'C', 'S', 'X', 'Pro', 'Lite', ''])}{rng.integers(1, 99)}"
        price_base = {
            "패션": 35000, "뷰티": 22000, "식품": 28000, "전자기기": 45000,
            "생활용품": 18000, "IT서비스": 50000, "교육": 30000, "스포츠": 40000,
        }.get(category, 25000)
        price = int(np.clip(rng.lognormal(np.log(price_base), 0.5), 3000, 500000))
        price = round(price, -2)

        status_r = rng.random()
        if status_r < 0.75:
            p_status = "active"
        elif status_r < 0.90:
            p_status = "sold_out"
        else:
            p_status = "hidden"

        listed_date = reference_date - timedelta(days=int(rng.integers(1, 365)))
        stock = int(rng.integers(0, 500)) if p_status != "sold_out" else 0

        products_data.append({
            "product_id": f"P{prod_idx:05d}", "shop_id": shop["shop_id"],
            "product_name": pname.strip(), "price": price, "category": category,
            "status": p_status, "listed_date": listed_date.strftime("%Y-%m-%d"),
            "stock_qty": stock,
        })

products_df = pd.DataFrame(products_data)
print(f"  - 상품: {len(products_df)}개")


# --------------------------------------------------------------------------
# 2.5 sellers.csv (300명 셀러, 쇼핑몰과 1:1 매칭)
# --------------------------------------------------------------------------
print("\n[2.5] 셀러 데이터 생성")

sellers_data = []
for _, shop in shops_df.iterrows():
    sid = shop["shop_id"].replace("S", "SEL")
    join_date = pd.to_datetime(shop["open_date"])
    days_since_join = (reference_date - join_date).days

    if shop["status"] == "churned":
        last_login = join_date + timedelta(days=int(rng.integers(1, max(2, days_since_join - 30))))
        total_orders = int(rng.integers(5, 200))
        total_revenue = int(total_orders * rng.integers(15000, 80000))
    elif shop["status"] == "dormant":
        last_login = reference_date - timedelta(days=int(rng.integers(14, 60)))
        total_orders = int(rng.integers(50, 800))
        total_revenue = int(total_orders * rng.integers(20000, 90000))
    else:
        last_login = reference_date - timedelta(days=int(rng.integers(0, 7)))
        total_orders = int(rng.integers(100, 5000))
        total_revenue = int(total_orders * rng.integers(25000, 120000))

    product_count = len(products_df[products_df["shop_id"] == shop["shop_id"]])
    sellers_data.append({
        "seller_id": sid,
        "plan_tier": shop["plan_tier"], "region": shop["region"],
        "join_date": join_date.strftime("%Y-%m-%d"),
        "last_login": last_login.strftime("%Y-%m-%d"),
        "status": shop["status"],
        "total_orders": total_orders, "total_revenue": total_revenue,
        "product_count": product_count,
    })

sellers_df = pd.DataFrame(sellers_data)
print(f"  - 셀러: {len(sellers_df)}명")


# --------------------------------------------------------------------------
# 2.6 operation_logs.csv (최대 30,000행)
# --------------------------------------------------------------------------
print("\n[2.6] 운영 로그 데이터 생성")

OP_EVENT_TYPES = [
    "order_received", "product_listed", "cs_ticket", "payment_settled",
    "refund_processed", "marketing_campaign", "product_updated", "login",
]
OP_EVENT_WEIGHTS = [0.25, 0.12, 0.15, 0.13, 0.08, 0.07, 0.10, 0.10]

op_logs_data = []
log_idx = 0
target_logs = 30000
logs_per_seller = target_logs // len(sellers_df)

for _, seller in sellers_df.iterrows():
    n_logs = min(logs_per_seller + rng.integers(-50, 50), 500)
    if seller["status"] == "churned":
        n_logs = max(20, n_logs // 4)
    elif seller["status"] == "dormant":
        n_logs = max(50, n_logs // 2)

    join_d = pd.to_datetime(seller["join_date"])
    last_d = pd.to_datetime(seller["last_login"])
    active_span = max(1, (last_d - join_d).days)

    for _ in range(n_logs):
        log_idx += 1
        evt = rng.choice(OP_EVENT_TYPES, p=OP_EVENT_WEIGHTS)
        evt_offset = int(rng.integers(0, active_span))
        evt_date = join_d + timedelta(
            days=evt_offset,
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )

        if evt == "order_received":
            detail = json.dumps({
                "order_amount": int(rng.integers(10000, 200000)),
                "payment_method": rng.choice(["card", "kakao", "naver", "bank"]),
            }, ensure_ascii=False)
        elif evt == "cs_ticket":
            detail = json.dumps({
                "category": rng.choice(["배송", "환불", "결제", "상품", "계정"]),
                "priority": rng.choice(["urgent", "high", "normal", "low"]),
            }, ensure_ascii=False)
        elif evt == "refund_processed":
            detail = json.dumps({
                "refund_amount": int(rng.integers(5000, 150000)),
                "reason": rng.choice(["상품 불량", "고객 변심", "배송 오류", "기타"]),
            }, ensure_ascii=False)
        else:
            detail = "{}"

        op_logs_data.append({
            "log_id": f"LOG{log_idx:07d}", "seller_id": seller["seller_id"],
            "event_type": evt, "event_date": evt_date.strftime("%Y-%m-%d %H:%M:%S"),
            "details_json": detail,
        })
        if log_idx >= target_logs:
            break
    if log_idx >= target_logs:
        break

op_logs_df = pd.DataFrame(op_logs_data)
print(f"  - 운영 로그: {len(op_logs_df)}건")


# --------------------------------------------------------------------------
# 2.7 seller_analytics.csv (300명)
# --------------------------------------------------------------------------
print("\n[2.7] 셀러 분석 데이터 생성")

seller_analytics_data = []
for _, seller in sellers_df.iterrows():
    days_since_reg = (reference_date - pd.to_datetime(seller["join_date"])).days
    days_since_login = (reference_date - pd.to_datetime(seller["last_login"])).days
    plan_encoded = PLAN_TIER_ENCODE.get(seller["plan_tier"], 0)

    cs_tickets = int(rng.integers(0, max(1, seller["total_orders"] // 5)))
    refund_rate = round(float(np.clip(
        rng.beta(2, 20) if seller["status"] == "active" else rng.beta(3, 10), 0.0, 0.5
    )), 4)
    avg_response_time = round(float(np.clip(rng.exponential(4) + 0.5, 0.5, 72)), 1)

    seller_analytics_data.append({
        "seller_id": seller["seller_id"],
        "total_orders": seller["total_orders"],
        "total_revenue": seller["total_revenue"],
        "product_count": seller["product_count"],
        "cs_tickets": cs_tickets,
        "refund_rate": refund_rate,
        "avg_response_time": avg_response_time,
        "days_since_last_login": days_since_login,
        "days_since_register": days_since_reg,
        "plan_tier_encoded": plan_encoded,
        "cluster": -1,
        "churn_risk": -1,
        "churn_probability": -1.0,
    })

seller_analytics_df = pd.DataFrame(seller_analytics_data)
print(f"  - 셀러 분석: {len(seller_analytics_df)}명")


# --------------------------------------------------------------------------
# 2.8 shop_performance.csv (300개 쇼핑몰)
# --------------------------------------------------------------------------
print("\n[2.8] 쇼핑몰 성과 데이터 생성")

shop_perf_data = []
for _, shop in shops_df.iterrows():
    if shop["status"] == "active":
        conv = round(float(rng.uniform(1.5, 6.0)), 2)
        rev = int(rng.lognormal(15, 0.8))
        orders = int(rev / rng.integers(25000, 80000))
        visitors = int(orders / (conv / 100))
    elif shop["status"] == "dormant":
        conv = round(float(rng.uniform(0.5, 2.5)), 2)
        rev = int(rng.lognormal(13, 0.7))
        orders = int(rev / rng.integers(20000, 60000))
        visitors = int(orders / max(0.005, conv / 100))
    else:
        conv = round(float(rng.uniform(0.1, 1.0)), 2)
        rev = int(rng.lognormal(11, 0.6))
        orders = max(0, int(rev / rng.integers(15000, 50000)))
        visitors = max(10, int(orders / max(0.001, conv / 100)))

    shop_perf_data.append({
        "shop_id": shop["shop_id"],
        "conversion_rate": conv,
        "avg_order_value": int(rng.integers(15000, 120000)),
        "return_rate": round(float(rng.beta(2, 20)), 4),
        "review_score": round(float(np.clip(rng.normal(4.2, 0.5), 1.0, 5.0)), 2),
        "monthly_revenue": rev,
        "monthly_orders": orders,
        "monthly_visitors": visitors,
        "customer_retention_rate": round(float(np.clip(rng.beta(5, 3), 0.1, 0.95)), 4),
    })

shop_perf_df = pd.DataFrame(shop_perf_data)
print(f"  - 쇼핑몰 성과: {len(shop_perf_df)}개")


# --------------------------------------------------------------------------
# 2.9 daily_metrics.csv (90일)
# --------------------------------------------------------------------------
print("\n[2.9] 일별 플랫폼 지표 생성")

daily_metrics_data = []
base_active = 72
base_gmv = 85000000
for day in range(90):
    d = reference_date - timedelta(days=89 - day)
    weekend = 1.12 if d.weekday() >= 5 else 1.0
    active = int(base_active * weekend * rng.uniform(0.90, 1.10))
    gmv = int(base_gmv * weekend * rng.uniform(0.85, 1.20))
    total_orders = int(gmv / rng.integers(30000, 70000))
    daily_metrics_data.append({
        "date": d.strftime("%Y-%m-%d"),
        "active_shops": active,
        "total_gmv": gmv,
        "new_signups": int(rng.integers(0, 5)),
        "total_orders": total_orders,
        "avg_settlement_time": round(float(rng.uniform(1.5, 4.5)), 1),
        "cs_tickets_open": int(rng.integers(20, 80)),
        "cs_tickets_resolved": int(rng.integers(15, 75)),
        "fraud_alerts": int(rng.integers(0, 6)),
        "total_sessions": int(active * rng.uniform(2.5, 4.5)),
        "avg_session_minutes": round(float(rng.uniform(12, 35)), 1),
    })

daily_metrics_df = pd.DataFrame(daily_metrics_data)
print(f"  - 일별 지표: {len(daily_metrics_df)}일")


# --------------------------------------------------------------------------
# 2.10 cs_stats.csv
# --------------------------------------------------------------------------
print("\n[2.10] CS 통계 데이터 생성")

CS_CATEGORIES = ["배송", "환불", "결제", "상품", "계정", "정산", "기술지원", "마케팅", "기타"]
cs_stats_data = []
for cat in CS_CATEGORIES:
    cs_stats_data.append({
        "category": cat,
        "total_tickets": int(rng.integers(50, 500)),
        "avg_resolution_hours": round(float(rng.uniform(1, 48)), 1),
        "satisfaction_score": round(float(np.clip(rng.normal(3.8, 0.6), 1.0, 5.0)), 2),
    })
cs_stats_df = pd.DataFrame(cs_stats_data)
print(f"  - CS 통계: {len(cs_stats_df)}개 카테고리")


# --------------------------------------------------------------------------
# 2.11 fraud_details.csv
# --------------------------------------------------------------------------
print("\n[2.11] 이상거래 상세 데이터 생성")

ANOMALY_TYPES = ["high_refund", "fake_review", "price_manipulation", "unusual_volume"]
fraud_data = []
anomaly_sellers = rng.choice(
    sellers_df["seller_id"].values, size=min(15, len(sellers_df)), replace=False
)
for sid in anomaly_sellers:
    fraud_data.append({
        "seller_id": sid,
        "anomaly_score": round(float(rng.uniform(0.6, 1.0)), 4),
        "anomaly_type": rng.choice(ANOMALY_TYPES),
        "detected_date": (reference_date - timedelta(days=int(rng.integers(0, 30)))).strftime("%Y-%m-%d"),
        "details": f"이상 패턴 감지: {rng.choice(['환불율 급증', '비정상 리뷰 패턴', '가격 이상 변동', '주문량 급변'])}",
    })
fraud_df = pd.DataFrame(fraud_data)
print(f"  - 이상거래 상세: {len(fraud_df)}건")


# --------------------------------------------------------------------------
# 2.12 cohort_retention.csv
# --------------------------------------------------------------------------
print("\n[2.12] 코호트 리텐션 데이터 생성")

cohort_data = []
cohort_months = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]
for m in cohort_months:
    w1 = round(float(np.clip(rng.normal(85, 5), 60, 98)), 1)
    w2 = round(float(np.clip(w1 * rng.uniform(0.75, 0.90), 40, 95)), 1)
    w4 = round(float(np.clip(w2 * rng.uniform(0.65, 0.85), 25, 85)), 1)
    w8 = round(float(np.clip(w4 * rng.uniform(0.60, 0.80), 15, 70)), 1)
    w12 = round(float(np.clip(w8 * rng.uniform(0.55, 0.75), 8, 55)), 1)
    cohort_data.append({
        "cohort_month": m, "week1": w1, "week2": w2,
        "week4": w4, "week8": w8, "week12": w12,
    })
cohort_df = pd.DataFrame(cohort_data)
print(f"  - 코호트 리텐션: {len(cohort_df)}개월")

# 전환 퍼널 데이터 (코호트별)
conversion_data = []
for m in cohort_months:
    registered = int(rng.integers(80, 150))
    activated = int(registered * rng.uniform(0.60, 0.85))
    engaged = int(activated * rng.uniform(0.50, 0.75))
    converted = int(engaged * rng.uniform(0.30, 0.60))
    retained = int(converted * rng.uniform(0.40, 0.70))
    conversion_data.append({
        "cohort": m, "registered": registered, "activated": activated,
        "engaged": engaged, "converted": converted, "retained": retained,
    })
conversion_df = pd.DataFrame(conversion_data)
print(f"  - 전환 퍼널: {len(conversion_df)}개월")


# --------------------------------------------------------------------------
# 2.13 seller_activity.csv (300 셀러 x 90일)
# --------------------------------------------------------------------------
print("\n[2.13] 셀러 일별 활동 데이터 생성")

seller_activity_data = []
for _, seller in sellers_df.iterrows():
    seller_last = pd.to_datetime(seller["last_login"])
    for day in range(90):
        d = reference_date - timedelta(days=89 - day)
        if d > seller_last and seller["status"] in ("dormant", "churned"):
            orders_p = 0
            products_u = 0
            cs_h = 0
            revenue = 0
        else:
            activity_mult = {
                "active": 1.0, "dormant": 0.3, "churned": 0.05,
            }.get(seller["status"], 0.5)
            orders_p = int(np.clip(rng.poisson(5 * activity_mult), 0, 50))
            products_u = int(np.clip(rng.poisson(3 * activity_mult), 0, 20))
            cs_h = int(rng.integers(0, max(1, int(4 * activity_mult))))
            revenue = int(orders_p * rng.integers(15000, 80000)) if orders_p > 0 else 0

        seller_activity_data.append({
            "seller_id": seller["seller_id"], "date": d.strftime("%Y-%m-%d"),
            "orders_processed": orders_p, "products_updated": products_u,
            "cs_handled": cs_h, "revenue": revenue,
        })

seller_activity_df = pd.DataFrame(seller_activity_data)
print(f"  - 셀러 일별 활동: {len(seller_activity_df)}건")


# --------------------------------------------------------------------------
# 2.14 platform_docs.csv
# --------------------------------------------------------------------------
print("\n[2.14] 플랫폼 문서 데이터 생성")

platform_docs_df = pd.DataFrame(PLATFORM_DOCS_DATA)
print(f"  - 플랫폼 문서: {len(platform_docs_df)}건")


# --------------------------------------------------------------------------
# 2.15 ecommerce_glossary.csv
# --------------------------------------------------------------------------
print("\n[2.15] 이커머스 용어 사전 생성")

glossary_df = pd.DataFrame(ECOMMERCE_GLOSSARY)
print(f"  - 용어 사전: {len(glossary_df)}건")


# --------------------------------------------------------------------------
# 2.16 seller_products.csv
# --------------------------------------------------------------------------
print("\n[2.16] 셀러-상품 매핑 데이터 생성")

seller_products_data = []
for _, seller in sellers_df.iterrows():
    shop_id = seller["seller_id"].replace("SEL", "S")
    shop_products = products_df[products_df["shop_id"] == shop_id]
    for _, prod in shop_products.iterrows():
        seller_products_data.append({
            "seller_id": seller["seller_id"],
            "product_id": prod["product_id"],
            "stock_qty": prod["stock_qty"],
            "price": prod["price"],
            "category": prod["category"],
            "status": prod["status"],
        })

seller_products_df = pd.DataFrame(seller_products_data)
print(f"  - 셀러 상품: {len(seller_products_df)}건")


# --------------------------------------------------------------------------
# 2.17 seller_resources.csv
# --------------------------------------------------------------------------
print("\n[2.17] 셀러 리소스 데이터 생성")

PLAN_QUOTA = {"Basic": 5, "Standard": 20, "Premium": 50, "Enterprise": 200}
seller_resources_data = []
for _, seller in sellers_df.iterrows():
    quota = PLAN_QUOTA.get(seller["plan_tier"], 10)
    used = round(float(np.clip(rng.uniform(0.1, 1.0) * quota, 0.1, quota * 0.95)), 2)
    seller_resources_data.append({
        "seller_id": seller["seller_id"],
        "plan_quota_gb": quota,
        "storage_used_gb": used,
        "api_calls_monthly": int(rng.integers(100, 50000)),
        "marketing_budget": int(rng.integers(0, 5000000)),
        "ad_spend": int(rng.integers(0, 3000000)),
    })

seller_resources_df = pd.DataFrame(seller_resources_data)
print(f"  - 셀러 리소스: {len(seller_resources_df)}건")

print("\n" + "=" * 70)
print("PART 2 완료: 모든 데이터 생성 완료 (18개 CSV)")
print("=" * 70)


# ============================================================================
# PART 3: 모델 학습 (10개 ML 모델)
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: 모델 학습 (10개)")
print("=" * 70)


# --------------------------------------------------------------------------
# 3.1 셀러 이탈 예측 (RandomForest + SHAP)
# --------------------------------------------------------------------------
print("\n[3.1] 셀러 이탈 예측 모델 (RandomForest + SHAP)")

CHURN_FEATURES = [
    "total_orders", "total_revenue", "product_count", "cs_tickets",
    "refund_rate", "avg_response_time", "days_since_last_login",
    "days_since_register", "plan_tier_encoded",
]
CHURN_FEATURE_NAMES_KR = {
    "total_orders": "총 주문수", "total_revenue": "총 매출",
    "product_count": "상품 수", "cs_tickets": "CS 문의 수",
    "refund_rate": "환불률", "avg_response_time": "평균 응답 시간",
    "days_since_last_login": "마지막 로그인 경과일",
    "days_since_register": "가입 후 경과일",
    "plan_tier_encoded": "플랜 등급",
}

is_churned = (sellers_df["status"] == "churned").astype(int).values
seller_analytics_df["is_churned"] = is_churned

X_churn = seller_analytics_df[CHURN_FEATURES].fillna(0).copy()
y_churn = seller_analytics_df["is_churned"].copy()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn,
)

churn_params = {
    "n_estimators": 200, "max_depth": 8, "min_samples_split": 5,
    "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42,
}
rf_churn = RandomForestClassifier(**churn_params, n_jobs=-1)
rf_churn.fit(X_train_c, y_train_c)

y_pred_c = rf_churn.predict(X_test_c)
acc_churn = accuracy_score(y_test_c, y_pred_c)
f1_churn = f1_score(y_test_c, y_pred_c, zero_division=0)
print(f"  정확도: {acc_churn:.4f}, F1: {f1_churn:.4f}")

feature_importances_churn = dict(zip(CHURN_FEATURES, rf_churn.feature_importances_))

# SHAP
shap_explainer = None
if SHAP_AVAILABLE:
    try:
        shap_explainer = shap.TreeExplainer(rf_churn)
        shap_values_raw = shap_explainer.shap_values(X_churn)
        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            shap_vals = shap_values_raw[1]
        elif hasattr(shap_values_raw, "values"):
            shap_vals = shap_values_raw.values
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            shap_vals = shap_values_raw[:, :, 1]
        else:
            shap_vals = np.array(shap_values_raw)
        for i, feat in enumerate(CHURN_FEATURES):
            seller_analytics_df[f"shap_{feat}"] = shap_vals[:, i]
        print("  SHAP 분석 완료")
    except Exception as e:
        print(f"  SHAP 분석 오류: {e}")
        SHAP_AVAILABLE = False

# 이탈 확률 기록
churn_proba = rf_churn.predict_proba(X_churn)[:, 1]
seller_analytics_df["churn_probability"] = np.round(churn_proba, 4)
seller_analytics_df["churn_risk"] = (churn_proba >= 0.5).astype(int)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="seller_churn_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.log_params(churn_params)
        mlflow.log_metrics({"accuracy": acc_churn, "f1_score": f1_churn})
        mlflow.sklearn.log_model(rf_churn, "model", registered_model_name="셀러이탈예측")


# --------------------------------------------------------------------------
# 3.2 이상거래 탐지 (Isolation Forest)
# --------------------------------------------------------------------------
print("\n[3.2] 이상거래 탐지 모델 (Isolation Forest)")

fraud_features = [
    "total_orders", "total_revenue", "product_count", "cs_tickets",
    "refund_rate", "avg_response_time",
]
X_fraud = seller_analytics_df[fraud_features].fillna(0).copy()
scaler_fraud = StandardScaler()
X_fraud_scaled = scaler_fraud.fit_transform(X_fraud)

fraud_params = {"n_estimators": 150, "contamination": 0.05, "random_state": 42}
iso_forest = IsolationForest(**fraud_params)
fraud_pred = iso_forest.fit_predict(X_fraud_scaled)
fraud_scores = iso_forest.decision_function(X_fraud_scaled)

fraud_count = int((fraud_pred == -1).sum())
print(f"  이상 셀러: {fraud_count}명 ({fraud_count/len(fraud_pred)*100:.1f}%)")

seller_analytics_df["is_anomaly"] = (fraud_pred == -1).astype(int)
raw_sc = -fraud_scores
norm_sc = (raw_sc - raw_sc.min()) / (raw_sc.max() - raw_sc.min() + 1e-8)
seller_analytics_df["anomaly_score"] = np.round(norm_sc, 4)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="fraud_detection_model"):
        mlflow.set_tag("model_type", "anomaly_detection")
        mlflow.log_params(fraud_params)
        mlflow.log_metrics({
            "anomaly_count": fraud_count,
            "anomaly_ratio": fraud_count / len(fraud_pred),
        })
        mlflow.sklearn.log_model(iso_forest, "model", registered_model_name="이상거래탐지")


# --------------------------------------------------------------------------
# 3.3 문의 자동 분류 (TF-IDF + RandomForest)
# --------------------------------------------------------------------------
print("\n[3.3] 문의 자동 분류 모델 (TF-IDF + RandomForest)")

inquiry_texts = []
inquiry_labels = []
for cat, templates in CS_INQUIRY_TEMPLATES.items():
    for tpl in templates:
        for _ in range(20):
            text = tpl.replace("{order_id}", f"ORD{rng.integers(10000, 99999)}")
            noise_words = rng.choice(
                ["요", "요.", "합니다.", "입니다.", "주세요.", ""], size=1
            )[0]
            inquiry_texts.append(text + " " + noise_words)
            inquiry_labels.append(cat)

tfidf_inquiry = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_inquiry = tfidf_inquiry.fit_transform(inquiry_texts)
le_inquiry_cat = LabelEncoder()
y_inquiry = le_inquiry_cat.fit_transform(inquiry_labels)

X_tr_inq, X_te_inq, y_tr_inq, y_te_inq = train_test_split(
    X_inquiry, y_inquiry, test_size=0.2, random_state=42, stratify=y_inquiry,
)

inq_params = {
    "n_estimators": 150, "max_depth": 10,
    "random_state": 42, "class_weight": "balanced",
}
rf_inquiry = RandomForestClassifier(**inq_params, n_jobs=-1)
rf_inquiry.fit(X_tr_inq, y_tr_inq)

y_pred_inq = rf_inquiry.predict(X_te_inq)
acc_inq = accuracy_score(y_te_inq, y_pred_inq)
f1_inq = f1_score(y_te_inq, y_pred_inq, average="macro")
print(f"  정확도: {acc_inq:.4f}, F1(매크로): {f1_inq:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="inquiry_classification_model"):
        mlflow.set_tag("model_type", "text_classification")
        mlflow.log_params(inq_params)
        mlflow.log_metrics({"accuracy": acc_inq, "f1_macro": f1_inq})
        mlflow.sklearn.log_model(rf_inquiry, "model", registered_model_name="문의자동분류")


# --------------------------------------------------------------------------
# 3.4 셀러 세그먼트 (K-Means)
# --------------------------------------------------------------------------
print("\n[3.4] 셀러 세그먼트 모델 (K-Means)")

segment_features = [
    "total_orders", "total_revenue", "product_count",
    "cs_tickets", "refund_rate", "avg_response_time",
]
X_seg = seller_analytics_df[segment_features].fillna(0).copy()
scaler_cluster = StandardScaler()
X_seg_scaled = scaler_cluster.fit_transform(X_seg)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_seg_scaled)
sil_score = silhouette_score(X_seg_scaled, cluster_labels)
print(f"  실루엣 점수: {sil_score:.4f}")

seller_analytics_df["cluster"] = cluster_labels

# 센트로이드 매출 기준으로 세그먼트 이름 자동 매핑
centroid_revenue = []
for c in range(n_clusters):
    mask = seller_analytics_df["cluster"] == c
    avg_rev = seller_analytics_df.loc[mask, "total_revenue"].mean() if mask.any() else 0
    avg_orders = seller_analytics_df.loc[mask, "total_orders"].mean() if mask.any() else 0
    centroid_revenue.append((c, avg_rev, avg_orders))

# 매출 내림차순 정렬
centroid_revenue.sort(key=lambda x: x[1], reverse=True)

# 매출 순서에 따라 이름 부여
ORDERED_NAMES = ["파워 셀러", "우수 셀러", "성장형 셀러", "관리 필요 셀러", "휴면 셀러"]
SEGMENT_NAMES = {}
for rank, (cid, avg_rev, avg_ord) in enumerate(centroid_revenue):
    SEGMENT_NAMES[cid] = ORDERED_NAMES[rank]
    print(f"  클러스터 {cid}: {ORDERED_NAMES[rank]} (평균 매출: {avg_rev:,.0f}, 평균 주문: {avg_ord:,.0f})")

# CSV에 세그먼트 이름 저장
seller_analytics_df["segment_name"] = seller_analytics_df["cluster"].map(SEGMENT_NAMES)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="seller_segmentation_model"):
        mlflow.set_tag("model_type", "clustering")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metrics({"silhouette_score": sil_score})
        mlflow.sklearn.log_model(kmeans, "model", registered_model_name="셀러세그먼트")


# --------------------------------------------------------------------------
# 3.5 매출 예측 (LightGBM / GradientBoosting)
# --------------------------------------------------------------------------
print("\n[3.5] 매출 예측 모델 (LightGBM / GradientBoosting)")

revenue_data = []
for _, seller in sellers_df.iterrows():
    sa_row = seller_analytics_df[
        seller_analytics_df["seller_id"] == seller["seller_id"]
    ].iloc[0]
    total_rev = seller["total_revenue"]
    txn_count = seller["total_orders"]
    unique_customers = max(1, int(txn_count * rng.uniform(0.4, 0.9)))
    aov = total_rev / max(1, txn_count)
    growth_rate = round(float(rng.normal(0.05, 0.15)), 4)

    cat = shops_df[
        shops_df["shop_id"] == seller["seller_id"].replace("SEL", "S")
    ]["category"].values
    category_name = cat[0] if len(cat) > 0 else "기타"
    industry_enc = CATEGORIES_KO.index(category_name) if category_name in CATEGORIES_KO else 0

    reg = shops_df[
        shops_df["shop_id"] == seller["seller_id"].replace("SEL", "S")
    ]["region"].values
    region_name = reg[0] if len(reg) > 0 else "서울"
    region_enc = REGIONS.index(region_name) if region_name in REGIONS else 0

    target_rev = int(total_rev * (1 + growth_rate) + rng.normal(0, total_rev * 0.1))
    target_rev = max(0, target_rev)

    revenue_data.append({
        "total_revenue": total_rev, "txn_count": txn_count,
        "unique_customers": unique_customers, "avg_order_value": round(aov, 0),
        "revenue_growth_rate": growth_rate, "industry_encoded": industry_enc,
        "region_encoded": region_enc, "target_revenue_next": target_rev,
    })

revenue_df = pd.DataFrame(revenue_data)
X_rev = revenue_df.drop(columns=["target_revenue_next"]).copy()
y_rev = revenue_df["target_revenue_next"].copy()

X_tr_rev, X_te_rev, y_tr_rev, y_te_rev = train_test_split(
    X_rev, y_rev, test_size=0.2, random_state=42,
)

if LIGHTGBM_AVAILABLE:
    model_revenue = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1,
    )
    algo_name_rev = "LightGBM"
else:
    model_revenue = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42,
    )
    algo_name_rev = "GradientBoosting"

model_revenue.fit(X_tr_rev, y_tr_rev)
y_pred_rev = model_revenue.predict(X_te_rev)
mae_rev = mean_absolute_error(y_te_rev, y_pred_rev)
r2_rev = r2_score(y_te_rev, y_pred_rev)
print(f"  알고리즘: {algo_name_rev}")
print(f"  MAE: {mae_rev:,.0f}, R2: {r2_rev:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="revenue_prediction_model"):
        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("algorithm", algo_name_rev)
        mlflow.log_metrics({"mae": mae_rev, "r2": r2_rev})
        mlflow.sklearn.log_model(model_revenue, "model", registered_model_name="매출예측")


# --------------------------------------------------------------------------
# 3.6 CS 응답 품질 (RandomForest Classifier)
# --------------------------------------------------------------------------
print("\n[3.6] CS 응답 품질 모델 (RandomForest Classifier)")

cs_quality_data = []
TICKET_CATS = ["배송", "환불", "결제", "상품", "계정", "정산", "기술지원"]
SELLER_TIERS_CS = ["Basic", "Standard", "Premium", "Enterprise"]
PRIORITIES = ["urgent", "high", "normal", "low"]
PRIORITY_WEIGHTS = [0.10, 0.20, 0.45, 0.25]

for _ in range(2000):
    t_cat = rng.choice(TICKET_CATS)
    s_tier = rng.choice(SELLER_TIERS_CS)
    sentiment = round(float(rng.uniform(-1, 1)), 3)
    order_val = int(rng.lognormal(10.5, 0.8))
    is_repeat = int(rng.random() < 0.25)
    text_len = int(rng.integers(20, 500))

    base_idx = TICKET_CATS.index(t_cat)
    tier_idx = SELLER_TIERS_CS.index(s_tier)

    if t_cat in ("환불", "결제") and sentiment < -0.3:
        priority = rng.choice(["urgent", "high"], p=[0.5, 0.5])
    elif is_repeat:
        priority = rng.choice(["urgent", "high", "normal"], p=[0.3, 0.4, 0.3])
    else:
        priority = rng.choice(PRIORITIES, p=PRIORITY_WEIGHTS)

    cs_quality_data.append({
        "ticket_category_encoded": base_idx,
        "seller_tier_encoded": tier_idx,
        "sentiment_score": sentiment,
        "order_value": order_val,
        "is_repeat_issue": is_repeat,
        "text_length": text_len,
        "priority": priority,
    })

cs_quality_df = pd.DataFrame(cs_quality_data)
le_ticket_cat = LabelEncoder()
le_seller_tier = LabelEncoder()
le_cs_priority = LabelEncoder()

le_ticket_cat.fit(TICKET_CATS)
le_seller_tier.fit(SELLER_TIERS_CS)
le_cs_priority.fit(PRIORITIES)

cs_feat_cols = [
    "ticket_category_encoded", "seller_tier_encoded", "sentiment_score",
    "order_value", "is_repeat_issue", "text_length",
]
X_cs = cs_quality_df[cs_feat_cols].copy()
y_cs = le_cs_priority.transform(cs_quality_df["priority"])

X_tr_cs, X_te_cs, y_tr_cs, y_te_cs = train_test_split(
    X_cs, y_cs, test_size=0.2, random_state=42, stratify=y_cs,
)

cs_params = {
    "n_estimators": 150, "max_depth": 10,
    "random_state": 42, "class_weight": "balanced",
}
rf_cs = RandomForestClassifier(**cs_params, n_jobs=-1)
rf_cs.fit(X_tr_cs, y_tr_cs)

y_pred_cs = rf_cs.predict(X_te_cs)
acc_cs = accuracy_score(y_te_cs, y_pred_cs)
f1_cs = f1_score(y_te_cs, y_pred_cs, average="macro")
print(f"  정확도: {acc_cs:.4f}, F1(매크로): {f1_cs:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="cs_quality_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.log_params(cs_params)
        mlflow.log_metrics({"accuracy": acc_cs, "f1_macro": f1_cs})
        mlflow.sklearn.log_model(rf_cs, "model", registered_model_name="CS응답품질")


# --------------------------------------------------------------------------
# 3.7 고객 LTV 예측 (GradientBoosting Regressor)
# --------------------------------------------------------------------------
print("\n[3.7] 고객 LTV 예측 모델 (GradientBoosting Regressor)")

n_customers = 3000
ltv_data = []
for i in range(n_customers):
    purchase_count = int(np.clip(rng.poisson(5), 1, 50))
    avg_ov = int(rng.lognormal(10.2, 0.6))
    total_purchases = purchase_count * avg_ov
    days_reg = int(rng.integers(30, 730))
    return_rate = round(float(rng.beta(2, 15)), 4)
    recency = int(rng.integers(0, min(days_reg, 180)))

    ltv = total_purchases * (1 + 0.3 * np.log1p(purchase_count)) * (1 - return_rate * 0.5)
    ltv *= rng.uniform(0.8, 1.2)
    ltv = max(0, int(ltv))

    ltv_data.append({
        "total_purchases": total_purchases,
        "purchase_count": purchase_count,
        "avg_order_value": avg_ov,
        "days_since_register": days_reg,
        "return_rate": return_rate,
        "recency_days": recency,
        "ltv": ltv,
    })

ltv_df = pd.DataFrame(ltv_data)
X_ltv = ltv_df.drop(columns=["ltv"]).copy()
y_ltv = ltv_df["ltv"].copy()

X_tr_ltv, X_te_ltv, y_tr_ltv, y_te_ltv = train_test_split(
    X_ltv, y_ltv, test_size=0.2, random_state=42,
)

model_ltv = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
)
model_ltv.fit(X_tr_ltv, y_tr_ltv)

y_pred_ltv = model_ltv.predict(X_te_ltv)
mae_ltv = mean_absolute_error(y_te_ltv, y_pred_ltv)
r2_ltv = r2_score(y_te_ltv, y_pred_ltv)
print(f"  MAE: {mae_ltv:,.0f}, R2: {r2_ltv:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="customer_ltv_model"):
        mlflow.set_tag("model_type", "regression")
        mlflow.log_metrics({"mae": mae_ltv, "r2": r2_ltv})
        mlflow.sklearn.log_model(model_ltv, "model", registered_model_name="고객LTV예측")


# --------------------------------------------------------------------------
# 3.8 리뷰 감성 분석 (TF-IDF + LogisticRegression)
# --------------------------------------------------------------------------
print("\n[3.8] 리뷰 감성 분석 모델 (TF-IDF + LogisticRegression)")

review_texts = []
review_labels = []
for tpl in REVIEW_TEMPLATES_POSITIVE:
    for _ in range(30):
        noise = rng.choice(
            ["", " 감사합니다!", " 좋아요!", " 강추!", " 최고에요!"],
            p=[0.3, 0.2, 0.2, 0.15, 0.15],
        )
        review_texts.append(tpl + noise)
        review_labels.append("positive")

for tpl in REVIEW_TEMPLATES_NEGATIVE:
    for _ in range(30):
        noise = rng.choice(
            ["", " 실망이에요.", " 별로에요.", " 비추입니다.", ""],
            p=[0.3, 0.2, 0.2, 0.15, 0.15],
        )
        review_texts.append(tpl + noise)
        review_labels.append("negative")

for tpl in REVIEW_TEMPLATES_NEUTRAL:
    for _ in range(30):
        noise = rng.choice(
            ["", " 그래요.", " 보통이에요.", ""],
            p=[0.3, 0.25, 0.25, 0.2],
        )
        review_texts.append(tpl + noise)
        review_labels.append("neutral")

tfidf_sentiment = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_sent = tfidf_sentiment.fit_transform(review_texts)
le_sentiment = LabelEncoder()
y_sent = le_sentiment.fit_transform(review_labels)

X_tr_sent, X_te_sent, y_tr_sent, y_te_sent = train_test_split(
    X_sent, y_sent, test_size=0.2, random_state=42, stratify=y_sent,
)

model_sentiment = LogisticRegression(
    max_iter=1000, random_state=42, class_weight="balanced",
)
model_sentiment.fit(X_tr_sent, y_tr_sent)

y_pred_sent = model_sentiment.predict(X_te_sent)
acc_sent = accuracy_score(y_te_sent, y_pred_sent)
f1_sent = f1_score(y_te_sent, y_pred_sent, average="macro")
print(f"  정확도: {acc_sent:.4f}, F1(매크로): {f1_sent:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="review_sentiment_model"):
        mlflow.set_tag("model_type", "text_classification")
        mlflow.log_metrics({"accuracy": acc_sent, "f1_macro": f1_sent})
        mlflow.sklearn.log_model(
            model_sentiment, "model", registered_model_name="리뷰감성분석",
        )


# --------------------------------------------------------------------------
# 3.9 상품 수요 예측 (XGBoost / GradientBoosting)
# --------------------------------------------------------------------------
print("\n[3.9] 상품 수요 예측 모델 (XGBoost / GradientBoosting)")

demand_data = []
for _ in range(2000):
    w1 = int(rng.poisson(15))
    w2 = int(rng.poisson(max(1, w1 + rng.integers(-5, 5))))
    w3 = int(rng.poisson(max(1, w2 + rng.integers(-5, 5))))
    w4 = int(rng.poisson(max(1, w3 + rng.integers(-5, 5))))
    price = int(rng.lognormal(10, 0.6))
    cat_enc = int(rng.integers(0, len(CATEGORIES_KO)))
    is_promo = int(rng.random() < 0.2)
    review_count = int(rng.integers(0, 200))

    trend = (w4 - w1) / max(1, w1)
    base_demand = (w1 + w2 + w3 + w4) / 4
    next_demand = int(max(0, base_demand * (1 + trend * 0.3 + is_promo * 0.2) + rng.normal(0, 3)))

    demand_data.append({
        "week1_orders": w1, "week2_orders": w2,
        "week3_orders": w3, "week4_orders": w4,
        "price": price, "category_encoded": cat_enc,
        "is_promotion": is_promo, "review_count": review_count,
        "next_week_demand": next_demand,
    })

demand_df = pd.DataFrame(demand_data)
X_demand = demand_df.drop(columns=["next_week_demand"]).copy()
y_demand = demand_df["next_week_demand"].copy()

X_tr_dem, X_te_dem, y_tr_dem, y_te_dem = train_test_split(
    X_demand, y_demand, test_size=0.2, random_state=42,
)

if XGBOOST_AVAILABLE:
    model_demand = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, verbosity=0,
    )
    algo_name_dem = "XGBoost"
else:
    model_demand = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    algo_name_dem = "GradientBoosting"

model_demand.fit(X_tr_dem, y_tr_dem)
y_pred_dem = model_demand.predict(X_te_dem)
mae_dem = mean_absolute_error(y_te_dem, y_pred_dem)
r2_dem = r2_score(y_te_dem, y_pred_dem)
print(f"  알고리즘: {algo_name_dem}")
print(f"  MAE: {mae_dem:.2f}, R2: {r2_dem:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="demand_forecast_model"):
        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("algorithm", algo_name_dem)
        mlflow.log_metrics({"mae": mae_dem, "r2": r2_dem})
        mlflow.sklearn.log_model(
            model_demand, "model", registered_model_name="상품수요예측",
        )


# --------------------------------------------------------------------------
# 3.10 정산 이상 탐지 (DBSCAN)
# --------------------------------------------------------------------------
print("\n[3.10] 정산 이상 탐지 모델 (DBSCAN)")

# 셀러당 3건 고정 (데모용 - 총 ~950건)
total_n = len(sellers_df) * 3
seller_ids_repeat = np.repeat(sellers_df["seller_id"].values, 3)

normal_df = pd.DataFrame({
    "seller_id": seller_ids_repeat,
    "amount": rng.lognormal(10.5, 0.7, size=total_n).astype(int),
    "fee_rate": np.round(rng.uniform(0.02, 0.04, size=total_n), 4),
    "settlement_days": np.round(np.clip(rng.exponential(3, size=total_n) + 1, 1, 14), 1),
    "txn_count": rng.integers(1, 20, size=total_n),
    "refund_ratio": np.round(rng.beta(1, 15, size=total_n), 4),
})

# 이상 데이터 50건 고정
anomaly_df = pd.DataFrame({
    "seller_id": rng.choice(sellers_df["seller_id"].values, size=50),
    "amount": rng.integers(500000, 5000000, size=50),
    "fee_rate": np.round(rng.uniform(0.08, 0.15, size=50), 4),
    "settlement_days": np.round(rng.uniform(10, 30, size=50), 1),
    "txn_count": rng.integers(50, 200, size=50),
    "refund_ratio": np.round(rng.uniform(0.3, 0.8, size=50), 4),
})
settlement_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
settle_features = ["amount", "fee_rate", "settlement_days", "txn_count", "refund_ratio"]
X_settle = settlement_df[settle_features].copy()
scaler_settle = StandardScaler()
X_settle_scaled = scaler_settle.fit_transform(X_settle)

dbscan = DBSCAN(eps=1.5, min_samples=5)
settle_labels = dbscan.fit_predict(X_settle_scaled)

n_noise = int((settle_labels == -1).sum())
n_clusters_db = len(set(settle_labels)) - (1 if -1 in settle_labels else 0)
print(f"  클러스터 수: {n_clusters_db}, 이상(noise): {n_noise}건 ({n_noise/len(settle_labels)*100:.1f}%)")

settlement_df["cluster_label"] = settle_labels
settlement_df["is_anomaly"] = (settle_labels == -1).astype(int)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="settlement_anomaly_model"):
        mlflow.set_tag("model_type", "anomaly_detection")
        mlflow.log_param("eps", 1.5)
        mlflow.log_param("min_samples", 5)
        mlflow.log_metrics({"n_clusters": n_clusters_db, "n_noise": n_noise})
        mlflow.sklearn.log_model(dbscan, "model", registered_model_name="정산이상탐지")

print("\n" + "=" * 70)
print("PART 3 완료: 모든 모델 학습 완료 (10개)")
print("=" * 70)


# ============================================================================
# PART 4: 저장 및 테스트
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: 저장 및 테스트")
print("=" * 70)

# --------------------------------------------------------------------------
# 4.1 CSV 저장 (17개)
# --------------------------------------------------------------------------
print("\n[4.1] CSV 파일 저장 (17개)")

csv_enc = "utf-8-sig"
shops_df.to_csv(BACKEND_DIR / "shops.csv", index=False, encoding=csv_enc)
categories_df.to_csv(BACKEND_DIR / "categories.csv", index=False, encoding=csv_enc)
services_df.to_csv(BACKEND_DIR / "services.csv", index=False, encoding=csv_enc)
products_df.to_csv(BACKEND_DIR / "products.csv", index=False, encoding=csv_enc)
sellers_df.to_csv(BACKEND_DIR / "sellers.csv", index=False, encoding=csv_enc)
op_logs_df.to_csv(BACKEND_DIR / "operation_logs.csv", index=False, encoding=csv_enc)
seller_analytics_df.to_csv(BACKEND_DIR / "seller_analytics.csv", index=False, encoding=csv_enc)
shop_perf_df.to_csv(BACKEND_DIR / "shop_performance.csv", index=False, encoding=csv_enc)
daily_metrics_df.to_csv(BACKEND_DIR / "daily_metrics.csv", index=False, encoding=csv_enc)
cs_stats_df.to_csv(BACKEND_DIR / "cs_stats.csv", index=False, encoding=csv_enc)
fraud_df.to_csv(BACKEND_DIR / "fraud_details.csv", index=False, encoding=csv_enc)
cohort_df.to_csv(BACKEND_DIR / "cohort_retention.csv", index=False, encoding=csv_enc)
conversion_df.to_csv(BACKEND_DIR / "conversion_funnel.csv", index=False, encoding=csv_enc)
seller_activity_df.to_csv(BACKEND_DIR / "seller_activity.csv", index=False, encoding=csv_enc)
platform_docs_df.to_csv(BACKEND_DIR / "platform_docs.csv", index=False, encoding=csv_enc)
glossary_df.to_csv(BACKEND_DIR / "ecommerce_glossary.csv", index=False, encoding=csv_enc)
seller_products_df.to_csv(BACKEND_DIR / "seller_products.csv", index=False, encoding=csv_enc)
seller_resources_df.to_csv(BACKEND_DIR / "seller_resources.csv", index=False, encoding=csv_enc)
print("  18개 CSV 파일 저장 완료")


# --------------------------------------------------------------------------
# 4.2 모델 저장 (10개 + 보조 파일)
# --------------------------------------------------------------------------
print("\n[4.2] 모델 파일 저장")

# 1. 셀러 이탈 예측
joblib.dump(rf_churn, BACKEND_DIR / "model_seller_churn.pkl")
if SHAP_AVAILABLE and shap_explainer is not None:
    joblib.dump(shap_explainer, BACKEND_DIR / "shap_explainer_churn.pkl")
    print("  - shap_explainer_churn.pkl 저장 완료")

churn_config = {
    "features": CHURN_FEATURES,
    "feature_names_kr": CHURN_FEATURE_NAMES_KR,
    "feature_importances": {k: float(v) for k, v in feature_importances_churn.items()},
    "shap_available": SHAP_AVAILABLE,
    "model_accuracy": float(acc_churn),
    "model_f1": float(f1_churn),
}
with open(BACKEND_DIR / "churn_model_config.json", "w", encoding="utf-8") as f:
    json.dump(churn_config, f, ensure_ascii=False, indent=2)

# 2. 이상거래 탐지
joblib.dump(iso_forest, BACKEND_DIR / "model_fraud_detection.pkl")

# 3. 문의 자동 분류
joblib.dump(rf_inquiry, BACKEND_DIR / "model_inquiry_classification.pkl")
joblib.dump(tfidf_inquiry, BACKEND_DIR / "tfidf_vectorizer.pkl")
joblib.dump(le_inquiry_cat, BACKEND_DIR / "le_inquiry_category.pkl")

# 4. 셀러 세그먼트
joblib.dump(kmeans, BACKEND_DIR / "model_seller_segment.pkl")
joblib.dump(scaler_cluster, BACKEND_DIR / "scaler_cluster.pkl")

# 5. 매출 예측
joblib.dump(model_revenue, BACKEND_DIR / "model_revenue_prediction.pkl")
revenue_config = {
    "algorithm": algo_name_rev,
    "features": list(X_rev.columns),
    "mae": float(mae_rev),
    "r2_score": float(r2_rev),
}
with open(BACKEND_DIR / "revenue_model_config.json", "w", encoding="utf-8") as f:
    json.dump(revenue_config, f, ensure_ascii=False, indent=2)

# 6. CS 응답 품질
joblib.dump(rf_cs, BACKEND_DIR / "model_cs_quality.pkl")
joblib.dump(le_ticket_cat, BACKEND_DIR / "le_ticket_category.pkl")
joblib.dump(le_seller_tier, BACKEND_DIR / "le_seller_tier.pkl")
joblib.dump(le_cs_priority, BACKEND_DIR / "le_cs_priority.pkl")

# 7. 고객 LTV
joblib.dump(model_ltv, BACKEND_DIR / "model_customer_ltv.pkl")

# 8. 리뷰 감성 분석
joblib.dump(model_sentiment, BACKEND_DIR / "model_review_sentiment.pkl")
joblib.dump(tfidf_sentiment, BACKEND_DIR / "tfidf_vectorizer_sentiment.pkl")

# 9. 상품 수요 예측
joblib.dump(model_demand, BACKEND_DIR / "model_demand_forecast.pkl")

# 10. 정산 이상 탐지
joblib.dump(dbscan, BACKEND_DIR / "model_settlement_anomaly.pkl")

print("  10개 모델 + 보조 파일 저장 완료")


# --------------------------------------------------------------------------
# 4.2.1 Guardian 감사 로그 이상탐지 (Isolation Forest) 학습
# --------------------------------------------------------------------------
print("\n[4.2.1] Guardian 감사 로그 IsolationForest 학습")

import sqlite3

GUARDIAN_DB_PATH = BACKEND_DIR / "guardian.db"
CORE_TABLES = {"orders", "payments", "users", "products", "shipments"}

# (A) 감사 로그 DB 준비 (없으면 생성 + 시드 데이터)
def _prepare_guardian_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        user_id TEXT NOT NULL,
        action TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_count INTEGER DEFAULT 0,
        affected_amount REAL DEFAULT 0,
        status TEXT DEFAULT 'executed',
        risk_level TEXT DEFAULT 'LOW',
        agent_reason TEXT DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        table_name TEXT NOT NULL,
        row_count INTEGER,
        was_mistake INTEGER DEFAULT 0,
        description TEXT
    )""")

    existing = c.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    if existing < 200:
        from datetime import datetime, timedelta
        _now = datetime.now()
        _users = ["kim", "park", "lee", "choi", "jung"]
        _tables = ["orders", "payments", "users", "products", "shipments", "logs", "temp_reports"]
        for _ in range(300):
            ts = (_now - timedelta(days=int(rng.integers(0, 60)), hours=int(rng.integers(0, 24)))).isoformat()
            u = rng.choice(_users)
            act = rng.choice(["INSERT", "UPDATE", "DELETE", "SELECT"])
            tbl = rng.choice(_tables)
            rc = int(rng.integers(1, 8)) if act == "DELETE" else int(rng.integers(1, 30))
            amt = rc * int(rng.integers(30000, 120000)) if tbl in ("orders", "payments") else 0
            c.execute(
                "INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, u, act, tbl, rc, amt, "executed", "LOW")
            )
        # 이상 패턴 데이터 (야간 대량 DELETE 등)
        for _ in range(20):
            ts = (_now - timedelta(days=int(rng.integers(0, 30)))).replace(hour=int(rng.integers(22, 24))).isoformat()
            u = rng.choice(["unknown_admin", "temp_user", rng.choice(_users)])
            tbl = rng.choice(["orders", "payments", "users"])
            rc = int(rng.integers(100, 5000))
            amt = rc * int(rng.integers(50000, 150000))
            c.execute(
                "INSERT INTO audit_log (timestamp,user_id,action,table_name,row_count,affected_amount,status,risk_level) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, u, "DELETE", tbl, rc, amt, "executed", "HIGH")
            )
        # 과거 사건 데이터
        _incidents = [
            ("DELETE", "orders", 250, 1, "신입 직원이 WHERE 없이 DELETE 실행, 전체 복구"),
            ("DELETE", "orders", 180, 1, "테스트 DB와 혼동하여 프로덕션에서 삭제"),
            ("DELETE", "payments", 320, 1, "정산 데이터 삭제 실수, DBA가 백업에서 복구"),
            ("DELETE", "orders", 150, 1, "퇴근 전 급하게 작업하다 실수"),
            ("DELETE", "users", 500, 1, "탈퇴 처리 스크립트 오류로 활성 유저 삭제"),
            ("DELETE", "products", 200, 1, "카테고리 정리 중 실수로 전체 삭제"),
            ("DELETE", "orders", 400, 1, "연말 정산 중 데이터 혼동"),
            ("DELETE", "logs", 10000, 0, "정기 로그 정리 (스케줄 작업)"),
            ("DELETE", "temp_reports", 5000, 0, "임시 리포트 정리"),
            ("UPDATE", "orders", 300, 1, "금액 필드 일괄 0으로 업데이트 실수"),
            ("UPDATE", "products", 150, 1, "가격 일괄 변경 시 WHERE 조건 누락"),
            ("UPDATE", "users", 1000, 1, "권한 일괄 변경 실수"),
        ]
        for act, tbl, rc, mis, desc in _incidents:
            c.execute("INSERT INTO incidents (action,table_name,row_count,was_mistake,description) VALUES (?,?,?,?,?)",
                      (act, tbl, rc, mis, desc))
        conn.commit()
        print(f"  guardian.db 시드 데이터 생성 완료 (audit_log: {c.execute('SELECT COUNT(*) FROM audit_log').fetchone()[0]}건)")
    else:
        print(f"  guardian.db 기존 데이터 사용 ({existing}건)")
    return conn

guardian_conn = _prepare_guardian_db(GUARDIAN_DB_PATH)

# (B) 학습 데이터 로드
rows = guardian_conn.execute(
    "SELECT user_id, action, table_name, row_count, affected_amount, timestamp "
    "FROM audit_log WHERE status='executed'"
).fetchall()
guardian_conn.close()

ACTION_MAP = {"INSERT": 0, "SELECT": 0, "UPDATE": 1, "DELETE": 2,
              "ALTER": 3, "DROP": 4, "TRUNCATE": 4}
guardian_features = []
for r in rows:
    ts = r["timestamp"] or ""
    hour = int(ts[11:13]) if len(ts) > 13 else 12
    guardian_features.append([
        ACTION_MAP.get(r["action"], 0),
        1 if r["table_name"] in CORE_TABLES else 0,
        r["row_count"],
        np.log1p(r["row_count"]),
        r["affected_amount"],
        hour,
        1 if (hour >= 22 or hour < 6) else 0,
    ])

X_guardian = np.array(guardian_features)
scaler_guardian = StandardScaler()
X_guardian_scaled = scaler_guardian.fit_transform(X_guardian)

# (C) IsolationForest 학습
guardian_iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
guardian_iso.fit(X_guardian_scaled)

# 학습 결과 확인
guardian_pred = guardian_iso.predict(X_guardian_scaled)
guardian_anomaly_count = int((guardian_pred == -1).sum())
print(f"  학습 데이터: {len(rows)}건, 7 features")
print(f"  이상 탐지: {guardian_anomaly_count}건 ({guardian_anomaly_count/len(rows)*100:.1f}%)")

# (D) 모델 저장
joblib.dump(guardian_iso, BACKEND_DIR / "model_guardian_anomaly.pkl")
joblib.dump(scaler_guardian, BACKEND_DIR / "scaler_guardian.pkl")
print("  model_guardian_anomaly.pkl, scaler_guardian.pkl 저장 완료")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="guardian_anomaly_model"):
        mlflow.set_tag("model_type", "anomaly_detection")
        mlflow.log_params({"n_estimators": 100, "contamination": 0.05, "random_state": 42})
        mlflow.log_metrics({
            "anomaly_count": guardian_anomaly_count,
            "anomaly_ratio": guardian_anomaly_count / len(rows),
            "n_samples": len(rows),
        })
        mlflow.sklearn.log_model(guardian_iso, "model", registered_model_name="Guardian감사로그이상탐지")


# --------------------------------------------------------------------------
# 4.3 셀러별 예측 결과 사전계산 → seller_analytics.csv 에 추가
# --------------------------------------------------------------------------
print("\n[4.3] 셀러별 모델 예측 결과 사전계산")

# (A) 매출 예측: 전 셀러에 대해 model_revenue로 예측
rev_feat_cols = ["total_revenue", "txn_count", "unique_customers",
                 "avg_order_value", "revenue_growth_rate",
                 "industry_encoded", "region_encoded"]
predicted_revenues = []
revenue_growth_rates = []
for idx, seller in sellers_df.iterrows():
    sa_row = seller_analytics_df[
        seller_analytics_df["seller_id"] == seller["seller_id"]
    ]
    if len(sa_row) == 0:
        predicted_revenues.append(0)
        revenue_growth_rates.append(0.0)
        continue
    total_rev = seller["total_revenue"]
    txn_count = seller["total_orders"]
    unique_customers = max(1, int(txn_count * rng.uniform(0.4, 0.9)))
    aov = total_rev / max(1, txn_count)
    growth_rate = round(float(rng.normal(0.05, 0.15)), 4)
    cat = shops_df[
        shops_df["shop_id"] == seller["seller_id"].replace("SEL", "S")
    ]["category"].values
    category_name = cat[0] if len(cat) > 0 else "기타"
    industry_enc = CATEGORIES_KO.index(category_name) if category_name in CATEGORIES_KO else 0
    reg = shops_df[
        shops_df["shop_id"] == seller["seller_id"].replace("SEL", "S")
    ]["region"].values
    region_name = reg[0] if len(reg) > 0 else "서울"
    region_enc = REGIONS.index(region_name) if region_name in REGIONS else 0

    X_pred = pd.DataFrame([{
        "total_revenue": total_rev, "txn_count": txn_count,
        "unique_customers": unique_customers, "avg_order_value": round(aov, 0),
        "revenue_growth_rate": growth_rate, "industry_encoded": industry_enc,
        "region_encoded": region_enc,
    }])
    pred_rev = int(model_revenue.predict(X_pred)[0])
    pred_rev = max(0, pred_rev)
    predicted_revenues.append(pred_rev)
    actual_growth = round((pred_rev / max(1, total_rev) - 1) * 100, 1)
    revenue_growth_rates.append(actual_growth)

seller_analytics_df["predicted_revenue"] = predicted_revenues
seller_analytics_df["revenue_growth_rate"] = revenue_growth_rates
print(f"  매출 예측 완료: {len(predicted_revenues)}명")

# (B) CS 응답 품질: 셀러별로 대표 티켓 생성 후 모델 예측 → 우선순위 분포 기반 점수화
cs_scores = []
cs_grades = []
for _, seller in sellers_df.iterrows():
    tier = seller.get("plan_tier", "Standard")
    tier_idx = SELLER_TIERS_CS.index(tier) if tier in SELLER_TIERS_CS else 1
    refund_rate = float(seller.get("refund_rate", 0.05))
    avg_resp = float(seller.get("avg_response_time", 2.0))
    # 셀러별 대표 티켓 5개 생성하여 평균 우선순위 예측
    sample_tickets = []
    for t_idx in range(min(5, len(TICKET_CATS))):
        sentiment = -0.5 + refund_rate * 3  # 환불률 높으면 부정적
        order_val = int(seller.get("total_revenue", 0) / max(1, seller.get("total_orders", 1)))
        is_repeat = 1 if refund_rate > 0.1 else 0
        text_len = 150
        sample_tickets.append({
            "ticket_category_encoded": t_idx,
            "seller_tier_encoded": tier_idx,
            "sentiment_score": round(sentiment, 3),
            "order_value": order_val,
            "is_repeat_issue": is_repeat,
            "text_length": text_len,
        })
    X_cs_pred = pd.DataFrame(sample_tickets)
    preds = rf_cs.predict(X_cs_pred)
    probas = rf_cs.predict_proba(X_cs_pred)
    # urgent=0, high=1, normal=2, low=3 순 (le_cs_priority 기준)
    # 점수 = low 비율 * 100 + normal 비율 * 70 + high 비율 * 40 + urgent 비율 * 10
    avg_proba = probas.mean(axis=0)
    priority_classes = le_cs_priority.classes_
    score_map = {"low": 100, "normal": 70, "high": 40, "urgent": 10}
    cs_score = 0
    for i, cls in enumerate(priority_classes):
        cs_score += avg_proba[i] * score_map.get(cls, 50)
    cs_score = max(0, min(100, int(cs_score)))
    cs_scores.append(cs_score)
    cs_grades.append("우수" if cs_score >= 80 else "보통" if cs_score >= 50 else "개선필요")

seller_analytics_df["cs_quality_score"] = cs_scores
seller_analytics_df["cs_quality_grade"] = cs_grades
print(f"  CS 품질 예측 완료: {len(cs_scores)}명")

# (C) 고객 LTV 예측: 셀러별 LTV
ltv_predictions = []
for _, seller in sellers_df.iterrows():
    total_rev = seller["total_revenue"]
    total_ord = seller["total_orders"]
    aov = total_rev / max(1, total_ord)
    months_active = max(1, seller.get("days_since_register", 180) / 30)
    monthly_rev = total_rev / months_active
    X_ltv = pd.DataFrame([{
        "total_revenue": total_rev, "total_orders": total_ord,
        "avg_order_value": round(aov, 0), "months_active": round(months_active, 1),
        "monthly_revenue": round(monthly_rev, 0),
    }])
    try:
        ltv_pred = int(model_ltv.predict(X_ltv)[0])
    except Exception:
        ltv_pred = int(total_rev * 1.5)
    ltv_predictions.append(max(0, ltv_pred))

seller_analytics_df["predicted_ltv"] = ltv_predictions
print(f"  LTV 예측 완료: {len(ltv_predictions)}명")

# CSV 다시 저장 (예측 결과 포함)
seller_analytics_df.to_csv(BACKEND_DIR / "seller_analytics.csv", index=False, encoding=csv_enc)
print("  seller_analytics.csv 업데이트 완료 (예측 컬럼 추가)")


# --------------------------------------------------------------------------
# 4.4 예측 함수 테스트
# --------------------------------------------------------------------------
print("\n[4.3] 예측 함수 테스트")

# 이탈 예측 테스트
sample_churn = {f: 0 for f in CHURN_FEATURES}
sample_churn.update({
    "total_orders": 50, "total_revenue": 5000000, "product_count": 15,
    "cs_tickets": 3, "refund_rate": 0.05, "avg_response_time": 2.5,
    "days_since_last_login": 2, "days_since_register": 180, "plan_tier_encoded": 1,
})
X_sample_churn = pd.DataFrame([sample_churn])[CHURN_FEATURES]
churn_pred = rf_churn.predict(X_sample_churn)[0]
churn_prob = rf_churn.predict_proba(X_sample_churn)[0]
print(f"  셀러 이탈 예측: {'이탈' if churn_pred else '유지'} (확률: {churn_prob[1]:.2%})")

# 문의 분류 테스트
test_inquiry = "배송이 너무 늦어요 언제 오나요?"
X_inq_test = tfidf_inquiry.transform([test_inquiry])
inq_pred = le_inquiry_cat.inverse_transform(rf_inquiry.predict(X_inq_test))[0]
print(f"  문의 분류: '{test_inquiry}' -> {inq_pred}")

# 감성 분석 테스트
test_review = "품질이 정말 좋고 배송도 빨라요! 추천합니다!"
X_rev_test = tfidf_sentiment.transform([test_review])
sent_pred = le_sentiment.inverse_transform(model_sentiment.predict(X_rev_test))[0]
print(f"  감성 분석: '{test_review}' -> {sent_pred}")

# 세그먼트 테스트
sample_seg = {
    "total_orders": 200, "total_revenue": 30000000, "product_count": 30,
    "cs_tickets": 5, "refund_rate": 0.03, "avg_response_time": 1.5,
}
X_seg_test = pd.DataFrame([sample_seg])[segment_features]
X_seg_test_scaled = scaler_cluster.transform(X_seg_test)
seg_pred = int(kmeans.predict(X_seg_test_scaled)[0])
print(f"  셀러 세그먼트: {SEGMENT_NAMES.get(seg_pred, '알 수 없음')}")


# --------------------------------------------------------------------------
# 완료 요약
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("완료! 카페24 이커머스 데이터 생성 및 모델 학습 성공")
print("=" * 70)
print(f"\n[요약]")
print(f"  데이터:")
print(f"    - 쇼핑몰: {len(shops_df)}개, 상품: {len(products_df)}개, 셀러: {len(sellers_df)}명")
print(f"    - 운영 로그: {len(op_logs_df)}건, 서비스: {len(services_df)}건")
print(f"    - 일별 지표: {len(daily_metrics_df)}일, 셀러 활동: {len(seller_activity_df)}건")
print(f"    - CSV 파일: 17개")
print(f"  모델 (11개):")
print(f"    1. 셀러 이탈 예측 (RandomForest + SHAP)")
print(f"    2. 이상거래 탐지 (Isolation Forest)")
print(f"    3. 문의 자동 분류 (TF-IDF + RandomForest)")
print(f"    4. 셀러 세그먼트 (K-Means)")
print(f"    5. 매출 예측 ({algo_name_rev})")
print(f"    6. CS 응답 품질 (RandomForest)")
print(f"    7. 고객 LTV 예측 (GradientBoosting)")
print(f"    8. 리뷰 감성 분석 (TF-IDF + LogisticRegression)")
print(f"    9. 상품 수요 예측 ({algo_name_dem})")
print(f"   10. 정산 이상 탐지 (DBSCAN)")
print(f"   11. Guardian 감사 로그 이상탐지 (IsolationForest, {len(rows)}건)")
print(f"  SHAP: {'활성화' if SHAP_AVAILABLE else '비활성화'}")
print(f"  MLflow: {'활성화' if MLFLOW_AVAILABLE else '비활성화'}")
print(f"\n백엔드 서버 시작: cd \"backend 리팩토링 시작\" && python main.py")
