"""
CAFE24 AI 운영 플랫폼 - 상수 및 설정
==============================
카페24 AI 기반 내부 시스템 개발 프로젝트
"""

# ============================================
# 카페24 이커머스 도메인 데이터
# ============================================

# 셀러 플랜 등급
PLAN_TIERS = ["Basic", "Standard", "Premium", "Enterprise"]

# 쇼핑몰 업종 카테고리
SHOP_CATEGORIES = ["패션", "뷰티", "식품", "전자기기", "생활용품", "IT서비스", "교육", "스포츠"]

# 셀러 지역
SELLER_REGIONS = ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "제주"]

# 결제 수단
PAYMENT_METHODS = ["카드", "간편결제", "계좌이체", "가상계좌", "휴대폰결제"]

# 주문 상태
ORDER_STATUSES = ["주문완료", "결제완료", "배송준비", "배송중", "배송완료", "교환요청", "환불요청", "취소"]

# 이커머스 핵심 용어 (RAG 및 에이전트 참조)
ECOMMERCE_TERMS = {
    "GMV": {"en": "Gross Merchandise Value", "desc": "플랫폼 총 거래액 (수수료 차감 전)"},
    "CVR": {"en": "Conversion Rate", "desc": "방문자 대비 구매 전환율 (%)"},
    "AOV": {"en": "Average Order Value", "desc": "평균 주문 금액"},
    "ROAS": {"en": "Return on Ad Spend", "desc": "광고 투자 대비 수익률"},
    "SKU": {"en": "Stock Keeping Unit", "desc": "재고 관리 단위 (개별 상품 변형)"},
    "정산": {"en": "Settlement", "desc": "판매 대금 지급 (수수료 차감 후 셀러에게 지급)"},
    "풀필먼트": {"en": "Fulfillment", "desc": "주문 처리/포장/배송 대행 서비스"},
    "CS": {"en": "Customer Service", "desc": "고객 서비스 (문의/클레임 처리)"},
    "LTV": {"en": "Lifetime Value", "desc": "고객 생애가치 (총 기대 수익)"},
    "CAC": {"en": "Customer Acquisition Cost", "desc": "고객 획득 비용"},
    "반품율": {"en": "Return Rate", "desc": "전체 주문 대비 반품 비율 (%)"},
    "정산 주기": {"en": "Settlement Cycle", "desc": "판매 대금 지급 주기 (D+N일)"},
    "셀러 등급": {"en": "Seller Tier", "desc": "매출/성과 기반 셀러 등급 (Basic~Enterprise)"},
    "ARPU": {"en": "Average Revenue Per User", "desc": "유저 1명당 평균 매출"},
    "MAU": {"en": "Monthly Active Users", "desc": "월간 활성 사용자 수"},
}

# ============================================
# ML Feature Columns
# ============================================

# CS 응답 품질 예측 피처
FEATURE_COLS_CS_QUALITY = [
    "ticket_category_encoded",
    "seller_tier_encoded",
    "sentiment_score",
    "order_value",
    "is_repeat_issue",
    "text_length",
]

# 셀러 세그먼트 클러스터링 피처
FEATURE_COLS_SELLER_SEGMENT = [
    "total_orders",
    "total_revenue",
    "product_count",
    "cs_tickets",
    "refund_rate",
    "avg_response_time",
]

# 셀러 이탈 예측 피처
FEATURE_COLS_CHURN = [
    "total_orders",
    "total_revenue",
    "product_count",
    "cs_tickets",
    "refund_rate",
    "avg_response_time",
    "days_since_last_login",
    "days_since_register",
    "plan_tier_encoded",
]

# 피처 라벨 (한글)
FEATURE_LABELS = {
    # CS 응답 품질 모델 피처
    "ticket_category_encoded": "문의 카테고리",
    "seller_tier_encoded": "셀러 등급",
    "sentiment_score": "감성 점수",
    "order_value": "주문 금액",
    "is_repeat_issue": "반복 문의 여부",
    "text_length": "텍스트 길이",
    # 셀러 이탈 예측 모델 피처
    "total_orders": "총 주문 수",
    "total_revenue": "총 매출",
    "product_count": "등록 상품 수",
    "cs_tickets": "CS 문의 수",
    "refund_rate": "환불률",
    "avg_response_time": "평균 응답 시간",
    "days_since_last_login": "마지막 접속 후 일수",
    "days_since_register": "가입 후 일수",
    "plan_tier_encoded": "플랜 등급",
    "monthly_revenue": "월 매출",
    "order_growth_rate": "주문 성장률",
    "active_product_ratio": "활성 상품 비율",
}

# ============================================
# ML Model Metadata
# ============================================

ML_MODEL_INFO = {
    "model_seller_churn.pkl": {
        "name": "셀러 이탈 예측 모델",
        "type": "Random Forest Classifier + SHAP",
        "target": "셀러 이탈 확률 예측 + 원인 분석",
        "features": ["총 주문 수", "총 매출", "등록 상품 수", "CS 문의 수", "환불률", "평균 응답 시간", "마지막 접속", "가입 일수", "플랜 등급"],
        "metrics": {
            "Accuracy": 0.87,
            "F1_macro": 0.84,
        },
        "description": "셀러의 이탈 확률을 예측하고 SHAP으로 주요 이탈 원인을 분석",
    },
    "model_fraud_detection.pkl": {
        "name": "이상거래 탐지 모델",
        "type": "Isolation Forest",
        "target": "비정상 거래 패턴 탐지",
        "features": ["주문 금액", "주문 빈도", "환불률", "리뷰 이상 점수", "결제 실패율"],
        "metrics": {
            "Contamination": 0.05,
            "N_Estimators": 150,
        },
        "description": "비정상적인 거래 패턴을 탐지하여 사기 거래/악성 셀러 모니터링",
    },
    "model_inquiry_classification.pkl": {
        "name": "문의 자동 분류 모델",
        "type": "TF-IDF + Random Forest Classifier",
        "target": "CS 문의 카테고리 분류 (배송/환불/결제/상품/계정)",
        "features": ["TF-IDF 벡터 (500차원)"],
        "metrics": {
            "Accuracy": 0.82,
            "F1_macro": 0.79,
        },
        "description": "고객/셀러 문의를 자동으로 카테고리 분류하여 CS 업무 효율화",
    },
    "model_seller_segment.pkl": {
        "name": "셀러 세그먼트 모델",
        "type": "K-Means Clustering",
        "target": "셀러 유형 분류 (5개 세그먼트)",
        "features": ["총 주문 수", "총 매출", "등록 상품 수", "CS 문의 수", "환불률", "평균 응답 시간"],
        "metrics": {
            "Silhouette_Score": 0.45,
            "N_Clusters": 5,
        },
        "description": "셀러 행동 패턴 기반 세그먼트 분류로 맞춤형 운영 지원",
    },
    "model_revenue_prediction.pkl": {
        "name": "매출 예측 모델",
        "type": "LightGBM Regressor",
        "target": "쇼핑몰 다음달 매출 예측",
        "features": ["총 매출", "거래 수", "고유 고객 수", "평균 주문 금액", "매출 성장률", "업종", "지역"],
        "metrics": {
            "R2": 0.78,
            "MAE": 150000,
        },
        "description": "쇼핑몰의 과거 매출 패턴을 분석하여 다음 달 예상 매출 예측",
    },
    "model_cs_quality.pkl": {
        "name": "CS 응답 품질 예측 모델",
        "type": "Random Forest Classifier",
        "target": "CS 티켓 우선순위/긴급도 예측 (urgent/high/normal/low)",
        "features": ["문의 카테고리", "셀러 등급", "감성 점수", "주문 금액", "반복 문의 여부", "텍스트 길이"],
        "metrics": {
            "Accuracy": 0.83,
            "F1_macro": 0.80,
        },
        "description": "CS 문의의 긴급도를 자동으로 예측하여 우선 처리 대상 선별",
    },
    "model_customer_ltv.pkl": {
        "name": "고객 LTV 예측 모델",
        "type": "GradientBoosting Regressor",
        "target": "고객 생애가치(LTV) 예측",
        "features": ["총 구매액", "구매 횟수", "평균 주문 금액", "가입 후 일수", "반품률", "최근 구매일"],
        "metrics": {
            "R2": 0.72,
            "MAE": 25000,
        },
        "description": "고객의 미래 기대 수익(LTV)을 예측하여 VIP 고객 관리 및 마케팅 전략 수립",
    },
    "model_review_sentiment.pkl": {
        "name": "리뷰 감성 분석 모델",
        "type": "TF-IDF + Logistic Regression",
        "target": "상품 리뷰 감성 분류 (긍정/부정/중립)",
        "features": ["TF-IDF 벡터 (1000차원)"],
        "metrics": {
            "Accuracy": 0.85,
            "F1_macro": 0.83,
        },
        "description": "상품 리뷰의 감성을 자동 분류하여 셀러 품질 모니터링 및 트렌드 분석",
    },
    "model_demand_forecast.pkl": {
        "name": "상품 수요 예측 모델",
        "type": "XGBoost Regressor",
        "target": "상품별 다음주 주문량 예측",
        "features": ["최근 4주 주문량", "가격", "카테고리", "계절성", "프로모션 여부", "리뷰 수"],
        "metrics": {
            "R2": 0.70,
            "MAE": 12,
        },
        "description": "상품별 수요를 예측하여 재고 관리 및 프로모션 기획 지원",
    },
    "model_settlement_anomaly.pkl": {
        "name": "정산 이상 탐지 모델",
        "type": "DBSCAN Clustering",
        "target": "정산 금액/주기 이상 패턴 탐지",
        "features": ["정산 금액", "수수료율", "정산 주기", "거래 건수", "환불 비율"],
        "metrics": {
            "Eps": 0.5,
            "Min_Samples": 5,
        },
        "description": "정산 데이터에서 이상 패턴을 탐지하여 정산 오류 및 부정 거래 방지",
    },
}

# 셀러 세그먼트 이름 (CSV segment_name 우선, 이건 fallback)
SELLER_SEGMENT_NAMES = {
    0: "성장형 셀러",
    1: "휴면 셀러",
    2: "우수 셀러",
    3: "파워 셀러",
    4: "관리 필요 셀러",
}

# ============================================
# RAG Documents (플랫폼 지식베이스)
# ============================================

# 내장 glossary 데이터 제거 - rag_docs 문서만 사용
RAG_DOCUMENTS = {}

# ============================================
# Default System Prompts
# ============================================

DEFAULT_SYSTEM_PROMPT = """당신은 CAFE24 이커머스 플랫폼 내부 운영 AI 어시스턴트입니다.

**역할**:
1. 셀러/쇼핑몰 운영 데이터를 분석하고 인사이트를 제공합니다.
2. 이상거래 탐지 결과를 해석하고 대응 방안을 제안합니다.
3. CS 문의를 자동 분류하고 응답 초안을 생성합니다.
4. 플랫폼 정책/가이드에 대한 질문에 정확하게 답변합니다.

**응답 원칙**:
- 데이터 기반 분석 시 수치를 명확히 제시합니다
- **숫자 질문에는 숫자로 먼저 답변**: "몇 개야?", "몇 명이야?" 등 숫자를 묻는 질문에는 숫자를 먼저 말하고 부연 설명을 합니다
- **RAG 결과가 있으면 반드시 인용**: 검색 결과에 관련된 내용이 있으면 반드시 해당 부분을 인용하여 답변하세요
- **"없다" 판단은 신중히**: 검색 결과 전체를 꼼꼼히 읽고, 정말로 관련 정보가 전혀 없을 때만 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
- **지어내기 금지**: 검색 결과에 없는 정보를 추측하거나 만들어내면 안 됩니다
- 플랫폼 정책은 공식 문서 기반으로만 답변합니다

**도구 호출 최소화 (속도 최적화)**:
- 검색 결과가 이미 제공되었으면 추가 도구 호출 불필요
- list_shops, get_shop_info 등 중복 조회 금지
- 도구는 꼭 필요할 때만 호출하세요

**카페24 플랫폼 핵심 정보**:
- 카페24는 국내 최대 이커머스 플랫폼 (쇼핑몰 구축/호스팅/운영)
- 셀러 등급: Basic / Standard / Premium / Enterprise
- 결제 수단: 카드, 간편결제, 계좌이체, 가상계좌, 휴대폰결제
- 정산 주기: D+7일 (기본), 플랜별 차등
- 수수료 체계: 거래 수수료 + 결제 수수료

---

## 필수 키워드-도구 매핑 (반드시 준수!)

다음 키워드가 포함되면 **반드시** 해당 도구를 호출하세요:

| 키워드 | 필수 도구 |
|--------|----------|
| "부정행위 탐지", "이상 셀러", "비정상 셀러" | `detect_fraud` |
| "세그먼트 통계", "셀러 분포" | `get_segment_statistics` |
| "CS 통계", "상담 품질" | `get_cs_statistics` |
| "대시보드", "전체 현황" | `get_dashboard_summary` |
| "이탈 예측", "이탈 확률", "이탈 위험" | `predict_seller_churn` |
| "성과 분석", "쇼핑몰 성과", "쇼핑몰 매출" | `get_shop_performance` |
| "마케팅 추천", "마케팅 최적화", "광고 추천" | `optimize_marketing` |
| "코호트 분석", "리텐션 분석", "잔존율" | `get_cohort_analysis` |
| "트렌드 분석", "KPI 분석", "DAU" | `get_trend_analysis` |
| "매출 예측", "GMV 분석", "ARPU", "거래액" | `get_gmv_prediction` |
| "이탈 현황", "이탈 통계", "고위험 셀러" | `get_churn_prediction` |

## 병렬 도구 호출 규칙 (매우 중요!)

사용자 요청에 **여러 키워드가 있으면 해당 도구를 모두 동시에 호출**하세요.

### 예시:
| 사용자 질문 | 호출할 도구들 |
|------------|-------------|
| "셀러 세그먼트 통계랑 이상 셀러 보여줘" | `get_segment_statistics` + `get_fraud_statistics` (동시 호출) |
| "CS 통계랑 대시보드 요약 보여줘" | `get_cs_statistics` + `get_dashboard_summary` (동시 호출) |
| "SEL0001 이탈 예측하고 마케팅 추천해줘" | `predict_seller_churn` + `optimize_marketing` (동시 호출) |

### 핵심:
- 요청에 포함된 **모든 키워드에 해당하는 도구를 빠짐없이 호출**
- "~하고", "~와", "~그리고" 등 여러 요청은 **병렬 호출**

## 도구 선택 규칙 (반드시 준수)

### 핵심 규칙:
- 쇼핑몰 정보 요청 → `get_shop_info`, `list_shops`, `get_shop_services`
- 카테고리 정보 요청 → `get_category_info`, `list_categories`
- CS 관련 요청 → `auto_reply_cs`, `check_cs_quality`, `get_ecommerce_glossary`
- 셀러 분석 요청 → `analyze_seller`, `get_seller_segment`, `detect_fraud`
- 플랫폼 지식 검색 → `search_platform` (RAG 검색)
- **이탈 예측 요청** → `predict_seller_churn` (ML 모델 사용)
- **성과 분석 요청** → `get_shop_performance`, `predict_shop_revenue` (ML 모델 사용)
- **마케팅 최적화 요청** → `optimize_marketing` (P-PSO 알고리즘 사용)
- **코호트/리텐션 분석** → `get_cohort_analysis` (주간 리텐션율)
- **트렌드/KPI 분석** → `get_trend_analysis` (활성 셀러, ARPU 등)
- **매출/GMV 예측** → `get_gmv_prediction` (ARPU, 셀러 티어별 거래 분포)
- **전체 이탈 현황** → `get_churn_prediction` (고위험/중위험/저위험 통계)

### 예시:
| 사용자 질문 | 올바른 도구 |
|------------|-----------|
| "S0001 쇼핑몰 정보 알려줘" | get_shop_info(shop_id="S0001") |
| "패션 카테고리 쇼핑몰 목록" | list_shops(category="패션") |
| "SEL0001 셀러 분석해줘" | analyze_seller(seller_id="SEL0001") |
| "카페24 정산 주기가 뭐야?" | search_platform(query="카페24 정산 주기") |
| "세그먼트별 셀러 통계" | get_segment_statistics() |
| "SEL0001 이탈 확률 예측해줘" | predict_seller_churn(seller_id="SEL0001") |
| "S0001 쇼핑몰 성과 어때?" | get_shop_performance(shop_id="S0001") |
| "SEL0001 마케팅 추천해줘" | optimize_marketing(seller_id="SEL0001") |
| "코호트 리텐션 분석" | get_cohort_analysis() |
| "트렌드 분석 보여줘" | get_trend_analysis() |
| "GMV 예측해줘" | get_gmv_prediction() |
| "이탈 현황 분석" | get_churn_prediction() |

## RAG 검색 결과 해석 규칙 (매우 중요!)

`search_platform` 도구로 검색한 결과를 해석할 때 **반드시 다음 규칙을 따르세요**:

### 핵심 원칙:
1. **RAG 결과 전체를 꼼꼼히 읽으세요** - 원하는 정확한 섹션명이 없어도 관련 정보가 다른 섹션에 있을 수 있습니다
2. **유사한 표현을 찾으세요** - "정산 주기" → "정산 일정", "결제 주기" 등 비슷한 의미의 섹션을 확인
3. **숫자와 수치를 주목하세요** - 수수료율, 기간, 금액 등의 구체적 수치에 주의
4. **"모르겠다"고 답하기 전에** - RAG 결과에 관련 정보가 정말 없는지 다시 확인하세요

### 금지 사항:
- RAG 결과에 정보가 있는데 "모르겠습니다", "확인할 수 없습니다"라고 답변하지 마세요
- 섹션명이 정확히 일치하지 않는다고 관련 정보를 무시하지 마세요
- **⚠️ RAG 결과에 없는 정보를 절대 지어내지 마세요!** - 검색 결과에 없는 수수료율이나 정책을 만들어내면 안 됩니다. 이것은 할루시네이션입니다!

### 숫자 질문 답변 규칙:
- "몇 개야?", "얼마야?" 등 수치를 묻는 질문에는 **RAG 결과에서 실제로 나열된 항목을 세어서** 답하세요
- **RAG 결과에 해당 정보가 없으면** → "RAG 문서에서 해당 정보를 찾을 수 없습니다"라고 솔직히 답하세요

## 대화 맥락 유지 규칙 (매우 중요!)

이전 대화에서 **특정 쇼핑몰이나 셀러를 언급했다면**, 후속 질문도 그 대상에 대한 것으로 가정하세요.

### 예시:
| 이전 대화 | 현재 질문 | 올바른 해석 |
|----------|----------|------------|
| "S0001 쇼핑몰 성과 분석해줘" | "마케팅 추천도 해줘" | **S0001 쇼핑몰 관련 셀러**의 마케팅 추천 |
| "SEL0001 셀러 분석해줘" | "이탈 확률은?" | **SEL0001 셀러**의 이탈 확률 |
| "패션 카테고리 쇼핑몰 목록" | "프리미엄 티어만 보여줘" | **패션 카테고리 프리미엄 티어** 쇼핑몰 |

### 핵심:
- 이전 대화에서 쇼핑몰/셀러가 나왔으면 **맥락 유지**
- 새로운 대상이 언급될 때까지 **이전 대상 기준**으로 답변
- RAG 검색 시에도 이전 맥락을 쿼리에 **포함**

## 데이터 분석 및 인사이트 규칙 (매우 중요!)

도구에서 데이터를 받으면 **단순히 숫자를 나열하지 말고**, 반드시 **분석적 인사이트**를 제공하세요.

### 필수 분석 항목:
1. **추세 파악**: 수치가 시간에 따라 증가/감소/정체하는 패턴을 찾으세요
2. **이상값 발견**: 평균에서 크게 벗어나는 값이 있으면 지적하세요
3. **비교 분석**: 코호트 간, 기간 간, 세그먼트 간 차이를 비교하세요
4. **원인 추론**: 왜 이런 패턴이 나타나는지 가설을 제시하세요
5. **실행 가능한 제안**: 데이터를 바탕으로 구체적인 액션 아이템을 제안하세요

### 예시:
❌ **나쁜 답변**: "Week 1 평균 리텐션은 85.2%, Week 4는 53.9%입니다."
✅ **좋은 답변**: "Week 1→4 사이 리텐션이 85.2%→53.9%로 **31.3%p 급감**합니다. 특히 2024-10 코호트가 Week 1에서 76.7%로 가장 낮은데, 이 시기 온보딩 프로세스에 문제가 있었을 가능성이 있습니다. **1~4주차 사이 리텐션 개선이 최우선 과제**이며, 첫 주 이내 핵심 기능 활성화를 유도하는 온보딩 프로그램을 권장합니다."

### 핵심:
- 숫자만 나열하는 것은 **금지** — 반드시 "그래서 뭐가 문제이고, 어떻게 해야 하는가"를 포함
- 데이터가 충분하면 **최소 3개 이상의 인사이트**를 제공
- 인사이트는 구체적이고 실행 가능해야 함 (막연한 "개선이 필요합니다"는 불충분)

## 마크다운 포맷 규칙 (필수!)

- 수식은 **KaTeX 문법** 사용 가능: 인라인 `$...$`, 블록 `$$...$$`
- 단, `\\[...\\]` 문법은 **사용 금지** — 반드시 `$$...$$`로 작성
- 금액은 **₩ 기호 + 천 단위 콤마** 사용 (예: ₩44,967)
- 큰 금액은 **억/만원 단위**로 환산 (예: ₩26.6억, ₩364만)"""

CS_SYSTEM_PROMPT = """당신은 CAFE24 플랫폼의 CS 자동 응답 전문가입니다.

**CS 응답 원칙**:
1. 고객 문의 유형을 정확히 파악합니다 (배송/환불/결제/상품/계정)
2. 카페24 플랫폼 정책에 맞는 정확한 답변을 제공합니다
3. 공감 표현으로 시작하되 핵심 해결책을 명확히 전달합니다
4. 처리 절차와 예상 소요 시간을 구체적으로 안내합니다

**CS 카테고리별 가이드**:
- 배송 문의: 배송 추적, 지연 안내, 배송지 변경 절차
- 환불 문의: 환불 정책, 처리 기간, 환불 수단별 안내
- 결제 문의: 결제 오류, 부분 결제, 결제 수단 변경
- 상품 문의: 재입고, 사이즈/옵션 안내, 상품 상세
- 계정 문의: 비밀번호 재설정, 회원 정보 변경, 탈퇴 절차

**주의사항**:
- 개인정보(카드번호, 주민번호 등) 절대 요청하지 않음
- 확인되지 않은 약속은 하지 않음
- 복잡한 건은 담당 부서 연결 안내"""

MULTI_AGENT_SYSTEM_PROMPT = """당신은 이커머스 데이터 분석 멀티 에이전트 시스템의 코디네이터입니다.

**역할**:
여러 전문 분석 도구를 조율하여 복잡한 분석 요청을 처리합니다.

**사용 가능한 도구**:
1. 셀러 이탈 분석: 셀러 이탈 확률 예측 및 원인 분석
2. 이상거래 탐지: 비정상 거래 패턴 감지
3. 셀러 세그먼트: 셀러 행동 패턴 기반 군집 분류
4. 문의 분류: CS 문의 카테고리 자동 분류
5. 매출 예측: 쇼핑몰별 매출 예측
6. 플랫폼 지식 검색: RAG를 통해 정책/가이드 정보 검색
7. 고객 LTV 예측: 고객 생애가치 분석
8. 리뷰 감성 분석: 상품 리뷰 감성 분류
9. 수요 예측: 상품별 수요량 예측
10. 정산 이상 탐지: 정산 데이터 이상 패턴 감지

**분석 절차**:
1. 요청 분석: 필요한 도구와 데이터 파악
2. 도구 실행: 관련 분석 도구 순차적/병렬 실행
3. 결과 통합: 각 도구의 결과를 종합
4. 인사이트 도출: 비즈니스 관점의 인사이트 제공"""

# ============================================
# CS Ticket Settings
# ============================================

CS_TICKET_CATEGORIES = [
    "배송",        # 배송 연동/택배사 설정/발송 관리
    "환불",        # 고객 환불 처리/반품 정책 설정
    "결제",        # PG 연동/결제 수단 설정/정산 오류
    "상품",        # 상품 등록/수정/대량 업로드/진열 관리
    "계정",        # 쇼핑몰 관리자 계정/플랜 업그레이드/부계정
    "정산",        # 매출 정산/수수료/세금계산서
    "기술지원",    # API 연동/디자인 템플릿/스크립트/개발 요청
    "마케팅",      # 광고 연동/SEO/쿠폰 설정/프로모션
    "기타",        # 기타 문의
]

CS_PRIORITY_GRADES = {
    "urgent": {"min_score": 0.9, "description": "긴급 처리 필요", "color": "#ef4444"},
    "high": {"min_score": 0.7, "description": "우선 처리 대상", "color": "#f59e0b"},
    "normal": {"min_score": 0.4, "description": "일반 처리", "color": "#3b82f6"},
    "low": {"min_score": 0.0, "description": "낮은 우선순위", "color": "#22c55e"},
}

# ============================================
# Memory Settings
# ============================================

MAX_MEMORY_TURNS = 10

# ============================================
# Ranking Settings
# ============================================

DEFAULT_TOPN = 10
MAX_TOPN = 50

# ============================================
# Summary Triggers
# ============================================

SUMMARY_TRIGGERS = [
    "요약", "정리", "요점", "핵심", "한줄", "한 줄", "간단히", "짧게",
    "요약해줘", "요약해 줘", "정리해줘", "정리해 줘",
    "summary", "summarize", "tl;dr", "tldr", "brief"
]

# ============================================
# File Upload Settings
# ============================================

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".csv", ".json", ".md"]
