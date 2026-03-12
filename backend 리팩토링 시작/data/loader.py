"""
CAFE24 AI 운영 플랫폼 - 데이터 로더
==============================
카페24 AI 기반 내부 시스템 개발 프로젝트

CSV 데이터 및 ML 모델 로딩
"""

import os
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd

import state as st


def get_data_path(filename: str) -> Path:
    """데이터 파일 경로 반환"""
    return Path(st.BASE_DIR) / filename


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame dtype 최적화 — Railway 메모리 절감용

    - int64 → int32 (값 범위가 int32 이내인 경우만)
    - float64 → float32
    - object 컬럼 중 고유값 비율 50% 이하 → category
    """
    for col in df.columns:
        col_dtype = df[col].dtype

        # int64 → int32 (값 범위 체크)
        if col_dtype == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if np.iinfo(np.int32).min <= col_min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        # float64 → float32
        elif col_dtype == np.float64:
            df[col] = df[col].astype(np.float32)

        # object → category (고유값 비율 50% 이하)
        elif col_dtype == object:
            nunique = df[col].nunique()
            if nunique <= len(df) * 0.5:
                df[col] = df[col].astype("category")

    return df


def load_data_safe(filepath: Path) -> Optional[pd.DataFrame]:
    """안전한 PKL 데이터 로딩 (dtype 자동 최적화 포함)"""
    pkl_path = filepath.with_suffix(".pkl") if filepath.suffix != ".pkl" else filepath
    if not pkl_path.exists():
        st.logger.warning(f"데이터 파일 없음: {pkl_path}")
        return None
    try:
        df = pd.read_pickle(pkl_path)
        df = _optimize_dtypes(df)
        st.logger.info(f"PKL 로드 완료: {pkl_path.name} ({len(df)} rows, dtype 최적화 적용)")
        return df
    except Exception as e:
        st.logger.error(f"PKL 로드 실패: {pkl_path} - {e}")
        return None


# 하위 호환
load_csv_safe = load_data_safe


def load_model_safe(filepath: Path):
    """안전한 모델 로딩"""
    if not filepath.exists():
        st.logger.warning(f"모델 파일 없음: {filepath}")
        return None
    try:
        model = joblib.load(filepath)
        st.logger.info(f"모델 로드 완료: {filepath.name}")
        return model
    except Exception as e:
        st.logger.error(f"모델 로드 실패: {filepath} - {e}")
        return None


def load_all_data():
    """모든 데이터 로드 (H23/cross-2: ThreadPoolExecutor 병렬화)"""
    st.logger.info("=" * 50)
    st.logger.info("CAFE24 AI 운영 플랫폼 데이터 로딩 시작 (PKL 병렬)")
    st.logger.info("=" * 50)

    _load_start = time.time()

    # ========================================
    # H23/cross-2: CSV 데이터 병렬 로드
    # ========================================
    data_tasks = {
        "SHOPS_DF": "shops.pkl",
        "CATEGORIES_DF": "categories.pkl",
        "SERVICES_DF": "services.pkl",
        "PRODUCTS_DF": "products.pkl",
        "SELLERS_DF": "sellers.pkl",
        "SELLER_ANALYTICS_DF": "seller_analytics.pkl",
        "SHOP_PERFORMANCE_DF": "shop_performance.pkl",
        "DAILY_METRICS_DF": "daily_metrics.pkl",
        "CS_STATS_DF": "cs_stats.pkl",
        "CS_TICKETS_DF": "cs_tickets.pkl",
        "FRAUD_DETAILS_DF": "fraud_details.pkl",
        "COHORT_RETENTION_DF": "cohort_retention.pkl",
        "CONVERSION_FUNNEL_DF": "conversion_funnel.pkl",
        "SELLER_ACTIVITY_DF": "seller_activity.pkl",
    }

    def _load_data_task(attr_name, filename):
        return attr_name, load_data_safe(get_data_path(filename))

    # max_workers=4: 피크 메모리 감소 (Railway 512MB 제한 대응)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_load_data_task, attr, fname)
            for attr, fname in data_tasks.items()
        ]
        for future in as_completed(futures):
            try:
                attr_name, df = future.result()
                setattr(st, attr_name, df)
            except Exception as e:
                st.logger.error(f"PKL 병렬 로드 실패: {e}")

    # SELLER_ANALYTICS_DF 후처리: plan_tier_encoded → plan_tier 디코딩
    if st.SELLER_ANALYTICS_DF is not None and "plan_tier_encoded" in st.SELLER_ANALYTICS_DF.columns:
        from core.constants import PLAN_TIERS
        st.SELLER_ANALYTICS_DF["plan_tier"] = (
            st.SELLER_ANALYTICS_DF["plan_tier_encoded"]
            .map({i: t for i, t in enumerate(PLAN_TIERS)})
            .fillna("Basic")
        )
        st.logger.info("SELLER_ANALYTICS_DF: plan_tier 컬럼 디코딩 완료")

    # 운영 로그
    st.OPERATION_LOGS_DF = load_data_safe(get_data_path("operation_logs.pkl"))

    _data_elapsed = time.time() - _load_start
    st.logger.info("PKL 데이터 로드 완료: %.1f초", _data_elapsed)

    # ========================================
    # ML 모델 — Lazy Loading (Railway 메모리 최적화)
    # 시작 시 전체 로드 대신, 각 모델을 처음 사용할 때 로드
    # ========================================
    _model_start = time.time()

    model_tasks = {
        "CS_QUALITY_MODEL": "model_cs_quality.pkl",
        "INQUIRY_CLASSIFICATION_MODEL": "model_inquiry_classification.pkl",
        "SELLER_SEGMENT_MODEL": "model_seller_segment.pkl",
        "FRAUD_DETECTION_MODEL": "model_fraud_detection.pkl",
        "SELLER_CHURN_MODEL": "model_seller_churn.pkl",
        "SHAP_EXPLAINER_CHURN": "shap_explainer_churn.pkl",
        "REVENUE_PREDICTION_MODEL": "model_revenue_prediction.pkl",
        "CUSTOMER_LTV_MODEL": "model_customer_ltv.pkl",
        "REVIEW_SENTIMENT_MODEL": "model_review_sentiment.pkl",
        "DEMAND_FORECAST_MODEL": "model_demand_forecast.pkl",
        "SETTLEMENT_ANOMALY_MODEL": "model_settlement_anomaly.pkl",
        "TFIDF_VECTORIZER": "tfidf_vectorizer.pkl",
        "TFIDF_VECTORIZER_SENTIMENT": "tfidf_vectorizer_sentiment.pkl",
        "SCALER_CLUSTER": "scaler_cluster.pkl",
    }

    def _load_model_task(attr_name, filename):
        return attr_name, load_model_safe(get_data_path(filename))

    # Lazy loading 모드: 파일 존재 여부만 확인, 실제 로드는 get_model()에서 수행
    st._LAZY_LOADING_ENABLED = True
    _available_models = []
    _missing_models = []
    for attr_name, filename in model_tasks.items():
        filepath = get_data_path(filename)
        if filepath.exists():
            _available_models.append(attr_name)
        else:
            _missing_models.append(attr_name)
            st._MODEL_LOAD_FAILED.add(attr_name)

    st.logger.info(
        "ML 모델 Lazy Loading 활성화: %d개 대기 (파일 있음), %d개 없음",
        len(_available_models), len(_missing_models),
    )

    # 이탈 예측 모델 설정 (JSON) — 작은 파일이므로 즉시 로드
    churn_config_path = get_data_path("churn_model_config.json")
    if churn_config_path.exists():
        try:
            import json
            with open(churn_config_path, "r", encoding="utf-8") as f:
                st.CHURN_MODEL_CONFIG = json.load(f)
            st.logger.info(f"이탈 예측 모델 설정 로드 완료: {churn_config_path.name}")
        except Exception as e:
            st.logger.warning(f"이탈 예측 모델 설정 로드 실패: {e}")
            st.CHURN_MODEL_CONFIG = None

    _model_elapsed = time.time() - _model_start
    st.logger.info("ML 모델 Lazy Loading 설정 완료: %.1f초", _model_elapsed)

    # ========================================
    # 마케팅 최적화 모듈 확인
    # ========================================
    try:
        from ml.marketing_optimizer import MarketingOptimizer
        st.MARKETING_OPTIMIZER_AVAILABLE = True
        st.logger.info("마케팅 최적화 모듈 로드 완료")
    except ImportError as e:
        st.MARKETING_OPTIMIZER_AVAILABLE = False
        st.logger.warning(f"마케팅 최적화 모듈 로드 실패: {e}")

    # ========================================
    # 라벨 인코더 — Lazy Loading (get_model로 접근 시 자동 로드)
    # ========================================
    le_tasks = {
        "LE_TICKET_CATEGORY": "le_ticket_category.pkl",
        "LE_SELLER_TIER": "le_seller_tier.pkl",
        "LE_CS_PRIORITY": "le_cs_priority.pkl",
        "LE_INQUIRY_CATEGORY": "le_inquiry_category.pkl",
    }
    for attr_name, filename in le_tasks.items():
        filepath = get_data_path(filename)
        if not filepath.exists():
            st._MODEL_LOAD_FAILED.add(attr_name)
    # _MODEL_FILE_MAP에 라벨 인코더도 등록 (get_model에서 찾을 수 있도록)
    for attr_name, filename in le_tasks.items():
        st._MODEL_FILE_MAP[attr_name] = filename
    st.logger.info("라벨 인코더 Lazy Loading 설정 완료 (%d개)", len(le_tasks))

    # ========================================
    # 매출 예측 모델 초기화 (학습 필요 시 백그라운드)
    # ========================================
    try:
        from ml.revenue_model import get_predictor, train_and_save
        predictor = get_predictor()

        if not predictor.is_fitted and st.SHOP_PERFORMANCE_DF is not None:
            def _train_revenue_bg():
                try:
                    result = train_and_save(st.SHOP_PERFORMANCE_DF)
                    st.logger.info(f"매출 예측 모델 학습 완료: R2={result['cv_r2_mean']:.3f}")
                except Exception as ex:
                    st.logger.warning(f"매출 예측 모델 백그라운드 학습 실패: {ex}")
            st.logger.info("매출 예측 모델 백그라운드 학습 시작...")
            import threading
            threading.Thread(target=_train_revenue_bg, daemon=True).start()
        elif predictor.is_fitted:
            st.logger.info("매출 예측 모델 로드 완료")
        else:
            st.logger.warning("매출 예측 모델 학습 불가 (shop_performance.csv 없음)")
    except Exception as e:
        st.logger.warning(f"매출 예측 모델 초기화 실패: {e}")

    # ========================================
    # 캐시 구성
    # ========================================
    build_caches()

    # ========================================
    # 시스템 상태 업데이트
    # ========================================
    st.SYSTEM_STATUS["data_loaded"] = True
    # lazy loading 모드에서는 파일 존재 여부로 모델 로드 가능 상태 판단
    st.SYSTEM_STATUS["models_loaded"] = (
        len(_available_models) > 0 or st.MARKETING_OPTIMIZER_AVAILABLE
    )

    st.logger.info("=" * 50)
    st.logger.info("데이터 로딩 완료")
    st.logger.info(f"  [기본 데이터]")
    st.logger.info(f"  - 쇼핑몰: {len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0}개")
    st.logger.info(f"  - 카테고리: {len(st.CATEGORIES_DF) if st.CATEGORIES_DF is not None else 0}개")
    st.logger.info(f"  - 상품: {len(st.PRODUCTS_DF) if st.PRODUCTS_DF is not None else 0}개")
    st.logger.info(f"  - 셀러: {len(st.SELLERS_DF) if st.SELLERS_DF is not None else 0}명")
    st.logger.info(f"  - 운영 로그: {len(st.OPERATION_LOGS_DF) if st.OPERATION_LOGS_DF is not None else 0}건")
    st.logger.info(f"  [분석용 데이터]")
    st.logger.info(f"  - 쇼핑몰 성과: {len(st.SHOP_PERFORMANCE_DF) if st.SHOP_PERFORMANCE_DF is not None else 0}개")
    st.logger.info(f"  - 일별 지표: {len(st.DAILY_METRICS_DF) if st.DAILY_METRICS_DF is not None else 0}일")
    st.logger.info(f"  - CS 통계: {len(st.CS_STATS_DF) if st.CS_STATS_DF is not None else 0}개")
    st.logger.info(f"  - 코호트: {len(st.COHORT_RETENTION_DF) if st.COHORT_RETENTION_DF is not None else 0}개")
    # Lazy loading 모드에서는 파일 존재=대기(L), 파일 없음=X로 표시
    _model_status = lambda name: 'L(lazy)' if name in _available_models else 'X'
    st.logger.info(f"  [ML 모델 (Lazy Loading, 10개)]")
    st.logger.info(f"  - 셀러 이탈 예측: {_model_status('SELLER_CHURN_MODEL')}")
    st.logger.info(f"  - 이상거래 탐지: {_model_status('FRAUD_DETECTION_MODEL')}")
    st.logger.info(f"  - 문의 자동 분류: {_model_status('INQUIRY_CLASSIFICATION_MODEL')}")
    st.logger.info(f"  - 셀러 세그먼트: {_model_status('SELLER_SEGMENT_MODEL')}")
    st.logger.info(f"  - 매출 예측: {_model_status('REVENUE_PREDICTION_MODEL')}")
    st.logger.info(f"  - CS 응답 품질: {_model_status('CS_QUALITY_MODEL')}")
    st.logger.info(f"  - 고객 LTV: {_model_status('CUSTOMER_LTV_MODEL')}")
    st.logger.info(f"  - 리뷰 감성: {_model_status('REVIEW_SENTIMENT_MODEL')}")
    st.logger.info(f"  - 수요 예측: {_model_status('DEMAND_FORECAST_MODEL')}")
    st.logger.info(f"  - 정산 이상: {_model_status('SETTLEMENT_ANOMALY_MODEL')}")
    st.logger.info(f"  - 마케팅 최적화: {'O' if st.MARKETING_OPTIMIZER_AVAILABLE else 'X'}")
    st.logger.info("=" * 50)

    # ========================================
    # 저장된 모델 선택 상태 로드 및 MLflow 모델 로드
    # ========================================
    load_selected_mlflow_models()


# MLflow 모델 경로 캐시 (version meta YAML → model_pkl_path)
_mlflow_path_cache: dict = {}


def load_selected_mlflow_models():
    """
    서버 시작 시 저장된 모델 선택 상태를 읽어서 MLflow 모델을 로드
    관리자가 선택한 모델이 서버 재시작 후에도 유지됨
    """
    import platform
    import yaml

    selected = st.load_selected_models()

    if not selected:
        st.logger.info("저장된 모델 선택 상태 없음 - 기본 pkl 모델 사용")
        return

    st.logger.info(f"저장된 모델 선택 상태 로드: {selected}")

    is_local = platform.system() == "Windows"
    st.logger.info(f"환경 감지: {'로컬(Windows)' if is_local else 'Docker(Linux)'}")

    # 모델 이름 → state 변수 매핑
    MODEL_STATE_MAP = {
        "셀러이탈예측": "SELLER_CHURN_MODEL",
        "이상거래탐지": "FRAUD_DETECTION_MODEL",
        "문의자동분류": "INQUIRY_CLASSIFICATION_MODEL",
        "셀러세그먼트": "SELLER_SEGMENT_MODEL",
        "매출예측": "REVENUE_PREDICTION_MODEL",
        "CS응답품질": "CS_QUALITY_MODEL",
        "고객LTV": "CUSTOMER_LTV_MODEL",
        "리뷰감성분석": "REVIEW_SENTIMENT_MODEL",
        "수요예측": "DEMAND_FORECAST_MODEL",
        "정산이상탐지": "SETTLEMENT_ANOMALY_MODEL",
    }

    ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
    if not os.path.exists(ml_mlruns):
        ml_mlruns = os.path.join(st.BASE_DIR, "mlruns")

    if not os.path.exists(ml_mlruns):
        st.logger.warning(f"MLflow 폴더 없음: {ml_mlruns}")
        return

    experiment_id = "660890565547137650"

    for model_name, version in selected.items():
        state_attr = MODEL_STATE_MAP.get(model_name)
        if not state_attr:
            st.logger.warning(f"알 수 없는 모델: {model_name}")
            continue

        loaded_model = None
        load_method = None

        # 1차 시도: MLflow API (Windows)
        if is_local:
            try:
                import mlflow
                mlflow.set_tracking_uri(f"file:///{ml_mlruns}")
                model_uri = f"models:/{model_name}/{version}"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                if hasattr(loaded_model, "_model_impl"):
                    loaded_model = loaded_model._model_impl.python_model
                    if hasattr(loaded_model, "model"):
                        loaded_model = loaded_model.model
                load_method = "MLflow API"
            except Exception as e:
                st.logger.debug(f"MLflow API 실패, fallback 시도: {e}")
                loaded_model = None

        # 2차 시도: joblib 직접 로드 (경로 캐싱)
        if loaded_model is None:
            cache_key = f"{model_name}:{version}"
            try:
                # 캐시된 pkl 경로 확인
                model_pkl_path = _mlflow_path_cache.get(cache_key)
                if model_pkl_path is None:
                    version_meta_path = os.path.join(
                        ml_mlruns, "models", model_name, f"version-{version}", "meta.yaml"
                    )
                    if not os.path.exists(version_meta_path):
                        st.logger.warning(f"버전 메타 없음: {version_meta_path}")
                        continue

                    with open(version_meta_path, "r", encoding="utf-8") as f:
                        version_meta = yaml.safe_load(f)

                    model_id = version_meta.get("model_id")
                    if not model_id:
                        st.logger.warning(f"model_id 없음: {model_name} v{version}")
                        continue

                    model_pkl_path = os.path.join(
                        ml_mlruns, experiment_id, "models", model_id, "artifacts", "model.pkl"
                    )
                    # 경로 캐싱
                    _mlflow_path_cache[cache_key] = model_pkl_path

                if not os.path.exists(model_pkl_path):
                    st.logger.warning(f"모델 파일 없음: {model_pkl_path}")
                    continue

                loaded_model = joblib.load(model_pkl_path)
                load_method = "직접 로드"
            except Exception as e:
                st.logger.warning(f"모델 로드 실패: {model_name} v{version} - {e}")
                continue

        if loaded_model is not None:
            setattr(st, state_attr, loaded_model)
            st.logger.info(f"[{load_method}] 모델 로드 완료: {model_name} v{version} → st.{state_attr}")


def build_caches():
    """캐시 데이터 구성 (groupby 벡터화)"""
    # 쇼핑몰별 서비스 매핑 — iterrows → groupby
    if st.SERVICES_DF is not None and st.SHOPS_DF is not None:
        svc_df = st.SERVICES_DF.dropna(subset=["shop_id"])
        cols = ["service_name", "service_type", "status", "description"]
        avail_cols = [c for c in cols if c in svc_df.columns]
        for shop_id, group in svc_df.groupby("shop_id"):
            st.SHOP_SERVICE_MAP[shop_id] = group[avail_cols].to_dict("records")
        st.logger.info(f"쇼핑몰 서비스 캐시 구성: {len(st.SHOP_SERVICE_MAP)}개")

    # 쇼핑몰별 성과 KPI 캐시 (O(1) 조회용)
    if st.SHOP_PERFORMANCE_DF is not None and "shop_id" in st.SHOP_PERFORMANCE_DF.columns:
        st.SHOP_PERF_MAP = st.SHOP_PERFORMANCE_DF.set_index("shop_id").to_dict("index")
        st.logger.info(f"쇼핑몰 성과 캐시 구성: {len(st.SHOP_PERF_MAP)}개")


def get_data_summary() -> dict:
    """데이터 요약 정보 반환"""
    return {
        "shops": {
            "count": len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0,
            "loaded": st.SHOPS_DF is not None,
        },
        "categories": {
            "count": len(st.CATEGORIES_DF) if st.CATEGORIES_DF is not None else 0,
            "loaded": st.CATEGORIES_DF is not None,
        },
        "services": {
            "count": len(st.SERVICES_DF) if st.SERVICES_DF is not None else 0,
            "loaded": st.SERVICES_DF is not None,
        },
        "products": {
            "count": len(st.PRODUCTS_DF) if st.PRODUCTS_DF is not None else 0,
            "loaded": st.PRODUCTS_DF is not None,
        },
        "sellers": {
            "count": len(st.SELLERS_DF) if st.SELLERS_DF is not None else 0,
            "loaded": st.SELLERS_DF is not None,
        },
        "operation_logs": {
            "count": len(st.OPERATION_LOGS_DF) if st.OPERATION_LOGS_DF is not None else 0,
            "loaded": st.OPERATION_LOGS_DF is not None,
        },
        "seller_analytics": {
            "count": len(st.SELLER_ANALYTICS_DF) if st.SELLER_ANALYTICS_DF is not None else 0,
            "loaded": st.SELLER_ANALYTICS_DF is not None,
        },
        "shop_performance": {
            "count": len(st.SHOP_PERFORMANCE_DF) if st.SHOP_PERFORMANCE_DF is not None else 0,
            "loaded": st.SHOP_PERFORMANCE_DF is not None,
        },
        "daily_metrics": {
            "count": len(st.DAILY_METRICS_DF) if st.DAILY_METRICS_DF is not None else 0,
            "loaded": st.DAILY_METRICS_DF is not None,
        },
        "cs_stats": {
            "count": len(st.CS_STATS_DF) if st.CS_STATS_DF is not None else 0,
            "loaded": st.CS_STATS_DF is not None,
        },
        "cohort_retention": {
            "count": len(st.COHORT_RETENTION_DF) if st.COHORT_RETENTION_DF is not None else 0,
            "loaded": st.COHORT_RETENTION_DF is not None,
        },
        "models": {
            "seller_churn": st.SELLER_CHURN_MODEL is not None or "SELLER_CHURN_MODEL" not in st._MODEL_LOAD_FAILED,
            "fraud_detection": st.FRAUD_DETECTION_MODEL is not None or "FRAUD_DETECTION_MODEL" not in st._MODEL_LOAD_FAILED,
            "inquiry_classification": st.INQUIRY_CLASSIFICATION_MODEL is not None or "INQUIRY_CLASSIFICATION_MODEL" not in st._MODEL_LOAD_FAILED,
            "seller_segment": st.SELLER_SEGMENT_MODEL is not None or "SELLER_SEGMENT_MODEL" not in st._MODEL_LOAD_FAILED,
            "revenue_prediction": st.REVENUE_PREDICTION_MODEL is not None or "REVENUE_PREDICTION_MODEL" not in st._MODEL_LOAD_FAILED,
            "cs_quality": st.CS_QUALITY_MODEL is not None or "CS_QUALITY_MODEL" not in st._MODEL_LOAD_FAILED,
            "customer_ltv": st.CUSTOMER_LTV_MODEL is not None or "CUSTOMER_LTV_MODEL" not in st._MODEL_LOAD_FAILED,
            "review_sentiment": st.REVIEW_SENTIMENT_MODEL is not None or "REVIEW_SENTIMENT_MODEL" not in st._MODEL_LOAD_FAILED,
            "demand_forecast": st.DEMAND_FORECAST_MODEL is not None or "DEMAND_FORECAST_MODEL" not in st._MODEL_LOAD_FAILED,
            "settlement_anomaly": st.SETTLEMENT_ANOMALY_MODEL is not None or "SETTLEMENT_ANOMALY_MODEL" not in st._MODEL_LOAD_FAILED,
            "marketing_optimizer": st.MARKETING_OPTIMIZER_AVAILABLE,
        },
    }


# 기존 함수 호환성을 위한 alias
def init_data_models():
    """데이터 로드 및 모델 초기화 (startup 시 호출)"""
    if st.SYSTEM_STATUS.get("data_loaded"):
        st.logger.info("데이터 이미 로드됨 - 스킵")
        return
    load_all_data()
