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
import pandas as pd

import state as st


def get_data_path(filename: str) -> Path:
    """데이터 파일 경로 반환"""
    return Path(st.BASE_DIR) / filename


def load_csv_safe(filepath: Path, encoding: str = "utf-8-sig") -> Optional[pd.DataFrame]:
    """안전한 CSV 로딩"""
    if not filepath.exists():
        st.logger.warning(f"CSV 파일 없음: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        st.logger.info(f"CSV 로드 완료: {filepath.name} ({len(df)} rows)")
        return df
    except Exception as e:
        st.logger.error(f"CSV 로드 실패: {filepath} - {e}")
        return None


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
    st.logger.info("CAFE24 AI 운영 플랫폼 데이터 로딩 시작 (병렬)")
    st.logger.info("=" * 50)

    _load_start = time.time()

    # ========================================
    # H23/cross-2: CSV 데이터 병렬 로드
    # ========================================
    csv_tasks = {
        "SHOPS_DF": "shops.csv",
        "CATEGORIES_DF": "categories.csv",
        "SERVICES_DF": "services.csv",
        "PRODUCTS_DF": "products.csv",
        "SELLERS_DF": "sellers.csv",
        "SELLER_ANALYTICS_DF": "seller_analytics.csv",
        "PLATFORM_DOCS_DF": "platform_docs.csv",
        "ECOMMERCE_GLOSSARY_DF": "ecommerce_glossary.csv",
        "SHOP_PERFORMANCE_DF": "shop_performance.csv",
        "DAILY_METRICS_DF": "daily_metrics.csv",
        "CS_STATS_DF": "cs_stats.csv",
        "FRAUD_DETAILS_DF": "fraud_details.csv",
        "COHORT_RETENTION_DF": "cohort_retention.csv",
        "CONVERSION_FUNNEL_DF": "conversion_funnel.csv",
        "SELLER_ACTIVITY_DF": "seller_activity.csv",
    }

    def _load_csv_task(attr_name, filename):
        return attr_name, load_csv_safe(get_data_path(filename))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_load_csv_task, attr, fname)
            for attr, fname in csv_tasks.items()
        ]
        for future in as_completed(futures):
            try:
                attr_name, df = future.result()
                setattr(st, attr_name, df)
            except Exception as e:
                st.logger.error(f"CSV 병렬 로드 실패: {e}")

    # 운영 로그 (nrows 제한이 있어 별도 처리)
    logs_path = get_data_path("operation_logs.csv")
    if logs_path.exists():
        try:
            st.OPERATION_LOGS_DF = pd.read_csv(logs_path, encoding="utf-8-sig", nrows=30000)
            st.logger.info(f"CSV 로드 완료: operation_logs.csv ({len(st.OPERATION_LOGS_DF)} rows, 제한됨)")
        except Exception as e:
            st.logger.error(f"CSV 로드 실패: operation_logs.csv - {e}")
            st.OPERATION_LOGS_DF = None
    else:
        st.logger.warning(f"CSV 파일 없음: {logs_path}")
        st.OPERATION_LOGS_DF = None

    _csv_elapsed = time.time() - _load_start
    st.logger.info("CSV 병렬 로드 완료: %.1f초", _csv_elapsed)

    # ========================================
    # ML 모델 로드 (M35: 병렬 로딩)
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

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(_load_model_task, attr, fname)
            for attr, fname in model_tasks.items()
        ]
        for future in as_completed(futures):
            try:
                attr_name, model = future.result()
                setattr(st, attr_name, model)
            except Exception as e:
                st.logger.error(f"모델 병렬 로드 실패: {e}")

    # 이탈 예측 모델 설정 (JSON)
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
    st.logger.info("ML 모델 병렬 로드 완료: %.1f초", _model_elapsed)

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
    # 라벨 인코더 로드
    # ========================================
    st.LE_TICKET_CATEGORY = load_model_safe(get_data_path("le_ticket_category.pkl"))
    st.LE_SELLER_TIER = load_model_safe(get_data_path("le_seller_tier.pkl"))
    st.LE_CS_PRIORITY = load_model_safe(get_data_path("le_cs_priority.pkl"))
    st.LE_INQUIRY_CATEGORY = load_model_safe(get_data_path("le_inquiry_category.pkl"))

    # ========================================
    # 매출 예측 모델 초기화
    # ========================================
    try:
        from ml.revenue_model import get_predictor, train_and_save
        predictor = get_predictor()

        if not predictor.is_fitted and st.SHOP_PERFORMANCE_DF is not None:
            st.logger.info("매출 예측 모델 학습 시작...")
            result = train_and_save(st.SHOP_PERFORMANCE_DF)
            st.logger.info(f"매출 예측 모델 학습 완료: R2={result['cv_r2_mean']:.3f}")
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
    st.SYSTEM_STATUS["models_loaded"] = (
        st.CS_QUALITY_MODEL is not None or
        st.INQUIRY_CLASSIFICATION_MODEL is not None or
        st.SELLER_SEGMENT_MODEL is not None or
        st.FRAUD_DETECTION_MODEL is not None or
        st.SELLER_CHURN_MODEL is not None or
        st.REVENUE_PREDICTION_MODEL is not None or
        st.CUSTOMER_LTV_MODEL is not None or
        st.REVIEW_SENTIMENT_MODEL is not None or
        st.DEMAND_FORECAST_MODEL is not None or
        st.MARKETING_OPTIMIZER_AVAILABLE
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
    st.logger.info(f"  [ML 모델 (10개)]")
    st.logger.info(f"  - 셀러 이탈 예측: {'O' if st.SELLER_CHURN_MODEL else 'X'}")
    st.logger.info(f"  - 이상거래 탐지: {'O' if st.FRAUD_DETECTION_MODEL else 'X'}")
    st.logger.info(f"  - 문의 자동 분류: {'O' if st.INQUIRY_CLASSIFICATION_MODEL else 'X'}")
    st.logger.info(f"  - 셀러 세그먼트: {'O' if st.SELLER_SEGMENT_MODEL else 'X'}")
    st.logger.info(f"  - 매출 예측: {'O' if st.REVENUE_PREDICTION_MODEL else 'X'}")
    st.logger.info(f"  - CS 응답 품질: {'O' if st.CS_QUALITY_MODEL else 'X'}")
    st.logger.info(f"  - 고객 LTV: {'O' if st.CUSTOMER_LTV_MODEL else 'X'}")
    st.logger.info(f"  - 리뷰 감성: {'O' if st.REVIEW_SENTIMENT_MODEL else 'X'}")
    st.logger.info(f"  - 수요 예측: {'O' if st.DEMAND_FORECAST_MODEL else 'X'}")
    st.logger.info(f"  - 정산 이상: {'O' if st.SETTLEMENT_ANOMALY_MODEL else 'X'}")
    st.logger.info(f"  - 마케팅 최적화: {'O' if st.MARKETING_OPTIMIZER_AVAILABLE else 'X'}")
    st.logger.info("=" * 50)

    # ========================================
    # 저장된 모델 선택 상태 로드 및 MLflow 모델 로드
    # ========================================
    load_selected_mlflow_models()


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

        # 2차 시도: joblib 직접 로드
        if loaded_model is None:
            try:
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
    """캐시 데이터 구성"""
    # 쇼핑몰별 서비스 매핑
    if st.SERVICES_DF is not None and st.SHOPS_DF is not None:
        for _, row in st.SERVICES_DF.iterrows():
            shop_id = row.get("shop_id")
            if shop_id:
                if shop_id not in st.SHOP_SERVICE_MAP:
                    st.SHOP_SERVICE_MAP[shop_id] = []
                st.SHOP_SERVICE_MAP[shop_id].append({
                    "service_name": row.get("service_name"),
                    "service_type": row.get("service_type"),
                    "status": row.get("status"),
                    "description": row.get("description"),
                })
        st.logger.info(f"쇼핑몰 서비스 캐시 구성: {len(st.SHOP_SERVICE_MAP)}개")


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
            "seller_churn": st.SELLER_CHURN_MODEL is not None,
            "fraud_detection": st.FRAUD_DETECTION_MODEL is not None,
            "inquiry_classification": st.INQUIRY_CLASSIFICATION_MODEL is not None,
            "seller_segment": st.SELLER_SEGMENT_MODEL is not None,
            "revenue_prediction": st.REVENUE_PREDICTION_MODEL is not None,
            "cs_quality": st.CS_QUALITY_MODEL is not None,
            "customer_ltv": st.CUSTOMER_LTV_MODEL is not None,
            "review_sentiment": st.REVIEW_SENTIMENT_MODEL is not None,
            "demand_forecast": st.DEMAND_FORECAST_MODEL is not None,
            "settlement_anomaly": st.SETTLEMENT_ANOMALY_MODEL is not None,
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
