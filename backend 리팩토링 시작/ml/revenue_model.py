"""
ml/revenue_model.py - 매출 예측 모델
=====================================
CAFE24 AI 운영 플랫폼

쇼핑몰 성과 기반 다음 달 매출 예측 모델 학습 및 추론

입력: monthly_revenue, monthly_orders, monthly_visitors, avg_order_value,
      customer_retention_rate, conversion_rate, review_score
출력: predicted_next_month_revenue (예측 매출)

[주피터 노트북에서 실행 시]
이 파일 전체를 복사해서 셀에 붙여넣고 실행하면 됩니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# ========================================
# MLflow 설정 (train_models.py와 동일)
# ========================================
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - skipping experiment tracking")

# 프로젝트 루트 (주피터/스크립트 호환)
try:
    # 스크립트 실행 시
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # 주피터 노트북 실행 시
    # 이미 BACKEND_DIR이 정의되어 있으면 사용
    if 'BACKEND_DIR' in dir():
        PROJECT_ROOT = BACKEND_DIR
    else:
        # ml 폴더에서 실행 시 부모 폴더로
        _cwd = Path(".").resolve()
        if _cwd.name == "ml":
            PROJECT_ROOT = _cwd.parent
        else:
            PROJECT_ROOT = _cwd

# 매출 예측 피처 컬럼
REVENUE_FEATURES = [
    'monthly_revenue',          # 월 매출
    'monthly_orders',           # 월 주문 수
    'monthly_visitors',         # 월 방문자 수
    'avg_order_value',          # 평균 주문 금액
    'customer_retention_rate',  # 고객 유지율 (%)
    'conversion_rate',          # 전환율 (%)
    'review_score',             # 리뷰 평점 (1~5)
]

# 모델 저장 경로
MODEL_PATH = PROJECT_ROOT / "model_revenue.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler_revenue.pkl"


class RevenuePredictor:
    """쇼핑몰 성과 기반 다음 달 매출 예측 모델"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def _generate_synthetic_data(
        self,
        base_df: pd.DataFrame,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        기존 데이터 기반 합성 데이터 생성

        전략:
        - 기존 쇼핑몰 성과 데이터에 노이즈를 추가해서 augmentation
        - 성과 지표 변화에 따른 다음 달 매출 변화 시뮬레이션
        """
        logger.info(f"Generating {n_samples} synthetic samples from {len(base_df)} base shops")

        # M32: numpy broadcasting으로 이중 루프 제거
        # 피처별 매출 영향도 (이커머스 도메인 기반 추정)
        impact_values = np.array([0.85, 150.0, 200.0, 0.5, 5000.0, 8000.0, 20000.0])

        samples_per_shop = max(1, n_samples // len(base_df))
        n_shops = len(base_df)
        n_features = len(REVENUE_FEATURES)

        # 원본 데이터를 행렬로 추출
        base_X = base_df[REVENUE_FEATURES].values  # (n_shops, n_features)
        base_y = base_df.get('next_month_revenue',
                            base_df['monthly_revenue'] * 1.02).values  # (n_shops,)

        # Broadcasting으로 합성 데이터 일괄 생성
        total_synthetic = n_shops * samples_per_shop
        noise_pct = np.random.uniform(-0.2, 0.2, (total_synthetic, n_features))

        # 각 샵별 반복 인덱스
        shop_indices = np.repeat(np.arange(n_shops), samples_per_shop)
        base_features_repeated = base_X[shop_indices]  # (total_synthetic, n_features)
        base_y_repeated = base_y[shop_indices]  # (total_synthetic,)

        # 변화량 계산
        changes = base_features_repeated * noise_pct  # (total_synthetic, n_features)
        new_features = np.maximum(0, base_features_repeated + changes)

        # 매출 변화 = 변화량 * 영향도의 내적
        delta_revenue = np.sum(changes * impact_values, axis=1)

        # 노이즈 추가
        noise_std = base_y_repeated * 0.05
        noise = np.random.normal(0, 1, total_synthetic) * noise_std
        new_y = np.maximum(0, base_y_repeated + delta_revenue + noise)

        # 원본 데이터 합치기
        X_all = np.vstack([new_features, base_X])
        y_all = np.concatenate([new_y, base_y])

        return X_all, y_all

    def train(self, shop_performance_df: pd.DataFrame, n_synthetic: int = 500) -> Dict:
        """
        모델 학습

        Args:
            shop_performance_df: 쇼핑몰 성과 데이터프레임
                필수 컬럼: monthly_revenue, monthly_orders, monthly_visitors,
                          avg_order_value, customer_retention_rate, conversion_rate, review_score
                선택 컬럼: next_month_revenue (없으면 monthly_revenue * 1.02 로 추정)
            n_synthetic: 생성할 합성 데이터 수

        Returns:
            학습 결과 (cv_score, feature_importance 등)
        """
        logger.info("Training revenue prediction model...")

        # 합성 데이터 생성
        X, y = self._generate_synthetic_data(shop_performance_df, n_synthetic)

        # 스케일링
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 모델 학습 (LightGBM)
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=3,
            random_state=42,
            verbose=-1
        )

        # 교차 검증
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')

        # 전체 데이터로 최종 학습
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # feature importance
        feature_importance = dict(zip(REVENUE_FEATURES, self.model.feature_importances_))

        result = {
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'n_samples': len(X),
            'feature_importance': feature_importance,
        }

        logger.info(f"Model trained: R2 = {result['cv_r2_mean']:.3f} (+/- {result['cv_r2_std']:.3f})")

        return result

    def predict(self, features: Dict[str, float]) -> float:
        """
        단일 쇼핑몰 다음 달 매출 예측

        Args:
            features: {
                'monthly_revenue': 50000000,
                'monthly_orders': 1200,
                'monthly_visitors': 800,
                'avg_order_value': 42000,
                'customer_retention_rate': 65.2,
                'conversion_rate': 3.1,
                'review_score': 4.3
            }

        Returns:
            예측 다음 달 매출 (원)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")

        X = np.array([[features.get(feat, 0) for feat in REVENUE_FEATURES]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]

        return float(max(0, pred))  # 음수 방지

    def save(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        """모델 저장"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> bool:
        """모델 로딩"""
        try:
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True
                logger.info(f"Model loaded from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# 전역 인스턴스 (싱글톤 패턴)
_predictor_instance: Optional[RevenuePredictor] = None


def get_predictor() -> RevenuePredictor:
    """매출 예측 모델 인스턴스 반환"""
    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = RevenuePredictor()
        # 저장된 모델 로딩 시도
        _predictor_instance.load()

    return _predictor_instance


def train_and_save(shop_performance_df: pd.DataFrame, register_mlflow: bool = True) -> Dict:
    """
    모델 학습 및 저장 (MLflow 등록 포함)

    Args:
        shop_performance_df: 쇼핑몰 성과 데이터프레임
        register_mlflow: MLflow에 등록 여부 (기본 True)

    Returns:
        학습 결과 dict
    """
    predictor = RevenuePredictor()
    result = predictor.train(shop_performance_df)
    predictor.save()

    # 전역 인스턴스 업데이트
    global _predictor_instance
    _predictor_instance = predictor

    # MLflow 등록
    if register_mlflow and MLFLOW_AVAILABLE:
        try:
            # MLflow 설정 (train_models.py와 동일한 경로)
            mlflow_tracking_uri = f"file:{PROJECT_ROOT / 'ml' / 'mlruns'}"
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            experiment_name = "cafe24-ops-ai"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="revenue_model"):
                # 태그
                mlflow.set_tag("model_type", "regression")
                mlflow.set_tag("target", "next_month_revenue")
                mlflow.set_tag("algorithm", "LightGBM")
                mlflow.set_tag("domain", "cafe24")

                # 하이퍼파라미터
                mlflow.log_params({
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "num_leaves": 15,
                    "min_child_samples": 3,
                    "n_features": len(REVENUE_FEATURES),
                    "n_samples": result['n_samples'],
                })

                # 메트릭
                mlflow.log_metrics({
                    "cv_r2_mean": result['cv_r2_mean'],
                    "cv_r2_std": result['cv_r2_std'],
                })

                # 모델 등록
                mlflow.sklearn.log_model(
                    predictor.model,
                    "revenue_model",
                    registered_model_name="매출예측"
                )

                print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")
                print(f"[MLflow] Model registered as '매출예측'")

        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
            print(f"[Warning] MLflow 등록 실패: {e}")

    return result

