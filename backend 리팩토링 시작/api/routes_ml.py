"""
api/routes_ml.py - MLflow/마케팅 최적화 API
"""
import os

import pandas as pd
from fastapi import APIRouter, Depends

from core.utils import safe_str, json_sanitize
from agent.tools import tool_optimize_marketing
import state as st
from api.common import verify_credentials, ModelSelectRequest, MarketingOptimizeRequest


router = APIRouter(prefix="/api", tags=["ml"])


# ============================================================
# MLflow
# ============================================================
@router.get("/mlflow/experiments")
def get_mlflow_experiments(user: dict = Depends(verify_credentials)):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        tracking_uri = _get_mlflow_uri()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        experiments = client.search_experiments()
        result = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=10)
            runs_data = [{"run_id": run.info.run_id, "run_name": run.info.run_name, "status": run.info.status, "start_time": run.info.start_time, "end_time": run.info.end_time, "params": dict(run.data.params), "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()}, "tags": dict(run.data.tags)} for run in runs]
            result.append({"experiment_id": exp.experiment_id, "name": exp.name, "artifact_location": exp.artifact_location, "lifecycle_stage": exp.lifecycle_stage, "runs": runs_data})
        return {"status": "success", "data": result}
    except ImportError:
        return {"status": "error", "message": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 조회 실패")
        return {"status": "error", "message": safe_str(e), "data": []}


@router.get("/mlflow/models")
def get_mlflow_registered_models(user: dict = Depends(verify_credentials)):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        tracking_uri = _get_mlflow_uri()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        registered_models = client.search_registered_models()
        result = []
        for rm in registered_models:
            versions = []
            try:
                all_versions = client.search_model_versions(filter_string=f"name='{rm.name}'")
                for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
                    versions.append({"version": v.version, "stage": v.current_stage, "status": v.status, "run_id": v.run_id, "source": v.source, "creation_timestamp": v.creation_timestamp})
            except Exception:
                for v in rm.latest_versions:
                    versions.append({"version": v.version, "stage": v.current_stage, "status": v.status, "run_id": v.run_id, "source": v.source, "creation_timestamp": v.creation_timestamp})
            result.append({"name": rm.name, "creation_timestamp": rm.creation_timestamp, "last_updated_timestamp": rm.last_updated_timestamp, "description": rm.description or "", "versions": versions, "model_type": "registry"})
        return {"status": "success", "data": result}
    except ImportError:
        return {"status": "error", "message": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 모델 조회 실패")
        return {"status": "error", "message": safe_str(e), "data": []}


@router.get("/mlflow/models/selected")
def get_selected_models(user: dict = Depends(verify_credentials)):
    st.load_selected_models()
    return {"status": "success", "data": st.SELECTED_MODELS, "message": f"{len(st.SELECTED_MODELS)}개 모델이 선택되어 있습니다"}


@router.post("/mlflow/models/select")
def select_mlflow_model(req: ModelSelectRequest, user: dict = Depends(verify_credentials)):
    MODEL_STATE_MAP = {"CS응답품질": "CS_QUALITY_MODEL", "문의분류": "INQUIRY_CLASSIFICATION_MODEL", "셀러세그먼트": "SELLER_SEGMENT_MODEL", "이상거래탐지": "FRAUD_DETECTION_MODEL", "셀러이탈예측": "SELLER_CHURN_MODEL", "매출예측": "REVENUE_PREDICTION_MODEL"}
    state_attr = MODEL_STATE_MAP.get(req.model_name)
    if not state_attr:
        return {"status": "error", "message": f"알 수 없는 모델: {req.model_name}. 지원 모델: {list(MODEL_STATE_MAP.keys())}"}
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        tracking_uri = _get_mlflow_uri()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        try:
            model_version = client.get_model_version(req.model_name, req.version)
            model_uri = f"models:/{req.model_name}/{req.version}"
            st.logger.info(f"모델 로드 시작: {model_uri}")
            loaded_model = mlflow.sklearn.load_model(model_uri)
            setattr(st, state_attr, loaded_model)
            st.logger.info(f"모델 로드 완료: st.{state_attr} = {model_uri}")
            st.SELECTED_MODELS[req.model_name] = req.version
            st.save_selected_models()
            return {"status": "success", "message": f"{req.model_name} v{req.version} 모델이 로드되었습니다", "data": {"model_name": req.model_name, "version": req.version, "stage": model_version.current_stage, "run_id": model_version.run_id, "state_variable": f"st.{state_attr}", "loaded": True}}
        except Exception as e:
            st.logger.warning(f"MLflow 모델 로드 실패: {e}")
            return {"status": "error", "message": f"모델 로드 실패: {safe_str(e)}", "data": {"model_name": req.model_name, "version": req.version}}
    except ImportError:
        return {"status": "error", "message": "MLflow가 설치되지 않았습니다."}
    except Exception as e:
        st.logger.exception("MLflow 모델 선택 실패")
        return {"status": "error", "message": safe_str(e)}


def _get_mlflow_uri():
    ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
    backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
    project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))
    if os.path.exists(ml_mlruns):
        return f"file:{ml_mlruns}"
    elif os.path.exists(backend_mlruns):
        return f"file:{backend_mlruns}"
    elif os.path.exists(project_mlruns):
        return f"file:{project_mlruns}"
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


# ============================================================
# 마케팅 예산 최적화
# ============================================================
@router.get("/marketing/seller/{seller_id}")
def get_marketing_seller_info(seller_id: str, user: dict = Depends(verify_credentials)):
    try:
        if st.SELLERS_DF is None:
            return {"status": "error", "message": "셀러 데이터 없음"}
        sid = seller_id.strip().upper()
        row = st.SELLERS_DF[st.SELLERS_DF["seller_id"].str.upper() == sid]
        if row.empty:
            return {"status": "error", "message": f"셀러 {seller_id}을(를) 찾을 수 없습니다"}
        seller = row.iloc[0]
        shops = []
        if st.SHOPS_DF is not None:
            seller_shops = st.SHOPS_DF[st.SHOPS_DF.get("seller_id", pd.Series()).str.upper() == sid] if "seller_id" in st.SHOPS_DF.columns else pd.DataFrame()
            if seller_shops.empty and st.SHOP_PERFORMANCE_DF is not None:
                shops = st.SHOP_PERFORMANCE_DF.head(5).to_dict("records")
            else:
                shops = seller_shops.head(5).to_dict("records")
        # shops에 cvr 필드 추가 (프론트엔드 차트용)
        for s in shops:
            if "conversion_rate" in s and "cvr" not in s:
                s["cvr"] = float(s["conversion_rate"])
        data = {
            "seller_id": seller.get("seller_id", sid),
            "total_revenue": float(seller.get("total_revenue", 0)),
            "total_orders": int(seller.get("total_orders", 0)),
            "product_count": int(seller.get("product_count", 0)),
            "resources": {
                "ad_budget": int(float(seller.get("total_revenue", 0)) * 0.1),
                "monthly_revenue": float(seller.get("total_revenue", 0)),
                "product_count": int(seller.get("product_count", 0)),
                "order_count": int(seller.get("total_orders", 0)),
            },
            "shops": shops,
        }
        return json_sanitize({"status": "success", "data": data})
    except Exception as e:
        st.logger.exception("마케팅 셀러 정보 조회 실패")
        return {"status": "error", "message": safe_str(e)}


@router.post("/marketing/optimize")
def optimize_marketing_budget(req: MarketingOptimizeRequest, user: dict = Depends(verify_credentials)):
    try:
        total_budget = None
        if req.budget_constraints and "total" in req.budget_constraints:
            total_budget = float(req.budget_constraints["total"])
        result = tool_optimize_marketing(seller_id=req.seller_id or "SEL0001", goal="maximize_roas", total_budget=total_budget)
        if result.get("status") == "FAILED":
            return {"status": "error", "message": result.get("error", "최적화 실패")}
        return {"status": "success", "data": result}
    except Exception as e:
        st.logger.exception("마케팅 최적화 실패")
        return {"status": "error", "message": f"마케팅 최적화 중 오류: {safe_str(e)}"}


@router.get("/marketing/status")
def get_marketing_optimizer_status(user: dict = Depends(verify_credentials)):
    return {"status": "success", "data": {"optimizer_available": st.MARKETING_OPTIMIZER_AVAILABLE, "shops_loaded": st.SHOPS_DF is not None, "shops_count": len(st.SHOPS_DF) if st.SHOPS_DF is not None else 0, "optimization_method": "P-PSO (Phasor Particle Swarm Optimization)"}}
