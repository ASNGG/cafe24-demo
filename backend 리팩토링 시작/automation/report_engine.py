"""
automation/report_engine.py - 운영 리포트 자동 생성 엔진
=====================================================
플랫폼 데이터 집계 → LLM 리포트 자동 작성
"""
import time
import uuid
from typing import Dict, Any

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str, safe_int, safe_float
import state as st
from agent.llm import get_llm, invoke_with_retry, pick_api_key
from automation.action_logger import save_report, get_report_history, log_action, create_pipeline_run, update_pipeline_step, complete_pipeline_run


# ── 리포트 타입별 라벨 ──
_REPORT_TYPE_LABELS = {
    "daily": "일간",
    "weekly": "주간",
    "monthly": "월간",
}


def collect_report_data() -> Dict[str, Any]:
    """모든 주요 DF에서 핵심 KPI를 수집합니다."""
    kpi: Dict[str, Any] = {}
    trends: Dict[str, Any] = {}
    segments: Dict[str, Any] = {}
    cs_summary: Dict[str, Any] = {}

    # ── DAILY_METRICS_DF: GMV, 활성셀러, 주문수, 신규가입 ──
    df = st.DAILY_METRICS_DF
    if df is not None and not df.empty:
        date_col = None
        for c in ("date", "날짜", "dt"):
            if c in df.columns:
                date_col = c
                break

        df_sorted = df.copy()
        if date_col:
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
            df_sorted = df_sorted.sort_values(date_col, ascending=False)

        recent_7 = df_sorted.head(7) if len(df_sorted) >= 7 else df_sorted
        recent_30 = df_sorted.head(30) if len(df_sorted) >= 30 else df_sorted

        for col in ("gmv", "active_sellers", "orders", "new_signups"):
            if col in df_sorted.columns:
                kpi[f"{col}_latest"] = safe_float(df_sorted[col].iloc[0]) if len(df_sorted) > 0 else 0
                kpi[f"{col}_7d_avg"] = round(safe_float(recent_7[col].mean()), 2)
                kpi[f"{col}_30d_avg"] = round(safe_float(recent_30[col].mean()), 2)

        # 트렌드: 7일 vs 이전 7일 비교
        if len(df_sorted) >= 14:
            prev_7 = df_sorted.iloc[7:14]
            for col in ("gmv", "active_sellers", "orders", "new_signups"):
                if col in df_sorted.columns:
                    cur = safe_float(recent_7[col].mean())
                    prev = safe_float(prev_7[col].mean())
                    if prev > 0:
                        trends[f"{col}_wow_change_pct"] = round((cur - prev) / prev * 100, 2)
                    else:
                        trends[f"{col}_wow_change_pct"] = 0.0

    # ── SHOPS_DF: 총 쇼핑몰 수, 플랜별 분포 ──
    df = st.SHOPS_DF
    if df is not None and not df.empty:
        kpi["total_shops"] = len(df)
        if "plan_tier" in df.columns:
            kpi["shops_by_plan"] = df["plan_tier"].value_counts().to_dict()

    # ── SELLERS_DF / SELLER_ANALYTICS_DF: 셀러 수, 세그먼트 분포 ──
    seller_df = st.SELLER_ANALYTICS_DF
    if seller_df is not None and not seller_df.empty:
        kpi["total_sellers"] = len(seller_df)
        if "cluster" in seller_df.columns:
            segments["seller_segments"] = seller_df["cluster"].value_counts().to_dict()
        if "is_anomaly" in seller_df.columns:
            kpi["anomaly_sellers"] = int(seller_df["is_anomaly"].sum())
    elif st.SELLERS_DF is not None and not st.SELLERS_DF.empty:
        kpi["total_sellers"] = len(st.SELLERS_DF)
        if "plan_tier" in st.SELLERS_DF.columns:
            segments["sellers_by_plan"] = st.SELLERS_DF["plan_tier"].value_counts().to_dict()

    # ── CS_STATS_DF: CS 문의 건수, 카테고리별 분포 ──
    df = st.CS_STATS_DF
    if df is not None and not df.empty:
        if "total_tickets" in df.columns:
            cs_summary["total_tickets"] = safe_int(df["total_tickets"].sum())
        else:
            cs_summary["total_tickets"] = len(df)

        if "satisfaction_score" in df.columns:
            cs_summary["avg_satisfaction"] = round(safe_float(df["satisfaction_score"].mean()), 2)

        if "avg_resolution_hours" in df.columns:
            cs_summary["avg_resolution_hours"] = round(safe_float(df["avg_resolution_hours"].mean()), 1)

        cat_col = "category" if "category" in df.columns else "ticket_category"
        if cat_col in df.columns and "total_tickets" in df.columns:
            cs_summary["by_category"] = {
                safe_str(row[cat_col]): safe_int(row["total_tickets"])
                for _, row in df.iterrows()
            }
        elif cat_col in df.columns:
            cs_summary["by_category"] = df[cat_col].value_counts().to_dict()

    # ── FRAUD_DETAILS_DF: 이상거래 건수 ──
    df = st.FRAUD_DETAILS_DF
    if df is not None and not df.empty:
        kpi["fraud_total"] = len(df)
        if "fraud_type" in df.columns:
            kpi["fraud_by_type"] = df["fraud_type"].value_counts().to_dict()

    # ── COHORT_RETENTION_DF: 코호트 리텐션율 ──
    df = st.COHORT_RETENTION_DF
    if df is not None and not df.empty:
        # 가장 최근 코호트의 리텐션율
        cohort_col = None
        for c in ("cohort", "cohort_month", "cohort_date"):
            if c in df.columns:
                cohort_col = c
                break
        if cohort_col:
            latest_cohort = df[cohort_col].max()
            latest_rows = df[df[cohort_col] == latest_cohort]
            retention_cols = [c for c in df.columns if c.startswith("month_") or c.startswith("m")]
            if retention_cols:
                kpi["latest_cohort"] = safe_str(latest_cohort)
                kpi["cohort_retention"] = {
                    c: round(safe_float(latest_rows[c].mean()), 2) for c in retention_cols
                }

    return {
        "kpi": kpi,
        "trends": trends,
        "segments": segments,
        "cs_summary": cs_summary,
        "collected_at": time.time(),
    }


def generate_report(report_type: str = "daily", api_key: str = "") -> Dict[str, Any]:
    """LLM으로 운영 리포트를 자동 생성합니다."""
    report_id = str(uuid.uuid4())[:8]
    type_label = _REPORT_TYPE_LABELS.get(report_type, report_type)
    run_id = create_pipeline_run("report", ["collect", "aggregate", "write", "save"])
    update_pipeline_step(run_id, "collect", "processing")

    try:
        # 데이터 수집
        data = collect_report_data()

        update_pipeline_step(run_id, "collect", "complete", {"kpi_count": len(data.get("kpi", {}))})
        update_pipeline_step(run_id, "aggregate", "complete", {"trends": len(data.get("trends", {}))})
        update_pipeline_step(run_id, "write", "processing")

        # LLM 호출 준비
        resolved_key = pick_api_key(api_key)
        if not resolved_key:
            return {
                "report_id": report_id,
                "report_type": report_type,
                "content": f"# {type_label} 운영 리포트\n\nAPI 키가 설정되지 않아 LLM 리포트를 생성할 수 없습니다.",
                "data_summary": data,
                "timestamp": time.time(),
                "error": "API 키 없음",
            }

        settings = st.get_active_llm_settings()
        model = settings.get("selectedModel", "gpt-4o-mini")
        max_tokens = settings.get("maxTokens", 4000)

        llm = get_llm(
            model=model,
            api_key=resolved_key,
            max_tokens=max_tokens,
            streaming=False,
            temperature=0.3,
        )

        system_prompt = (
            "당신은 카페24 이커머스 플랫폼 운영 분석가입니다.\n"
            f"제공된 KPI 데이터를 기반으로 {type_label} 운영 리포트를 작성합니다.\n\n"
            "리포트 구성:\n"
            "1. 핵심 지표 요약 (GMV, 주문수, 활성셀러 등)\n"
            "2. 주요 변화 및 트렌드\n"
            "3. 이슈 & 주의사항 (이상거래, 이탈위험 등)\n"
            "4. 권장 조치사항\n\n"
            "마크다운 형식으로 작성하세요."
        )

        import json
        data_text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        user_prompt = (
            f"다음은 카페24 플랫폼의 최신 운영 데이터입니다.\n"
            f"이 데이터를 기반으로 {type_label} 운영 리포트를 작성해 주세요.\n\n"
            f"```json\n{data_text}\n```"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        content = invoke_with_retry(llm, messages)
        update_pipeline_step(run_id, "write", "complete")
        update_pipeline_step(run_id, "save", "processing")

        result = {
            "report_id": report_id,
            "report_type": report_type,
            "content": content,
            "data_summary": data,
            "timestamp": time.time(),
            "pipeline_run_id": run_id,
        }

        # 히스토리 저장
        save_report(result)
        log_action("report_generate", report_id, {
            "report_type": report_type,
            "model": model,
            "kpi_keys": list(data.get("kpi", {}).keys()),
        })

        update_pipeline_step(run_id, "save", "complete")
        complete_pipeline_run(run_id)

        st.logger.info("REPORT_GENERATED id=%s type=%s model=%s", report_id, report_type, model)
        return result

    except Exception as e:
        if run_id:
            update_pipeline_step(run_id, "write", "error", {"error": safe_str(e)})
        st.logger.error("REPORT_GENERATE_FAIL id=%s err=%s", report_id, safe_str(e))
        log_action("report_generate", report_id, {
            "report_type": report_type,
            "error": safe_str(e),
        }, status="error")
        return {
            "report_id": report_id,
            "report_type": report_type,
            "content": f"# {type_label} 운영 리포트 생성 실패\n\n오류: {safe_str(e)}",
            "data_summary": {},
            "timestamp": time.time(),
            "error": safe_str(e),
        }


def get_history(limit: int = 20) -> Dict[str, Any]:
    """리포트 히스토리를 조회합니다."""
    reports = get_report_history(limit)
    return {
        "total": len(reports),
        "reports": reports,
    }
