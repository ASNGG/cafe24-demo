"""
automation/faq_engine.py - CS FAQ 자동 생성 엔진
================================================
CS 문의 원문 TF-IDF 임베딩 → 실루엣 계수 최적 K → K-Means 클러스터링 → LLM FAQ 생성
"""
import json
import random
import re
import uuid
import time
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str, safe_int
from core.constants import CS_TICKET_CATEGORIES
from agent.llm import get_llm, invoke_with_retry, pick_api_key
from automation.action_logger import (
    save_faq,
    get_faq,
    get_all_faqs,
    delete_faq,
    update_faq_status,
    log_action,
    create_pipeline_run,
    update_pipeline_step,
    complete_pipeline_run,
)
import state as st


# ── FAQ 생성용 시스템 프롬프트 ──
_FAQ_SYSTEM_PROMPT = (
    "당신은 카페24 이커머스 플랫폼 CS 전문가입니다.\n"
    "실제 고객 문의 클러스터링 결과를 기반으로 FAQ를 생성합니다.\n"
    "각 FAQ는 question, answer, category, tags 형식으로 JSON 배열로 반환하세요.\n"
    "반드시 유효한 JSON 배열만 출력하세요. 다른 텍스트는 포함하지 마세요.\n\n"
    "중요 규칙:\n"
    "- 실제 문의 패턴에 기반하여 대표 질문을 정리하고 정확한 답변을 작성하세요.\n"
    "- 특정 카테고리가 지정된 경우, 모든 FAQ의 category 값은 반드시 해당 카테고리와 동일해야 합니다.\n"
    "- 허용 카테고리: 배송, 환불, 결제, 상품, 계정, 정산, 기술지원, 마케팅, 기타"
)


def _find_optimal_k(tfidf_matrix) -> dict:
    """실루엣 계수로 최적 K를 탐색합니다. k_max는 데이터 크기에 비례."""
    n_samples = tfidf_matrix.shape[0]
    # 데이터 크기 대비 적절한 k_max (너무 많으면 의미 없는 1~2건 클러스터만 생김)
    k_max = min(max(n_samples // 8, 3), 10, n_samples - 1)
    k_min = 2
    if k_max < k_min:
        return {"optimal_k": k_min, "silhouette": 0.0, "scores": []}

    scores = []
    for k in range(k_min, k_max + 1):
        # MiniBatchKMeans: 메모리 효율적, batch_size로 메모리 사용량 제어
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=256)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        scores.append({"k": k, "silhouette": round(float(score), 4)})

    best = max(scores, key=lambda x: x["silhouette"])
    return {
        "optimal_k": best["k"],
        "silhouette": best["silhouette"],
        "scores": scores,
    }


def _cluster_with_optimal_k(texts: List[str]) -> Dict[str, Any]:
    """TF-IDF + 실루엣 최적 K + MiniBatchKMeans 클러스터링 (메모리 효율적)."""
    if len(texts) < 3:
        return {
            "optimal_k": 1,
            "silhouette": 0.0,
            "scores": [],
            "clusters": [{"cluster_id": 0, "size": len(texts), "representative": texts[0] if texts else "", "samples": texts[:3]}],
        }

    vectorizer = TfidfVectorizer(max_features=500, max_df=0.95, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 실루엣 계수로 최적 K 탐색
    k_result = _find_optimal_k(tfidf_matrix)
    optimal_k = k_result["optimal_k"]

    # 최적 K로 최종 클러스터링 (MiniBatchKMeans: 메모리 효율적)
    kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init=3, batch_size=256)
    labels = kmeans.fit_predict(tfidf_matrix)

    # PCA 2D 좌표 (산점도용)
    n_components = min(2, tfidf_matrix.shape[1])
    pca = PCA(n_components=n_components)
    coords_2d = pca.fit_transform(tfidf_matrix.toarray())
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    clusters = []
    points = []
    for cid in range(optimal_k):
        mask = labels == cid
        indices = np.where(mask)[0]
        cluster_texts = [texts[i] for i in indices]

        # 대표 질문: 클러스터 중심에 가장 가까운 문의
        center = kmeans.cluster_centers_[cid]
        dists = np.asarray(tfidf_matrix[indices].dot(center.T)).flatten()
        rep_idx = indices[np.argmax(dists)]

        clusters.append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "representative": texts[rep_idx],
            "samples": cluster_texts[:5],
        })

        # 각 문의의 2D 좌표
        for idx in indices:
            points.append({
                "x": round(float(coords_2d[idx][0]), 4),
                "y": round(float(coords_2d[idx][1]), 4),
                "cluster": cid,
                "text": texts[idx][:40],
            })

    clusters.sort(key=lambda x: x["size"], reverse=True)

    # 중심점 2D 좌표
    centroid_points = [
        {"x": round(float(centroids_2d[cid][0]), 4),
         "y": round(float(centroids_2d[cid][1]), 4),
         "cluster": cid}
        for cid in range(optimal_k)
    ]

    return {
        "optimal_k": optimal_k,
        "silhouette": k_result["silhouette"],
        "scores": k_result["scores"],
        "clusters": clusters,
        "points": points,
        "centroids": centroid_points,
    }


def _cluster_with_llm(texts: List[str], category: str = "", api_key: str = "") -> Dict[str, Any]:
    """LLM으로 문의 텍스트를 의미 기반 그룹핑합니다."""
    api_key = pick_api_key(api_key)
    if not api_key:
        return {"optimal_k": 0, "silhouette": 0.0, "scores": [], "clusters": []}

    # 중복 제거 후 LLM에 전달 (토큰 절약)
    unique_texts = list(set(texts))
    sample = unique_texts[:50]  # 최대 50개만

    prompt = (
        f"아래는 카페24 CS '{category or '전체'}' 카테고리의 고객 문의 목록입니다.\n"
        f"비슷한 질문끼리 그룹으로 묶고, 각 그룹에 주제 라벨을 붙여주세요.\n\n"
        f"[문의 목록]\n" + "\n".join(f"- {t}" for t in sample) + "\n\n"
        f"다음 JSON 형식으로만 반환하세요:\n"
        f'[{{"label": "주제 라벨", "representative": "대표 질문", "members": ["질문1", "질문2", ...]}}]\n'
        f"반드시 유효한 JSON 배열만 출력하세요."
    )

    settings = st.get_active_llm_settings()
    llm = get_llm(
        model=settings.get("selectedModel", "gpt-4o-mini"),
        api_key=api_key, max_tokens=4000, streaming=False, temperature=0.3,
    )
    raw = invoke_with_retry(llm, [
        SystemMessage(content="당신은 CS 문의 분류 전문가입니다. 비슷한 질문을 그룹핑합니다."),
        HumanMessage(content=prompt),
    ])
    groups = _parse_faq_json(raw)
    if not groups:
        return {"optimal_k": 0, "silhouette": 0.0, "scores": [], "clusters": []}

    # 원본 텍스트에서 건수 매칭
    clusters = []
    for i, g in enumerate(groups):
        members = g.get("members", [])
        # 원본에서 해당 그룹에 속하는 문의 수 계산 (부분 매칭)
        count = 0
        matched_samples = []
        for t in texts:
            for m in members:
                if m in t or t in m:
                    count += 1
                    if len(matched_samples) < 5:
                        matched_samples.append(t)
                    break
        if count == 0:
            count = len(members)
            matched_samples = members[:5]

        clusters.append({
            "cluster_id": i,
            "size": count,
            "representative": g.get("representative", members[0] if members else g.get("label", "")),
            "label": g.get("label", ""),
            "samples": matched_samples,
        })

    clusters.sort(key=lambda x: x["size"], reverse=True)
    return {
        "optimal_k": len(clusters),
        "silhouette": 0.0,
        "scores": [],
        "clusters": clusters,
    }


def analyze_cs_patterns(
    category: Optional[str] = None,
    top_n: int = 10,
    mode: str = "kmeans",
    api_key: str = "",
) -> Dict[str, Any]:
    """CS 문의 패턴 분석. mode='kmeans' (TF-IDF+K-Means) 또는 mode='llm' (LLM 의미 분류)."""
    if st.CS_TICKETS_DF is None or "inquiry_text" not in st.CS_TICKETS_DF.columns:
        return _analyze_stats_fallback(category)

    df = st.CS_TICKETS_DF.copy()
    cat_col = "category" if "category" in df.columns else None

    # 카테고리별 통계
    categories = []
    if cat_col:
        total = len(df)
        for cat, grp in df.groupby(cat_col):
            categories.append({
                "category": str(cat),
                "count": len(grp),
                "percentage": round(len(grp) / total * 100, 1),
            })
        categories.sort(key=lambda x: x["count"], reverse=True)

    # 클러스터링 함수 선택
    cluster_fn = _cluster_with_llm if mode == "llm" else _cluster_with_optimal_k
    method_name = "llm" if mode == "llm" else "clustering"

    # 특정 카테고리 선택 시
    if category:
        if cat_col:
            df = df[df[cat_col] == category]
        if len(df) == 0:
            return {"total_inquiries": 0, "clusters": [], "categories": categories,
                    "category_results": [], "method": method_name}

        texts = df["inquiry_text"].dropna().tolist()
        if mode == "llm":
            result = cluster_fn(texts, category=category, api_key=api_key)
        else:
            result = cluster_fn(texts)
        cat_result = {
            "category": category,
            "count": len(texts),
            "optimal_k": result["optimal_k"],
            "silhouette": result["silhouette"],
            "scores": result["scores"],
            "clusters": result["clusters"],
            "points": result.get("points", []),
            "centroids": result.get("centroids", []),
        }
        return {
            "total_inquiries": len(texts),
            "clusters": result["clusters"],
            "categories": categories,
            "category_results": [cat_result],
            "method": method_name,
        }

    # 전체: 카테고리별로 각각 클러스터링
    category_results = []
    all_clusters = []

    if cat_col:
        cat_groups = [(str(cat), grp) for cat, grp in df.groupby(cat_col)]

        # LLM 모드: 전체 카테고리 분석 시 최대 3개 랜덤 선택 (속도)
        if mode == "llm" and len(cat_groups) > 3:
            cat_groups_sorted = sorted(cat_groups, key=lambda x: len(x[1]), reverse=True)
            cat_groups = random.sample(cat_groups_sorted[:6], min(3, len(cat_groups_sorted)))

        for cat, grp in cat_groups:
            texts = grp["inquiry_text"].dropna().tolist()
            if len(texts) < 2:
                continue
            if mode == "llm":
                result = cluster_fn(texts, category=cat, api_key=api_key)
            else:
                result = cluster_fn(texts)
            for cl in result["clusters"]:
                cl["category"] = cat
            cat_result = {
                "category": cat,
                "count": len(texts),
                "optimal_k": result["optimal_k"],
                "silhouette": result["silhouette"],
                "scores": result["scores"],
                "clusters": result["clusters"],
                "points": result.get("points", []),
                "centroids": result.get("centroids", []),
            }
            category_results.append(cat_result)
            all_clusters.extend(result["clusters"])
    else:
        texts = df["inquiry_text"].dropna().tolist()
        if mode == "llm":
            result = cluster_fn(texts, api_key=api_key)
        else:
            result = cluster_fn(texts)
        all_clusters = result["clusters"]
        category_results = [{"category": "전체", "count": len(texts),
                             "optimal_k": result["optimal_k"],
                             "silhouette": result["silhouette"],
                             "scores": result["scores"],
                             "clusters": result["clusters"]}]

    # 전체 클러스터를 건수 순 정렬
    all_clusters.sort(key=lambda x: x["size"], reverse=True)

    return {
        "total_inquiries": len(df),
        "clusters": all_clusters[:top_n],
        "categories": categories,
        "category_results": category_results,
        "method": method_name,
    }


def _analyze_stats_fallback(category: Optional[str] = None) -> Dict[str, Any]:
    """cs_stats.csv 기반 fallback 분석."""
    if st.CS_STATS_DF is None:
        return {"total_inquiries": 0, "clusters": [], "categories": [],
                "category_results": [], "method": "no_data"}

    df = st.CS_STATS_DF
    cat_col = "category" if "category" in df.columns else "ticket_category"
    if cat_col not in df.columns or "total_tickets" not in df.columns:
        return {"total_inquiries": 0, "clusters": [], "categories": [],
                "category_results": [], "method": "stats_fallback"}

    total = safe_int(df["total_tickets"].sum())
    categories = []
    for _, row in df.iterrows():
        count = safe_int(row["total_tickets"])
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        categories.append({"category": safe_str(row[cat_col]), "count": count, "percentage": pct})
    categories.sort(key=lambda x: x["count"], reverse=True)

    if category:
        categories = [c for c in categories if c["category"] == category]

    return {
        "total_inquiries": total,
        "clusters": [],
        "categories": categories,
        "category_results": [],
        "method": "stats_fallback",
    }


def generate_faq_items(
    category: Optional[str] = None,
    count: int = 5,
    mode: str = "kmeans",
    api_key: str = "",
    selected_clusters: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """클러스터링 결과 기반으로 LLM을 사용하여 FAQ를 자동 생성합니다.
    selected_clusters가 주어지면 재분석 없이 해당 클러스터만 사용합니다."""
    run_id = None
    try:
        if category and category not in CS_TICKET_CATEGORIES:
            return {
                "generated_count": 0, "faqs": [],
                "error": f"유효하지 않은 카테고리입니다: {category}. "
                         f"사용 가능: {', '.join(CS_TICKET_CATEGORIES)}",
            }

        api_key = pick_api_key(api_key)
        if not api_key:
            return {"generated_count": 0, "faqs": [], "error": "API 키가 설정되지 않았습니다."}

        run_id = create_pipeline_run("faq", ["analyze", "generate", "review", "approve"])

        # 선택된 클러스터가 있으면 재분석 생략
        if selected_clusters:
            clusters = selected_clusters
            method = mode
            update_pipeline_step(run_id, "analyze", "complete", {
                "method": method,
                "clusters": len(clusters),
                "category": category or "all",
                "note": "사용자 선택 클러스터",
            })
        else:
            update_pipeline_step(run_id, "analyze", "processing")
            patterns = analyze_cs_patterns(category=category, top_n=count * 2, mode=mode, api_key=api_key)
            clusters = patterns.get("clusters", [])
            method = patterns.get("method", "no_data")

            update_pipeline_step(run_id, "analyze", "complete", {
                "method": method,
                "total": patterns.get("total_inquiries", 0),
                "clusters": len(clusters),
                "category": category or "all",
            })

            if not clusters and not patterns.get("categories"):
                return {"generated_count": 0, "faqs": [], "warning": "분석할 CS 문의 데이터가 없습니다."}

        update_pipeline_step(run_id, "generate", "processing")

        # 프롬프트: 클러스터 정보 포함
        if clusters:
            cluster_text = ""
            for i, cl in enumerate(clusters[:count], 1):
                samples = "\n    ".join((cl.get("samples") or [])[:3])
                cat_label = f" [{cl['category']}]" if cl.get("category") else ""
                cluster_text += (
                    f"\n클러스터 {i}{cat_label} ({cl.get('size', 0)}건):\n"
                    f"  대표 질문: {cl.get('representative', '')}\n"
                    f"  유사 질문 예시:\n    {samples}\n"
                )

            user_prompt = (
                f"아래는 카페24 CS 문의를 클러스터링한 결과입니다.\n"
                f"각 클러스터의 대표 질문과 유사 질문을 참고하여 FAQ {count}개를 생성하세요.\n"
                f"빈도가 높은 클러스터(건수가 많은 것)를 우선적으로 FAQ로 만드세요.\n\n"
                f"[클러스터링 결과]{cluster_text}\n"
            )
        else:
            cat_text = "\n".join(
                f"- {c['category']}: {c['count']}건 ({c['percentage']}%)"
                for c in patterns.get("categories", [])[:5]
            )
            user_prompt = (
                f"아래 카페24 CS 문의 패턴을 분석하여 FAQ {count}개를 생성하세요.\n\n"
                f"[문의 패턴]\n{cat_text}\n"
            )

        if category:
            user_prompt += (
                f"\n[필수 조건]\n"
                f"반드시 '{category}' 카테고리에 해당하는 FAQ만 생성하세요.\n"
                f"모든 FAQ의 category 값은 '{category}'이어야 합니다.\n"
            )

        user_prompt += (
            f"\n각 FAQ는 다음 형식의 JSON 배열로 반환하세요:\n"
            f'[{{"question": "...", "answer": "...", "category": "...", "tags": ["태그1", "태그2"]}}]\n\n'
            f"답변은 카페24 이커머스 플랫폼 맥락에 맞게 구체적으로 작성하세요."
        )

        settings = st.get_active_llm_settings()
        llm = get_llm(
            model=settings.get("selectedModel", "gpt-4o-mini"),
            api_key=api_key,
            max_tokens=settings.get("maxTokens", 4000),
            streaming=False,
            temperature=0.7,
        )

        messages = [
            SystemMessage(content=_FAQ_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        raw = invoke_with_retry(llm, messages)
        faq_list = _parse_faq_json(raw)
        if not faq_list:
            st.logger.error("FAQ JSON 파싱 실패: %s", raw[:200])
            return {"generated_count": 0, "faqs": [], "error": "LLM 응답 파싱 실패"}

        saved_faqs = []
        for item in faq_list[:count]:
            faq_id = str(uuid.uuid4())[:8]
            faq_data = {
                "id": faq_id,
                "question": safe_str(item.get("question")),
                "answer": safe_str(item.get("answer")),
                "category": category if category else safe_str(item.get("category")),
                "tags": item.get("tags", []),
                "status": "draft",
                "created_at": time.time(),
            }
            save_faq(faq_id, faq_data)
            saved_faqs.append(faq_data)

        update_pipeline_step(run_id, "generate", "complete", {"count": len(saved_faqs)})

        log_action(
            "faq_generate",
            "system",
            {"count": len(saved_faqs), "category": category or "all", "method": method},
        )

        return {
            "generated_count": len(saved_faqs),
            "faqs": saved_faqs,
            "pipeline_run_id": run_id,
            "method": method,
            "clusters_used": len(clusters),
        }

    except Exception as e:
        st.logger.error("FAQ 생성 실패: %s", str(e))
        if run_id:
            update_pipeline_step(run_id, "generate", "error", {"error": str(e)})
        return {"generated_count": 0, "faqs": [], "error": str(e)}


def approve_faq(faq_id: str) -> Dict[str, Any]:
    ok = update_faq_status(faq_id, "approved")
    if not ok:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}
    log_action("faq_approve", faq_id, {"faq_id": faq_id})
    return {"status": "success", "faq_id": faq_id}


def update_faq(faq_id: str, question: Optional[str] = None, answer: Optional[str] = None) -> Dict[str, Any]:
    existing = get_faq(faq_id)
    if not existing:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}
    updated_fields = []
    if question is not None:
        existing["question"] = question
        updated_fields.append("question")
    if answer is not None:
        existing["answer"] = answer
        updated_fields.append("answer")
    if not updated_fields:
        return {"status": "error", "message": "수정할 필드가 없습니다."}
    existing["updated_at"] = time.time()
    save_faq(faq_id, existing)
    log_action("faq_update", faq_id, {"updated_fields": updated_fields})
    return {"status": "success", "faq_id": faq_id, "updated_fields": updated_fields}


def delete_faq_item(faq_id: str) -> Dict[str, Any]:
    ok = delete_faq(faq_id)
    if not ok:
        return {"status": "error", "message": f"FAQ '{faq_id}'를 찾을 수 없습니다."}
    log_action("faq_delete", faq_id, {"faq_id": faq_id})
    return {"status": "success", "faq_id": faq_id}


def list_faqs(status: Optional[str] = None) -> Dict[str, Any]:
    all_faqs = get_all_faqs()
    if status and status != "all":
        all_faqs = [f for f in all_faqs if f.get("status") == status]
    return {"total": len(all_faqs), "faqs": all_faqs}


def _parse_faq_json(raw: str) -> List[Dict[str, Any]]:
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    return []
