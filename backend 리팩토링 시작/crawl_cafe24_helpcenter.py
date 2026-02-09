"""
crawl_cafe24_helpcenter.py - CAFE24 Help Center 크롤러
======================================================
CAFE24 Help Center (support.cafe24.com) 의 전체 기사를
Zendesk API로 가져와 rag_docs/ 에 Markdown 파일로 저장합니다.

사용법 (Jupyter에서):
    from crawl_cafe24_helpcenter import crawl_all
    results = crawl_all()

    # 특정 카테고리만:
    results = crawl_all(category_filter=["쇼핑몰"])

터미널:
    python crawl_cafe24_helpcenter.py
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from typing import Optional

# ============================================================
# 설정
# ============================================================
API_BASE = "https://support.cafe24.com/api/v2/help_center/ko"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_docs")
PER_PAGE = 100  # Zendesk API 최대
REQUEST_DELAY = 0.3  # API 요청 간 딜레이 (초)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ============================================================
# API 호출
# ============================================================
def api_get(endpoint: str, params: Optional[dict] = None) -> dict:
    """Zendesk API GET 요청"""
    url = f"{API_BASE}/{endpoint}" if not endpoint.startswith("http") else endpoint
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  [API ERROR] {url} -> {e}")
        return {}


def fetch_all_pages(endpoint: str, key: str, params: Optional[dict] = None) -> list:
    """페이지네이션 처리하여 모든 결과 수집"""
    all_items = []
    p = params or {}
    p["per_page"] = PER_PAGE
    page = 1

    while True:
        p["page"] = page
        data = api_get(endpoint, p)
        items = data.get(key, [])
        if not items:
            break
        all_items.extend(items)

        # 다음 페이지 확인
        if not data.get("next_page"):
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return all_items


# ============================================================
# 데이터 수집
# ============================================================
def fetch_categories() -> dict:
    """카테고리 목록 (id -> name 매핑)"""
    cats = fetch_all_pages("categories.json", "categories")
    return {c["id"]: c["name"] for c in cats}


def fetch_sections() -> dict:
    """섹션 목록 (id -> {name, category_id} 매핑)"""
    secs = fetch_all_pages("sections.json", "sections")
    return {s["id"]: {"name": s["name"], "category_id": s["category_id"]} for s in secs}


def fetch_articles(category_filter: Optional[list] = None) -> list:
    """전체 기사 목록"""
    articles = fetch_all_pages("articles.json", "articles", {"sort_by": "created_at", "sort_order": "desc"})

    if category_filter:
        # 카테고리 필터가 있으면 해당 카테고리만
        cats = fetch_categories()
        allowed_cat_ids = set()
        for cid, cname in cats.items():
            for f in category_filter:
                if f in cname:
                    allowed_cat_ids.add(cid)

        secs = fetch_sections()
        allowed_section_ids = {sid for sid, s in secs.items() if s["category_id"] in allowed_cat_ids}
        articles = [a for a in articles if a.get("section_id") in allowed_section_ids]

    return articles


# ============================================================
# 텍스트 변환
# ============================================================
def html_to_markdown(html_body: str) -> str:
    """HTML 본문을 정제된 텍스트로 변환"""
    if not html_body:
        return ""

    soup = BeautifulSoup(html_body, "html.parser")

    # 불필요 태그 제거
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    lines = []
    _convert_element(soup, lines)

    text = "\n".join(lines)
    # 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 중복 라인 제거
    deduped = []
    for line in text.split("\n"):
        if not deduped or line.strip() != deduped[-1].strip() or line.strip() == "":
            deduped.append(line)
    text = "\n".join(deduped)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _convert_element(el, lines: list):
    """재귀적으로 HTML -> Markdown 변환"""
    from bs4 import NavigableString, Tag

    if isinstance(el, NavigableString):
        text = str(el).strip()
        if text:
            lines.append(text)
        return

    if not isinstance(el, Tag):
        return

    tag = el.name

    if tag in ("h1", "h2", "h3", "h4", "h5"):
        text = el.get_text(strip=True)
        if text:
            level = int(tag[1])
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        return

    if tag == "li":
        text = el.get_text(strip=True)
        if text:
            lines.append(f"- {text}")
        return

    if tag == "p":
        text = el.get_text(strip=True)
        if text:
            lines.append(text)
            lines.append("")
        return

    if tag == "table":
        rows = el.find_all("tr")
        if rows:
            lines.append("")
            for i, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                cell_texts = [c.get_text(strip=True).replace("|", "/") for c in cells]
                if cell_texts:
                    lines.append("| " + " | ".join(cell_texts) + " |")
                    if i == 0 and row.find("th"):
                        lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
            lines.append("")
        return

    if tag == "br":
        lines.append("")
        return

    # 나머지 태그는 자식 재귀
    for child in el.children:
        _convert_element(child, lines)


def sanitize_filename(s: str) -> str:
    """파일명에 사용할 수 없는 문자 제거"""
    s = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", s)
    s = re.sub(r"\s+", "_", s)
    # ASCII + 한글만 남기기
    s = re.sub(r"[^\w가-힣_\-.]", "", s)
    return s[:60]


# ============================================================
# 섹션별 합치기 & 저장
# ============================================================
def crawl_all(category_filter: Optional[list] = None) -> list:
    """
    전체 크롤링 → 섹션별로 합쳐서 .md 파일로 저장.

    Args:
        category_filter: 카테고리 이름 필터 (예: ["쇼핑몰"]이면 쇼핑몰 관련만)
                         None이면 전체.
    """
    print("=" * 60)
    print("CAFE24 Help Center 크롤러 (Zendesk API)")
    print("=" * 60)
    print(f"저장 경로: {OUTPUT_DIR}")
    if category_filter:
        print(f"필터: {category_filter}")
    print()

    # 1) 메타데이터 수집
    print("[1/3] 카테고리 & 섹션 정보 수집...")
    categories = fetch_categories()
    sections = fetch_sections()
    print(f"  -> {len(categories)}개 카테고리, {len(sections)}개 섹션")

    # 2) 기사 수집
    print("[2/3] 기사 목록 수집 중 (31페이지)...")
    articles = fetch_articles(category_filter)
    print(f"  -> {len(articles)}개 기사")
    print()

    # 3) 섹션별로 그룹핑 & 저장
    print("[3/3] 섹션별 파일 저장 중...")

    # 섹션별 기사 그룹핑
    section_articles: dict[int, list] = {}
    skipped = 0
    for art in articles:
        content = html_to_markdown(art.get("body", ""))
        if not content or len(content) < 20:
            skipped += 1
            continue
        sid = art.get("section_id", 0)
        if sid not in section_articles:
            section_articles[sid] = []
        section_articles[sid].append({
            "title": art.get("title", ""),
            "id": art.get("id", 0),
            "content": content,
        })

    # 섹션별 파일 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for sid, arts in section_articles.items():
        sec_info = sections.get(sid, {})
        section_name = sec_info.get("name", "기타")
        cat_id = sec_info.get("category_id")
        category_name = categories.get(cat_id, "기타")

        safe_cat = sanitize_filename(category_name)
        safe_sec = sanitize_filename(section_name)
        filename = f"cafe24_hc_{safe_cat}_{safe_sec}_{sid}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 섹션 내 전체 기사를 하나의 Markdown으로 합치기
        parts = [
            f"# {category_name} - {section_name}",
            "",
            f"> source: https://support.cafe24.com/hc/ko/sections/{sid}",
            f"> category: {category_name}",
            f"> section: {section_name}",
            f"> articles: {len(arts)}개",
            "",
            "---",
            "",
        ]

        for art in arts:
            url = f"https://support.cafe24.com/hc/ko/articles/{art['id']}"
            parts.append(f"## {art['title']}")
            parts.append(f"<!-- {url} -->")
            parts.append("")
            parts.append(art["content"])
            parts.append("")
            parts.append("---")
            parts.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        total_chars = sum(len(a["content"]) for a in arts)
        results.append({
            "section_id": sid,
            "category": category_name,
            "section": section_name,
            "articles_count": len(arts),
            "total_chars": total_chars,
            "file": filepath,
        })

    # 결과 요약
    print()
    print("=" * 60)
    print("크롤링 완료!")
    total_articles = sum(r["articles_count"] for r in results)
    total_chars = sum(r["total_chars"] for r in results)
    print(f"  섹션 파일: {len(results)}개")
    print(f"  포함 기사: {total_articles}개")
    print(f"  스킵: {skipped}개")
    print(f"  총 텍스트: {total_chars:,}자")
    print(f"  저장 위치: {OUTPUT_DIR}")

    # 카테고리별 통계
    print()
    print("카테고리별:")
    cat_stats: dict[str, dict] = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"sections": 0, "articles": 0, "chars": 0}
        cat_stats[c]["sections"] += 1
        cat_stats[c]["articles"] += r["articles_count"]
        cat_stats[c]["chars"] += r["total_chars"]

    for cat, st in sorted(cat_stats.items(), key=lambda x: -x[1]["articles"]):
        print(f"  {cat:35s} {st['sections']:>3}섹션 {st['articles']:>5}기사 {st['chars']:>8,}자")

    print()
    print("다음 단계: /api/rag/reload 호출하여 RAG 인덱스에 반영하세요.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    crawl_all()
else:
    print("crawl_cafe24_helpcenter 모듈 로드 완료.")
    print("  crawl_all()           -> 전체 크롤링")
    print("  crawl_all(['쇼핑몰'])  -> 쇼핑몰 카테고리만")
