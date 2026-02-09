"""
crawl_cafe24_guide.py - CAFE24 서비스 가이드 크롤러
====================================================
CAFE24 서비스 가이드(https://serviceguide.cafe24.com/IN/ko_KR/)
페이지를 크롤링하여 rag_docs/ 에 Markdown 파일로 저장합니다.

사용법 (Jupyter에서):
    %run crawl_cafe24_guide.py

또는 터미널에서:
    python crawl_cafe24_guide.py
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ============================================================
# 설정
# ============================================================
BASE_URL = "https://serviceguide.cafe24.com/IN/ko_KR/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_docs")
REQUEST_DELAY = 1.5  # 요청 간 딜레이 (초)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 페이지 코드 → 한글 이름 매핑
PAGE_MAP = {
    "SG.SI": "기본설정",
    "SG.PA": "결제관리",
    "SG.DH": "배송설정",
    "PT.PE": "상품등록",
    "PT.PL": "상품목록",
    "PT.PC": "상품분류관리",
    "OD.AO": "전체주문목록",
    "OD.OF.DB": "배송관리",
    "OD.OW.CM": "취소교환반품환불",
    "CR.CR": "고객관리",
    "BD.BE": "게시판설정",
    "BD.BA": "게시물관리",
    "DN.DK": "디자인보관함",
    "DN.DA": "디자인추가",
    "DN.BR": "디자인백업복구",
    "PN.MB.BR": "혜택관리",
    "PN.PN.CU": "쿠폰관리",
    "PN.MD": "적립금관리",
}


# ============================================================
# 크롤링 함수
# ============================================================
def fetch_page(url: str) -> str | None:
    """URL에서 HTML 가져오기"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        print(f"  [ERROR] {url} → {e}")
        return None


def discover_pages(html: str) -> list[tuple[str, str]]:
    """메인 페이지에서 하위 링크 자동 수집 (코드, URL 튜플 리스트)"""
    soup = BeautifulSoup(html, "html.parser")
    pages = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # 잘못된 URL 수정 (hhttps → https)
        if href.startswith("hhttps://"):
            href = href[1:]
        if not href.startswith(("http://", "https://")):
            href = urljoin(BASE_URL, href)

        # serviceguide.cafe24.com/IN/ko_KR/ 하위 .html 파일만
        if "/IN/ko_KR/" not in href or not href.endswith(".html"):
            continue

        # 코드 추출 (예: SG.SI.html → SG.SI)
        filename = href.rsplit("/", 1)[-1]
        code = filename.replace(".html", "")

        if code not in seen:
            seen.add(code)
            pages.append((code, href))

    return pages


def _extract_element(el, lines: list, depth: int = 0):
    """재귀적으로 HTML 요소를 Markdown 텍스트로 변환 (중복 방지)"""
    from bs4 import NavigableString, Tag

    if isinstance(el, NavigableString):
        text = str(el).strip()
        if text:
            lines.append(text)
        return

    if not isinstance(el, Tag):
        return

    tag = el.name

    if tag in ("h1", "h2", "h3", "h4"):
        text = el.get_text(strip=True)
        if text:
            level = int(tag[1])
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        return  # 자식 재탐색 안 함

    if tag == "li":
        text = el.get_text(strip=True)
        if text:
            lines.append(f"- {text}")
        return  # 자식 재탐색 안 함 (li 내부 중복 방지)

    if tag == "p":
        text = el.get_text(strip=True)
        if text:
            lines.append(text)
            lines.append("")
        return

    if tag == "table":
        _extract_table(el, lines)
        return

    # 기타 태그는 자식 재귀 탐색
    for child in el.children:
        _extract_element(child, lines, depth + 1)


def _extract_table(table, lines: list):
    """테이블을 Markdown 표로 변환"""
    rows = table.find_all("tr")
    if not rows:
        return

    lines.append("")
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [c.get_text(strip=True).replace("|", "/") for c in cells]
        if cell_texts:
            lines.append("| " + " | ".join(cell_texts) + " |")
            # 헤더 행 뒤에 구분선
            if i == 0 and row.find("th"):
                lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
    lines.append("")


def extract_content(html: str) -> tuple[str, str]:
    """HTML에서 제목과 본문 텍스트 추출"""
    soup = BeautifulSoup(html, "html.parser")

    # 불필요한 요소 제거
    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    # 네비게이션/사이드바 제거
    for sel in ["#gnb", "#sidebar", ".inner", "header", "nav", "footer"]:
        for el in soup.select(sel):
            el.decompose()

    # 제목 추출
    title = ""
    # 1) h1 태그
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    # 2) title 태그 fallback
    if not title:
        t = soup.find("title")
        if t:
            title = t.get_text(strip=True)

    # 본문 영역 추출
    content_area = soup.select_one("#contents") or soup.find("body")
    if not content_area:
        return title, ""

    # 본문을 구조화된 텍스트로 변환
    # children (not descendants) 단위로 재귀 처리하여 중복 방지
    lines = []
    _extract_element(content_area, lines)

    # 텍스트 정리
    raw = "\n".join(lines)
    # 페이지 내부 마커 제거 (예: --SG.SI.10--, --OD.OW.CM.250--)
    raw = re.sub(r"--[A-Z]{2}\.[A-Z]{2}(?:\.[A-Z]{2,3})?\.?\d+--", "", raw)
    # 연속 빈 줄 정리 (3개 이상 → 2개)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    # 연속 중복 라인 제거
    deduped = []
    for line in raw.split("\n"):
        if not deduped or line.strip() != deduped[-1].strip() or line.strip() == "":
            deduped.append(line)
    raw = "\n".join(deduped)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = raw.strip()

    return title, raw


def save_markdown(code: str, name: str, url: str, title: str, content: str) -> str:
    """Markdown 파일로 저장"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"cafe24_guide_{name}_{code}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    md = f"""# {title or name}

> source: {url}
> category: {name}
> code: {code}

---

{content}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)

    return filepath


# ============================================================
# 메인 실행
# ============================================================
def crawl_all():
    """전체 크롤링 실행"""
    print("=" * 60)
    print("CAFE24 서비스 가이드 크롤러")
    print("=" * 60)
    print(f"대상: {BASE_URL}")
    print(f"저장 경로: {OUTPUT_DIR}")
    print()

    # 1) 메인 페이지에서 링크 수집
    print("[1/3] 메인 페이지에서 링크 수집 중...")
    main_html = fetch_page(BASE_URL)
    if not main_html:
        print("메인 페이지 접근 실패!")
        return []

    discovered = discover_pages(main_html)
    print(f"  → {len(discovered)}개 페이지 발견")

    # PAGE_MAP에 없는 페이지도 포함 (자동 발견)
    pages_to_crawl = []
    for code, url in discovered:
        name = PAGE_MAP.get(code, code)
        pages_to_crawl.append((code, name, url))

    # PAGE_MAP에 있지만 발견되지 않은 페이지 추가
    discovered_codes = {code for code, _ in discovered}
    for code, name in PAGE_MAP.items():
        if code not in discovered_codes:
            url = f"{BASE_URL}{code}.html"
            pages_to_crawl.append((code, name, url))
            print(f"  + 추가: {code} ({name})")

    print(f"  → 총 {len(pages_to_crawl)}개 크롤링 예정")
    print()

    # 2) 각 페이지 크롤링
    print("[2/3] 페이지 크롤링 시작...")
    results = []

    for i, (code, name, url) in enumerate(pages_to_crawl, 1):
        print(f"  [{i}/{len(pages_to_crawl)}] {name} ({code})...", end=" ")

        html = fetch_page(url)
        if not html:
            print("SKIP")
            continue

        title, content = extract_content(html)

        if not content or len(content) < 50:
            print(f"내용 부족 ({len(content)}자)")
            continue

        filepath = save_markdown(code, name, url, title, content)
        size_kb = os.path.getsize(filepath) / 1024
        results.append({
            "code": code,
            "name": name,
            "title": title,
            "file": filepath,
            "content_length": len(content),
        })
        print(f"OK ({len(content):,}자, {size_kb:.1f}KB)")

        # 딜레이
        if i < len(pages_to_crawl):
            time.sleep(REQUEST_DELAY)

    # 3) 결과 요약
    print()
    print("[3/3] 크롤링 완료!")
    print("=" * 60)
    print(f"성공: {len(results)}/{len(pages_to_crawl)}개")
    total_chars = sum(r["content_length"] for r in results)
    print(f"총 텍스트: {total_chars:,}자")
    print(f"저장 위치: {OUTPUT_DIR}")
    print()
    print("다음 단계: /api/rag/reload 호출하여 RAG 인덱스에 반영하세요.")
    print("=" * 60)

    return results


# Jupyter / 직접 실행 모두 지원
if __name__ == "__main__":
    results = crawl_all()
else:
    # Jupyter에서 import 시 자동 실행하지 않음
    # crawl_all() 직접 호출하세요
    print("crawl_cafe24_guide 모듈 로드 완료. crawl_all() 을 호출하세요.")
