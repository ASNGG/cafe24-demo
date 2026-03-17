"""
rag/chunking.py - 텍스트 청킹 로직

문서를 검색 가능한 청크로 분할하는 기능:
- Parent-Child Chunking
- 섹션 기반 분할
- 불릿/목록 항목 분리
- PDF 텍스트 추출
"""
import os
import re
from collections import Counter
from typing import List, Dict, Any, Tuple

from core.utils import safe_str
from rag.utils import sha1_text, clean_text, extract_text_from_pdf
import state as st

# ============================================================
# Optional imports
# ============================================================
Document = None
RecursiveCharacterTextSplitter = None

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    pass

# ============================================================
# 내부 유틸 (M21, M22: 공통 유틸로 위임)
# ============================================================
_sha1_text = sha1_text
_clean_text_for_rag = clean_text


# ============================================================
# Breadcrumb 계층 경로 추출
# ============================================================
def _extract_breadcrumb(source: str) -> str:
    """파일명에서 카테고리 계층 경로 추출
    cafe24_hc_쇼핑몰_자주_묻는_질문_결제_설정_8456870914585.md
    → '쇼핑몰 > 자주 묻는 질문 > 결제 설정'
    cafe24_guide_상품등록_PT.PE.md
    → '가이드 > 상품등록'
    """
    if not source:
        return ""
    name = source
    # 확장자 + 숫자 ID 제거
    name = re.sub(r'_\d{5,}\.md$', '', name)
    name = re.sub(r'\.md$', '', name)
    # 시스템 코드 제거 (예: _SG.PA, _PN.MD, _PT.PE, _BD.BA)
    name = re.sub(r'_[A-Z]{2}\.[A-Z]{2,4}(\.[A-Z]{2,4})?$', '', name)

    # cafe24_hc_ / cafe24_guide_ prefix 처리
    if name.startswith('cafe24_hc_'):
        name = name[len('cafe24_hc_'):]
    elif name.startswith('cafe24_guide_'):
        parts = name[len('cafe24_guide_'):].replace('_', ' ').strip()
        return f"가이드 > {parts}" if parts else "가이드"

    # 알려진 복합 키워드를 먼저 치환 (분리 방지)
    compound_map = {
        '자주_묻는_질문': '자주 묻는 질문',
        '동영상_가이드': '동영상 가이드',
        '기본_설정': '기본 설정',
        '유튜브_쇼핑': '유튜브 쇼핑',
        '카페24_PRO': '카페24 PRO',
        '상품_등록': '상품 등록',
        '상품_관리': '상품 관리',
        '주문_관리': '주문 관리',
        '결제_설정': '결제 설정',
        '결제_관리': '결제 관리',
        '배송_설정': '배송 설정',
        '배송_관리': '배송 관리',
        '고객_설정': '고객 설정',
        '회원_관리': '회원 관리',
        '쇼핑몰_설정': '쇼핑몰 설정',
        '사이트_설정': '사이트 설정',
        '운영자_설정': '운영자 설정',
        '도메인_설정': '도메인 설정',
        '디자인_추가': '디자인 추가',
        '채널_설정': '채널 설정',
    }
    for k, v in compound_map.items():
        name = name.replace(k, v.replace(' ', '\x00'))
    # 남은 언더스코어를 계층 구분으로
    parts = [p.replace('\x00', ' ').strip() for p in name.split('_') if p.strip()]
    if not parts:
        return ""
    return ' > '.join(parts)


# ============================================================
# 보일러플레이트 섹션 감지
# ============================================================
_BOILERPLATE_TITLES = {
    '연관 콘텐츠', 'info', 'caution', 'note', 'warning', 'tip',
}

def _is_boilerplate_section(title: str, content: str) -> bool:
    """의미 없는 보일러플레이트 섹션인지 판별"""
    if not title:
        return False
    title_lower = title.strip().lower()
    # 제목이 보일러플레이트 패턴이고 내용이 짧으면 스킵
    if title_lower in _BOILERPLATE_TITLES and len(content) < 300:
        return True
    # '자세히 알아보기' 단독 (내용 없이 링크만)
    if title_lower == '자세히 알아보기' and len(content) < 100:
        return True
    return False


def _is_garbage_text(txt: str) -> bool:
    if not txt:
        return True
    t = txt.strip()
    if len(t) < 50:
        return True
    uniq = len(set(t))
    # 한국어 대형 문서 보호: unique 문자 100개 이상이면 정상 텍스트로 판정
    # (한글은 ~2000자 범위라 5만자 이상 문서도 ratio가 0.02 이하로 떨어짐)
    if uniq < 100 and uniq / max(1, len(t)) < 0.005:
        return True
    meaningful = re.findall(r"[가-힣A-Za-z0-9]", t)
    if len(meaningful) / max(1, len(t)) < 0.15:
        return True
    return False


# ============================================================
# 핵심 명사 추출 (BM25 키워드 매칭 강화)
# ============================================================
def _extract_key_nouns(text: str, top_k: int = 15) -> List[str]:
    """텍스트에서 핵심 명사 추출 (빈도 기반)"""
    if not text:
        return []

    nouns = re.findall(r'[가-힣]{2,6}', text)
    compound_nouns = re.findall(r'[가-힣]{2,6}\s+[가-힣]{2,6}', text)

    stopwords = {
        '이다', '하다', '있다', '없다', '되다', '않다', '것이', '수가', '등의',
        '으로', '에서', '까지', '부터', '에게', '한다', '된다', '이며', '이고',
        '그리고', '하지만', '그러나', '따라서', '그래서', '때문에', '위해서',
        '경우', '통해', '대한', '관한', '대해', '관해', '사용', '기능', '설명',
    }

    all_terms = nouns + compound_nouns
    counter = Counter(all_terms)

    result = []
    seen = set()
    for term, count in counter.most_common(top_k * 3):
        term_clean = term.strip()
        if term_clean in stopwords or term_clean in seen or len(term_clean) < 2:
            continue
        seen.add(term_clean)
        result.append(term_clean)
        if len(result) >= top_k:
            break

    return result


# ============================================================
# 불릿/목록 청킹
# ============================================================
BULLET_PATTERNS = [
    r'^[-•*○●◦▪▸►→]\s+',
    r'^[가-힣][.)]\s+',
    r'^[a-zA-Z][.)]\s+',
    r'^※\s*',                  # ※ 주의사항
    r'^\d+\)\s+',              # 1) 2) 형식
]
BULLET_REGEX = re.compile('|'.join(BULLET_PATTERNS))
SECTION_TITLE_PATTERN = re.compile(r'^\d+\.(?:\d+\.)*\s+.{2,50}$')


def _is_bullet_line(line: str) -> bool:
    """줄이 불릿/목록 항목인지 확인"""
    stripped = line.strip()
    if SECTION_TITLE_PATTERN.match(stripped):
        return False
    return bool(BULLET_REGEX.match(stripped))


def _extract_bullet_blocks(text: str) -> List[Dict[str, Any]]:
    """텍스트에서 불릿/목록 블록과 일반 텍스트 블록을 분리"""
    lines = text.split('\n')
    blocks: List[Dict[str, Any]] = []

    current_prose: List[str] = []
    current_header = ""
    current_bullet_items: List[Dict[str, str]] = []
    current_item_title = ""
    current_item_desc: List[str] = []
    in_bullet_block = False
    empty_line_count = 0

    def save_current_item():
        nonlocal current_item_title, current_item_desc
        if current_item_title:
            desc = '\n'.join(current_item_desc).strip()
            current_bullet_items.append({
                "title": current_item_title,
                "description": desc
            })
            current_item_title = ""
            current_item_desc = []

    def save_bullet_block():
        nonlocal current_bullet_items, current_header, in_bullet_block
        save_current_item()
        if current_bullet_items:
            blocks.append({
                "type": "bullet",
                "header": current_header,
                "items": current_bullet_items
            })
        current_bullet_items = []
        current_header = ""
        in_bullet_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            empty_line_count += 1
            if empty_line_count >= 2 and in_bullet_block:
                save_bullet_block()
            elif in_bullet_block and current_item_title:
                current_item_desc.append("")
            elif current_prose:
                current_prose.append(line)
            continue

        empty_line_count = 0

        if _is_bullet_line(stripped):
            if not in_bullet_block:
                if current_prose:
                    for j in range(len(current_prose) - 1, -1, -1):
                        if current_prose[j].strip():
                            current_header = current_prose[j].strip()
                            current_prose = current_prose[:j]
                            break
                    prose_text = '\n'.join(current_prose).strip()
                    if prose_text:
                        blocks.append({"type": "prose", "content": prose_text})
                    current_prose = []
                in_bullet_block = True
            else:
                save_current_item()
            current_item_title = BULLET_REGEX.sub('', stripped).strip()
        else:
            if in_bullet_block and current_item_title:
                current_item_desc.append(stripped)
            elif in_bullet_block and not current_item_title:
                save_bullet_block()
                current_prose.append(line)
            else:
                current_prose.append(line)

    if in_bullet_block:
        save_bullet_block()
    elif current_prose:
        prose_text = '\n'.join(current_prose).strip()
        if prose_text:
            blocks.append({"type": "prose", "content": prose_text})

    return blocks


def _create_bullet_chunks(
    blocks: List[Dict[str, Any]],
    section_title: str,
    source: str,
    base_metadata: Dict[str, Any]
) -> List[Any]:
    """불릿 블록을 청크로 변환 (Breadcrumb + 키워드 포함, 중복 방지)"""
    chunks: List[Any] = []

    # Breadcrumb 경로
    breadcrumb = _extract_breadcrumb(source)

    # 불릿 블록 존재 여부 — 있으면 prose는 preamble로만 사용 (독립 청크 안 만듦)
    has_bullets = any(b["type"] == "bullet" for b in blocks)

    # 산문 블록에서 preamble 추출 (불릿 청크에 맥락 부여용)
    preamble = ""
    if has_bullets:
        for block in blocks:
            if block["type"] == "prose":
                prose_text = block.get("content", "").strip()
                if prose_text and len(prose_text) < 500:
                    preamble = prose_text
                    break

    def _make_tags(extra_text: str = "") -> List[str]:
        """태그 생성 (경로 + 섹션 + 키워드) — 본문 앞에 몰아서 배치"""
        tags = []
        if breadcrumb:
            tags.append(f"[경로: {breadcrumb}]")
        if section_title:
            tags.append(f"[섹션: {section_title}]")
        kw_source = (section_title or "") + " " + (extra_text or "")
        keywords = _extract_key_nouns(kw_source, top_k=8)
        if keywords:
            tags.append(f"[키워드: {', '.join(keywords)}]")
        return tags

    MAX_GROUP_CHARS = 1500

    for block in blocks:
        if block["type"] == "bullet":
            header = block.get("header", "")
            items = block.get("items", [])

            total_len = sum(
                len(item.get("title", "") if isinstance(item, dict) else str(item))
                + len(item.get("description", "") if isinstance(item, dict) else "")
                for item in items
            )
            all_bullet_text = " ".join(
                (item.get("title", "") + " " + item.get("description", "")) if isinstance(item, dict) else str(item)
                for item in items
            )

            if total_len <= MAX_GROUP_CHARS:
                # 태그 → preamble → header → 불릿 항목 순서
                content_parts = _make_tags(all_bullet_text)
                if preamble:
                    content_parts.append(preamble)
                if header:
                    content_parts.append(f"{header}:")
                for item in items:
                    if isinstance(item, dict):
                        t, d = item.get("title", ""), item.get("description", "")
                        content_parts.append(f"- {t}: {d}" if d else f"- {t}")
                    else:
                        content_parts.append(f"- {item}")
                chunk_content = '\n'.join(content_parts)
                chunk_meta = {
                    **base_metadata, "source": source,
                    "section_title": section_title, "chunk_type": "bullet_group",
                    "bullet_header": header, "breadcrumb": breadcrumb,
                }
                chunks.append(Document(page_content=chunk_content, metadata=chunk_meta))
            else:
                # 서브그룹으로 묶기 (MAX_GROUP_CHARS 단위)
                sub_groups: List[List] = []
                current_group: List = []
                current_len = 0
                for item in items:
                    if isinstance(item, dict):
                        item_len = len(item.get("title", "")) + len(item.get("description", ""))
                    else:
                        item_len = len(str(item))
                    if current_group and current_len + item_len > MAX_GROUP_CHARS:
                        sub_groups.append(current_group)
                        current_group = [item]
                        current_len = item_len
                    else:
                        current_group.append(item)
                        current_len += item_len
                if current_group:
                    sub_groups.append(current_group)

                for sub_items in sub_groups:
                    sub_text = " ".join(
                        (it.get("title", "") + " " + it.get("description", "")) if isinstance(it, dict) else str(it)
                        for it in sub_items
                    )
                    content_parts = _make_tags(sub_text)
                    if preamble:
                        content_parts.append(preamble)
                    if header:
                        content_parts.append(f"{header}:")
                    for item in sub_items:
                        if isinstance(item, dict):
                            t, d = item.get("title", ""), item.get("description", "")
                            content_parts.append(f"- {t}: {d}" if d else f"- {t}")
                        else:
                            content_parts.append(f"- {item}")
                    chunk_content = '\n'.join(content_parts)
                    chunk_meta = {
                        **base_metadata, "source": source,
                        "section_title": section_title, "chunk_type": "bullet_group",
                        "bullet_header": header, "breadcrumb": breadcrumb,
                    }
                    chunks.append(Document(page_content=chunk_content, metadata=chunk_meta))

        else:
            prose_content = block.get("content", "")
            if not prose_content or len(prose_content) < 50:
                continue
            # 불릿이 있는 섹션이면 prose는 preamble로만 사용 → 독립 청크 안 만듦
            if has_bullets and prose_content == preamble:
                continue
            content_parts = _make_tags(prose_content)
            content_parts.append(prose_content)
            chunk_meta = {
                **base_metadata, "source": source,
                "section_title": section_title, "chunk_type": "prose",
                "breadcrumb": breadcrumb,
            }
            chunks.append(Document(page_content='\n'.join(content_parts), metadata=chunk_meta))

    return chunks


# ============================================================
# 섹션 분할
# ============================================================
def _is_qa_header(title: str) -> bool:
    """FAQ 질문 형태의 섹션 헤더인지 판별"""
    if not title:
        return False
    # 물음표로 끝나거나, ~싶어요/~하나요/~되나요/~인가요 등 질문 패턴
    if title.rstrip().endswith('?'):
        return True
    qa_suffixes = ['싶어요', '하나요', '되나요', '인가요', '있나요', '없나요',
                   '알려줘', '알려주세요', '궁금해요', '할까요', '볼까요',
                   '해야 하나요', '수 있나요', '안 되나요', '안되나요']
    for suffix in qa_suffixes:
        if title.rstrip('.').endswith(suffix):
            return True
    return False


def _is_table_line(line: str) -> bool:
    """마크다운 테이블 줄인지 판별"""
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|')


def _split_by_sections(text: str, source: str = "") -> List[Tuple[str, str]]:
    """문서를 섹션 단위로 분리 (번호 패턴 + 마크다운 ## 헤더 + Q-A 보존 + 테이블 보존)"""
    if not text:
        return [("", text)]

    # 번호 패턴: "1. 제목", "1.2. 제목"
    number_pattern = re.compile(r'^(\d+\.(?:\d+\.)*\s*.+)$', re.MULTILINE)
    # 마크다운 헤더: "## 제목", "### 제목" (# 1개는 문서 전체 제목이므로 제외)
    md_header_pattern = re.compile(r'^(#{2,4})\s+(.+)$')

    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_content: List[str] = []
    in_table = False  # 테이블 보존 플래그

    for line in lines:
        stripped = line.strip()

        # 테이블 보존: 테이블 안에서는 섹션 분리 안 함
        if _is_table_line(stripped):
            in_table = True
            current_content.append(line)
            continue
        elif in_table and not stripped:
            # 테이블 후 빈 줄 → 테이블 종료
            in_table = False
            current_content.append(line)
            continue
        elif in_table:
            in_table = False

        num_match = number_pattern.match(stripped)
        md_match = md_header_pattern.match(stripped)

        is_section = False
        new_title = ""

        if num_match and len(stripped) < 100:
            is_section = True
            new_title = stripped
        elif md_match and len(stripped) < 100:
            is_section = True
            new_title = md_match.group(2).strip()

        if is_section:
            # Q-A 페어 보존: 현재 섹션이 FAQ 질문이고, 새 섹션도 FAQ 질문이면
            # 이전 Q-A를 완결 후 새 Q-A 시작
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_title, content))
            current_title = new_title
            current_content = [line]
        else:
            current_content.append(line)

    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_title, content))

    if not sections:
        return [("", text)]

    # 짧은 섹션 합치기: 200자 미만 인접 섹션을 병합 (정보 손실 방지)
    SHORT_SECTION_THRESHOLD = 200
    coalesced: List[Tuple[str, str]] = []
    i = 0
    while i < len(sections):
        title, content = sections[i]
        if len(content.strip()) < SHORT_SECTION_THRESHOLD:
            # 인접한 짧은 섹션들을 모아서 하나로 합침
            merged_title = title
            merged_parts = [content]
            j = i + 1
            while j < len(sections) and len(sections[j][1].strip()) < SHORT_SECTION_THRESHOLD:
                merged_parts.append(sections[j][0] + "\n" + sections[j][1])
                j += 1
            # 다음 긴 섹션이 있으면 그 앞에 붙이기
            if j < len(sections) and len(merged_parts) == 1:
                # 짧은 섹션 1개만이면 다음 섹션에 prepend
                next_title, next_content = sections[j]
                coalesced.append((next_title, content + "\n\n" + next_content))
                i = j + 1
            elif len(merged_parts) > 1:
                # 여러 짧은 섹션 → 하나로 합침
                coalesced.append((merged_title, '\n\n'.join(merged_parts)))
                i = j
            else:
                coalesced.append((title, content))
                i += 1
        else:
            coalesced.append((title, content))
            i += 1
    sections = coalesced

    # Q-A 페어 병합: 연속된 짧은 FAQ 질문+답변 섹션을 합치기
    merged_sections: List[Tuple[str, str]] = []
    i = 0
    while i < len(sections):
        title, content = sections[i]
        # FAQ 질문 섹션이고 답변이 짧으면(500자 미만) 다음 하위 섹션과 합치기
        if _is_qa_header(title) and len(content) < 500 and i + 1 < len(sections):
            next_title, next_content = sections[i + 1]
            # 다음 섹션이 ### 수준(하위) 이거나 짧은 보충이면 합침
            if not _is_qa_header(next_title) and len(next_content) < 800:
                merged_content = content + "\n\n" + next_title + "\n" + next_content
                merged_sections.append((title, merged_content))
                i += 2
                continue
        merged_sections.append((title, content))
        i += 1

    # 동일 제목 섹션 중복 제거: 같은 제목이 2번 나오면 긴 것만 유지
    title_best: Dict[str, Tuple[int, str]] = {}  # title → (길이, content)
    for title, content in merged_sections:
        # 제목 정규화 (마침표/공백 차이 무시)
        norm_title = title.rstrip('.').strip().lower()
        if norm_title in title_best:
            if len(content) > title_best[norm_title][0]:
                title_best[norm_title] = (len(content), content)
        else:
            title_best[norm_title] = (len(content), content)

    # 순서 유지하면서 중복 제거
    seen_titles: set = set()
    final_sections: List[Tuple[str, str]] = []
    for title, content in merged_sections:
        norm_title = title.rstrip('.').strip().lower()
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        # 가장 긴 버전 사용
        best_content = title_best[norm_title][1]
        final_sections.append((title, best_content))

    dedup_removed = len(merged_sections) - len(final_sections)
    st.logger.info("SECTIONS_SPLIT source=%s sections=%d (merged=%d, dedup=%d, original=%d)",
                   source, len(final_sections), len(sections) - len(merged_sections),
                   dedup_removed, len(sections))
    return final_sections


# ============================================================
# PDF 텍스트 추출 (M24: 공통 유틸로 위임)
# ============================================================
_extract_text_from_pdf = extract_text_from_pdf


def _deep_clean_document(txt: str) -> str:
    """문서 전처리: 노이즈 제거 + 구조 정규화"""
    if not txt:
        return ""

    # 1. HTML 주석 제거 (<!-- ... -->)
    txt = re.sub(r'<!--.*?-->', '', txt, flags=re.DOTALL)

    # 2. 보일러플레이트 제거
    boilerplate_patterns = [
        r'^콘텐츠\s*목차\s*$',           # "콘텐츠 목차" 단독 줄
        r'^\[바로가기\]\s*$',            # "[바로가기]" 단독 줄
        r'^---+\s*$',                    # 구분선 (메타데이터 아래)
    ]
    bp_regex = re.compile('|'.join(boilerplate_patterns), re.MULTILINE)
    txt = bp_regex.sub('', txt)

    # 3. 메타데이터 헤더 제거 (문서 앞부분의 > source:, > category:, > section:, > articles: 등)
    meta_pattern = re.compile(r'^>\s*(source|category|code|section|articles)\s*:.*$', re.MULTILINE)
    txt = meta_pattern.sub('', txt)

    # 4. HTML 엔티티 디코딩
    txt = txt.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

    # 5. 연속 빈 줄 정리 (3줄 이상 → 2줄로)
    txt = re.sub(r'\n{3,}', '\n\n', txt)

    # 6. ㆍ 글머리 → 표준 불릿(-)으로 정규화
    txt = re.sub(r'^ㆍ\s*', '- ', txt, flags=re.MULTILINE)

    return txt.strip()


def _rag_read_file(path: str) -> str:
    """RAG용 파일 읽기 (PDF/텍스트) + 전처리"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            txt = _extract_text_from_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()

        txt = (txt or "").strip()
        if len(txt) > st.RAG_MAX_DOC_CHARS:
            txt = txt[:st.RAG_MAX_DOC_CHARS]

        txt = _clean_text_for_rag(txt)
        txt = _deep_clean_document(txt)
        if _is_garbage_text(txt):
            st.logger.warning("RAG_SKIP_GARBAGE path=%s len=%d", os.path.basename(path), len(txt or ""))
            return ""
        return txt
    except (OSError, UnicodeDecodeError, ValueError):
        return ""


# ============================================================
# Parent-Child Chunking
# ============================================================
def _create_parent_child_chunks(
    docs: List[Any],
    parent_size: int = 3000,
    parent_overlap: int = 500,
    child_size: int = 500,
    child_overlap: int = 100,
    enable_contextual: bool = True,
    contextual_prefix_func=None,
    contextual_cache_load_func=None,
    contextual_cache_save_func=None,
    contextual_client_func=None,
    contextual_max_workers: int = 5,
) -> Tuple[List[Any], Dict[str, Any], Dict[str, str]]:
    """
    Parent-Child Chunking 구현

    Args:
        docs: 원본 Document 리스트
        parent_size: Parent 청크 크기
        parent_overlap: Parent 청크 오버랩
        child_size: Child 청크 크기
        child_overlap: Child 청크 오버랩
        enable_contextual: Contextual Retrieval 활성화 여부
        contextual_prefix_func: Contextual Prefix 생성 함수 (외부 주입)
        contextual_cache_load_func: 캐시 로드 함수
        contextual_cache_save_func: 캐시 저장 함수
        contextual_client_func: OpenAI 클라이언트 반환 함수
        contextual_max_workers: 병렬 처리 워커 수

    Returns:
        (child_chunks, parent_store, child_to_parent)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if RecursiveCharacterTextSplitter is None:
        parent_store = {}
        child_to_parent = {}
        for i, doc in enumerate(docs):
            pid = f"p_{i}"
            parent_store[pid] = doc
            child_to_parent[_sha1_text(safe_str(getattr(doc, "page_content", "")))[:16]] = pid
        return docs, parent_store, child_to_parent

    # 1단계: 문서를 섹션 단위로 분리
    section_docs: List[Any] = []
    bullet_child_chunks: List[Any] = []
    bullet_section_texts: List[str] = []  # 불릿 청크의 원본 섹션 텍스트 (parent 용)

    for doc in docs:
        content = safe_str(getattr(doc, "page_content", ""))
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "")

        sections = _split_by_sections(content, source)
        for section_title, section_content in sections:
            if not section_content or len(section_content.strip()) < 200:
                continue
            # 보일러플레이트 섹션 스킵
            if _is_boilerplate_section(section_title, section_content):
                continue

            blocks = _extract_bullet_blocks(section_content)
            has_bullets = any(b["type"] == "bullet" for b in blocks)

            if has_bullets:
                bullet_chunks = _create_bullet_chunks(blocks, section_title, source, metadata)
                bullet_child_chunks.extend(bullet_chunks)
                # 각 불릿 청크에 원본 섹션 텍스트 매핑 (parent 컨텍스트 복원용)
                for _ in bullet_chunks:
                    bullet_section_texts.append(section_content)

                for block in blocks:
                    if block["type"] == "prose":
                        prose_content = block.get("content", "")
                        if prose_content and len(prose_content) >= 100:
                            new_meta = {**metadata, "section_title": section_title}
                            section_docs.append(Document(page_content=prose_content, metadata=new_meta))
            else:
                new_meta = {**metadata, "section_title": section_title}
                section_docs.append(Document(page_content=section_content, metadata=new_meta))

    if not section_docs:
        section_docs = docs

    st.logger.info("SECTION_DOCS_CREATED original=%d sections=%d bullet_items=%d",
                   len(docs), len(section_docs), len(bullet_child_chunks))

    separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]

    try:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=parent_overlap,
            separators=separators, length_function=len,
        )
        parent_chunks = parent_splitter.split_documents(section_docs)

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap,
            separators=separators, length_function=len,
        )

        parent_store: Dict[str, Any] = {}
        child_to_parent: Dict[str, str] = {}
        all_child_chunks: List[Any] = []

        contextual_generated_count = 0
        contextual_client = None

        if enable_contextual and contextual_cache_load_func and contextual_client_func:
            contextual_cache_load_func()
            contextual_client = contextual_client_func()
            st.logger.info("CONTEXTUAL_RETRIEVAL_STATUS enabled=%s client=%s workers=%d",
                          True, "OK" if contextual_client is not None else "None",
                          contextual_max_workers)
        else:
            st.logger.info("CONTEXTUAL_RETRIEVAL_SKIPPED")

        child_infos: List[Dict[str, Any]] = []

        for i, parent in enumerate(parent_chunks):
            parent_id = f"p_{i}"
            parent_content = safe_str(getattr(parent, "page_content", ""))
            parent_metadata = getattr(parent, "metadata", {})
            source = parent_metadata.get("source", "")

            contextual_content = f"[문서: {source}]\n\n{parent_content}"
            parent_store[parent_id] = Document(
                page_content=contextual_content,
                metadata={**parent_metadata, "parent_id": parent_id}
            )

            section_title = parent_metadata.get("section_title", "")
            if not section_title:
                for line in parent_content.strip().split('\n')[:15]:
                    line = line.strip()
                    if not line or len(line) > 100:
                        continue
                    if re.match(r'^\d+\.\d*\.?\s*\S', line):
                        section_title = line
                        break
                    elif line.startswith('## ') or line.startswith('### '):
                        section_title = line.lstrip('#').strip()
                        break

            parent_first = parent_content.strip()[:500].replace('\n', ' ').strip()
            parent_context = parent_first[:300] if len(parent_first) > 300 else parent_first

            subsection_keywords = []
            for line in parent_content.strip().split('\n')[:30]:
                line = line.strip()
                if not line or len(line) > 50:
                    continue
                if re.match(r'^[가-힣]{2,10}\s+[가-힣]{2,10}$', line):
                    subsection_keywords.append(line)
                elif re.match(r'^\d+\.\d+\.?\s*(.+)$', line):
                    match = re.match(r'^\d+\.\d+\.?\s*(.+)$', line)
                    if match:
                        subsection_keywords.append(match.group(1).strip())

            key_nouns = _extract_key_nouns(parent_content)
            keywords_tag = f"[키워드: {', '.join(key_nouns[:10])}]" if key_nouns else ""
            subsections_tag = f"[하위섹션: {', '.join(subsection_keywords[:5])}]" if subsection_keywords else ""

            try:
                temp_doc = Document(page_content=parent_content, metadata=parent_metadata)
                children = child_splitter.split_documents([temp_doc])

                for child in children:
                    child_content = safe_str(getattr(child, "page_content", ""))
                    if not child_content or len(child_content) < 30:
                        continue

                    child_infos.append({
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "parent_metadata": parent_metadata,
                        "child_content": child_content,
                        "source": source,
                        "section_title": section_title,
                        "subsections_tag": subsections_tag,
                        "keywords_tag": keywords_tag,
                        "parent_context": parent_context,
                    })

            except Exception as e:
                st.logger.warning("PARENT_CHILD_SPLIT_FAIL parent=%s err=%s", parent_id, safe_str(e))
                child_hash = _sha1_text(parent_content)[:16]
                child_to_parent[child_hash] = parent_id
                all_child_chunks.append(parent)

        # 병렬 Contextual Prefix 생성
        contextual_prefixes: Dict[int, str] = {}

        if enable_contextual and contextual_client is not None and contextual_prefix_func:
            st.logger.info("CONTEXTUAL_PARALLEL_START total_children=%d workers=%d",
                          len(child_infos), contextual_max_workers)

            def process_child(idx: int) -> Tuple[int, str]:
                info = child_infos[idx]
                prefix = contextual_prefix_func(
                    doc_content=info["parent_content"],
                    chunk_content=info["child_content"],
                    source=info["source"],
                    section_title=info["section_title"]
                )
                return idx, prefix

            with ThreadPoolExecutor(max_workers=contextual_max_workers) as executor:
                futures = {executor.submit(process_child, i): i for i in range(len(child_infos))}
                completed = 0
                _contextual_errors = 0
                for future in as_completed(futures):
                    # M29: 개별 future 에러 처리
                    try:
                        idx, prefix = future.result()
                        contextual_prefixes[idx] = prefix
                        if prefix:
                            contextual_generated_count += 1
                    except Exception as e:
                        _contextual_errors += 1
                        if _contextual_errors <= 5:
                            st.logger.warning("CONTEXTUAL_CHILD_FAIL idx=%d err=%s",
                                            futures[future], safe_str(e)[:80])
                    completed += 1
                    if completed % 100 == 0 or completed == len(child_infos):
                        st.logger.info("CONTEXTUAL_PROGRESS %d/%d (%.1f%%) errors=%d",
                                      completed, len(child_infos),
                                      100.0 * completed / max(1, len(child_infos)),
                                      _contextual_errors)

        # Document 조합
        for idx, info in enumerate(child_infos):
            section_title = info["section_title"]
            subsections_tag = info["subsections_tag"]
            keywords_tag = info["keywords_tag"]
            parent_context = info["parent_context"]
            source = info["source"]
            child_content = info["child_content"]
            parent_id = info["parent_id"]
            parent_metadata = info["parent_metadata"]

            tags = []
            if section_title:
                pure_title = re.sub(r'^\d+\.[\d.]*\s*', '', section_title).strip()
                tags.append(f"[섹션: {section_title}]")
                tags.append(f"[섹션제목: {section_title}]")
                tags.append(f"[제목: {pure_title}]")
                if pure_title and pure_title != section_title:
                    tags.append(f"[주제: {pure_title}]")

            if subsections_tag:
                tags.append(subsections_tag)
            if keywords_tag:
                tags.append(keywords_tag)
            tags.append(f"[컨텍스트: {parent_context}]")
            tags.append(f"[문서: {source}]")

            contextual_prefix = contextual_prefixes.get(idx, "")
            if contextual_prefix:
                tags.insert(0, f"[맥락: {contextual_prefix}]")

            contextual_child_content = " ".join(tags) + " " + child_content

            child_meta = {**parent_metadata, "parent_id": parent_id}
            new_child = Document(page_content=contextual_child_content, metadata=child_meta)
            all_child_chunks.append(new_child)

            child_hash = _sha1_text(contextual_child_content)[:16]
            child_to_parent[child_hash] = parent_id

        if contextual_generated_count > 0 and contextual_cache_save_func:
            contextual_cache_save_func()

        # 불릿 청크 추가 — parent를 원본 섹션 텍스트로 설정 + 기본 태그 부여
        for i, bullet_chunk in enumerate(bullet_child_chunks):
            bullet_parent_id = f"bullet_{i}"
            section_text = bullet_section_texts[i] if i < len(bullet_section_texts) else ""
            if section_text:
                bullet_meta = getattr(bullet_chunk, "metadata", {})
                parent_store[bullet_parent_id] = Document(
                    page_content=section_text,
                    metadata={**bullet_meta, "parent_id": bullet_parent_id}
                )
            else:
                parent_store[bullet_parent_id] = bullet_chunk

            # bullet chunk에 최소 태그 추가 (contextual tag 파이프라인 우회 보정)
            bullet_content = safe_str(getattr(bullet_chunk, "page_content", ""))
            bullet_meta = getattr(bullet_chunk, "metadata", {})
            b_source = bullet_meta.get("source", "")
            b_section = bullet_meta.get("section_title", "")
            b_tags = []
            if b_source:
                b_tags.append(f"[문서: {b_source}]")
            if b_section:
                b_tags.append(f"[섹션: {b_section}]")
            if b_tags:
                bullet_chunk = Document(
                    page_content=" ".join(b_tags) + " " + bullet_content,
                    metadata=bullet_meta,
                )

            child_hash = _sha1_text(safe_str(getattr(bullet_chunk, "page_content", "")))[:16]
            child_to_parent[child_hash] = bullet_parent_id
            bullet_chunk.metadata["parent_id"] = bullet_parent_id
            all_child_chunks.append(bullet_chunk)

        # 중복 청크 제거 (dedup): 같은 문서 내에서만 n-gram Jaccard 비교
        _TAG_RE = re.compile(r'\[(?:경로|섹션|키워드|문서|제목|섹션제목|하위섹션|컨텍스트|맥락|주제):.*?\]')

        def _extract_body(text: str) -> str:
            return _TAG_RE.sub('', text).strip()

        def _ngram_set(text: str, n: int = 3) -> set:
            t = re.sub(r'\s+', '', text)
            return {t[i:i+n] for i in range(len(t) - n + 1)} if len(t) >= n else {t}

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        DEDUP_THRESHOLD = 0.75

        before_dedup = len(all_child_chunks)
        deduped_chunks: List[Any] = []
        # 문서별로 그룹핑하여 dedup (O(n²) → O(Σ m_i²), m_i << n)
        seen_per_source: Dict[str, List[set]] = {}

        for chunk in all_child_chunks:
            text = safe_str(getattr(chunk, "page_content", ""))
            body = _extract_body(text)
            ngrams = _ngram_set(body)
            source = getattr(chunk, "metadata", {}).get("source", "")

            is_dup = False
            seen_list = seen_per_source.get(source, [])
            for seen in seen_list:
                if _jaccard(ngrams, seen) >= DEDUP_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                deduped_chunks.append(chunk)
                if source not in seen_per_source:
                    seen_per_source[source] = []
                seen_per_source[source].append(ngrams)

        dedup_removed = before_dedup - len(deduped_chunks)
        all_child_chunks = deduped_chunks

        st.logger.info("PARENT_CHILD_CHUNKS_CREATED parents=%d children=%d bullet=%d contextual=%d dedup_removed=%d",
                       len(parent_store), len(all_child_chunks), len(bullet_child_chunks),
                       contextual_generated_count, dedup_removed)

        return all_child_chunks, parent_store, child_to_parent

    except Exception as e:
        st.logger.warning("PARENT_CHILD_CHUNK_FAIL err=%s", safe_str(e))
        parent_store = {}
        child_to_parent = {}
        for i, doc in enumerate(docs):
            pid = f"p_{i}"
            parent_store[pid] = doc
            child_to_parent[_sha1_text(safe_str(getattr(doc, "page_content", "")))[:16]] = pid
        return docs, parent_store, child_to_parent
