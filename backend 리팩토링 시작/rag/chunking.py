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
import hashlib
from typing import List, Dict, Any, Tuple

from core.utils import safe_str
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
# 내부 유틸
# ============================================================
def _sha1_text(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


def _clean_text_for_rag(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _is_garbage_text(txt: str) -> bool:
    if not txt:
        return True
    t = txt.strip()
    if len(t) < 50:
        return True
    uniq = len(set(t))
    if uniq / max(1, len(t)) < 0.02:
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

    from collections import Counter
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
    """불릿 블록을 개별 청크로 변환"""
    chunks: List[Any] = []

    for block in blocks:
        if block["type"] == "bullet":
            header = block.get("header", "")
            items = block.get("items", [])

            for item in items:
                if isinstance(item, dict):
                    item_title = item.get("title", "")
                    item_desc = item.get("description", "")
                else:
                    item_title = item
                    item_desc = ""

                content_parts = []
                if section_title:
                    content_parts.append(f"[섹션: {section_title}]")
                if header:
                    content_parts.append(f"{header}:")
                if item_desc:
                    content_parts.append(f"{item_title}: {item_desc}")
                else:
                    content_parts.append(item_title)

                chunk_content = '\n'.join(content_parts)
                chunk_meta = {
                    **base_metadata,
                    "source": source,
                    "section_title": section_title,
                    "chunk_type": "bullet_item",
                    "bullet_header": header,
                }
                chunks.append(Document(page_content=chunk_content, metadata=chunk_meta))

        else:
            prose_content = block.get("content", "")
            if prose_content and len(prose_content) >= 50:
                chunk_meta = {
                    **base_metadata,
                    "source": source,
                    "section_title": section_title,
                    "chunk_type": "prose",
                }
                chunks.append(Document(page_content=prose_content, metadata=chunk_meta))

    return chunks


# ============================================================
# 섹션 분할
# ============================================================
def _split_by_sections(text: str, source: str = "") -> List[Tuple[str, str]]:
    """문서를 섹션 단위로 분리"""
    if not text:
        return [("", text)]

    section_pattern = re.compile(r'^(\d+\.(?:\d+\.)*\s*.+)$', re.MULTILINE)

    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_content: List[str] = []

    for line in lines:
        match = section_pattern.match(line.strip())
        if match and len(line.strip()) < 100:
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_title, content))
            current_title = line.strip()
            current_content = [line]
        else:
            current_content.append(line)

    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_title, content))

    if not sections:
        return [("", text)]

    st.logger.info("SECTIONS_SPLIT source=%s sections=%d", source, len(sections))
    return sections


# ============================================================
# PDF 텍스트 추출
# ============================================================
def _extract_text_from_pdf(path: str) -> str:
    """PDF에서 텍스트 추출"""
    raw_text = ""

    try:
        import fitz
        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
        raw_text = "\n".join(text_parts).strip()
        if raw_text:
            st.logger.info("PDF_EXTRACTED_PYMUPDF path=%s chars=%d",
                          os.path.basename(path), len(raw_text))
    except ImportError:
        st.logger.debug("pymupdf not available, falling back to pypdf")
    except Exception as e:
        st.logger.warning("PYMUPDF_FAIL path=%s err=%s", path, safe_str(e))

    if not raw_text:
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return ""
            reader = PdfReader(path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            raw_text = "\n".join(text_parts).strip()
            if raw_text:
                st.logger.info("PDF_EXTRACTED_PYPDF path=%s chars=%d",
                              os.path.basename(path), len(raw_text))
        except Exception as e:
            st.logger.warning("PYPDF_FAIL path=%s err=%s", path, safe_str(e))
            return ""

    return raw_text


def _rag_read_file(path: str) -> str:
    """RAG용 파일 읽기 (PDF/텍스트)"""
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
        if _is_garbage_text(txt):
            st.logger.warning("RAG_SKIP_GARBAGE path=%s len=%d", os.path.basename(path), len(txt or ""))
            return ""
        return txt
    except Exception:
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

    for doc in docs:
        content = safe_str(getattr(doc, "page_content", ""))
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "")

        sections = _split_by_sections(content, source)
        for section_title, section_content in sections:
            if not section_content or len(section_content.strip()) < 50:
                continue

            blocks = _extract_bullet_blocks(section_content)
            has_bullets = any(b["type"] == "bullet" for b in blocks)

            if has_bullets:
                bullet_chunks = _create_bullet_chunks(blocks, section_title, source, metadata)
                bullet_child_chunks.extend(bullet_chunks)

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
                for future in as_completed(futures):
                    idx, prefix = future.result()
                    contextual_prefixes[idx] = prefix
                    if prefix:
                        contextual_generated_count += 1
                    completed += 1
                    if completed % 100 == 0 or completed == len(child_infos):
                        st.logger.info("CONTEXTUAL_PROGRESS %d/%d (%.1f%%)",
                                      completed, len(child_infos),
                                      100.0 * completed / max(1, len(child_infos)))

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

        # 불릿 청크 추가
        for i, bullet_chunk in enumerate(bullet_child_chunks):
            bullet_parent_id = f"bullet_{i}"
            parent_store[bullet_parent_id] = bullet_chunk
            child_hash = _sha1_text(safe_str(getattr(bullet_chunk, "page_content", "")))[:16]
            child_to_parent[child_hash] = bullet_parent_id
            bullet_chunk.metadata["parent_id"] = bullet_parent_id
            all_child_chunks.append(bullet_chunk)

        st.logger.info("PARENT_CHILD_CHUNKS_CREATED parents=%d children=%d bullet=%d contextual=%d",
                       len(parent_store), len(all_child_chunks), len(bullet_child_chunks), contextual_generated_count)

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
