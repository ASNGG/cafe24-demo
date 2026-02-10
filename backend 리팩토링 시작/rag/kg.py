"""
rag/kg.py - Knowledge Graph (Simple Entity-Relation Extraction)

카페24 플랫폼 도메인의 개체명 및 관계 추출
"""
import re
from typing import List, Dict, Any

from core.utils import safe_str
import state as st

# ============================================================
# Knowledge Graph 상태
# ============================================================
KNOWLEDGE_GRAPH: Dict[str, List[Dict]] = {}


# ============================================================
# Entity Extraction (카페24 플랫폼)
# ============================================================
def _extract_entities_simple(text: str) -> List[str]:
    """카페24 플랫폼 개체명 추출 (정규식 기반)"""
    entities = []

    # 쇼핑몰 ID 패턴: S0001, S0123
    shop_ids = re.findall(r'S\d{4}', text)
    entities.extend(shop_ids)

    # 카테고리 키워드
    categories = re.findall(r'(?:패션|뷰티|식품|전자기기|생활용품|IT서비스|교육|스포츠)', text)
    entities.extend(categories)

    # 플랜 티어
    tiers = re.findall(r'(?:Basic|Standard|Premium|Enterprise)\s*(?:플랜|요금제)?', text)
    entities.extend(tiers)

    # 결제/정산 관련 용어
    payment_terms = re.findall(r'(?:PG|이니시스|토스페이먼츠|KCP|카카오페이|네이버페이)', text)
    entities.extend(payment_terms)

    # 주요 키워드
    keywords = re.findall(r'(?:정산|환불|배송|결제|상품|주문|회원|쿠폰|적립금|SEO|API)', text)
    entities.extend(keywords)

    # 영문 고유명사 (대문자로 시작)
    english_entities = re.findall(r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*', text)
    entities.extend([e for e in english_entities if len(e) > 3])

    return list(set(entities))


def _extract_relations_simple(text: str, entities: List[str]) -> List[Dict]:
    """카페24 플랫폼 관계 추출 (패턴 기반)"""
    relations = []

    relation_patterns = [
        # 소속 관계: "쇼핑몰 S0001의 상품"
        (r'(S\d{4})(?:의|에서|에)\s*(\w+)', 'belongs_to'),
        # 카테고리 관계
        (r'(패션|뷰티|식품|전자기기|생활용품)\s*(?:전문|카테고리|분야)', 'has_category'),
    ]

    for pattern, rel_type in relation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 2:
                relations.append({
                    "source": match[0].strip(),
                    "target": match[1].strip(),
                    "type": rel_type,
                })

    return relations


def build_knowledge_graph(chunks: List[Any]) -> Dict:
    """청크에서 Knowledge Graph 구축"""
    global KNOWLEDGE_GRAPH

    KNOWLEDGE_GRAPH = {}
    entity_docs: Dict[str, List[str]] = {}
    all_relations = []

    for chunk in chunks:
        try:
            content = safe_str(getattr(chunk, "page_content", ""))
            source = getattr(chunk, "metadata", {}).get("source", "unknown")

            entities = _extract_entities_simple(content)
            for entity in entities:
                if entity not in entity_docs:
                    entity_docs[entity] = []
                if source not in entity_docs[entity]:
                    entity_docs[entity].append(source)

            relations = _extract_relations_simple(content, entities)
            all_relations.extend(relations)
        except Exception:
            continue

    # 중복 관계 제거
    seen_relations = set()
    unique_relations = []
    for rel in all_relations:
        key = (rel.get("source", ""), rel.get("target", ""), rel.get("type", ""))
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(rel)

    KNOWLEDGE_GRAPH = {
        "entities": entity_docs,
        "relations": unique_relations,
        "stats": {
            "entity_count": len(entity_docs),
            "relation_count": len(unique_relations),
        }
    }

    st.logger.info("KNOWLEDGE_GRAPH_BUILT entities=%d relations=%d",
                   len(entity_docs), len(unique_relations))
    return KNOWLEDGE_GRAPH


def search_knowledge_graph(query: str, top_k: int = 5) -> List[Dict]:
    """Knowledge Graph에서 관련 엔티티 검색"""
    global KNOWLEDGE_GRAPH

    if not KNOWLEDGE_GRAPH or "entities" not in KNOWLEDGE_GRAPH:
        return []

    results = []
    query_lower = query.lower()

    for entity, sources in KNOWLEDGE_GRAPH.get("entities", {}).items():
        entity_lower = entity.lower()
        score = 0

        if entity_lower in query_lower or query_lower in entity_lower:
            score = 10
        elif any(word in entity_lower for word in query_lower.split() if len(word) >= 2):
            score = 5

        if score > 0:
            related_relations = [
                r for r in KNOWLEDGE_GRAPH.get("relations", [])
                if r.get("source") == entity or r.get("target") == entity
            ]

            results.append({
                "entity": entity,
                "sources": sources,
                "relations": related_relations[:3],
                "score": score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
