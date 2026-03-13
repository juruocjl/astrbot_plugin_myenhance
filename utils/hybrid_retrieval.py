from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from rank_bm25 import BM25Okapi
from pypinyin import lazy_pinyin, Style

from .jargon_store import JargonRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
EMOJI_TOKEN_RE = re.compile(
    r"(?:[\U0001F1E6-\U0001F1FF]{2}|[\U0001F300-\U0001FAFF\u2600-\u27BF](?:\uFE0F|\u200D[\U0001F300-\U0001FAFF\u2600-\u27BF])*)"
)


@dataclass(slots=True)
class ScoredJargon:
    record: JargonRecord
    score: float
    bm25_score: float
    embedding_score: float


def hybrid_search(
    query: str,
    records: list[JargonRecord],
    top_k: int,
    bm25_weight: float = 0.55,
    embedding_weight: float = 0.45,
    rrf_k: int = 60,
    embedding_scores: list[float] | None = None,
) -> list[ScoredJargon]:
    normalized_query = str(query or "").strip()
    if not normalized_query or not records or top_k <= 0:
        return []

    learned_terms = _build_learned_terms(normalized_query, records)
    query_tokens = tokenize(normalized_query, learned_terms=learned_terms)
    if not query_tokens:
        return []

    doc_tokens = [
        tokenize(record.keyword or record.content, learned_terms=learned_terms)
        for record in records
    ]
    
    # 使用专业的 rank_bm25 库替代手动实现
    bm25 = BM25Okapi(doc_tokens)
    bm25_scores = bm25.get_scores(query_tokens)
    
    final_embedding_scores = embedding_scores
    if final_embedding_scores is None:
        query_vector = _hashed_embedding(normalized_query)
        final_embedding_scores = [
            _cosine_similarity(query_vector, _hashed_embedding(record.content))
            for record in records
        ]
    if len(final_embedding_scores) != len(records):
        raise ValueError("embedding_scores length mismatch")

    bm25_ranks = _build_ranks(bm25_scores)
    embedding_ranks = _build_ranks(final_embedding_scores)

    merged: list[ScoredJargon] = []
    for index, record in enumerate(records):
        bm25_rank = bm25_ranks[index]
        embedding_rank = embedding_ranks[index]
        score = (
            bm25_weight / (rrf_k + bm25_rank)
            + embedding_weight / (rrf_k + embedding_rank)
        )
        if bm25_scores[index] <= 0 and final_embedding_scores[index] <= 0:
            continue
        merged.append(
            ScoredJargon(
                record=record,
                score=score,
                bm25_score=bm25_scores[index],
                embedding_score=final_embedding_scores[index],
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def tokenize(text: str, learned_terms: set[str] | None = None) -> list[str]:
    lowered = str(text or "").lower()
    base_tokens = TOKEN_RE.findall(lowered)
    emoji_tokens = _extract_emoji_tokens(lowered)
    joined_cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
    bigrams = [joined_cjk[i : i + 2] for i in range(len(joined_cjk) - 1)]
    pinyin_units = _pinyin_units(joined_cjk)
    adaptive_tokens = []
    if learned_terms:
        for token in base_tokens:
            adaptive_tokens.extend(_adaptive_alnum_subtokens(token, learned_terms))
    tokens = base_tokens + emoji_tokens + bigrams + pinyin_units + adaptive_tokens
    return [token for token in tokens if token.strip()]


def _extract_emoji_tokens(text: str) -> list[str]:
    return [match.group(0) for match in EMOJI_TOKEN_RE.finditer(text)]


def _build_learned_terms(query: str, records: list[JargonRecord]) -> set[str]:
    learned_terms: set[str] = set()
    texts = [query]
    texts.extend((record.keyword or record.content or "") for record in records)
    for text in texts:
        lowered = str(text or "").lower()
        learned_terms.update(_extract_alnum_terms(lowered))
        joined_cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
        learned_terms.update(_pinyin_units(joined_cjk))
    return {term for term in learned_terms if len(term) >= 2}


def _extract_alnum_terms(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text) if len(token) >= 2 and token.isascii()]


def _adaptive_alnum_subtokens(token: str, learned_terms: set[str]) -> list[str]:
    normalized = str(token or "").strip().lower()
    if len(normalized) < 4 or not normalized.isascii():
        return []
    matches: set[str] = set()
    max_window = min(len(normalized), 8)
    for start in range(len(normalized)):
        for size in range(3, max_window + 1):
            end = start + size
            if end > len(normalized):
                break
            fragment = normalized[start:end]
            if fragment == normalized:
                continue
            if fragment in learned_terms:
                matches.add(fragment)
    return sorted(matches)


def _pinyin_units(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    units = lazy_pinyin(cleaned, style=Style.NORMAL, errors="ignore")
    normalized = [unit for unit in units if unit]
    if not normalized:
        return []
    acronym = "".join(unit[0] for unit in normalized if unit)
    pinyin_bigrams = ["".join(normalized[i : i + 2]) for i in range(len(normalized) - 1)] if len(normalized) > 1 else []
    extras = [acronym] if acronym else []
    return normalized + pinyin_bigrams + extras


def _hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    units = tokenize(text)
    condensed = str(text or "").lower().replace(" ", "")
    units.extend(condensed[i : i + 3] for i in range(max(0, len(condensed) - 2)))

    if not units:
        return vector

    for unit in units:
        index = hash(unit) % dimensions
        idx = index if index >= 0 else index + dimensions
        vector[idx % dimensions] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 <= 0 or norm2 <= 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _build_ranks(scores: list[float]) -> list[int]:
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(indexed, 1):
        ranks[index] = rank
    return ranks



def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    return sum(left * right for left, right in zip(vector_a, vector_b))


def _build_ranks(scores: list[float]) -> list[int]:
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)

    ranks = [len(scores) + 1 for _ in scores]
    for rank, (index, _score) in enumerate(indexed_scores, start=1):
        ranks[index] = rank
    return ranks
