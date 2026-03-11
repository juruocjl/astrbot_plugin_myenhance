from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from rank_bm25 import BM25Okapi

from .memory_store import MemoryRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class ScoredMemory:
    record: MemoryRecord
    score: float
    bm25_score: float
    embedding_score: float


def hybrid_search(
    query: str,
    records: list[MemoryRecord],
    top_k: int,
    bm25_weight: float = 0.55,
    embedding_weight: float = 0.45,
    rrf_k: int = 60,
    embedding_scores: list[float] | None = None,
) -> list[ScoredMemory]:
    normalized_query = str(query or "").strip()
    if not normalized_query or not records or top_k <= 0:
        return []

    query_tokens = tokenize(normalized_query)
    if not query_tokens:
        return []

    doc_tokens = [tokenize(record.content) for record in records]
    
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

    merged: list[ScoredMemory] = []
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
            ScoredMemory(
                record=record,
                score=score,
                bm25_score=bm25_scores[index],
                embedding_score=final_embedding_scores[index],
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def tokenize(text: str) -> list[str]:
    lowered = str(text or "").lower()
    base_tokens = TOKEN_RE.findall(lowered)
    joined_cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
    bigrams = [joined_cjk[i : i + 2] for i in range(len(joined_cjk) - 1)]
    return [token for token in base_tokens + bigrams if token.strip()]


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
