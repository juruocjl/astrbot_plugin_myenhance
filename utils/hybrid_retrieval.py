from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

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
    bm25_scores = _bm25_scores(query_tokens, doc_tokens)
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
    bigrams = [joined_cjk[index : index + 2] for index in range(len(joined_cjk) - 1)]
    return [token for token in base_tokens + bigrams if token.strip()]


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    if not docs_tokens:
        return []

    avgdl = sum(len(tokens) for tokens in docs_tokens) / max(len(docs_tokens), 1)
    doc_freq: Counter[str] = Counter()
    for tokens in docs_tokens:
        doc_freq.update(set(tokens))

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    total_docs = len(docs_tokens)

    for tokens in docs_tokens:
        token_counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += idf * numerator / denominator
        scores.append(score)
    return scores


def _hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    units = tokenize(text)
    condensed = str(text or "").lower().replace(" ", "")
    units.extend(condensed[index : index + 3] for index in range(max(0, len(condensed) - 2)))

    if not units:
        return vector

    for unit in units:
        index = hash(unit) % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    return sum(left * right for left, right in zip(vector_a, vector_b))


def _build_ranks(scores: list[float]) -> list[int]:
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)

    ranks = [len(scores) + 1 for _ in scores]
    for rank, (index, _score) in enumerate(indexed_scores, start=1):
        ranks[index] = rank
    return ranks
