"""BM25 passage ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .index import Index, Passage


@dataclass(frozen=True, eq=False)
class ScoredPassage:
    passage: Passage
    text_score: float  # BM25 over the sentence text only
    score: float  # text_score plus a small title bonus
    text_matched: frozenset[str]
    matched: frozenset[str]  # text matches plus title matches


def bm25_rank(
    index: Index,
    query_terms: Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    title_weight: float = 0.5,
) -> list[ScoredPassage]:
    """Rank passages that share at least one *text* term with the query."""
    terms = list(dict.fromkeys(query_terms))
    ranked: list[ScoredPassage] = []
    for passage in index.passages:
        text_matched = frozenset(term for term in terms if term in passage.tf)
        if not text_matched:
            continue
        length_norm = k1 * (1 - b + b * len(passage.terms) / index.avg_passage_length)
        text_score = sum(
            index.idf[term] * passage.tf[term] * (k1 + 1) / (passage.tf[term] + length_norm)
            for term in text_matched
        )
        title_matched = frozenset(
            term for term in terms if term in index.title_terms[passage.doc_id]
        )
        title_score = title_weight * sum(index.idf[term] for term in title_matched)
        ranked.append(
            ScoredPassage(
                passage=passage,
                text_score=text_score,
                score=text_score + title_score,
                text_matched=text_matched,
                matched=text_matched | title_matched,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.passage.doc_index, item.passage.position))
    return ranked
