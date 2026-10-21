"""Abstention gate: decides whether the docs support an answer, and which sentences.

Decision order (first failing check wins; each produces a machine-readable reason):

1. no_content_terms  - nothing left after stop-word removal
2. query_too_short   - bare keyword, too ambiguous to answer
3. uncovered_terms   - a question term appears nowhere in the docs
4. no_match          - no sentence shares a term with the question
5. ambiguous         - two different articles match about equally well
6. weak_match        - best sentence matches too few question terms
7. low_coverage      - selected sentences do not cover enough of the question
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GateConfig
from .index import Index
from .retrieval import ScoredPassage, bm25_rank


@dataclass(frozen=True, eq=False)
class GateResult:
    answered: bool
    reason: str
    detail: str
    confidence: float
    selected: tuple[ScoredPassage, ...] = ()
    unknown_terms: tuple[str, ...] = ()
    related: tuple[tuple[str, str], ...] = ()
    candidates: tuple[dict, ...] = ()


def _coverage(index: Index, query_terms: list[str], matched: frozenset[str]) -> float:
    total = sum(index.idf.get(term, 0.0) for term in query_terms)
    if total <= 0:
        return 0.0
    return sum(index.idf[term] for term in query_terms if term in matched) / total


def _select_window(
    index: Index,
    ranked: list[ScoredPassage],
    top: ScoredPassage,
    query_terms: list[str],
    cfg: GateConfig,
) -> list[ScoredPassage]:
    """Expand the best sentence into a short, same-article answer.

    1. Add contiguous neighbouring sentences that are nearly as relevant
       (procedures and caveats usually span adjacent sentences).
    2. Greedily add the sentence that covers the most still-uncovered
       question terms (multi-part questions).
    """
    same_doc = {
        item.passage.position: item
        for item in ranked
        if item.passage.doc_id == top.passage.doc_id
    }
    selected = {top.passage.position: top}
    floor = cfg.neighbor_ratio * top.text_score

    for step in (1, -1):
        position = top.passage.position + step
        while (
            len(selected) < cfg.max_passages
            and position in same_doc
            and same_doc[position].text_score >= floor
        ):
            selected[position] = same_doc[position]
            position += step

    while len(selected) < cfg.max_passages:
        covered = frozenset().union(*(item.matched for item in selected.values()))
        uncovered = [term for term in query_terms if term not in covered]
        if not uncovered:
            break
        best, best_gain = None, 0.0
        for position, item in same_doc.items():
            if position in selected:
                continue
            gain = sum(index.idf[t] for t in uncovered if t in item.text_matched)
            if gain > best_gain:
                best, best_gain = item, gain
        if best is None:
            break
        selected[best.passage.position] = best

    return [selected[position] for position in sorted(selected)]


def _candidates(ranked: list[ScoredPassage], limit: int = 3) -> tuple[dict, ...]:
    return tuple(
        {
            "passage_id": item.passage.passage_id,
            "score": round(item.score, 3),
            "matched": sorted(item.matched),
        }
        for item in ranked[:limit]
    )


def evaluate(
    index: Index,
    analysis: list[tuple[str, str]],
    raw_token_count: int,
    cfg: GateConfig,
) -> GateResult:
    """Run the gate for one analysed question."""
    display: dict[str, str] = {}
    for raw, term in analysis:
        display.setdefault(term, raw)
    query_terms = list(display)

    if not query_terms:
        return GateResult(False, "no_content_terms", "no searchable words in the question", 0.0)
    if raw_token_count < cfg.min_query_tokens:
        return GateResult(
            False, "query_too_short", "please ask a full question, not a single keyword", 0.0
        )

    unknown = tuple(display[term] for term in query_terms if term not in index.vocabulary)
    known = [term for term in query_terms if term in index.vocabulary]
    ranked = bm25_rank(index, known)
    candidates = _candidates(ranked)

    if unknown:
        related: tuple[tuple[str, str], ...] = ()
        if ranked and len(ranked[0].matched) >= 2:
            top_doc = ranked[0].passage.doc_id
            related = ((top_doc, index.doc_titles[top_doc]),)
        confidence = _coverage(index, query_terms, ranked[0].matched) if ranked else 0.0
        return GateResult(
            False,
            "uncovered_terms",
            f"the documentation does not mention: {', '.join(unknown)}",
            confidence,
            unknown_terms=unknown,
            related=related,
            candidates=candidates,
        )

    if not ranked:
        return GateResult(False, "no_match", "no documentation sentence matches", 0.0)

    top = ranked[0]
    runner_up = next((r for r in ranked if r.passage.doc_id != top.passage.doc_id), None)
    if runner_up and runner_up.score >= cfg.max_ambiguity_ratio * top.score:
        related = tuple(
            (item.passage.doc_id, index.doc_titles[item.passage.doc_id])
            for item in (top, runner_up)
        )
        return GateResult(
            False,
            "ambiguous",
            "the question matches more than one article about equally",
            _coverage(index, query_terms, top.matched),
            related=related,
            candidates=candidates,
        )

    if len(top.matched) < min(cfg.min_matched_terms, len(query_terms)):
        return GateResult(
            False,
            "weak_match",
            "the best sentence shares too few terms with the question",
            _coverage(index, query_terms, top.matched),
            candidates=candidates,
        )

    selected = _select_window(index, ranked, top, query_terms, cfg)
    union = frozenset().union(*(item.matched for item in selected))
    coverage = _coverage(index, query_terms, union)
    if coverage < cfg.min_coverage:
        missing = [display[term] for term in query_terms if term not in union]
        top_doc = top.passage.doc_id
        return GateResult(
            False,
            "low_coverage",
            f"the closest sentences do not address: {', '.join(missing)}",
            coverage,
            related=((top_doc, index.doc_titles[top_doc]),),
            candidates=candidates,
        )

    return GateResult(
        True,
        "answered",
        "enough of the question is supported by one article",
        coverage,
        selected=tuple(selected),
        candidates=candidates,
    )
