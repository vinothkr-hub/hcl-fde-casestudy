"""Document loading and the in-memory passage index."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DOCS_PATH
from .text import Lexicon, split_sentences


def load_documents(path: Path = DOCS_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        payload: Any = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("docs"), list):
        raise ValueError(f"Expected a JSON object containing a 'docs' list: {path}")

    documents: list[dict[str, str]] = []
    for index, document in enumerate(payload["docs"]):
        if not isinstance(document, dict) or not all(
            isinstance(document.get(field), str)
            for field in ("doc_id", "title", "content")
        ):
            raise ValueError(f"Document at index {index} is missing required text fields")
        documents.append(document)
    return documents


@dataclass(frozen=True, eq=False)
class Passage:
    """One sentence of one article, the unit of retrieval and citation."""

    passage_id: str  # e.g. "D-01#3" (1-based sentence position)
    doc_id: str
    title: str
    doc_index: int
    position: int  # 0-based sentence position within the article
    text: str
    terms: tuple[str, ...]
    tf: dict[str, int]


class Index:
    """Passage index with document-level IDF, built once at startup.

    Retrieval unit: sentence. Term weights: IDF over articles (title and body),
    which keeps weights stable on a small corpus. The title is indexed as a
    separate field so that topic words in a title can support, but never alone
    create, a match.
    """

    def __init__(self, documents: list[dict[str, str]], lexicon: Lexicon) -> None:
        self.lexicon = lexicon
        self.documents = documents
        self.passages: list[Passage] = []
        self.title_terms: dict[str, frozenset[str]] = {}
        self.doc_titles: dict[str, str] = {}
        self.doc_passages: dict[str, list[Passage]] = {}

        doc_term_sets: list[set[str]] = []
        for doc_index, document in enumerate(documents):
            doc_id, title = document["doc_id"], document["title"]
            self.doc_titles[doc_id] = title
            title_terms = frozenset(lexicon.tokenize(title))
            self.title_terms[doc_id] = title_terms
            all_terms: set[str] = set(title_terms)
            doc_passages: list[Passage] = []
            for position, sentence in enumerate(split_sentences(document["content"])):
                terms = tuple(lexicon.tokenize(sentence))
                passage = Passage(
                    passage_id=f"{doc_id}#{position + 1}",
                    doc_id=doc_id,
                    title=title,
                    doc_index=doc_index,
                    position=position,
                    text=sentence,
                    terms=terms,
                    tf=dict(Counter(terms)),
                )
                doc_passages.append(passage)
                all_terms.update(terms)
            self.doc_passages[doc_id] = doc_passages
            self.passages.extend(doc_passages)
            doc_term_sets.append(all_terms)

        doc_count = len(documents)
        doc_freq = Counter(term for terms in doc_term_sets for term in terms)
        self.vocabulary: frozenset[str] = frozenset(doc_freq)
        self.idf: dict[str, float] = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        lengths = [len(passage.terms) for passage in self.passages]
        self.avg_passage_length = (sum(lengths) / len(lengths)) if lengths else 1.0
        self.fingerprint = hashlib.sha256(
            json.dumps(documents, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    @classmethod
    def build(cls, documents: list[dict[str, str]], lexicon: Lexicon | None = None) -> "Index":
        return cls(documents, lexicon or Lexicon.load())
