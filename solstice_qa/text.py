"""Text analysis: normalisation, light stemming, stop words and synonyms."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .config import LEXICON_PATH

_CONTRACTION = re.compile(r"(?<=[a-z])'(?:s|t|ve|ll|re|d|m)\b")
_NUMBER_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9'\"(`])")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]


def stem(token: str) -> str:
    """Very light English suffix stripping (plurals, -ing, -ed, trailing e).

    Deliberately simple and deterministic. Irregular forms and true synonyms
    are handled by the data-driven synonym groups, not by special cases here.
    """
    if token.isdigit() or len(token) <= 3:
        return token
    result = token
    if result.endswith("ies") and len(result) > 4:
        result = result[:-3] + "y"
    elif result.endswith("sses"):
        result = result[:-2]
    elif result.endswith("s") and not result.endswith(("ss", "us", "is")):
        result = result[:-1]
    for suffix in ("ing", "ed"):
        if suffix == "ed" and result.endswith("eed"):
            break  # exceed, proceed, need: not a past tense
        if result.endswith(suffix) and len(result) - len(suffix) >= 3:
            result = result[: -len(suffix)]
            if result[-1] == result[-2] and result[-1] not in "lsz":
                result = result[:-1]
            break
    if result.endswith("e") and len(result) > 3:
        result = result[:-1]
    return result


@dataclass(frozen=True)
class Lexicon:
    stop_words: frozenset[str]
    synonyms: dict[str, str]

    @classmethod
    def load(cls, path: Path = LEXICON_PATH) -> "Lexicon":
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        stop_words = frozenset(str(word).lower() for word in payload["stop_words"])
        synonyms: dict[str, str] = {}
        for group in payload["synonym_groups"]:
            canonical = stem(str(group[0]).lower())
            for word in group:
                synonyms[stem(str(word).lower())] = canonical
        return cls(stop_words=stop_words, synonyms=synonyms)

    def analyze(self, text: str) -> list[tuple[str, str]]:
        """Return ``(raw_token, normalised_term)`` pairs for the content words."""
        normalised = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
        normalised = _CONTRACTION.sub("", normalised)
        normalised = _NUMBER_COMMA.sub("", normalised)
        pairs: list[tuple[str, str]] = []
        for raw in _TOKEN.findall(normalised):
            if raw in self.stop_words or (len(raw) < 2 and not raw.isdigit()):
                continue
            term = stem(raw)
            if term in self.stop_words:
                continue
            pairs.append((raw, self.synonyms.get(term, term)))
        return pairs

    def tokenize(self, text: str) -> list[str]:
        return [term for _, term in self.analyze(text)]
