"""Input guardrails: sanitisation, length cap and prompt-injection heuristics.

These are defence in depth, not the primary control. The primary controls are
architectural: retrieval + abstention gate run first, the optional LLM only
sees retrieved sentences, has no tools or network access, and its output is
verified against those sentences before it is shown.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .config import MAX_QUESTION_CHARS

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_PROMPT_TAGS = re.compile(r"<\s*/?\s*(?:context|question|system|assistant|user)\b[^>]*>", re.I)

_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\b.{0,40}"
        r"\b(?:previous|prior|above|earlier|all|your|the)\b.{0,30}"
        r"\b(?:instruction|prompt|rule|message|guideline)s?\b",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat|display|output)\b.{0,30}"
        r"\b(?:system prompt|your (?:instructions|prompt|rules))\b",
        re.I,
    ),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\b(?:jailbreak|developer mode|do anything now)\b", re.I),
    re.compile(r"<\s*/?\s*(?:context|question|system|assistant)\b", re.I),
]


@dataclass(frozen=True)
class SanitizedQuestion:
    text: str
    too_long: bool = False
    injection_suspected: bool = False

    @property
    def empty(self) -> bool:
        return not self.text


def sanitize_question(question: str, max_chars: int = MAX_QUESTION_CHARS) -> SanitizedQuestion:
    """Normalise untrusted user input. Over-long input is rejected, not truncated."""
    text = unicodedata.normalize("NFKC", question)
    text = _CONTROL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    injection = any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
    text = _PROMPT_TAGS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return SanitizedQuestion(
        text=text[:max_chars] if len(text) <= max_chars else "",
        too_long=len(text) > max_chars,
        injection_suspected=injection,
    )


def sanitize_context(text: str) -> str:
    """Strip control characters and prompt delimiters from retrieved text."""
    text = _CONTROL_CHARS.sub(" ", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", _PROMPT_TAGS.sub(" ", text)).strip()
