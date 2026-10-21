"""Paths, constants and tunable thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "starter_kit" / "data" / "docs.json"
QUESTIONS_PATH = ROOT / "starter_kit" / "data" / "example_questions.json"
LEXICON_PATH = Path(__file__).resolve().parent / "lexicon.json"
PROMPT_PATH = ROOT / "prompts" / "grounded_answer.md"
EVAL_QUESTIONS_PATH = ROOT / "eval" / "questions.json"

ABSTAIN_MESSAGE = "I don't have an answer for that in the provided documentation."

MAX_QUESTION_CHARS = 300


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the abstention gate.

    These are heuristics tuned against ``eval/questions.json``; they are not
    calibrated probabilities. Precision is deliberately weighted over recall:
    a wrong answer is worse than "I don't know".
    """

    # Bare keywords ("password") are too ambiguous to answer.
    min_query_tokens: int = 2
    # Distinct question terms the best passage must match.
    min_matched_terms: int = 2
    # IDF-weighted share of question terms the selected passages must cover.
    min_coverage: float = 0.8
    # Runner-up article within this fraction of the best score => ambiguous.
    max_ambiguity_ratio: float = 0.9
    # Neighbouring sentence is added if its score >= ratio * best score.
    neighbor_ratio: float = 0.4
    # Never return more than this many sentences.
    max_passages: int = 3
