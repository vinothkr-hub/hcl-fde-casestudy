"""Solstice docs Q&A: retrieval-first, abstention-first documentation assistant."""

from .answer import Answer, Engine, answer_question
from .config import ABSTAIN_MESSAGE, GateConfig
from .index import Index, load_documents

__all__ = [
    "ABSTAIN_MESSAGE",
    "Answer",
    "Engine",
    "GateConfig",
    "Index",
    "answer_question",
    "load_documents",
]
