"""Answer engine: sanitise -> analyse -> retrieve -> gate -> (optional LLM) -> log."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import gate
from .config import ABSTAIN_MESSAGE, GateConfig
from .guardrails import sanitize_question
from .index import Index
from .llm import (
    ChatClient,
    LLMConfig,
    LLMError,
    OpenAICompatibleClient,
    PromptTemplate,
    generate_grounded_answer,
    load_prompt,
)
from .observability import QueryLogger
from .text import Lexicon

MODES = ("extractive", "llm")


@dataclass(frozen=True, eq=False)
class Answer:
    text: str
    doc_id: str | None = None
    title: str | None = None
    confidence: float = 0.0
    reason: str = "answered"
    detail: str = ""
    mode: str = "extractive"
    passage_ids: tuple[str, ...] = ()
    unknown_terms: tuple[str, ...] = ()
    related: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    candidates: tuple[dict, ...] = ()
    latency_ms: float = 0.0

    @property
    def answered(self) -> bool:
        return self.doc_id is not None

    def to_dict(self) -> dict:
        return {
            "answered": self.answered,
            "text": self.text,
            "doc_id": self.doc_id,
            "title": self.title,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "detail": self.detail,
            "mode": self.mode,
            "passage_ids": list(self.passage_ids),
            "unknown_terms": list(self.unknown_terms),
            "related": [{"doc_id": d, "title": t} for d, t in self.related],
            "notes": list(self.notes),
        }


class Engine:
    """Build the index once, then answer many questions."""

    def __init__(
        self,
        documents: list[dict[str, str]],
        *,
        lexicon: Lexicon | None = None,
        config: GateConfig | None = None,
        llm: ChatClient | None = None,
        llm_config: LLMConfig | None = None,
        prompt: PromptTemplate | None = None,
        query_logger: QueryLogger | None = None,
    ) -> None:
        self.lexicon = lexicon or Lexicon.load()
        self.index = Index(documents, self.lexicon)
        self.config = config or GateConfig()
        self._llm = llm
        self._llm_config = llm_config
        self._prompt = prompt
        self.query_logger = query_logger

    def _client(self) -> ChatClient:
        if self._llm is None:
            self._llm = OpenAICompatibleClient(self._llm_config or LLMConfig.from_env())
        return self._llm

    def _template(self) -> PromptTemplate:
        if self._prompt is None:
            self._prompt = load_prompt()
        return self._prompt

    def ask(self, question: str, mode: str = "extractive") -> Answer:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        started = time.perf_counter()
        answer = self._answer(question, mode)
        elapsed = (time.perf_counter() - started) * 1000
        answer = Answer(**{**answer.__dict__, "latency_ms": elapsed})
        if self.query_logger:
            self.query_logger.log(question, answer)
        return answer

    def _abstain(self, mode: str, reason: str, detail: str, **extra) -> Answer:
        return Answer(ABSTAIN_MESSAGE, reason=reason, detail=detail, mode=mode, **extra)

    def _answer(self, question: str, mode: str) -> Answer:
        clean = sanitize_question(question)
        if clean.too_long:
            return self._abstain(mode, "question_too_long", "please shorten the question")
        if clean.empty:
            return self._abstain(mode, "empty_question", "no question text provided")
        if clean.injection_suspected:
            return self._abstain(
                mode, "blocked_input", "the question looks like an instruction to the assistant"
            )

        analysis = self.lexicon.analyze(clean.text)
        raw_tokens = len(clean.text.split())
        result = gate.evaluate(self.index, analysis, raw_tokens, self.config)
        if not result.answered:
            return self._abstain(
                mode,
                result.reason,
                result.detail,
                confidence=result.confidence,
                unknown_terms=result.unknown_terms,
                related=result.related,
                candidates=result.candidates,
            )

        passages = [item.passage for item in result.selected]
        first = passages[0]
        extractive_text = " ".join(passage.text for passage in passages)
        common = dict(
            doc_id=first.doc_id,
            title=first.title,
            confidence=result.confidence,
            reason="answered",
            detail=result.detail,
            passage_ids=tuple(passage.passage_id for passage in passages),
            candidates=result.candidates,
        )

        if mode == "extractive":
            return Answer(extractive_text, mode="extractive", **common)

        try:
            outcome = generate_grounded_answer(
                self._client(), self._template(), clean.text, passages, self.lexicon
            )
        except LLMError as exc:
            from .llm import LLMOutcome

            outcome = LLMOutcome("error", note=str(exc))

        if outcome.status == "ok" and outcome.text:
            return Answer(outcome.text, mode="llm", **common)
        if outcome.status == "declined":
            return self._abstain(
                "llm",
                "llm_declined",
                "the gate accepted the sources but the model judged them insufficient",
                confidence=result.confidence,
                related=((first.doc_id, first.title),),
                candidates=result.candidates,
                notes=(outcome.note,),
            )
        note = f"LLM unavailable or unverified ({outcome.note}); showing extractive answer"
        return Answer(extractive_text, mode="extractive", notes=(note,), **common)


def answer_question(
    question: str,
    documents: list[dict[str, str]],
    *,
    mode: str = "extractive",
    config: GateConfig | None = None,
) -> Answer:
    """Convenience wrapper: build an index and answer a single question."""
    return Engine(documents, config=config).ask(question, mode=mode)
