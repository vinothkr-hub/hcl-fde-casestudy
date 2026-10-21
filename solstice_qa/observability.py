"""Opt-in JSONL query log for answered/abstained decisions.

Questions are hashed by default. With ``log_raw_questions`` the text is stored
after masking e-mail addresses and long digit runs. Logging never raises.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .answer import Answer

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_DIGITS = re.compile(r"\d[\d\s-]{4,}\d")


def redact(text: str) -> str:
    return _LONG_DIGITS.sub("[number]", _EMAIL.sub("[email]", text))


class QueryLogger:
    def __init__(self, path: Path | str, log_raw_questions: bool = False) -> None:
        self.path = Path(path)
        self.log_raw_questions = log_raw_questions

    def log(self, question: str, answer: "Answer") -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            "question_chars": len(question),
            "mode": answer.mode,
            "decision": "answered" if answer.answered else "abstained",
            "reason": answer.reason,
            "confidence": round(answer.confidence, 3),
            "doc_id": answer.doc_id,
            "passage_ids": list(answer.passage_ids),
            "unknown_term_count": len(answer.unknown_terms),
            "notes": list(answer.notes),
            "latency_ms": round(answer.latency_ms, 1),
        }
        if self.log_raw_questions:
            record["question"] = redact(question)[:300]
            record["unknown_terms"] = [redact(term) for term in answer.unknown_terms]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"warning: could not write query log: {exc}", file=sys.stderr)
