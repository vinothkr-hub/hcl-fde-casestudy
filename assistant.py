"""A small, dependency-free documentation Q&A assistant."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DOCS_PATH = ROOT / "starter_kit" / "data" / "docs.json"
QUESTIONS_PATH = ROOT / "starter_kit" / "data" / "example_questions.json"
ABSTAIN_MESSAGE = "I don't have an answer for that in the provided documentation."
MIN_MATCHED_TERMS = 2
MIN_CONFIDENCE = 0.35

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "can", "do",
    "for", "from", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "the", "this", "to", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    doc_id: str | None = None
    title: str | None = None
    confidence: float = 0.0


def _normalize_token(token: str) -> str:
    if token in {"forgot", "forgotten"}:
        return "reset"

    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        elif token.endswith("us"):
            token += "e"
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
        if token.endswith("ir"):
            token += "e"

    if len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        normalized
        for token in TOKEN_PATTERN.findall(text.lower())
        if (normalized := _normalize_token(token)) not in STOP_WORDS
    ]


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


def _sentence_for_question(question_terms: set[str], content: str) -> str:
    sentences = SENTENCE_PATTERN.split(content.strip())
    if not sentences:
        return content.strip()

    def sentence_score(sentence: str) -> tuple[int, int]:
        matches = question_terms.intersection(tokenize(sentence))
        return len(matches), sum(len(term) for term in matches)

    return max(sentences, key=sentence_score).strip()


def answer_question(question: str, documents: list[dict[str, str]]) -> Answer:
    question_terms = set(tokenize(question))
    if not question_terms or not documents:
        return Answer(ABSTAIN_MESSAGE)

    document_terms = [set(tokenize(document["content"])) for document in documents]
    document_frequency = {
        term: sum(term in terms for terms in document_terms)
        for term in question_terms
    }
    document_count = len(documents)
    weights = {
        term: math.log(
            1 + (document_count - frequency + 0.5) / (frequency + 0.5)
        )
        for term, frequency in document_frequency.items()
    }
    total_weight = sum(weights.values())

    ranked: list[tuple[float, int, int]] = []
    for index, terms in enumerate(document_terms):
        matched_terms = question_terms.intersection(terms)
        matched_weight = sum(weights[term] for term in matched_terms)
        confidence = matched_weight / total_weight if total_weight else 0.0
        ranked.append((confidence, len(matched_terms), index))

    confidence, matched_count, best_index = max(
        ranked, key=lambda result: (result[0], result[1], -result[2])
    )
    if matched_count < MIN_MATCHED_TERMS or confidence < MIN_CONFIDENCE:
        return Answer(ABSTAIN_MESSAGE, confidence=confidence)

    document = documents[best_index]
    excerpt = _sentence_for_question(question_terms, document["content"])
    if not excerpt or not question_terms.intersection(tokenize(excerpt)):
        return Answer(ABSTAIN_MESSAGE, confidence=confidence)

    return Answer(
        text=excerpt,
        doc_id=document["doc_id"],
        title=document["title"],
        confidence=confidence,
    )


def _format_answer(question: str, answer: Answer, question_id: str | None = None) -> str:
    prefix = f"{question_id}: " if question_id else ""
    lines = [f"{prefix}{question}", f"  Answer: {answer.text}"]
    if answer.doc_id and answer.title:
        lines.append(f"  Source: {answer.doc_id} - {answer.title}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions using the Solstice docs.")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="answer all provided example questions",
    )
    parser.add_argument(
        "--question",
        help="ask one question; omit this option to start interactive mode",
    )
    args = parser.parse_args()
    documents = load_documents()

    if args.examples:
        with QUESTIONS_PATH.open(encoding="utf-8") as file:
            examples = json.load(file)["example_questions"]
        for example in examples:
            answer = answer_question(example["question"], documents)
            print(_format_answer(example["question"], answer, example["question_id"]))
    elif args.question:
        print(_format_answer(args.question, answer_question(args.question, documents)))
    else:
        print("Ask a question about the Solstice documentation. Type 'exit' to quit.")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                answer = answer_question(question, documents)
                print(_format_answer(question, answer))


if __name__ == "__main__":
    main()
