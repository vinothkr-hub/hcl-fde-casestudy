"""Command-line interface for the Solstice docs Q&A assistant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import MODES, Answer, Engine
from .config import DOCS_PATH, QUESTIONS_PATH
from .index import load_documents
from .llm import LLMConfig, LLMError, OpenAICompatibleClient
from .observability import QueryLogger


def format_answer(
    question: str,
    answer: Answer,
    *,
    question_id: str | None = None,
    expected: tuple[bool, str | None] | None = None,
    debug: bool = False,
) -> str:
    prefix = f"{question_id}: " if question_id else ""
    lines = [f"{prefix}{question}", f"  Answer: {answer.text}"]
    if answer.answered:
        passages = ", ".join(answer.passage_ids)
        lines.append(f"  Source: {answer.doc_id} - {answer.title} [{passages}]")
        lines.append(f"  Confidence: {answer.confidence:.2f}  (mode: {answer.mode})")
    else:
        if answer.detail:
            lines.append(f"  Why: {answer.detail}")
        for doc_id, title in answer.related:
            lines.append(f"  Related: {doc_id} - {title}")
    for note in answer.notes:
        lines.append(f"  Note: {note}")
    if expected is not None:
        ok, expected_doc = expected
        want = expected_doc or "abstain"
        lines.append(f"  Check: {'PASS' if ok else 'FAIL'} (expected {want})")
    if debug:
        lines.append(f"  Debug: reason={answer.reason} latency={answer.latency_ms:.1f}ms")
        for candidate in answer.candidates:
            lines.append(
                f"    candidate {candidate['passage_id']} score={candidate['score']} "
                f"matched={candidate['matched']}"
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assistant.py",
        description="Ask questions answered strictly from the Solstice documentation.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--examples", action="store_true", help="run the supplied example questions")
    target.add_argument("--question", "-q", help="ask one question (omit for interactive mode)")
    target.add_argument("--check-llm", action="store_true", help="check the configured LLM endpoint and exit")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="extractive",
        help="extractive (default, no LLM) or llm (grounded generation, falls back to extractive)",
    )
    parser.add_argument("--compare", action="store_true", help="show extractive and llm answers together")
    parser.add_argument("--debug", action="store_true", help="show the reason code and top candidates")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--docs", type=Path, default=DOCS_PATH, help="path to docs.json")
    parser.add_argument("--llm-model", help="model name (default: $LLM_MODEL or llama3.2:3b)")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible base URL (default: $LLM_BASE_URL or Ollama)")
    parser.add_argument("--log", type=Path, help="append decisions to this JSONL file (questions are hashed)")
    parser.add_argument(
        "--log-raw",
        action="store_true",
        help="also store the redacted question text in the log (off by default)",
    )
    return parser


def _check_llm(config: LLMConfig) -> int:
    print(f"Endpoint: {config.base_url}\nModel:    {config.model}")
    try:
        OpenAICompatibleClient(config).complete(
            "You are checking whether this model endpoint is ready.",
            "Reply with OK.",
        )
    except LLMError as exc:
        print(f"NOT READY: {exc}")
        print("Check that the endpoint is running, the model is available, and credentials are valid.")
        return 1
    print("READY")
    return 0


def _show(engine: Engine, question: str, args: argparse.Namespace, **kwargs) -> Answer:
    modes = MODES if args.compare else (args.mode,)
    last = None
    for mode in modes:
        answer = engine.ask(question, mode=mode)
        last = answer
        if args.json:
            print(json.dumps({"question": question, **answer.to_dict()}, ensure_ascii=False))
        else:
            if args.compare:
                print(f"--- mode: {mode} ---")
            print(format_answer(question, answer, debug=args.debug, **kwargs))
    assert last is not None
    return last


def _run_examples(engine: Engine, args: argparse.Namespace) -> int:
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        examples = json.load(file)["example_questions"]
    passed = 0
    for example in examples:
        modes = MODES if args.compare else (args.mode,)
        for mode in modes:
            answer = engine.ask(example["question"], mode=mode)
            expected_doc = example.get("expected_doc_id")
            ok = answer.doc_id == expected_doc
            passed += ok
            if args.json:
                print(json.dumps({"question_id": example["question_id"], "pass": ok,
                                  "question": example["question"], **answer.to_dict()},
                                 ensure_ascii=False))
                continue
            if args.compare:
                print(f"--- mode: {mode} ---")
            print(format_answer(example["question"], answer, question_id=example["question_id"],
                                expected=(ok, expected_doc), debug=args.debug))
    total = len(examples) * (len(MODES) if args.compare else 1)
    if not args.json:
        print(f"\n{passed}/{total} matched the expected source (or expected abstention).")
    return 0 if passed == total else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    llm_config = LLMConfig.from_env(base_url=args.llm_base_url, model=args.llm_model)
    if args.check_llm:
        return _check_llm(llm_config)

    logger = QueryLogger(args.log, log_raw_questions=args.log_raw) if args.log else None
    engine = Engine(load_documents(args.docs), llm_config=llm_config, query_logger=logger)

    if args.examples:
        return _run_examples(engine, args)
    if args.question:
        _show(engine, args.question, args)
        return 0

    print("Ask a question about the Solstice documentation. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            _show(engine, question, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
