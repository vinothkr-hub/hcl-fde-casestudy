"""Evaluate the assistant against eval/questions.json.

Usage:
    python eval/run_eval.py                 # extractive mode (no LLM needed)
    python eval/run_eval.py --mode llm      # needs a local model, see README
    python eval/run_eval.py --verbose       # also print every failure in detail

Metrics (precision is weighted over recall: a wrong answer is worse than "I don't know"):
    false_answer  expected abstention, but the assistant answered
    wrong_source  answered from the wrong article
    bad_content   right article, but a required fact is missing from the answer
    false_abstain expected an answer, but the assistant abstained
Exit code is 1 if any false answer, wrong source or bad content occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solstice_qa import Engine, load_documents  # noqa: E402
from solstice_qa.answer import MODES  # noqa: E402
from solstice_qa.config import EVAL_QUESTIONS_PATH  # noqa: E402


def classify(item: dict, answer) -> str:
    expected = item["expected_doc_id"]
    if expected is None:
        return "false_answer" if answer.answered else "ok"
    if not answer.answered:
        return "false_abstain"
    if answer.doc_id != expected:
        return "wrong_source"
    if any(fact.lower() not in answer.text.lower() for fact in item.get("must_include", [])):
        return "bad_content"
    return "ok"


def evaluate(engine: Engine, questions: list[dict], mode: str) -> list[tuple[dict, str, object]]:
    return [(item, *_ask(engine, item, mode)) for item in questions]


def _ask(engine: Engine, item: dict, mode: str):
    answer = engine.ask(item["question"], mode=mode)
    return classify(item, answer), answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="extractive")
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    engine = Engine(load_documents())
    results = evaluate(engine, questions, args.mode)

    by_category: dict[str, list[str]] = defaultdict(list)
    for item, outcome, _ in results:
        by_category[item["category"]].append(outcome)

    print(f"Mode: {args.mode}   Questions: {len(questions)}\n")
    print(f"{'category':<14}{'n':>3}{'pass':>6}{'false_ans':>11}{'wrong_src':>11}{'bad_content':>13}{'false_abst':>12}")
    for category, outcomes in by_category.items():
        count = outcomes.count
        print(
            f"{category:<14}{len(outcomes):>3}{count('ok'):>6}{count('false_answer'):>11}"
            f"{count('wrong_source'):>11}{count('bad_content'):>13}{count('false_abstain'):>12}"
        )

    outcomes = [outcome for _, outcome, _ in results]
    answerable = [o for (item, o, _) in results if item["expected_doc_id"]]
    unanswerable = [o for (item, o, _) in results if not item["expected_doc_id"]]
    total_fail = len(outcomes) - outcomes.count("ok")
    print(f"\nOverall pass:        {outcomes.count('ok')}/{len(outcomes)}")
    print(f"False-answer rate:   {unanswerable.count('false_answer')}/{len(unanswerable)} unanswerable questions answered")
    print(f"Answer recall:       {answerable.count('ok')}/{len(answerable)} answerable questions answered correctly")

    failures = [(item, outcome, answer) for item, outcome, answer in results if outcome != "ok"]
    if failures:
        print("\nFailures:")
        for item, outcome, answer in failures:
            print(f"  {item['id']} [{outcome}] {item['question']}")
            if args.verbose:
                print(f"      expected={item['expected_doc_id']} got={answer.doc_id} reason={answer.reason}")
                print(f"      text={answer.text[:160]}")
    unsafe = [o for o in outcomes if o in {"false_answer", "wrong_source", "bad_content"}]
    return 1 if unsafe else (0 if total_fail == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
