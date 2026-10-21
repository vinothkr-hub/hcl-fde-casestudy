"""Entry point and backwards-compatible exports for the Solstice docs Q&A assistant.

Run ``python assistant.py --help``. The implementation lives in the
``solstice_qa`` package.
"""

from __future__ import annotations

import sys

from solstice_qa import ABSTAIN_MESSAGE, Answer, Engine, answer_question, load_documents
from solstice_qa.cli import main

__all__ = ["ABSTAIN_MESSAGE", "Answer", "Engine", "answer_question", "load_documents", "main"]

if __name__ == "__main__":
    sys.exit(main())
