"""Optional grounded LLM answer generation (local by default) with verification.

Contract: the LLM is only ever called *after* the abstention gate has accepted
the question, it only sees the retrieved sentences, and its output is checked
against those sentences. Anything that fails the checks falls back to the
extractive answer. The client uses only the standard library and speaks the
OpenAI-compatible chat API, which Ollama, vLLM, LM Studio and Azure OpenAI
(behind a gateway or with a base URL change) all support.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlparse

from .config import ABSTAIN_MESSAGE, PROMPT_PATH
from .guardrails import sanitize_context
from .index import Passage
from .text import Lexicon

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2:3b"
MAX_RESPONSE_BYTES = 1_000_000
MAX_ANSWER_CHARS = 1200
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LLMError(RuntimeError):
    """The model endpoint is unreachable, misconfigured or returned garbage."""


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 300

    @classmethod
    def from_env(cls, base_url: str | None = None, model: str | None = None) -> "LLMConfig":
        try:
            timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError:
            timeout = 60.0
        return cls(
            base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY") or None,
            timeout_s=timeout,
        )


def validate_base_url(base_url: str) -> None:
    """Allow only http(s); require TLS for anything that is not loopback."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMError(f"LLM_BASE_URL must be an http(s) URL, got: {base_url!r}")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise LLMError("Refusing plain http to a non-local host; use https.")


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        validate_base_url(self.config.base_url)

    def _request(self, path: str, payload: dict | None = None) -> dict:
        url = self.config.base_url.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data else "GET"
        )
        try:
            # Scheme is restricted to http(s) by validate_base_url().
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:  # noqa: S310
                body = response.read(MAX_RESPONSE_BYTES)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM endpoint not reachable at {self.config.base_url}: {exc}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("LLM endpoint returned invalid JSON") from exc

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        result = self._request("/chat/completions", payload)
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from LLM endpoint") from exc
        if not isinstance(content, str):
            raise LLMError("LLM returned non-text content")
        return content

    def list_models(self) -> list[str]:
        result = self._request("/models")
        # Ollama returns "data": null when no model is installed.
        data = result.get("data") or []
        return [str(item.get("id")) for item in data if isinstance(item, dict)]


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str


def load_prompt(path: Path = PROMPT_PATH) -> PromptTemplate:
    """Read the SYSTEM_PROMPT and USER_TEMPLATE fenced blocks from the prompt file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError(f"Cannot read prompt file {path}: {exc}") from exc

    def block(name: str) -> str:
        match = re.search(rf"##\s+{name}\s*\n```text\n(.*?)\n```", text, re.S)
        if not match:
            raise LLMError(f"Prompt file {path} is missing a '## {name}' block")
        return match.group(1).strip()

    return PromptTemplate(system=block("SYSTEM_PROMPT"), user=block("USER_TEMPLATE"))


def build_messages(
    prompt: PromptTemplate, question: str, passages: Sequence[Passage]
) -> tuple[str, str]:
    context = "\n".join(
        f"[{passage.passage_id}] {sanitize_context(passage.text)}" for passage in passages
    )
    user = prompt.user.replace("{{context}}", context).replace("{{question}}", question)
    return prompt.system, user


_CITATION = re.compile(r"\[\s*(D-\d+#\d+)\s*\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_URL_OR_EMAIL = re.compile(r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.-]+")
_ALWAYS_ALLOWED_ENTITIES = {"solstice"}


@dataclass(frozen=True)
class Verification:
    ok: bool
    problems: tuple[str, ...] = ()


def _numbers(text: str) -> set[str]:
    return {match.rstrip(",").replace(",", "") for match in _NUMBER.findall(text)}


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", _CITATION.sub("", text)).strip()


def verify_grounded(
    answer: str,
    passages: Sequence[Passage],
    question: str,
    lexicon: Lexicon,
    max_unsupported_ratio: float = 0.25,
) -> Verification:
    """Check that an LLM answer introduces nothing absent from the sources."""
    problems: list[str] = []
    context = " ".join(passage.text for passage in passages)
    context_lower = context.lower()

    bad_citations = sorted(set(_CITATION.findall(answer)) - {p.passage_id for p in passages})
    if bad_citations:
        problems.append(f"cites passages that were not provided: {', '.join(bad_citations)}")

    body = strip_citations(answer)
    if not body:
        return Verification(False, ("empty answer",))
    if len(body) > MAX_ANSWER_CHARS:
        problems.append("answer is unreasonably long")

    new_numbers = sorted(_numbers(body) - _numbers(context))
    if new_numbers:
        problems.append(f"numbers not found in sources: {', '.join(new_numbers)}")

    for item in _URL_OR_EMAIL.findall(body):
        if item.lower().rstrip(".,)") not in context_lower:
            problems.append(f"link or address not found in sources: {item}")

    allowed_text = f"{context_lower} {question.lower()}"
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", sentence)
        for word in words[1:]:
            lowered = word.lower()
            if (
                word[0].isupper()
                and lowered not in _ALWAYS_ALLOWED_ENTITIES
                and lowered not in allowed_text
            ):
                problems.append(f"name not found in sources: {word}")

    answer_terms = set(lexicon.tokenize(body))
    supported = set(lexicon.tokenize(context)) | set(lexicon.tokenize(question))
    unsupported = sorted(term for term in answer_terms if term not in supported)
    if answer_terms and len(unsupported) / len(answer_terms) > max_unsupported_ratio:
        problems.append(f"too many terms not in sources: {', '.join(unsupported[:8])}")

    return Verification(not problems, tuple(dict.fromkeys(problems)))


@dataclass(frozen=True)
class LLMOutcome:
    status: str  # "ok" | "declined" | "unverified" | "error"
    text: str | None = None
    note: str = ""


def _is_decline(text: str) -> bool:
    cleaned = text.replace("\u2019", "'").casefold()
    return ABSTAIN_MESSAGE.casefold().rstrip(".") in cleaned


def generate_grounded_answer(
    client: ChatClient,
    prompt: PromptTemplate,
    question: str,
    passages: Sequence[Passage],
    lexicon: Lexicon,
) -> LLMOutcome:
    system, user = build_messages(prompt, question, passages)
    try:
        raw = client.complete(system, user)
    except LLMError as exc:
        return LLMOutcome("error", note=str(exc))
    raw = raw.strip().strip("`").strip()
    if _is_decline(raw):
        return LLMOutcome("declined", note="the model judged the sources insufficient")
    check = verify_grounded(raw, passages, question, lexicon)
    if not check.ok:
        return LLMOutcome("unverified", note="; ".join(check.problems))
    return LLMOutcome("ok", text=strip_citations(raw))
