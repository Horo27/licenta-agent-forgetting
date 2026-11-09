"""Deterministic stand-ins for unit tests: no network, no model, same output every run."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from pydantic import BaseModel

from fademem.providers.base import LLMResult, Message, Usage

Responder = Callable[[Sequence[Message], type[BaseModel] | None], str]

_TOKEN = re.compile(r"\w+")


class FakeLLM:
    """Replies from a queue of canned strings, else from ``responder``, else echoes.

    Every call is recorded in ``calls`` so tests can assert on the prompts that were sent.
    """

    def __init__(
        self,
        replies: Iterable[str] = (),
        responder: Responder | None = None,
        model: str = "fake-llm",
    ) -> None:
        self.model = model
        self._replies = list(replies)
        self._responder = responder
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if self._replies:
            text = self._replies.pop(0)
        elif self._responder is not None:
            text = self._responder(messages, schema)
        elif schema is not None:
            raise ValueError("FakeLLM got a schema request but has no reply or responder")
        else:
            text = messages[-1].content if messages else ""
        parsed = schema.model_validate_json(text) if schema is not None else None
        usage = Usage(
            input_tokens=sum(len(_TOKEN.findall(m.content)) for m in messages),
            output_tokens=len(_TOKEN.findall(text)),
        )
        return LLMResult(text=text, model=self.model, usage=usage, parsed=parsed)


class FakeEmbedder:
    """Hashed bag-of-words embeddings.

    Texts that share words get positive cosine similarity, so similarity-threshold logic
    (conflict candidates, fusion clusters) can be tested meaningfully without a real model.
    """

    def __init__(self, dim: int = 64, model: str = "fake-embedder") -> None:
        self.model = model
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        return np.stack([self._vector(t) for t in texts])
