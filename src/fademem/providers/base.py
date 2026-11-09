"""Provider protocols.

Everything that talks to a model goes through these two interfaces. Code in ``memory/``,
``systems/`` and ``eval/`` must depend only on them, never on a concrete client, so that
switching Ollama <-> OpenAI <-> vLLM is a profile change and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    # Validated object when ``complete`` was called with ``schema``; otherwise None.
    parsed: BaseModel | None = None


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResult:
        """Return one completion. With ``schema``, the reply must validate against it."""
        ...


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalised rows."""
        ...
