"""Model providers and the factory that builds them from a profile entry."""

from __future__ import annotations

from fademem.config import ProviderConfig
from fademem.providers.base import Embedder, LLMClient, LLMResult, Message, Usage
from fademem.providers.fake import FakeEmbedder, FakeLLM

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "FakeLLM",
    "LLMClient",
    "LLMResult",
    "Message",
    "Usage",
    "build_embedder",
    "build_llm",
]


def build_llm(cfg: ProviderConfig) -> LLMClient:
    if cfg.kind == "fake":
        return FakeLLM(model=cfg.model)
    raise NotImplementedError(f"LLM provider kind {cfg.kind!r} arrives in M2")


def build_embedder(cfg: ProviderConfig) -> Embedder:
    if cfg.kind == "fake":
        return FakeEmbedder(dim=cfg.dim or 64, model=cfg.model)
    raise NotImplementedError(f"Embedder provider kind {cfg.kind!r} arrives in M2")
