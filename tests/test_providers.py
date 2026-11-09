import numpy as np
import pytest
from pydantic import BaseModel

from fademem.config import ProviderConfig
from fademem.providers import (
    Embedder,
    FakeEmbedder,
    FakeLLM,
    LLMClient,
    Message,
    build_embedder,
    build_llm,
)


class Label(BaseModel):
    relation: str


def test_fakes_satisfy_protocols():
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(FakeEmbedder(), Embedder)


def test_fake_llm_queue_then_echo_and_records_calls():
    llm = FakeLLM(replies=["first"])
    assert llm.complete([Message("user", "hi")]).text == "first"
    assert llm.complete([Message("user", "echo me")]).text == "echo me"
    assert [c[-1].content for c in llm.calls] == ["hi", "echo me"]


def test_fake_llm_structured_output_is_validated():
    llm = FakeLLM(replies=['{"relation": "contradictory"}'])
    result = llm.complete([Message("user", "classify")], schema=Label)
    assert isinstance(result.parsed, Label)
    assert result.parsed.relation == "contradictory"


def test_fake_llm_schema_without_reply_fails_loudly():
    with pytest.raises(ValueError):
        FakeLLM().complete([Message("user", "x")], schema=Label)


def test_fake_llm_responder_sees_messages_and_schema():
    seen = []

    def responder(messages, schema):
        seen.append(schema)
        return '{"relation": "compatible"}'

    result = FakeLLM(responder=responder).complete([Message("user", "x")], schema=Label)
    assert result.parsed.relation == "compatible"
    assert seen == [Label]


def test_fake_embedder_is_deterministic_normalised_and_word_sensitive():
    emb = FakeEmbedder(dim=128)
    a, b, c, a2 = emb.embed(
        [
            "Alice moved to Paris last year",
            "Alice moved to Berlin last year",
            "the quarterly tax report is due",
            "Alice moved to Paris last year",
        ]
    )
    assert emb.embed(["x", "y"]).shape == (2, 128)
    assert np.allclose(np.linalg.norm([a, b, c], axis=1), 1.0)
    assert np.array_equal(a, a2)
    assert a @ b > a @ c  # shared words -> more similar


def test_fake_embedder_empty_inputs():
    emb = FakeEmbedder(dim=8)
    assert emb.embed([]).shape == (0, 8)
    assert not emb.embed([""]).any()


def test_factory_builds_fakes_and_rejects_unimplemented_kinds():
    assert isinstance(build_llm(ProviderConfig(kind="fake", model="m")), FakeLLM)
    assert build_embedder(ProviderConfig(kind="fake", model="e", dim=16)).dim == 16
    with pytest.raises(NotImplementedError):
        build_llm(ProviderConfig(kind="openai_compat", model="gpt-4o-mini"))
