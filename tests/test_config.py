from pathlib import Path

import pytest
from pydantic import ValidationError

from fademem.config import deep_merge, load_params, load_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = ROOT / "configs/fademem/default.yaml"
PROFILES = sorted((ROOT / "configs/profiles").glob("*.yaml"))


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_profile_validates(path):
    profile = load_profile(path)
    assert profile.name == path.stem
    assert profile.embedder.dim is not None


def test_profiles_never_contain_api_keys():
    for path in PROFILES:
        text = path.read_text()
        assert "sk-" not in text, f"{path.name} looks like it contains an API key"


def test_profile_lookup_by_name(in_repo_root):
    assert load_profile("local-dev").memory_llm.base_url == "http://localhost:11434/v1"


def test_default_params_keep_the_paper_values():
    p = load_params(DEFAULT_PARAMS)
    assert (p.layers.theta_promote, p.layers.theta_demote) == (0.7, 0.3)
    assert (p.layers.capacity_lml, p.layers.capacity_sml) == (1000, 500)
    assert p.decay.lambda_base == 0.1
    assert (p.decay.shape_lml, p.decay.shape_sml) == (0.8, 1.2)
    assert p.fusion.theta_fusion == 0.75


def test_overrides_merge_without_touching_other_values():
    p = load_params(DEFAULT_PARAMS, {"decay": {"mu": 2.5}, "retrieval": {"score": "sim"}})
    assert p.decay.mu == 2.5
    assert p.decay.lambda_base == 0.1
    assert p.retrieval.score == "sim"


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):  # breaks hysteresis
        load_params(DEFAULT_PARAMS, {"layers": {"theta_demote": 0.8}})
    with pytest.raises(ValidationError):  # typo'd key must not be silently ignored
        load_params(DEFAULT_PARAMS, {"decay": {"lamda_base": 0.2}})


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
