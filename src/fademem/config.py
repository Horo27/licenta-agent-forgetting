"""Configuration: model profiles (which models) and FadeMem hyperparameters (how it behaves).

Both are loaded from YAML. Hyperparameter models have no defaults on purpose: the YAML file is
the single source of truth, so every value used in a run is explicit and gets snapshotted
with the results.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

PROFILES_DIR = Path("configs/profiles")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- profiles


class ProviderConfig(_Strict):
    kind: Literal["fake", "openai_compat", "sentence_transformers"]
    model: str
    base_url: str | None = None
    # Name of the environment variable holding the key. The key itself never goes in a config.
    api_key_env: str | None = None
    dim: PositiveInt | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Profile(_Strict):
    name: str
    description: str = ""
    memory_llm: ProviderConfig  # fact extraction, conflict classification, fusion
    answer_llm: ProviderConfig  # answers benchmark questions from retrieved memories
    judge_llm: ProviderConfig  # FCR and other LLM-judged metrics; keep fixed across systems
    embedder: ProviderConfig


# --------------------------------------------------------------------------- hyperparameters
# Field comments: (paper) = value stated in FadeMem, (ours) = our choice, see reproduction-notes.


class PipelineParams(_Strict):
    memory_unit: Literal["fact", "turn"]
    update_schedule: Literal["per_memory", "per_session"]
    context_turns: PositiveInt  # turns forming Q_t for rel()


class ImportanceParams(_Strict):
    w_rel: float = Field(ge=0)  # alpha in Eq. 2
    w_freq: float = Field(ge=0)  # beta in Eq. 2 (renamed: beta is reused in Eq. 4/6)
    w_recency: float = Field(ge=0)  # gamma in Eq. 2
    recency_rate: PositiveFloat  # delta, 1/day
    access_decay_rate: PositiveFloat  # kappa, 1/day


class LayerParams(_Strict):
    theta_promote: float = Field(ge=0, le=1)
    theta_demote: float = Field(ge=0, le=1)
    capacity_lml: PositiveInt
    capacity_sml: PositiveInt

    @model_validator(mode="after")
    def _hysteresis(self) -> LayerParams:
        if self.theta_demote >= self.theta_promote:
            raise ValueError("theta_demote must be < theta_promote (hysteresis band)")
        return self


class DecayParams(_Strict):
    lambda_base: PositiveFloat
    mu: float = Field(ge=0)
    shape_lml: PositiveFloat  # beta for the long-term layer in Eq. 6
    shape_sml: PositiveFloat  # beta for the short-term layer in Eq. 6
    initial_strength: float = Field(gt=0, le=1)


class ReinforcementParams(_Strict):
    delta_v: float = Field(ge=0, le=1)
    saturation_n: PositiveFloat  # N in Eq. 7
    window_days: PositiveFloat  # W in Eq. 7


class PruningParams(_Strict):
    epsilon: float = Field(ge=0, lt=1)
    max_dormant_days: PositiveFloat


class ConflictParams(_Strict):
    theta_sim: float = Field(ge=-1, le=1)
    omega: float = Field(ge=0, le=1)
    rho: float = Field(ge=0)
    age_window_days: PositiveFloat


class FusionParams(_Strict):
    theta_fusion: float = Field(ge=-1, le=1)
    time_window_days: PositiveFloat
    min_cluster_size: int = Field(ge=2)
    theta_preserve: float = Field(ge=0, le=1)
    variance_bonus: float = Field(ge=0)  # epsilon in Eq. 12


class RetrievalParams(_Strict):
    top_k: PositiveInt
    score: Literal["sim_times_strength", "sim"]


class FadeMemParams(_Strict):
    pipeline: PipelineParams
    importance: ImportanceParams
    layers: LayerParams
    decay: DecayParams
    reinforcement: ReinforcementParams
    pruning: PruningParams
    conflict: ConflictParams
    fusion: FusionParams
    retrieval: RetrievalParams


# --------------------------------------------------------------------------- loading


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` applied recursively. Inputs are not modified."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_profile_path(name_or_path: str | Path) -> Path:
    """Accept a file path, or a bare profile name looked up in ``configs/profiles/``."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} or path.exists():
        return path
    return PROFILES_DIR / f"{name_or_path}.yaml"


def load_profile(name_or_path: str | Path) -> Profile:
    return Profile.model_validate(_read_yaml(resolve_profile_path(name_or_path)))


def load_params(path: str | Path, overrides: dict[str, Any] | None = None) -> FadeMemParams:
    data = _read_yaml(Path(path))
    if overrides:
        data = deep_merge(data, overrides)
    return FadeMemParams.model_validate(data)
