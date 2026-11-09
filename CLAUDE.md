# licenta-agent-forgetting

Bachelor's thesis (licență) on forgetting in LLM-agent memory.

1. **Reproduce FadeMem** (arXiv 2601.18642v2, no public code). Plan, open questions and milestones are in `docs/reproduction-plan.md`.
2. **Contribution A:** learned importance. Replace the hand-set weights in Eq. 2 with logistic regression or a decision tree.
3. **Contribution B:** multimodal memory. Image memories use ScrapMem-style resolution tiers driven by FadeMem strength, plus V-Mem-style modality-routed retrieval.

## Commands

```bash
uv sync                                   # install deps into .venv
uv run pytest                             # all tests (fake providers, no network)
uv run ruff check . && uv run ruff format --check .
uv run fademem show-config --profile local-dev
uv run fademem check-providers --profile local-dev   # needs `ollama serve`
```

## Layout

- `src/fademem/providers/`: `LLMClient` and `Embedder` protocols (`base.py`), plus implementations. `fake.py` is for tests.
- `src/fademem/config.py`: `Profile` (which models) and `FadeMemParams` (hyperparameters), both loaded from YAML.
- `src/fademem/memory/`: FadeMem itself. `systems/` holds FadeMem and the baselines behind one `MemorySystem` interface; `benchmarks/` and `eval/` hold loaders and metrics.
- `configs/profiles/*.yaml`: model profiles (`test`, `local-dev`, `local-full`, `cloud-openai`, `cloud-vllm`).
- `configs/fademem/default.yaml`: every hyperparameter, each marked `(paper)` or `(ours)`.
- `prompts/`: versioned prompt templates. `results/` and `cache/` are git-ignored.

## Rules

- **Only call models through the protocols.** Code in `memory/`, `systems/` and `eval/` must not import a concrete client or mention a model name. Switching local ↔ cloud is a profile change only.
- **Keep the math pure.** Importance, decay, layers, reinforcement and pruning are plain functions of (state, time in days) with no LLM calls, and each has unit tests.
- **Every value we guess goes into `docs/reproduction-notes.md`.** Each entry gives the value, the reason and the sweep range, and the YAML entry is marked `(ours)`. Don't put hyperparameter defaults in Python; the YAML is the source of truth.
- **Notation:** the paper's β is used twice. In code it's `w_freq` for the importance weight and `shape_lml` / `shape_sml` for the decay exponent.
- **Secrets:** API keys come from env vars named in a profile (`api_key_env`), never from a config file.
- Tests use the `test` profile or the fakes directly. No network in `pytest`.
