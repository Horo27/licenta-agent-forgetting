# FadeMem reproduction plan

Paper: *FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory*, Wei et al., arXiv [2601.18642v2](https://arxiv.org/html/2601.18642v2) (6 Feb 2026). No public code. No appendix: prompts and most hyperparameters are not in the paper.

This document has four parts: what the paper **gives**, what we **infer** from it, what we **must decide** ourselves, and the **architecture + milestones** for building it local-first and running the real benchmarks in the cloud.

---

## 0. What counts as a successful reproduction

The paper is underspecified, so exact numbers are not a realistic target. Proposed criteria (agree these with your supervisor):

1. **Direction holds:** FadeMem ≥ Mem0 on LoCoMo multi-hop F1 and MSC RP@10, using the same LLM and embedder for both.
2. **Storage claim holds:** SRR around 0.45 (≈55% of memories retained) without a large drop in critical-fact retention.
3. **Ablations point the same way:** removing the dual layer, fusion or conflict resolution each lowers multi-hop F1.
4. **Sensitivity is reported:** every value we chose (§3) gets a small sweep, so the conclusions don't hinge on one guess.

Target numbers from the paper (Table 3): LoCoMo multi-hop F1 29.43 (Mem0 28.37), FCR 85.9%, SRR 0.45; MSC RP@10 77.2% (Mem0 74.8%), TCS 0.82. Ablation (multi-hop F1): w/o LML-SML 19.45, w/o fusion 13.63, w/o conflict 22.88.

---

## 1. What you need

| Item | Local (your Mac) | Cloud (real runs) |
|---|---|---|
| Machine | Apple M5, 24 GB RAM: fine for 7–14B instruct models at 4-bit | API calls need no GPU (a laptop or cheap VM is enough). A GPU box is needed only to serve an open model with vLLM |
| Python | 3.11+ via `uv` (system Python is 3.9, too old) | same `uv.lock` |
| LLM runtime | [Ollama](https://ollama.com) (`brew install ollama`), with an 8B-class instruct model for development and a 14B-class one for local "real" runs | OpenAI API (`gpt-4o-mini`, which is what the paper used; **check it's still offered**, else take the closest successor and record it) |
| Embeddings | `nomic-embed-text` via Ollama, or a sentence-transformers model | `text-embedding-3-small` (paper) |
| Data | LoCoMo: `locomo10.json` from [snap-research/locomo](https://github.com/snap-research/locomo) (10 conversations, 1,986 QA). MSC: Multi-Session Chat (ParlAI / Hugging Face mirror) | same |
| Baseline | `mem0ai` open source, which supports Ollama and OpenAI through config | same |
| Accounts | none | OpenAI API key; optional GPU rental (RunPod, Lambda, Modal) if you want open-model results |

**Rough API cost (estimate, to verify after the first run):** LoCoMo is about 6k turns. Ingesting them costs about 1 embedding per turn plus a few small conflict/fusion calls, and answering costs 1 call per question. That's on the order of 10–20M tokens with `gpt-4o-mini`, a few dollars per full LoCoMo run. MSC is much bigger (5,000 dialogues): evaluate on a fixed random subsample first. The LLM-call cache (§4) means a rerun with unchanged prompts costs nothing.

---

## 2. Given by the paper

**Memory item:** `m_i(t) = (c_i, s_i, v_i(t), τ_i, f_i)`, where c is the embedding, s the text, v the strength ∈ [0,1], τ the creation time and f the access frequency. Time is measured in **days**.

**Importance (Eq. 2):** `I_i(t) = α·rel(c_i, Q_t) + β·f̃_i/(1+f̃_i) + γ·exp(−δ(t−τ_i))`, with `f̃_i = Σ_j exp(−κ(t − t_j))` over access times t_j. Q_t is the "recent context".

**Layers (Eq. 3):** long-term layer (LML) if `I ≥ θ_promote = 0.7`, short-term layer (SML) if `I < θ_demote = 0.3`, with hysteresis between them. Capacity is 1,000 for LML and 500 for SML.

**Decay (Eq. 4–6):** `v_i(t) = v_i(0)·exp(−λ_i·(t−τ_i)^{β_i})`, with `λ_i = λ_base·exp(−μ·I_i(t))` and `λ_base = 0.1`. The shape parameter is β = 0.8 for LML and 1.2 for SML. Half-lives at I=0 are 11.25 days (LML) and 5.02 days (SML); I verified both, and they become a unit test.

**Reinforcement on access (Eq. 7):** `v ← v + Δv·(1−v)·exp(−n_i/N)`, where n_i counts accesses in the last W days.

**Pruning:** a memory is removed if `v < ε_prune` or it has been dormant for more than `T_max` days.

**Conflict resolution (Eq. 8–10):** candidates are memories with cosine similarity > θ_sim. The LLM labels each pair as one of *compatible / contradictory / subsumes / subsumed*.
- Compatible: both are kept, and the old one is down-weighted with `I ← I·(1−ω·sim)`.
- Contradictory: the old strength decays with `v ← v·exp(−ρ·clip((τ_new−τ_i)/W_age, 0, 1))`.
- Subsumes / subsumed: the LLM merges the two into the more general memory.

**Fusion (Eq. 11–13):** memories cluster if `sim > θ_fusion = 0.75` and they're within `T_window` of each other in time. A cluster above a size threshold is fused by the LLM. The fused memory gets `v(0) = max v + ε·var(v)` (clipped), and its decay rate is slowed by `1/(1+log|C|)`. The fusion is rejected if an LLM "preservation" check scores below θ_preserve.

**Update order (Eq. 14):** `M_{t+Δt} = Fusion(Resolution(Decay(M_t, Δt) ∪ {m_new}))`.

**Models:** GPT-4o-mini for conflict resolution and fusion; text-embedding-3-small for embeddings.

**Metrics:**
- SRR = 1 − |retained| / |total|.
- RP@K: "precision of top-K retrieved memories".
- TCS: chronological coherence, 0–1.
- F1: on downstream QA.
- FCR: LLM fact-checking, citing SelfCheckGPT.
- Conflict Acc. = correct strategy chosen; Cons. = coherence after resolution.

**Benchmarks:** LTI-Bench (the authors' own, 10,780 synthetic 30-day sequences with 4,075 injected conflicts; **not released**), MSC, LoCoMo.

**Baselines:** Fixed-16K FIFO window, LangChain Memory, **Mem0 (primary)**, MemGPT.

---

## 3. Inferred and to decide

Each row is something the paper does not state. "Proposed" is the starting choice, and "Sweep" is the sensitivity range for the cloud runs. Every choice goes into `docs/reproduction-notes.md` with its reason.

### 3a. Pipeline semantics

| Question | Proposed | Why / alternative |
|---|---|---|
| What is a memory unit `s_i`? | **LLM-extracted atomic facts** per turn (config switch `memory_unit: fact \| turn`) | The 4-way conflict labels only make sense between atomic statements, and Mem0, the main baseline, also extracts facts. Raw turns become an ablation. The extraction prompt is ours |
| Where does time `t` come from? | LoCoMo/MSC **session timestamps**, converted to days; turns inside a session get small offsets | Paper: "time in days". MSC has no real dates, so use a fixed synthetic gap between sessions (decide: e.g. 1–7 days) |
| When does Eq. 14 run? | On **every new memory**: decay to now, then resolution against candidates, then fusion check | Literal reading of Eq. 14. Alternative is once per session (cheaper); make it a config option |
| What is `Q_t` and `rel()`? | Q_t = embedding of the last *n* turns (n=3); rel = cosine(c_i, Q_t), rescaled to [0,1] | Paper only says "recent context" |
| When is importance / layer recomputed? | At each ingest step and before each retrieval | Q_t changes, so I_i(t) is query-dependent |
| Initial layer and `v(0)` | v(0)=1.0; initial layer from the first I computed | Not stated |
| Hysteresis zone (0.3 ≤ I < 0.7) | Memory **keeps its current layer** | That's what hysteresis means |
| Layer over capacity | Demote (LML→SML) or prune (SML) the lowest-v item | Not stated |
| "Access" for reinforcement | A memory counts as accessed when it is returned in the top-K for a query | Not stated |
| Retrieval score | `cosine(query, c_i) × v_i`, top-K over non-pruned memories | Not stated. Similarity alone is an ablation |
| Top-K | K=10 (matches RP@10) | Not stated for QA |
| Answering LLM | Same as the memory LLM (gpt-4o-mini in the cloud) | Not stated |
| Notation clash: β is used twice | Rename to `w_freq` (importance weight) and `shape_lml/shape_sml` (decay exponent) | Mention in the thesis |

### 3b. Unspecified numbers

| Parameter | Meaning | Proposed | Sweep |
|---|---|---|---|
| α, β (w_freq), γ | importance weights | 0.4, 0.3, 0.3 | grid over the simplex in 0.1 steps (this is also where your **learned-importance** contribution plugs in) |
| δ | recency decay rate (1/day) | 0.1 | 0.05–0.5 |
| κ | access-time decay rate (1/day) | 0.1 | 0.05–0.5 |
| μ | importance → slower decay | 1.0 | 0.5–3 |
| Δv, N, W | reinforcement step, saturation, window (days) | 0.2, 5, 7 | Δv 0.1–0.4 |
| ε_prune, T_max | prune threshold, dormancy limit (days) | 0.05, 30 | ε 0.01–0.2 (this mostly controls SRR) |
| θ_sim | conflict-candidate similarity | 0.8 | 0.7–0.9 |
| ω | compatible down-weight | 0.3 | 0.1–0.5 |
| ρ, W_age | contradiction decay strength, age window (days) | 2.0, 7 | ρ 1–4 |
| T_window, min cluster size | fusion time window (days), size threshold | 3, 3 | – |
| θ_preserve | fusion acceptance | 0.8 | – |
| temperature, seed | LLM determinism | 0, fixed | – |

**Similarity thresholds depend on the embedding model.** θ_fusion=0.75 and θ_sim are calibrated for `text-embedding-3-small`, and cosine values from `nomic-embed-text` have a different spread. Local runs map each threshold to the **same percentile** of the pairwise-similarity distribution, measured once per embedder on a LoCoMo sample. The cloud paper-faithful run uses the raw values.

### 3c. Evaluation (also undefined in the paper)

| Metric | Proposed definition |
|---|---|
| LoCoMo F1 | The **official token-level F1 from the LoCoMo repo**, reported per category. The magnitudes in the paper (29.43) fit token F1, not an LLM judge. **Check the category ID → name mapping against the file**: published papers disagree on it |
| LoCoMo categories | Report all of 1–4; exclude adversarial (5) from the headline number, as Mem0 does; report it separately |
| SRR | as defined, measured at the end of each conversation |
| FCR | LLM judge: for each answer, the fraction of claims supported by the conversation. Fixed prompt and a fixed judge model, kept the same across all systems |
| MSC RP@K | MSC has no relevance labels. Use MSC's per-session **persona summaries** as gold facts: a retrieved memory counts as relevant if a judge (or similarity > threshold) matches it to a gold persona fact needed for the next session |
| TCS | Kendall τ between the retrieved memories' timestamps and their true order, rescaled to [0,1]. A guess: document it as ours |
| Conflict Acc./Cons. | Only on our synthetic benchmark (§5, M6), where the right label is known by construction |

### 3d. Inconsistencies to note in the thesis

- β is used for two different things (Eq. 2 vs Eq. 4/6).
- Table 3 gives SRR 0.00 for MemGPT, but Table 1 gives MemGPT 85.3% storage.
- The "w/o conflict" drop is stated as 22.4%; the numbers give 22.3%.
- Hyperparameters are said to come both from "grid search on validation sets" and from "systematic ablation studies".
- No code, no prompts, no seeds and no variance are reported. A paired t-test is mentioned, but no statistics are given.
- **Mem0 has changed a lot since January 2026.** Pin a Mem0 version from around the paper's date for the faithful comparison, and optionally report the current version too.

---

## 4. Architecture

Swapping local ↔ API is a **config change only**. Ollama, vLLM, LM Studio, OpenRouter and OpenAI all speak the OpenAI chat and embeddings API, so a single adapter with a `base_url` covers almost everything.

```
licenta-agent-forgetting/
├── pyproject.toml / uv.lock
├── configs/
│   ├── profiles/                # WHICH models (swap these)
│   │   ├── local-dev.yaml       # Ollama 8B + nomic-embed-text
│   │   ├── local-full.yaml      # Ollama 14B
│   │   ├── cloud-openai.yaml    # gpt-4o-mini + text-embedding-3-small (paper-faithful)
│   │   └── cloud-vllm.yaml      # open model on a rented GPU
│   ├── fademem/default.yaml     # all §3b hyperparameters
│   └── experiments/*.yaml       # benchmark × system × profile × overrides
├── prompts/                     # extraction, conflict, fusion, preserve, answer, judge (versioned)
├── src/fademem/
│   ├── providers/
│   │   ├── base.py              # LLMClient / Embedder protocols
│   │   ├── openai_compat.py     # OpenAI, Ollama, vLLM… via base_url
│   │   ├── local_st.py          # sentence-transformers embedder (optional)
│   │   ├── cached.py            # SQLite cache wrapper + token/cost log
│   │   └── fake.py              # deterministic fake for unit tests
│   ├── clock.py                 # SimulatedClock (days)
│   ├── memory/
│   │   ├── types.py             # MemoryItem
│   │   ├── store.py             # numpy vector index + persistence
│   │   ├── importance.py  decay.py  layers.py  reinforcement.py  pruning.py
│   │   ├── conflict.py  fusion.py   # the only parts that call the LLM
│   │   └── system.py            # FadeMem: orchestrates Eq. 14
│   ├── systems/                 # everything implements MemorySystem
│   │   ├── base.py              # ingest(turn, t) / retrieve(query, t, k) / stats()
│   │   ├── fixed_window.py
│   │   └── mem0_adapter.py
│   ├── benchmarks/  locomo.py  msc.py  synthetic_lti.py
│   ├── eval/        f1.py  srr.py  rp_at_k.py  tcs.py  judge.py
│   └── cli.py                   # fademem run configs/experiments/x.yaml --profile local-dev
├── results/<run_id>/            # config snapshot, git SHA, metrics.json, calls.jsonl
├── tests/
└── docs/  reproduction-plan.md  reproduction-notes.md
```

**Key interfaces**

```python
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResult: ...

    # LLMResult: text, parsed (validated pydantic object if schema), usage(in/out tokens), model


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...  # L2-normalised


class MemorySystem(Protocol):
    def ingest(self, turn: Turn, now: float) -> None: ...
    def retrieve(self, query: str, now: float, k: int) -> list[Retrieved]: ...
    def stats(self) -> MemoryStats: ...  # counts per layer, pruned, fused…
```

**Design rules**

- **Separate roles per profile:** `memory_llm` (extraction, conflict, fusion), `answer_llm`, `judge_llm` and `embedder` can each point at a different provider. For example, all memory work runs on Ollama while the judge uses the API, and the judge stays fixed across all systems so scores stay comparable.
- **Structured outputs everywhere:** conflict labels and fusion results are pydantic schemas. Ollama and OpenAI both accept a JSON schema. On a parse failure, retry once, then fall back to "compatible". Count the fallbacks: small local models fail more often, and that's a finding.
- **The math is pure and LLM-free:** importance, decay, layers, reinforcement and pruning are plain functions of (state, t). They get unit tests with no model at all, including the 11.25 / 5.02-day half-lives.
- **Determinism and cost control:** temperature 0, a fixed seed, and an SQLite cache keyed on (model, prompt, params). Every call is logged with tokens. Runs checkpoint per conversation, so a crashed cloud run resumes.
- **Mem0 gets the same profile:** the adapter translates our profile into Mem0's config dict, so the baseline uses the same LLM and embedder.
- **Each run records:** the resolved config, git SHA, prompt hashes, package versions and the model names actually used.

---

## 5. Local first, then cloud

| Tier | Where | Data | Purpose |
|---|---|---|---|
| **T0 unit** | laptop, `fake` provider | hand-made cases | the math and pipeline logic are correct. Runs in seconds in `pytest` |
| **T1 smoke** | laptop, `local-dev` (Ollama 8B) | 1 LoCoMo conversation, first 3 sessions + its QA; 20 synthetic conflict sequences | end-to-end works, prompts parse, metrics compute. Minutes |
| **T2 local full** | laptop overnight, `local-full` | all of LoCoMo, MSC sample of 100 | first real numbers, threshold calibration, catching bugs before spending money |
| **T3 cloud** | `cloud-openai` (from laptop or a small VM) | all of LoCoMo, MSC sample of 500–1,000, synthetic LTI | paper-faithful numbers, ablations, sensitivity sweeps |
| **T4 optional** | `cloud-vllm` on a rented GPU | same as T3 | open-model results (relevant later for the image work) |

The step from T2 to T3 is one flag: `--profile cloud-openai`.

---

## 6. Milestones

1. **M0 – Setup:** install `uv` and Python 3.11, create the package skeleton, write CLAUDE.md, and add the `fake` provider, pytest and ruff.
2. **M1 – Core math:** `MemoryItem`, clock, importance, decay, layers with hysteresis, reinforcement, pruning, capacity handling. Unit tests, including the half-lives.
3. **M2 – Providers:** OpenAI-compatible adapter, cache, cost log, profiles. Install Ollama and pull one chat and one embedding model.
4. **M3 – LLM operations:** fact extraction, conflict classification and the four strategies, fusion with the preservation check. Prompts in `prompts/`.
5. **M4 – LoCoMo end-to-end:** loader with timestamps and the category mapping checked, replay runner, retrieval, answering, token F1, SRR, fixed-window baseline. **T1 smoke passes.**
6. **M5 – Mem0 baseline:** pinned version, shared profile.
7. **M6 – Synthetic LTI-style benchmark:** generator for 30-day sequences with injected contradiction, update and overlap conflicts, with known gold labels. Measures conflict Acc./Cons. and critical-fact retention.
8. **M7 – MSC:** loader, persona-based RP@K, TCS.
9. **M8 – Ablations and sweeps:** w/o dual layer, w/o fusion, w/o conflict, retrieval score with and without v, fact vs turn memory units, and the §3b sweeps. T2 locally, then T3 in the cloud.
10. **M9 – Write-up:** reproduction chapter built from `reproduction-notes.md` plus results.

LangChain Memory and MemGPT (now Letta) are optional extra baselines after M8. Mem0 is the one the paper calls primary.

---

## 7. Decisions needed from you

1. **"Cloud" for the real runs:** paper-faithful with the OpenAI API (recommended, cheap, no GPU), or an open model on a rented GPU, or both.
2. **Memory unit:** extracted facts as the main setting, with raw turns as an ablation (recommended), or the other way round.
3. **Success criterion:** confirm §0 with your supervisor.
4. **Budget cap** for API spend, so the runner can stop at a limit.
