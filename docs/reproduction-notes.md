# Reproduction notes

A running log of every choice the paper leaves open. It becomes the methodology section of the reproduction chapter.
Each entry records what we chose, why, and how we'll test whether it matters.
Proposed defaults and sweep ranges are in `reproduction-plan.md` §3. A row moves here once the code uses it.

| # | Date | Topic | Choice | Reason | Sweep / check |
|---|---|---|---|---|---|
| 1 | 2026-11-09 | Notation | Paper's β (Eq. 2 importance weight) → `w_freq`; β (Eq. 4/6 decay exponent) → `shape_lml`, `shape_sml` | The paper uses the same symbol for both | – |
| 2 | 2026-11-09 | Hyperparameter defaults | All values in `configs/fademem/default.yaml` marked `(ours)` are starting points from the plan, §3b | Not stated in the paper | Ranges in plan §3b |
| 3 | 2026-11-09 | Eq. 12 ε | `fusion.variance_bonus = 0.1` | The paper reuses ε from the pruning threshold, which can't be what's meant; no value given | 0–0.3 |
