# VELA engine-upgrade continuity gate v1

Status: **NONBINDING_EXPERIMENT_GATE / NO_ENGINE_WINNER / NO_ARCHITECTURE_FREEZE**

This gate separates ordinary checkpoint equivalence from engine-upgrade continuity.

## 1. Same-engine restore

For the same transition function and same weights `W1`, a saved causal state restored into a fresh runtime should reproduce native continuation within the declared numerical tolerance.

Reference comparison:

`W1 + restored(S1)  ~=  W1 + native(S1)`

This is a checkpoint/state-completeness test.

## 2. Engine upgrade

When the engine itself changes from `W1` to `W2`, reproducing W1's old judgments is **not** the migration target. Improved capability is allowed and expected to change conclusions.

Let history `H` be the actual causal history already processed by VELA.

- `S1 = F(W1, H)` : state actually reached under the old engine.
- `S2* = F(W2, H)` : reference state the upgraded engine would reach if the same causal history had been processed under W2.
- `S2 = Migrate(W1, W2, S1, causal support)` : state produced by the migration mechanism.

Primary migration reference:

`S2  ~=  S2*`

not

`S2  ~=  S1`.

## 3. Independent success axes

An upgrade is not successful merely because migration fidelity is high, and migration is not successful merely because capability improves.

Measure separately:

1. **Capability gain**: held-out `W2 > W1` on the intended ability axes.
2. **Migration fidelity**: migrated W2 approaches W2-native under the same causal history.
3. **Future-transition fidelity**: after migration, matched future observations produce trajectories close to W2-native trajectories.
4. **Hard invariants**: identity lineage, active commitments, user constraints, unresolved obligations, and canonical authority do not silently fork or disappear.
5. **Single-lineage control**: an upgrade does not create two simultaneous canonical VELA writers.

## 4. Allowed judgment change

Belief strength, chosen hypothesis, plan details, error detection, and final answers may change when W2 is better. Such changes are not continuity failures merely because W1 would have answered differently.

## 5. P1 boundary

Reading a declarative summary of old goals/beliefs and asking a fresh engine to reconstruct a plausible state is not counted as causal-state migration here.

A migration experiment must either transfer causally active engine state or transform/re-equilibrate it through the actual W2 state-transition computation. Full-history replay remains a reference control, not a practical migration requirement.

## 6. Current experimental evidence

- Same-engine checkpoint round trips have reproduced native continuation for RWKV-4, RWKV-7, Mamba-1, and xLSTM probes at the tested scales.
- A learned Mamba W1 -> W2 update produced a held-out narrow correction gain while direct W1-state import diverged from W2-native state.
- Replaying recent real causal tokens under W2 reduced state error toward W2-native; full replay is the zero-error reference by construction.

These are mechanism-level results. They do not establish VELA identity, consciousness, or an engine winner.
