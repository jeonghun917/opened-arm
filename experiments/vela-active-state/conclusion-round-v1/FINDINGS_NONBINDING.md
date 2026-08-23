# VELA core-hypothesis conclusion round v1 — NONBINDING

Status: `NONBINDING_RESEARCH / NO_ARCHITECTURE_FREEZE / NO_WINNER`

## Evidence boundary

These results are from a narrow synthetic correction curriculum on `state-spaces/mamba-130m-hf`, plus previously completed real-weight checkpoint probes. They test learned capability change, recurrent-state migration, restart exactness, and decision fidelity. They do **not** establish general reasoning, personal identity, consciousness, or a final VELA architecture.

## What survived

1. **Capability gain and causal-state migration are not contradictory.**
   - Primary seed: held-out accuracy `0.3333 -> 0.9167`; correction `0.125 -> 1.0`; control stayed `0.75`.
   - Replication seed 918 reached the same final accuracy/correction/control values.
   - Therefore the migration target should remain `migrated W2 ≈ W2-native after the same causal history`, not reproduction of W1 outputs.

2. **More learned change increased W1→W2 state mismatch in this run.**
   - Primary direct W1-state RMS error vs W2-native after epochs 1/2/3: `0.04175 / 0.05192 / 0.05656`.
   - Capability rose `0.75 / 0.8333 / 0.9167` over the same epochs.
   - This is a tradeoff observation, not a scaling law.

3. **Checkpoint serialization remains exact after migration.**
   - At 16- and 32-token replay points, save/load round-trip cache max error and next-logit max error were both `0.0`.

4. **Exact causal replay can move the old state toward W2-native.**
   - More replay generally reduced state error, but behavioral agreement and raw tensor distance were not interchangeable.

## What failed or weakened

1. **A fixed short recent-replay window is not a universal migration solution.**
   - On the main 59-token history, partial replay retained only `5/6` W2-native hard-gate decisions; full replay reached `6/6`.
   - Different histories showed different behavioral sensitivity even at similar state-distance scales.

2. **The location of the causally important event matters more than replay length alone.**
   - If the correction event occurred early, every tested partial recent window failed the W2-native codeword decision; full replay recovered it.
   - If the correction occurred late, a recent replay window covering that event recovered the W2-native codeword decision.
   - A dedicated anchor test reproduced this: replay from a checkpoint **before** the correction gave `4/4` decision agreement, while replay from a checkpoint **after** the correction gave `3/4` and kept the wrong old codeword.

3. **Simple gradual weight morphing was not better than abrupt W2 activation in state fidelity.**
   - After the same 41 real future tokens, state RMS vs W2-native was:
     - abrupt W2: `0.02711`
     - 2-stage morph: `0.03041`
     - 4-stage morph: `0.03311`
     - 8-stage morph: `0.03453`
   - All four modes matched the tested W2-native decisions, so this is a state-fidelity negative result rather than a behavioral failure.

4. **Uninterrupted future computation does not guarantee monotonic convergence to W2-native.**
   - Partial-migration state/logit errors sometimes contracted and sometimes expanded over the 51-token neutral future, although the tested next-token argmax stayed aligned.

## Current interpretation

The narrow evidence supports the following research direction:

- Keep the real causal engine state as the continuity substrate.
- Treat an engine upgrade as a change in transition dynamics, so the old state is generally not already W2-native.
- Do not require W2 to preserve W1 conclusions; require W2 capability gain plus migration toward the W2-native trajectory.
- Recent replay is only sufficient when it covers the events whose W1 processing is incompatible with W2.
- A practical migration mechanism therefore needs at least one of:
  1. a learned/nontrivial state translator,
  2. exact causal replay from a sufficiently early checkpoint/anchor,
  3. upgrade training that explicitly constrains old-state compatibility,
  4. another mechanism that demonstrates equivalent W2-native future dynamics.

A declarative summary or typed record read by a fresh engine is still not counted as causal-state migration merely because it recreates similar conclusions.

## Next unresolved tests

Before promoting any backbone candidate, the next high-information tests are cross-architecture replication (RWKV-7/xLSTM), learned state-translation beyond affine mapping, compatibility-aware upgrade training, and long-horizon non-neutral/adversarial future trajectories.
