# VELA engine transparency gate v1

Status: NONBINDING_EXPERIMENT_GATE / NO_ENGINE_WINNER / NO_ARCHITECTURE_FREEZE

Purpose: define the minimum observability needed to make causal-state continuity claims identifiable rather than merely behavioral.

## Core-engine gates

1. **Pinned transition implementation**
   - Exact source revision or equivalent executable specification of the state-transition computation.
   - Kernel/runtime substitutions must be recorded when they can change numerical or state semantics.

2. **Pinned parameter artifact**
   - Exact model/parameter revision.
   - Prefer a cryptographic hash of the weight artifact used by the experiment.
   - A model family name or moving repository tag is not sufficient.

3. **Enumerated causally active state**
   - The state that can affect the next computation must be inspectable or exportable.
   - Examples include recurrent tensors, SSM/conv caches, controller cursors, scheduler state, RNG state, and any engine-local mutable memory that survives between steps.

4. **Checkpoint round-trip equivalence**
   - Save the active state before continuation.
   - Continue once natively and once from a fresh runtime restored from the checkpoint.
   - Under deterministic matched inputs, the two branches must match within a declared numerical tolerance.

5. **Replay control**
   - Compare full-history replay with checkpoint restoration.
   - If replay reconstructs the same causal state, checkpointing may still be valuable by avoiding recomputation; this is not evidence for process-object identity.

6. **Missing-state ablation**
   - Remove or reset candidate state components and verify whether future computation changes.
   - Divergence is evidence that the removed component is causally load-bearing; it does not by itself establish cognitive identity.

7. **Role boundary**
   - An opaque model/API may still be used as a peripheral VELA organ if it does not own identity-bearing active state or canonical commit authority.
   - A candidate carrying the identity-bearing cognitive state is not promoted on behavioral similarity alone while its causal state is hidden.

## Current interpretation

For the continuity research path, open/inspectable engines are the strongest reference candidates because failures can be attributed to weights, transition code, runtime state, or replay semantics rather than guessed from outputs. This is an experiment-identifiability requirement, not a claim that every eventual VELA module must be open-weight.

Current evidence examples:
- RWKV recurrent state: serialized state reproduced native continuation; full replay also reconstructed the state in the earlier probe.
- Mamba-1 130M: serialized `MambaCache` reproduced native next-token logits exactly in the current M0 probe.
- ONA/Soar ordinary persistence omits live state, while an external whole-process checkpoint preserved the tested live-state signatures.

No engine is selected by this gate.
