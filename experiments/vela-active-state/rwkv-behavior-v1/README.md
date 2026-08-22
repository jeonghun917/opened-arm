# VELA RWKV behavioral-state probe v1

Status: EXPERIMENT_ONLY / M1_BEHAVIORAL_PROBE / NO_ENGINE_SELECTION

Purpose: extend the already-passed RWKV recurrent-state M0 checkpoint test into a small deterministic behavioral probe without pretending the 169M base model is a VELA-quality cognitive engine.

Each of five fixtures has a pre-cut prefix that carries task state and a post-cut suffix with two candidate continuations.

Conditions:
1. native recurrent continuation;
2. serialized recurrent state restored into the same model;
3. fresh model state rebuilt by replaying the full prefix;
4. fresh model with the pre-cut state omitted.

Measurements:
- candidate log-probability scores;
- expected-candidate margin;
- native vs restored score equivalence;
- native vs full-prefix replay equivalence;
- whether omitting the pre-cut state changes the expected-candidate margin;
- prefix tokens avoided when a checkpoint is restored instead of recomputed.

Interpretation gate:
- restored == native is checkpoint-equivalence evidence;
- full-prefix replay == native shows replay can reconstruct the same recurrent state when the complete prefix is available;
- fresh != native shows the pre-cut state is causally relevant to the measured behavior;
- checkpointing can then be valued for avoiding replay/recomputation, not as identity proof.

The five fixtures are intentionally simple. Accuracy is descriptive only and does not promote this small RWKV model as the VELA engine.
