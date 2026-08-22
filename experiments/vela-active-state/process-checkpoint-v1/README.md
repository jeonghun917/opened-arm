# VELA full-process checkpoint reference v1

Status: EXPERIMENT_ONLY / REFERENCE_MECHANISM / NO_WINNER / NO_ARCHITECTURE_FREEZE

Purpose: add a stronger control after the ordinary ONA/Soar persistence test. Their built-in save paths preserved durable content but omitted live queues or working memory. This suite asks whether an external whole-process checkpoint that captures causally active runtime state can restore the same observable continuation.

Mechanism: DMTCP checkpoints the running Linux process computation. It is used here only as a reference for full causal-state preservation, not as the proposed VELA architecture.

Conditions for each candidate:
1. build a live state and stop at a stable cut;
2. checkpoint the whole computation;
3. let the original computation continue and record a state signature;
4. restart from the checkpoint in a new OS process instance;
5. issue the same post-cut inspection and compare signatures.

ONA checkpoint includes the Python harness, ONA shell process, pipes, and ONA's live cycling belief/goal queues.

Soar checkpoint includes the Python runtime, Soar kernel thread, loaded production, agent state, and live input-link working memory.

Interpretation gate:
- equal signatures: whole-process causal state was sufficient for this fixture;
- mismatch: either checkpoint coverage is incomplete or restart semantics differ;
- neither result proves VELA identity, consciousness, or engine superiority.
