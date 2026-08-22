# VELA Mamba M0 state checkpoint probe v1

Status: EXPERIMENT_ONLY / M0_FAMILY_PROBE / NO_WINNER

Purpose: check a second recurrent/state-space family after RWKV. This uses the pretrained `state-spaces/mamba-130m-hf` model on CPU and tests the model's causally active inference cache at a supported one-token continuation boundary.

Conditions:
1. native continuation from the live pre-cut Mamba cache;
2. continuation from the same cache serialized to disk and reloaded;
3. continuation from a fresh state with the pre-cut cache omitted.

Gate:
- native == restored logits within tolerance;
- fresh-state continuation differs.

This is a Mamba-1 130M family probe. It is not evidence about Mamba-3 specifically, nor an engine-selection or cognitive-continuity result.
