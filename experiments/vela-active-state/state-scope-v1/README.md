# VELA ONA/Soar state-scope comparison v1

Status: EXPERIMENT_ONLY / NONBINDING / NO_WINNER

Purpose: test the exact boundary left open by the persistence-classification note. The question is not whether ONA or Soar can save *something*, but whether their ordinary persistence path preserves the live state that is causally active at the cut point.

Conditions:
1. same-process/native state reference;
2. fresh runtime restored through the product's ordinary durable save/reload path;
3. fresh runtime with matched external pre-cut state replay.

ONA fixture:
- durable eternal belief;
- current temporal belief;
- current goal;
- inspect concept memory and live cycling belief/goal queues.
- durable-reload control restores only the durable declarative belief.

Soar fixture:
- a procedural production plus an input-link working-memory marker;
- `save agent` to disk;
- fresh runtime sources the saved agent;
- matched replay condition re-adds the working-memory marker.

Interpretation:
- If ordinary durable restoration keeps long-term/declarative/procedural content but drops the live queue/working-memory state, classify it as partial reconstruction rather than a full Active-State checkpoint.
- If matched replay reconstructs the visible state, that shows re-execution can rebuild the state; it does not by itself make replay equivalent to a native causal checkpoint.
- None of these outcomes proves VELA identity or selects an engine.
