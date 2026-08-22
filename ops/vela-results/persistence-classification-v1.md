# VELA candidate persistence classification v1

Status: NONBINDING_RESEARCH / EXPERIMENT_PREP / NO_WINNER

## ONA / OpenNARS for Applications

Official `misc/Python/persistentNAR.py` persists a Python-side `memory` dictionary to `mem.json`. On restart it reloads that JSON, restores stamp IDs, and re-injects remembered beliefs into ONA when queried.

Classification for the VELA continuity experiment:

- `persistentNAR.py`: **external declarative-memory reconstruction path**.
- It is useful as a durable-memory organ/reference control.
- It is **not** evidence that the native ONA process restores its full causally active inference/control state.
- A future ONA continuity test must therefore distinguish same-process continuation from `mem.json` reload and from any genuine native/process checkpoint mechanism.

Source: https://github.com/opennars/OpenNARS-for-Applications/blob/master/misc/Python/persistentNAR.py

## Soar

Current Soar `save agent` implementation writes selected settings, procedural memory/rules, and semantic memory. The source explicitly notes episodic-memory saving there is not implemented. This path does not claim to serialize current working memory, active decision phase/cursor, or every piece of transient control state.

`save rete-net`/`load rete-net` persists the Rete network representation; this is useful runtime structure but is not by itself a full cognitive-state checkpoint.

Classification for the VELA continuity experiment:

- `save agent`: **partial durable semantic/procedural state**, not a full Active-State checkpoint.
- `save/load rete-net`: **runtime structure persistence**, not sufficient by itself for same-state continuation.
- A future Soar continuity test must compare same-process continuation against fresh `save agent` restoration and, separately, a fuller working-memory/process checkpoint if one can be exposed.

Source: https://github.com/SoarGroup/Soar/blob/development/Core/CLI/src/cli_load_save.cpp

## Specialist organs

- Z3: treat solver context as a specialist computation state. Rebuilding the context is acceptable unless later experiments show solver-internal continuation adds independent value.
- Fast Downward: treat search/planning state as a specialist state. It can later be checkpointed for efficiency experiments, but it is not treated as VELA identity state by default.

## Experimental consequence

For ONA and Soar, a persistence feature name must not be counted as continuity evidence. The next matched-control test is:

1. same-process continuation;
2. fresh runtime from the product's ordinary save/reload path;
3. full causally active checkpoint restoration if available;
4. matched future input/event stream.

If ordinary save/reload diverges but full checkpoint matches same-process continuation, the missing value is transient causal state. If ordinary save/reload already matches, no independent advantage should be attributed to process identity.
