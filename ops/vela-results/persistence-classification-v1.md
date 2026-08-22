# VELA candidate persistence classification v1

Status: NONBINDING_RESEARCH / EMPIRICALLY_CHECKED_SCOPE / NO_WINNER

## ONA / OpenNARS for Applications

Official `misc/Python/persistentNAR.py` persists a Python-side `memory` dictionary to `mem.json`. On restart it reloads that JSON, restores stamp IDs, and re-injects remembered beliefs into ONA when queried.

Classification for the VELA continuity experiment:

- `persistentNAR.py`: **external declarative-memory reconstruction path**.
- It is useful as a durable-memory organ/reference control.
- It is **not** a native full causal-state checkpoint.

Empirical state-scope result on the isolated CPU fixture:

- same-process ONA retained the active cycling belief and active goal;
- durable-memory-style fresh reload did **not** retain either live queue state;
- the durable declarative concept was retained;
- matched pre-cut transcript replay rebuilt the observable live state, but by re-execution.

External whole-process checkpoint reference (DMTCP):

- checkpoint command succeeded;
- restart succeeded in a new process instance;
- native and restored observable belief/goal queue signatures were identical for the fixture.

Interpretation: ONA's ordinary persistence path behaves like partial reconstruction, while preserving the full causally active runtime state is sufficient to reproduce this fixture's continuation. DMTCP is only a reference mechanism, not a proposed VELA architecture.

Source: https://github.com/opennars/OpenNARS-for-Applications/blob/master/misc/Python/persistentNAR.py

## Soar

Current Soar `save agent` implementation writes selected settings, procedural memory/rules, and semantic memory. The source explicitly notes episodic-memory saving there is not implemented. This path does not claim to serialize current working memory, active decision phase/cursor, or every piece of transient control state.

`save rete-net`/`load rete-net` persists the Rete network representation; this is useful runtime structure but is not by itself a full cognitive-state checkpoint.

Classification for the VELA continuity experiment:

- `save agent`: **partial durable semantic/procedural state**, not a full Active-State checkpoint.
- `save/load rete-net`: **runtime structure persistence**, not sufficient by itself for same-state continuation.

Empirical state-scope result on the isolated CPU fixture:

- `save agent` created a valid artifact and restored the test production;
- it did **not** restore the live input-link working-memory marker;
- matched external working-state replay rebuilt that marker.

External whole-process checkpoint reference (DMTCP):

- checkpoint command succeeded;
- restart succeeded in a new process instance;
- the live input marker, loaded production, and printed agent-state signature were identical between native and restored continuation.

Interpretation: Soar's ordinary `save agent` path is partial durable-state restoration, while a fuller causal runtime checkpoint can preserve the tested live state. This does not establish that DMTCP, process identity, or Soar itself should become VELA's final substrate.

Source: https://github.com/SoarGroup/Soar/blob/development/Core/CLI/src/cli_load_save.cpp

## Specialist organs

- Z3: treat solver context as a specialist computation state. Rebuilding the context is acceptable unless later experiments show solver-internal continuation adds independent value.
- Fast Downward: treat search/planning state as a specialist state. It can later be checkpointed for efficiency experiments, but it is not treated as VELA identity state by default.

## Current experimental consequence

The earlier open comparison now has one concrete result:

1. ordinary product save/reload can omit live causally active state;
2. matched transcript/working-state replay can reconstruct the visible state;
3. whole-process checkpoint restoration can reproduce the native observable state for both tested ONA and Soar fixtures;
4. therefore fresh OS process identity is not, by itself, the relevant boundary in these fixtures; the preserved causal state is.

This is a mechanism result only. It does **not** prove cognitive continuity, identity, or engine superiority. The next useful gate is behavioral: hold future inputs fixed and test whether preserved engine state changes decision trajectory, correction response, scope adherence, and recomputation on real multi-step fixtures.
