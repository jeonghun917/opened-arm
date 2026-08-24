# VELA G7-G9 validation v1 — predeclared program

Date: 2026-08-24  
Status: EXECUTION STARTED; formal Gate verdicts remain PENDING.

This file is intentionally committed before G7-G9 outcome inspection. It fixes the first-pass questions and required reporting fields while keeping implementation choices such as final checkpoint interval, selector threshold, backbone and canonical state representation open.

## Inputs

Promoted baseline evidence:

- `ops/vela-results/sequential-w1-w2-w3-v3-latest.json` — G4 small Mamba chained-source PASS
- `ops/vela-results/target-free-anchor-selector-v2-stress-latest.json` — G5 normal stress
- `ops/vela-results/target-free-selector-v3-fallback-latest.json` — G5 injected fallback
- `ops/vela-results/rwkv7-target-free-anchor-selector-v1-latest.json` — G6 RWKV-7 0.1B cross-architecture protocol
- `ops/vela-results/rwkv7-0p4b-residual-diagnostic-v1-latest.json` — 0.4B retention diagnostic; patched arms 13/13 superseded and 38/38 diagnostic suite
- `ops/vela-results/rwkv7-0p4b-chain-v1-latest.json` — larger carried-lineage qualification; pending until produced

No Gate may silently exclude a predeclared fixture because the new-generation native trajectory is inconvenient. Invalid-native cases must be reported separately and retained in the result.

## G7 — long-horizon trajectory stability

Primary question: after migration is immediately functionally correct, how long does the migrated trajectory continue to match the same-generation full-history native trajectory?

First-pass engine: RWKV-7 0.4B CPU float32 using the same pinned public checkpoint and the retention-safe W2/W3 recipe established by the preceding diagnostics.

Predeclared horizon knobs:

- 128 steps
- 512 steps
- 2048 steps

Predeclared fixture families:

- same-slot overwrite
- independent persistent facts
- repeated overwrite
- correction followed by long neutral distractors
- multiple live causal fields
- old-value echo/adversarial vocabulary

Required per-fixture fields:

- native_valid
- immediate_functional_agreement
- first_divergence_step
- divergence_free_steps
- decision_agreement_by_horizon
- invariant_violation_count
- correction_retained
- control_degradation
- terminal_functional_agreement
- selected_anchor, replay_fraction, source_generation

Averages cannot substitute for first-divergence and invariant reporting.

Initial safety criterion: zero fatal invariant violations among native-valid predeclared fixtures. This criterion does not imply final long-horizon promotion by itself.

Target result: `ops/vela-results/g7-long-horizon-v1-latest.json`.

## G8 — checkpoint retention/storage policy

Primary question: how much storage can be saved without increasing unsafe migration or fallback beyond the same validation envelope?

Policies to compare:

- every event boundary
- fixed every 2 events
- fixed every 4 events
- fixed every 8 events
- fixed every 16 events
- semantic-event only
- hybrid semantic-event + maximum-gap cap

Required fields per policy:

- checkpoint_count
- checkpoint_bytes
- bytes_per_event
- storage_amplification
- replay_work_p50
- replay_work_p95
- replay_work_worst
- functional_success_count
- full_replay_fallback_count
- recovery_success_count under missing/corrupt checkpoint injection

Safety comparison is primary; storage/replay efficiency ranks only policies that remain safety-qualified. No N becomes a final interval from this gate alone.

Target result: `ops/vela-results/g8-checkpoint-policy-v1-latest.json`.

## G9 — adversarial/failure/rollback

Predeclared injected failure classes:

1. large upgrade jump
2. selector miss / unsafe late anchor
3. corrupted checkpoint bytes
4. missing selected checkpoint
5. incompatible state schema
6. replay budget exceeded
7. selected-candidate validation failure
8. stale canonical writer / CAS conflict
9. incomplete semantic-analysis coverage
10. partial checkpoint write / crash remnant
11. full replay also fails validation

Required fields per injection:

- failure_class
- detected
- initial_candidate_committed
- fallback_mode
- fallback_passed
- previous_canonical_preserved_when_unrecoverable
- audit_outcome_codes
- final_engine_generation

First-pass safety criterion:

- invalid candidate commits = 0
- every injection yields an explicit audited fallback or abort
- unrecoverable cases leave the previous canonical state unchanged

Target result: `ops/vela-results/g9-failure-matrix-v1-latest.json`.

## Development bridge

Actual runtime integration belongs in `jeonghun917/Ars-Mentis`, branch `vela-foundation-runtime`, behind existing engine-neutral interfaces. The development-side handoff is `docs/G7_G9_INTEGRATION_HANDOFF.md`.

G7-G9 work must not change unrelated branches or mainline. Paid GPU, irreversible storage migration and final freezes require separate approval.
