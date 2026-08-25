# VELA Experiment Handoff — CURRENT

Date: 2026-08-26 KST  
Status: CORE BASELINE PROMOTED / RWKV-7 2.9B FULL G7 PASS (TESTED SYNTHETIC SCOPE) / FOUNDATION TARGET-FREE EVIDENCE PORT CI PASS / G8 PARTIAL / G9 PASS / FINAL FREEZE BLOCKED

This is the current continuity entrypoint. Historical details remain in authoritative result JSONs and older commits; do not infer success from workflow triggers alone.

## 1. Repositories and boundaries

Experiment:
- `jeonghun917/opened-arm @ vela-experiment-infra`

Foundation:
- `jeonghun917/Ars-Mentis @ vela-foundation-runtime`

Isolated foundation CI:
- `jeonghun917/opened-arm @ vela-foundation-ci`

Do not touch unrelated projects/mainline. Paid GPU, mainline merge, irreversible storage migration, final backbone/schema/checkpoint/selector freeze remain separate approval points.

## 2. Promoted core baseline

`checkpoint store + causal/event lineage + semantic-change analysis + live-cause replay scope + replaceable anchor selector + replay executor + functional validation + rollback/full-replay fallback`

G4/G5/G6 remain PASS and the engine-neutral core baseline remains promoted.

## 3. G7 history needed for interpretation

### 0.4B selector-v3 migration subgate

Authoritative result:
- `ops/vela-results/g7-rwkv7-0p4b-selector-v3-latest.json`
- Run `32802306230`

Observed:
- initial chain `4/4`
- native-stable `6/8`
- migration-only excess failure `4 -> 0`
- `migration_excess_gate=true`
- `selector_v3_migration_subgate_pass=true`
- remaining 0.4B blocker was `superseded_old_value_echoes` native `0/2`

Selector-v3 rule under test: among immediate-functionally-equivalent carried W2 candidates, back off to the earliest W2-origin anchor. No W3-native/future oracle is used for selection.

### 1.5B focused probe

- `ops/vela-results/g7-rwkv7-1p5b-native-scale-probe-v1-latest.json`
- Run `32802357103`
- immediate W3-native accuracy `0.75`, native-stable `0/2`

Interpretation: that checkpoint+recipe is NO-GO; this was not a clean proof that larger RWKV cannot work.

## 4. RWKV-7 2.9B focused rescue — PASS

Authoritative result:
- `ops/vela-results/g7-rwkv7-2p9b-native-scale-probe-v2-latest.json`
- Run `32814109278`
- source/head `4cf22827cc5d9f012d42a5f38e40506b26f97ead`

Observed on `superseded_old_value_echoes`:
- immediate native qualification PASS
- telemetry/inventory native-stable `2/2`
- 128/512/2048 expected accuracy `1.0`

The older Run `32810175701` remains infrastructure/runtime-before-result only and must not be used as scientific evidence.

## 5. RWKV-7 2.9B FULL G7 — PASS

Authoritative result:
- `ops/vela-results/g7-rwkv7-2p9b-full-selector-v3-v1-latest.json`
- source commit `db96d09fcb0893280590b449523a10db31c1ec4f`
- result snapshot commit `1bc953fe99a640d8047e5fbc580f1cb796f6ca99`
- Run `32835329386`

Workflow jobs all SUCCESS:
- one reusable W2/W3 adaptation build
- eval shard A
- eval shard B
- aggregate
- record

Audited summary:
- fixture count `4`
- case count `8`
- initial chain qualified `4/4`
- native stable `8/8`
- per fixture native stable `2/2` for base / suffix4 / suffix8 / old_value_echoes
- migration-only excess failure `0`
- `native_coverage_gate=true`
- `migration_excess_gate=true`
- `formal_g7_2p9b_pass_candidate=true`

All four fixtures use actual carried W2-origin anchors under selector-v3; the result serializes selector-v3 and provenance candidate traces. Functional decision agreement remains `1.0` at 128/512/2048 for the tested paths. W3-native/future data remains evaluation-only, not anchor-selection input.

**Gate consequence:** G7 is PASS in the currently defined 4-fixture × 2-stream synthetic scope.

This does NOT freeze the final backbone, selector formula, canonical state schema, checkpoint interval, production hardware, or prove general identity/real-world long-horizon robustness.

## 6. Foundation integration — current code state

Earlier generic selector-v3 consequence remains implemented:
- `EquivalentCarriedAnchorGuardSelector`
- selector decision propagated into the real `ValidatedReplayExecutor` audit path

New M2 integration port:
- `TargetFreeAnchorEvidence`
- `TargetFreeAnchorEvidenceProvider`
- `EvidenceBackedAnchorSelector`

Purpose: concrete RWKV/model code produces target-free equivalence evidence; foundation consumes only normalized checkpoint IDs/provenance. Core contains no RWKV tensors, threshold, score formula, or future/native oracle. Non-target-free evidence is refused and an inner full-replay decision cannot be weakened.

Latest code-changing tested private commit:
- `7f369f6dbb11bf3fcd936232a776e64a071c2b5f`

Isolated CI:
- Run `32874626696`
- Python 3.11 / ubuntu-latest
- `58/58 PASS`

## 7. Current next order

1. Implement the concrete RWKV evidence provider/ReplayAdapter bridge against the new foundation port without introducing PyTorch/RWKV dependencies into engine-neutral core.
2. Run an integrated foundation G7 regression proving: model evidence -> guarded anchor -> replay -> validate/fallback -> canonical commit/audit.
3. Then strengthen durable G8 integration and actual application/agent lifecycle wiring.
4. CPU inference backend, quantization, replay throughput and production hardware tuning are deferred until the correctness path is complete.

## 8. Gate snapshot

- G7: `PASS / RWKV-7 2.9B FULL G7 (tested synthetic scope)`
- G8: `PARTIAL / POLICY-SAFETY BENCHMARK PASS (small synthetic)`
- G9: `PASS`
- core baseline: `PROMOTED / ACTIVE`
- final backbone/schema/checkpoint/selector/mainline: `NOT FROZEN`

Resume by reading Drive criteria -> Gate Ledger -> Drive CURRENT handoff -> this file -> authoritative JSONs -> private foundation handoff/CI summary.
