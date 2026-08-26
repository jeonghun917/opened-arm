# VELA Experiment Handoff — CURRENT / FROZEN SNAPSHOT OVERRIDE

Date: 2026-08-26 KST  
Status: **CURRENT STATE FROZEN / HISTORICAL LOG PRESERVED**

This override freezes the current verified development state. **No future milestone or next-action plan is active from this file.** Any `Next`, `next order`, or milestone language retained below is historical experiment/development log only and must not be treated as an instruction to continue work.

Current verified snapshot:
- core: `PROMOTED / ACTIVE`
- G7: `PASS / RWKV-7 2.9B FULL G7 (tested synthetic scope)`
- G8: `PARTIAL / POLICY-SAFETY BENCHMARK PASS (small synthetic)`
- G9: `PASS`
- authoritative full G7 result: `ops/vela-results/g7-rwkv7-2p9b-full-selector-v3-v1-latest.json`
- experiment source: `db96d09fcb0893280590b449523a10db31c1ec4f`
- result snapshot: `1bc953fe99a640d8047e5fbc580f1cb796f6ca99`
- Foundation latest CI-verified source at freeze: `e3e16abb948f7714b7081a1484132bf643cdb924`
- Foundation CI Run `32877549681`: `72/72 PASS`
- audited 2.9B evidence bridge on the same tested source: selected-anchor `8/8`, eligible-set `8/8`, status `PASS`
- later application/reference-host commit `799c02aea762d07517f251cfc88e5923add8134d` exists but is **not part of the frozen verified baseline** because it has not been CI-verified.

Freeze meaning:
- this freezes the CURRENT handoff/development-state snapshot only;
- it does **not** finalize backbone, active-state schema, checkpoint policy, selector final rule, production backend/hardware, or mainline;
- all experiment results, historical interpretation, failure records, prior milestone notes, and authoritative paths below are intentionally preserved.

---

## HISTORICAL EXPERIMENT / DEVELOPMENT LOG — PRESERVED VERBATIM BELOW

# VELA Experiment Handoff — CURRENT

Date: 2026-08-26 KST  
Status: CORE BASELINE PROMOTED / RWKV-7 2.9B FULL G7 PASS (TESTED SYNTHETIC SCOPE) / FOUNDATION M2-M5 PASS / G8 PARTIAL / G9 PASS / FINAL FREEZE BLOCKED

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

`checkpoint store + causal/event lineage + semantic-change analysis + live-cause replay scope + replaceable selector + target-free model evidence + replay executor + functional validation + rollback/full-replay fallback`

G4/G5/G6 remain PASS and the engine-neutral core baseline remains promoted.

## 3. G7 history needed for interpretation

0.4B selector-v3:
- Run `32802306230`
- native-stable `6/8`
- migration-only excess failure `4 -> 0`
- remaining blocker `old_value_echoes 0/2`

1.5B focused probe:
- Run `32802357103`
- immediate W3-native accuracy `0.75`
- native-stable `0/2`
- checkpoint+recipe NO-GO only; not a general scaling verdict.

2.9B focused rescue:
- Run `32814109278`
- `old_value_echoes` immediate qualification PASS
- telemetry/inventory native-stable `2/2`
- 128/512/2048 expected accuracy `1.0`

The older Run `32810175701` remains infrastructure/runtime-before-result only.

## 4. RWKV-7 2.9B FULL G7 — PASS

Authoritative result:
- `ops/vela-results/g7-rwkv7-2p9b-full-selector-v3-v1-latest.json`
- source commit `db96d09fcb0893280590b449523a10db31c1ec4f`
- result snapshot commit `1bc953fe99a640d8047e5fbc580f1cb796f6ca99`
- Run `32835329386`

Audited:
- 4 fixtures x 2 streams
- horizons 128 / 512 / 2048
- initial chain qualified `4/4`
- native stable `8/8`
- each fixture `2/2`
- migration-only excess failure `0`
- `native_coverage_gate=true`
- `migration_excess_gate=true`
- `formal_g7_2p9b_pass_candidate=true`

All selected anchors are actual carried W2-origin anchors under selector-v3. Candidate/provenance trace is serialized. W3-native/future data remains evaluation-only, not selector input.

Gate consequence: **G7 PASS in the defined synthetic scope.** This does not freeze final backbone, selector formula, canonical state schema, checkpoint interval, production hardware, or prove general identity.

## 5. Foundation M2 — concrete RWKV edge — PASS

Implemented on `Ars-Mentis@vela-foundation-runtime`:
- `TargetFreeAnchorEvidence`
- `EvidenceBackedAnchorSelector`
- `RWKVRuntimeBackend`
- `RWKVReplayAdapter`
- `RWKVFunctionalReplayProbe`
- `RWKVTargetFreeAnchorEvidenceProvider`

Concrete RWKV/model logic stays outside the engine-neutral core. The provider receives the actual baseline decision and only considers carried-generation checkpoints no later than that baseline. Functional-equivalence evidence is current-lineage target-free evidence; future/native oracle evidence is not accepted.

## 6. Foundation M3 — audited 2.9B evidence bridge — PASS

Public bridge result:
- `opened-arm@vela-foundation-ci/ops/vela-foundation-rwkv-g7-evidence-bridge-latest.json`

Exact audited 2.9B selector/provenance evidence is replayed through:

`RWKVTargetFreeAnchorEvidenceProvider -> EvidenceBackedAnchorSelector -> EquivalentCarriedAnchorGuardSelector`

Latest result on tested private source `4eb76be5cf7704b174732333dfce01b1798069c5`:
- cases `8`
- selected-anchor match `8/8`
- eligible-set match `8/8`
- oracle-free `8/8`
- native-stable `8/8`
- migration-clean `8/8`
- PASS

This is a selector/evidence integration proof. It does not rerun the 2.9B model or measure production backend performance.

## 7. Foundation M4 — guarded upgrade transaction — PASS

Implemented:
- `build_rwkv_guarded_upgrade_executor(...)`
- upgrade-aware baseline -> RWKV target-free evidence -> conservative guard -> guarded preflight -> RWKV adapter -> validate -> commit/fallback/rollback.

Integration also fixed duplicate selector evaluation: guarded preflight computes the expensive selector/evidence decision once and the actual validated transaction reuses that exact decision. This avoids duplicate RWKV evidence replay and preserves one audited selection.

Durable regression covers `FileCheckpointRepository`, `FileCanonicalStateStore`, replay budget, one evidence evaluation per candidate, commit persistence, and transaction-id continuity from `anchor_selection` through prepare/commit.

## 8. Foundation M5 — durable restart lifecycle — PASS

Implemented:
- `recover_lineage_from_event_log(...)`
- incomplete trailing event recovery only; complete corruption remains fatal.
- durable event log -> in-memory causal lineage rebuild.

Regression covers:
1. durable events/checkpoints;
2. simulated process death during event append;
3. safe trailing-partial recovery;
4. lineage reconstruction;
5. guarded RWKV upgrade and W3 canonical commit;
6. fresh-process reopen of canonical/checkpoint/event stores;
7. recovery of the same committed W3 canonical state and lineage.

Latest code-changing tested private source:
- `4eb76be5cf7704b174732333dfce01b1798069c5`

Foundation CI:
- Run `32876465620`
- Python 3.11 / ubuntu-latest
- `65/65 PASS`
- audited 2.9B bridge also PASS in the same run.

## 9. Current next order

Completed: M0 / M1 / M2 / M3 / M4 / M5.

Next: **M6 application/agent lifecycle integration.** Add the orchestration boundary around the proven transaction path: quiesce/flush current engine state, run guarded upgrade, activate target only after valid canonical commit, and preserve/reactivate previous engine on failure.

CPU inference backend, quantization, replay throughput and production hardware tuning are deliberately deferred until correctness/application integration is complete.

## 10. Gate snapshot

- G7: `PASS / RWKV-7 2.9B FULL G7 (tested synthetic scope)`
- G8: `PARTIAL / POLICY-SAFETY BENCHMARK PASS (small synthetic)`
- G9: `PASS`
- core: `PROMOTED / ACTIVE`
- M2-M5 foundation integration: `PASS`
- final backbone/schema/checkpoint/selector/mainline: `NOT FROZEN`

Resume: Drive criteria -> Gate Ledger -> Drive CURRENT handoff -> this file -> authoritative G7 JSON -> private foundation handoff -> latest Foundation CI + bridge summaries.
