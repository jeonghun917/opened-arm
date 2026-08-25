# VELA Experiment Handoff — CURRENT

Date: 2026-08-25 KST  
Status: CORE BASELINE PROMOTED / CHAIN-V2 PASS / G7 MIGRATION SUBGATE PASS / NATIVE BACKBONE OPEN / SELECTOR-V3 FOUNDATION INTEGRATION PASS / G8-G9 STRONG EVIDENCE / FINAL FREEZE BLOCKED

This document is continuity insurance for a new session or maintainer. Active work continues.

## 1. Read this first

Experiment repo/branch:

- `jeonghun917/opened-arm`
- `vela-experiment-infra`

Foundation repo/branch:

- `jeonghun917/Ars-Mentis`
- `vela-foundation-runtime`

Public isolated foundation CI:

- `jeonghun917/opened-arm`
- `vela-foundation-ci`

Do not infer success from a workflow trigger. Re-read authoritative result JSONs and distinguish scientific failure from infrastructure failure. Do not touch unrelated projects or mainline. Paid GPU, mainline merge, irreversible storage migration, and final backbone/schema/checkpoint/selector freeze require separate user approval.

## 2. Project target and promoted core

VELA preserves active/cognitive state continuity across engine/model upgrades without forcing a new model to retain an old model's conclusions.

Promoted engine-neutral baseline:

`checkpoint store + causal/event lineage + semantic-change analysis + live-cause replay scope + anchor selector + replay executor + functional validation + rollback/full-replay fallback`

G4/G5/G6 and core architecture promotion remain valid. G7 does not revoke the core; it blocks final backbone/selector freeze until long-horizon native coverage is adequate.

## 3. Closed chain qualification — RWKV-7 0.4B chain-v2 PASS

Authoritative result:

- `ops/vela-results/rwkv7-0p4b-chain-v2-latest.json`

Result:

- capability W1 -> W2 -> W3: `0.48 -> 0.84 -> 1.00`
- W3-native valid: `4/4`
- hop1 / hop2 / final functional agreement: `4/4 / 4/4 / 4/4`
- hop2 selected anchor source generation W2: `4/4`
- `suite_pass=true`

The v1 provenance-blind W1 fallback was fixed by chain-v2. Do not reopen chain-v3 without contradictory evidence.

## 4. 0.4B retention diagnostic — narrow synthetic success only

The old `11/13 supersession` result was misclassified. Overwrite/codeword was already `13/13`; the two misses were verification persistent-fact retention after neutral suffixes.

Retention-aware adaptation reached:

- correction/control `100/100%`
- disjoint status holdout `4/4`
- superseded full fixture `13/13`
- long-history `38/38`
- residual failures `0`

This is synthetic diagnostic evidence, not a final-backbone or general identity proof.

## 5. G7 — cause separated

### 5.1 Cause-isolation v2 — full gate FAIL

Authoritative result:

- `ops/vela-results/g7-rwkv7-0p4b-cause-isolation-v2-latest.json`
- source commit `f1173aaa7054eb4ffaeb5996c51f8742a0c58c96`

Protocol: 4 fixtures x 2 neutral future streams, horizons `128 / 512 / 2048`.

Observed:

- immediate chain qualification `4/4`
- W3-native stable `6/8`
- W3-native unstable `2/8`, both `superseded_old_value_echoes`
- migration-only excess failures on native-stable cases `4`
- failures are `suffix4` and `suffix8`, both streams
- one-step older W2 anchor rescue `1/4`
- formal candidate `false`

### 5.2 Anchor-depth v3 — migration cause isolated

Authoritative result:

- `ops/vela-results/g7-rwkv7-0p4b-anchor-depth-v3-latest.json`
- source commit `67cd1e871616e3c7322b34ee1a5b7ddcdb757169`

Oracle evaluation over the four native-stable migration failures showed:

- selected failures `4/4`
- rescued by some carried W2-origin bounded anchor `4/4`
- W1-only rescue `0`
- full-replay-only rescue `0`
- `all_selected_failures_rescuable_by_w2_anchor=true`

This proves bounded replay can work on all four; chain-v2 late pruning was the migration-side defect. W3-native future outcomes were used only to label safe anchors, so this diagnostic cannot be used as a production selector.

### 5.3 Target-free selector-v3 — migration subgate PASS

Authoritative result:

- `ops/vela-results/g7-rwkv7-0p4b-selector-v3-latest.json`
- source commit `87ae0eff4c7cff81df91cbc7a3f04f4c1441583f`

Policy:

- preserve chain-v2 target-free candidate detection and immediate functional equivalence;
- among equivalent carried W2-origin candidates choose the **earliest** carried W2-origin anchor rather than the latest;
- no W3-native/future outcome in anchor choice;
- no final selector threshold/score freeze.

Observed:

- initial chain qualified `4/4`
- native-stable `6/8`
- migration-only excess failure `4 -> 0`
- `migration_excess_gate=true`
- `selector_v3_migration_subgate_pass=true`
- full G7 remains blocked because `old_value_echoes` is native-stable `0/2`.

Current G7 conclusion: **migration-specific long-horizon drift is removed on the native-stable synthetic denominator; native backbone stability remains open.**

## 6. Foundation selector-v3 integration — implemented and latest CI PASS

Private foundation branch now contains an engine-neutral selector-v3 consequence:

- `EquivalentCarriedAnchorGuardSelector`
- latest code-changing tested commit `c3eaf6d0c21e30c60e18848c88446fdb8e1e1d53`

Properties:

- wraps an existing selector instead of replacing foundation boundaries;
- receives checkpoint ids already judged equivalent by target-free upstream analysis;
- filters to the specified carried `EngineGeneration`;
- chooses the earliest eligible checkpoint no later than the wrapped selector boundary;
- never weakens a required full replay;
- contains no RWKV tensor assumptions, native/future oracle, frozen score or threshold.

Replay integration now also records an `anchor_selection` audit event before prepare, carrying selected/baseline/eligible checkpoints, selected generation, escalation flag, replay start/count and selector/scope metadata under the same replay transaction id. This makes selector-depth decisions traceable through the real prepare -> validate -> commit/fallback path.

Latest isolated CI:

- run `32812722263`
- tested source commit `c3eaf6d0c21e30c60e18848c88446fdb8e1e1d53`
- Python 3.11 / ubuntu-latest
- `54/54 PASS`

This is a replaceable integration primitive, not a final universal selector rule. The remaining integration step is model-specific target-free equivalence/candidate evidence feeding this generic guard in an end-to-end experiment path.

## 7. Backbone scale track

### 7.1 Existing 1.5B probe — not a clean scale verdict

Authoritative result:

- `ops/vela-results/g7-rwkv7-1p5b-native-scale-probe-v1-latest.json`
- source commit `658f02b2eba12c783590e49f5946432594acb0db`
- checkpoint `rwkv7-g1i-1.5b-20260805-ctx16384.pth`

Observed on `superseded_old_value_echoes`:

- W3-native expected accuracy already `0.75` at time zero
- initial qualification fails
- native-stable `0/2`

Therefore this is a **recipe/checkpoint NO-GO**, not clean evidence that scale cannot help. The long-horizon question is confounded because immediate native validity is not established.

### 7.2 2.9B probe — infrastructure failure, no scientific result

Run `32810175701`, head `888641c178d7444c314c8987ad0dcb83db825876`.

Verified:

- checkout/python/frozen dependency/resource inspection/runtime install PASS
- scientific probe step exited `143`
- runner reported shutdown signal
- artifact upload skipped
- record job failed because no artifact existed
- no authoritative `g7-rwkv7-2p9b-native-scale-probe-v1-latest.json` result was produced

Classification: **infrastructure/runtime-before-result**, not scientific FAIL. Do not claim anything about 2.9B native stability from this run.

Do not grow 0.4B into 1.5B from scratch now. Larger-model work should first make time-zero W3-native qualification valid, then test long-horizon retention. Paid GPU still requires explicit approval.

## 8. G8 current evidence

Retention/storage evidence is strong but production interval is not frozen.

The small suite showed `fixed_n2` as a promising cost/safety point, and the durable follow-up used real `FileCheckpointRepository` I/O with roughly `6.5 MB` state payloads. Keep retention policy replaceable and above repository semantics.

## 9. G9 current evidence

Failure-matrix and process-boundary crash/recovery evidence are strong. Invariant:

- invalid/corrupt/incompatible candidates never silently become canonical;
- unrecoverable paths preserve prior canonical state;
- full replay failure does not commit invalid state;
- CAS/stale-writer/crash remnants remain explicit recovery paths.

Official gate authority remains the Drive Gate Ledger plus audited source/workflow/result evidence.

## 10. What is NOT frozen

Do not freeze or merge without explicit approval:

- final backbone
- canonical active-state layout/schema
- checkpoint interval/policy
- final selector score/threshold/rule
- production storage backend
- distributed locking design
- mainline integration

## 11. Immediate work order

1. Feed actual model-specific target-free equivalence/candidate evidence into the generic `EquivalentCarriedAnchorGuardSelector`.
2. Run focused native-stable G7 regression through that integrated foundation path, then full G7 with native-stable/native-unstable denominators separated.
3. For scale, first qualify the larger checkpoint at time zero; only then compare native long-horizon stability. Treat the 2.9B exit-143 run as infrastructure only.
4. Keep G8/G9 regression coverage while selector integration changes.
5. Only after native G7 coverage is resolved review final backbone/state/checkpoint/selector freeze and end-to-end adapter integration.

## 12. Resume sequence

A new session should:

1. read Drive CURRENT 기준관리 and Gate Ledger;
2. read Drive `00_CURRENT_VELA_개발_인수인계` and the overall summary;
3. read this file and private `Ars-Mentis/docs/G7_G9_INTEGRATION_HANDOFF.md`;
4. re-fetch chain-v2, G7 cause-isolation, anchor-depth-v3, selector-v3 and 1.5B result JSONs;
5. treat 2.9B run `32810175701` as infrastructure failure unless a later authoritative result JSON exists;
6. verify latest private foundation CI before calling newer runtime changes passed.

Key fact to preserve: **selector-v3 has eliminated migration-only G7 failures on the native-stable denominator, and that guard is now wired into generic selector/audit infrastructure; native backbone coverage is still the blocker.**
