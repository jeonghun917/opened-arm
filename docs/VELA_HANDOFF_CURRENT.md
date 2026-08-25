# VELA Experiment Handoff — CURRENT

Date: 2026-08-25 KST  
Status: CORE BASELINE PROMOTED / CHAIN-V2 PASS / G7 BLOCKED-DIAGNOSED / G8-G9 STRONG EVIDENCE / FINAL FREEZE BLOCKED

This document is continuity insurance for a new session or maintainer. Active work continues; do not interpret this handoff as a stop-work notice.

## 1. Read this first

Repository/branch for experiment work:

- `jeonghun917/opened-arm`
- branch `vela-experiment-infra`

Do not infer success from workflow triggers. Re-read the authoritative result JSON before changing any gate or freeze decision.

Do not touch unrelated projects or mainline. Paid GPU, mainline merge, irreversible storage migration, final backbone/schema/checkpoint/selector freeze require separate user approval.

## 2. Project target

VELA preserves active/cognitive state continuity across engine/model upgrades without forcing a new model to retain an old model's conclusions.

The promoted engine-neutral baseline is:

`checkpoint store + causal/event lineage + semantic-change analysis + live-cause replay scope + anchor selector + replay executor + functional validation + rollback/full-replay fallback`

G4/G5/G6 and the core architecture promotion remain valid. G7 failure does not revoke the promoted core; it blocks final selector/backbone freeze.

## 3. Closed qualification: RWKV-7 0.4B chain-v2

Authoritative result:

- `ops/vela-results/rwkv7-0p4b-chain-v2-latest.json`

Result:

- capability W1 -> W2 -> W3: `0.48 -> 0.84 -> 1.00`
- W3-native valid: `4/4`
- hop1 functional agreement: `4/4`
- hop2 functional agreement: `4/4`
- final chain agreement: `4/4`
- hop2 selected anchor source generation W2: `4/4`
- `suite_pass=true`

The v1 provenance-blind W1 fallback was fixed by the chain-v2 target-free provenance tiebreak. Do not reopen chain-v3 unless contradictory evidence appears.

## 4. RWKV-7 0.4B retention diagnostic

The old `11/13 supersession` result was misclassified. Overwrite/codeword was already `13/13`; the two misses were verification persistent-fact retention after neutral suffixes.

Retention-aware adaptation reached, on the narrow diagnostic suite:

- correction/control `100/100%`
- disjoint status holdout `4/4`
- superseded full fixture `13/13`
- long-history `38/38`
- residual failures `0`

This is synthetic diagnostic evidence only, not a final-backbone proof.

## 5. G7 formal result — FAIL, with cause separated

### G7 cause-isolation v2

Authoritative result:

- `ops/vela-results/g7-rwkv7-0p4b-cause-isolation-v2-latest.json`
- source commit recorded in result: `f1173aaa7054eb4ffaeb5996c51f8742a0c58c96`

Predeclared protocol: 4 fixtures x 2 neutral future streams, horizons `128 / 512 / 2048`.

Observed:

- immediate chain qualification: `4/4`
- W3-native long-horizon stable cases: `6/8`
- W3-native unstable cases: `2/8`
- both native-unstable cases are `superseded_old_value_echoes`
- on the six native-stable cases, migration-only excess failures: `4`
- all four are `superseded_suffix4` or `superseded_suffix8`, across both future streams
- one-step older W2 anchor rescued only `1/4`
- `formal_g7_v2_pass_candidate=false`

Interpretation at this point: both native 0.4B long-horizon weakness and migration-path drift exist.

### G7 anchor-depth v3 — decisive migration-side diagnosis

Authoritative result:

- `ops/vela-results/g7-rwkv7-0p4b-anchor-depth-v3-latest.json`
- source commit recorded in result: `67cd1e871616e3c7322b34ee1a5b7ddcdb757169`

This is an evaluation-only exhaustive anchor ablation. W3-native is used only to label which historical anchors are safe; therefore this run must NOT be copied directly into a production selector.

Observed over the four native-stable migration-failure cases:

- selected failures: `4/4`
- rescued by some carried W2-origin anchor: `4/4`
- W1-only rescue: `0`
- full-replay-only rescue: `0`
- `all_selected_failures_rescuable_by_w2_anchor=true`

Latest safe W2 anchor segment:

- suffix4 / telemetry: `seg 7`
- suffix4 / inventory: `seg 5`
- suffix8 / telemetry: `seg 5`
- suffix8 / inventory: `seg 5`

**Current migration-side conclusion:** bounded replay is sufficient on all four native-stable failure cases. The main migration defect is that the target-free selector/pruning policy can choose an anchor that is too late/aggressive for long-horizon stability. Full replay is not intrinsically required for these four cases.

**Separate backbone conclusion:** `superseded_old_value_echoes` still fails W3-native long-horizon in both neutral streams at 0.4B. That is a backbone-capacity/robustness question, not a migration-only failure.

## 6. Immediate plan

### Track A — selector/replay v3, PRIMARY

Build a production-target-free policy that uses no W3-native oracle and is more conservative when a late anchor is risky.

Required properties:

1. preserve chain-v2 provenance requirement: hop2 must use actual carried W2 lineage when appropriate;
2. do not use future/native target outcomes to choose an anchor;
3. add a conservative long-horizon-risk proxy / anchor-depth guard / replay-budget escalation mechanism;
4. retain full-replay fallback for validation or budget failure;
5. rerun the four G7 migration-failure cases first, then the complete formal G7 suite;
6. report native-stable denominator separately from native-unstable cases.

Do not freeze a specific score threshold yet. The anchor-depth-v3 oracle ablation is diagnostic evidence, not a deployable selector rule.

### Track B — backbone scale probe, SECONDARY/PARALLEL

Do **not** spend time growing the 0.4B model into 1.5B from scratch now.

Use an existing public RWKV-7 1.5B checkpoint to repeat the native long-horizon / G7 protocol, especially `superseded_old_value_echoes`.

Decision rule:

- if 1.5B materially removes native instability, treat 0.4B capacity as an important bottleneck;
- if native instability persists, consider 2.9B or another recurrent backbone;
- if scale fixes native stability but migrated paths still fail, continue selector/replay work rather than scaling blindly.

No paid GPU without explicit approval.

## 7. G8 current evidence

Authoritative experiment result:

- `ops/vela-results/g8-rwkv7-0p4b-retention-v1-latest.json`
- source commit recorded in result: `501adc83a353f79da0e1218959153401b7d0eefe`

0.4B serialized checkpoint size in this run: `6,507,147 bytes`.

Small-suite tradeoff highlights:

- every-event: safe-anchor availability `1.0`, fallback `0`, 27 checkpoints, ~175.69 MB total
- fixed N=2: safe-anchor availability `1.0`, fallback `0`, 13 checkpoints, ~84.59 MB total, storage amplification `0.4815`
- fixed N=4: safe-anchor availability `0.5`, fallback `0.5`
- fixed N=8: safe-anchor availability `0.25`, fallback `0.75`
- fixed N=16: fallback `1.0`
- semantic-only/hybrid: safe-anchor availability `0.75`, fallback `0.25`

All policies recover in the small test because fallback remains active. `fixed_n2` is a strong candidate, NOT a frozen production interval.

## 8. G9 current evidence

Authoritative first result:

- `ops/vela-results/g9-runtime-failure-matrix-v1-latest.json`

Foundation work has also exercised process-level crash/recovery paths and regression CI. Safety evidence is strong: corrupted/missing/incomplete candidates must not silently become canonical state, and unrecoverable paths preserve the prior canonical state.

Official gate authority remains the Drive Gate Ledger plus audited source/workflow/result evidence; implementation/CI success alone must not silently flip the gate.

## 9. What is NOT frozen

Do not freeze or merge without explicit approval:

- final backbone
- canonical active-state layout/schema
- checkpoint interval/policy
- final selector score/threshold/rule
- production storage backend
- distributed locking design
- mainline integration

## 10. Handoff resume sequence

A new session should:

1. read Drive `00_CURRENT_VELA_개발_기준관리` and `01_CURRENT_VELA_개발_GATE_LEDGER`;
2. read Drive `00_CURRENT_VELA_개발_인수인계`;
3. read this file;
4. re-fetch `g7-rwkv7-0p4b-cause-isolation-v2-latest.json` and `g7-rwkv7-0p4b-anchor-depth-v3-latest.json`;
5. inspect current branch HEAD/workflows before claiming anything is complete;
6. continue Track A selector/replay v3; run Track B 1.5B scale probe when compute route is available/approved.

The key non-obvious fact to preserve is: **G7's four native-stable migration failures are all recoverable with a carried W2-origin anchor, so the migration-side problem is unsafe late-anchor selection, while 0.4B also has an independent native long-horizon weakness.**
