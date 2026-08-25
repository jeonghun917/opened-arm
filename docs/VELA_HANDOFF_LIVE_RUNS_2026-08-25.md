# VELA Handoff — Result Audit 2026-08-25

Read after `docs/VELA_HANDOFF_CURRENT.md`. This file is the latest override for the runs recorded below. Gate authority remains the Drive Gate Ledger plus audited source/workflow/result evidence.

## A. G7 selector/replay v3 — PRIMARY / COMPLETED / MIGRATION SUBGATE PASS

- Repo/branch: `jeonghun917/opened-arm@vela-experiment-infra`
- Workflow: `.github/workflows/vela-g7-rwkv7-0p4b-selector-v3.yml`
- Run ID: `32802306230`
- Workflow result: run job SUCCESS; record job SUCCESS.
- Experiment source: `experiments/vela-active-state/g7-selector-v3/rwkv7_0p4b_g7_selector_v3.py`
- Authoritative result: `ops/vela-results/g7-rwkv7-0p4b-selector-v3-latest.json`
- Result source commit: `87ae0eff4c7cff81df91cbc7a3f04f4c1441583f`
- Result snapshot commit: `e2789f4ef9fb850a6e1be83472235ae3f26ce71b`

Audited result:

- initial chain qualified: `4/4`
- W3-native long-horizon stable: `6/8`
- W3-native unstable: `2/8`
- `superseded_old_value_echoes` native-stable streams: `0/2`
- migration-only excess failures on the native-stable denominator: `4 -> 0`
- `migration_excess_gate=true`
- `selector_v3_migration_subgate_pass=true`
- `native_coverage_gate=false`
- `formal_g7_v2_pass_candidate=false`

Interpretation: the observed migration-only long-horizon failures are eliminated on the same native-stable G7 denominator by the conservative earliest-carried-W2 functional-equivalence guard. This materially strengthens the diagnosis that unsafe late-anchor pruning, not bounded replay itself, caused the four migration-only failures.

The selector-v3 policy remains diagnostic. It uses no W3-native/future oracle for anchor choice, but it must not be frozen as the final production selector from this suite alone.

Result-report limitation: the final selected anchor segment is recorded in the inherited G7 case report, but selector-v3's internal candidate/provenance trace is not serialized into this result JSON. Source inspection confirms that the wrapper replaces `chosen_seg/state/rows` before the inherited report consumes the selector output, but a future audit-friendly revision should serialize the selector-v3 trace explicitly.

## B. RWKV-7 1.5B native scale probe — COMPLETED / NO-GO FOR THAT CHECKPOINT+RECIPE

- Workflow: `.github/workflows/vela-g7-rwkv7-1p5b-native-scale-probe-v1.yml`
- Run ID: `32802357103`
- Workflow result: run job SUCCESS; record job SUCCESS.
- Experiment source: `experiments/vela-active-state/g7-scale-1p5b-v1/rwkv7_1p5b_g7_native_scale_probe.py`
- Public checkpoint: `BlinkDL/rwkv7-g1 / rwkv7-g1i-1.5b-20260805-ctx16384.pth`
- Authoritative result: `ops/vela-results/g7-rwkv7-1p5b-native-scale-probe-v1-latest.json`
- Result source commit: `658f02b2eba12c783590e49f5946432594acb0db`
- Result snapshot commit: `3731ecebbf7847905c9c58ea7a4b65276151766f`
- Paid GPU: no.

Audited result on preregistered `superseded_old_value_echoes` x telemetry/inventory x 128/512/2048:

- `native_stable_case_count=0/2`
- `native_unstable_case_count=2/2`
- `native_coverage_gate=false`
- `initial_chain_qualified=0`
- immediate W3-native expected accuracy is `0.75` in both streams; the codeword probe is already wrong before long-horizon continuation.

Interpretation: the released 1.5B checkpoint under the same adaptation recipe does not rescue the 0.4B native-stability blocker. Because it fails immediate qualification, this is not a clean size-only causal test and does not prove that larger RWKV cannot help.

## C. Current Gate interpretation

Drive Gate Ledger current state:

- G7: `PARTIAL / MIGRATION SUBGATE PASS / NATIVE COVERAGE FAIL`
- G8: unchanged `PARTIAL / POLICY-SAFETY BENCHMARK PASS (small synthetic)`
- G9: unchanged `PASS`
- core architecture baseline: still PROMOTED / ACTIVE
- final backbone, canonical state schema, checkpoint interval, final selector rule, production storage, and mainline: NOT FROZEN

G7 overall is not PASS. The migration-side blocker observed in cause-isolation v2 is cleared on the tested denominator; the remaining blocker is native validity/backbone robustness.

## D. User-directed 2.9B follow-up

The user explicitly requested a 2.9B follow-up because 1.5B may itself be too small. This overrides the previous recommendation not to scale blindly, but does not alter the Gate or freeze policy.

Official released source checkpoint verified before launch:

- `BlinkDL/rwkv7-g1 / rwkv7-g1i-2.9b-20260805-ctx16384.pth`
- source revision: `ede85bf8ab2e59aff7d7ca909fbbc73317866d89`
- shape: 32 layers, width 2560, head size 64, vocab 65536, ctx16384

## E. RWKV-7 2.9B native scale probe — RUNNING

- Experiment source: `experiments/vela-active-state/g7-scale-2p9b-v1/rwkv7_2p9b_g7_native_scale_probe.py`
- Source-add commit: `0023abc085d92bec61a4741c21828c321cfe5111`
- Workflow: `.github/workflows/vela-g7-rwkv7-2p9b-native-scale-probe-v1.yml`
- Workflow-add/source-run commit: `888641c178d7444c314c8987ad0dcb83db825876`
- Run ID: `32810175701`
- Last verified state: setup/checkout/python/frozen dependency verification/resource inspection PASS; runtime install in progress; scientific probe pending.
- Scope: same preregistered `superseded_old_value_echoes` x telemetry/inventory x horizons 128/512/2048.
- Same adaptation recipe and CPU float32 path are preserved initially; no paid GPU.
- Planned authoritative result: `ops/vela-results/g7-rwkv7-2p9b-native-scale-probe-v1-latest.json`

Decision discipline:

- `2/2` native stable: strong scale evidence; consider full 2.9B G7, but no backbone freeze.
- `1/2`: partial scale benefit; inspect horizons/margins before expanding.
- `0/2`: no focused benefit under this recipe; do not infer all larger RWKV are impossible.
- OOM/timeout/download/runtime failure before a valid result: infrastructure failure, NOT scientific failure. The next step may be a separately labelled memory-adapted probe; do not silently change precision/optimizer and call it the same protocol.

No mainline merge, irreversible storage migration, final freeze, or paid GPU without separate approval.
