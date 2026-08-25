# VELA Handoff — Result Audit 2026-08-25

Read after `docs/VELA_HANDOFF_CURRENT.md`. This file is the latest override for the two runs that were previously recorded as RUNNING. Gate authority remains the Drive Gate Ledger plus audited source/workflow/result evidence.

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

## B. RWKV-7 1.5B native scale probe — SECONDARY / COMPLETED / NO-GO FOR CURRENT ROUTE

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

Interpretation: the released 1.5B checkpoint under the same adaptation recipe does not rescue the 0.4B native-stability blocker. Per the preregistered decision rule, do not scale blindly to 2.9B.

This is not a general proof that larger RWKV models or scaling cannot help. Because the 1.5B probe fails immediate qualification before the long-horizon test, it is specifically a NO-GO for this checkpoint + adaptation-recipe route and is not a clean size-only causal test.

## C. Current Gate interpretation

Drive Gate Ledger current state should be read as:

- G7: `PARTIAL / MIGRATION SUBGATE PASS / NATIVE COVERAGE FAIL`
- G8: unchanged `PARTIAL / POLICY-SAFETY BENCHMARK PASS (small synthetic)`
- G9: unchanged `PASS`
- core architecture baseline: still PROMOTED / ACTIVE
- final backbone, canonical state schema, checkpoint interval, final selector rule, production storage, and mainline: NOT FROZEN

G7 overall is not PASS. The migration-side blocker observed in cause-isolation v2 is cleared on the tested denominator; the remaining blocker is native validity/backbone robustness.

## D. Next action

Do not launch a larger model merely because 1.5B failed. The next backbone/recipe experiment must preregister and first satisfy immediate native qualification before long-horizon stability is interpreted. Keep selector-v3 as a conservative diagnostic candidate and preserve validation -> deeper replay/full replay fallback.

No paid GPU, mainline merge, irreversible storage migration, or final freeze without separate approval.
