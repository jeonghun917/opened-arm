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

## E. RWKV-7 2.9B native scale probe v1 — EXECUTION FAIL / NO SCIENTIFIC RESULT

- Experiment source: `experiments/vela-active-state/g7-scale-2p9b-v1/rwkv7_2p9b_g7_native_scale_probe.py`
- Source-add commit: `0023abc085d92bec61a4741c21828c321cfe5111`
- Workflow: `.github/workflows/vela-g7-rwkv7-2p9b-native-scale-probe-v1.yml`
- Workflow-add/source-run commit: `888641c178d7444c314c8987ad0dcb83db825876`
- Run ID: `32810175701`
- Workflow conclusion: FAILURE.
- Exact scientific step exit: `143`; GitHub runner reported shutdown signal.
- No scientific artifact/result JSON was produced.
- Classification: `EXECUTION FAIL / INFRASTRUCTURE-RUNTIME TERMINATION / SCIENTIFIC RESULT UNAVAILABLE`.

Root-cause analysis: the v1 runner exposed 15 GiB RAM + 3 GiB swap. The 2.9B reference path loads the full model as CPU float32, makes about 629,145,600 K/V/R parameters trainable under AdamW, and reused the full G7 chain/cause-isolation path including W1/W2 lineage and multiple KVR snapshots even though the scale question was native-only. Exit 143 is not direct proof of an OOM kill, but this memory structure is the strongest technical cause candidate and makes v1 structurally inappropriate for the runner.

## F. Reproducibility audit after user challenge

The historical W1->W2->W3 work did have a real provenance issue: an earlier chain could appear valid while hop2 selected a W1-origin path. That issue was later closed by chain-v2's carried-W2 provenance requirement. The user therefore asked whether the failed G7/1.5B evidence could have been distorted by a similar hash/value/JSON problem.

Audit of Run `32802357103`:

- run and record jobs both completed successfully;
- the workflow's frozen dependency verification passed for all seven declared core dependency blobs;
- the authoritative result is a full result, not `FAIL_BEFORE_RESULT`, and contains protocol/model/training/summary/case/horizon data;
- the failure is already present at immediate W3-native qualification (`0.75`, codeword failure), before long-horizon aggregation, so a missing later summary field cannot explain the result;
- no evidence was found that a dependency hash drift or missing JSON value actually occurred in this 1.5B run.

However, a genuine workflow reproducibility weakness exists: historical run jobs check out the mutable branch name `vela-experiment-infra` rather than exact `${{ github.sha }}`. Frozen hash steps protect listed dependencies but do not necessarily assert the experiment wrapper blob. Therefore a branch move between trigger and checkout could theoretically execute a different wrapper while retaining the trigger SHA as metadata.

For the audited 1.5B run, no evidence was found that this theoretical race actually affected the result.

## G. 2.9B v2 hardening and rerun — CURRENT OVERRIDE

New source:

- `experiments/vela-active-state/g7-scale-2p9b-v2/rwkv7_2p9b_native_scale_probe_v2.py`
- wrapper blob: `6571c6c086d519a4afd645db52b8607e880573bf`

Protocol boundary:

- same `superseded_old_value_echoes` x telemetry/inventory x `128/512/2048` native readout;
- same sequential W2 then W3 KVR-only AdamW adaptation rows, learning rates, order, clipping and zero weight decay;
- native-scale-irrelevant W1/W2 lineage, migration/selector evaluation and W1/W2/W3 KVR RAM snapshots removed;
- AdamW `foreach=False` to reduce peak allocation; this is an execution-memory change, not a different target or readout.

Workflow hardening:

1. checkout exact `${{ github.sha }}`;
2. assert `git rev-parse HEAD == $GITHUB_SHA`;
3. assert wrapper blob plus transitive frozen protocol blobs;
4. serialize actual checked-out HEAD and wrapper blob into result JSON;
5. install CPU-only PyTorch explicitly;
6. add 16 GiB swap, heartbeat/max-RSS telemetry, and always-uploaded failure artifact/result marker.

Run `32813968989` validated exact SHA/blob/swap/CPU-only runtime but failed before scientific execution because `numpy` was omitted from the minimized dependency set. It produced a failure artifact with `scientific_failure=false`; this is not a 2.9B model result.

The missing `numpy` dependency was fixed without changing the experiment wrapper/protocol. Current rerun:

- Run ID: `32814109278`
- triggering/head SHA: `4cf22827cc5d9f012d42a5f38e40506b26f97ead`
- exact-SHA/frozen-blob/swap/runtime-install steps: PASS at last audit
- scientific 2.9B native-only step: RUNNING at last audit
- paid GPU: no

Do not infer 2.9B native stability until a scientific result JSON exists. Gate remains `G7 PARTIAL / MIGRATION SUBGATE PASS / NATIVE COVERAGE FAIL`.

## H. CPU-backbone selection constraint

For VELA, `runs inference on CPU` is insufficient. A useful backbone should also expose checkpointable active state, support deterministic restore/continuation, keep working state fixed-size or bounded with history length, and allow a small learned-upgrade/adapter experiment on CPU without requiring a proprietary GPU kernel.

Current practical order:

1. RWKV — primary. Existing VELA restore/upgrade/chain/selector evidence and mature CPU inference paths make it the cheapest architecture to continue testing.
2. Mamba — secondary cross-architecture candidate. SSM state is attractive and VELA already has Mamba restore/learned-upgrade evidence; CPU feasibility for the exact learned-upgrade path must remain an explicit gate because the fastest upstream kernels are GPU-oriented.
3. RecurrentGemma/Griffin — exploratory third candidate. CPU execution and fixed-size recurrent state are viable, but local sliding-window attention makes active-state serialization/migration more complex than RWKV or pure Mamba.
4. xLSTM — deprioritized under the current no-paid-GPU constraint; toy state restore and a practical full G7 learned-upgrade comparison are different questions.

No mainline merge, irreversible storage migration, final freeze, or paid GPU without separate approval.
