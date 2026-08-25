# VELA Handoff — Result Audit 2026-08-26

Read after `docs/VELA_HANDOFF_CURRENT.md`. Gate authority remains the Drive Gate Ledger plus audited source/workflow/result evidence.

## A. 0.4B selector-v3 — migration subgate PASS

- Run `32802306230`
- result `ops/vela-results/g7-rwkv7-0p4b-selector-v3-latest.json`
- native stable `6/8`
- migration-only excess failure `4 -> 0`
- `selector_v3_migration_subgate_pass=true`
- remaining 0.4B blocker: `old_value_echoes 0/2` native-stable

## B. 1.5B focused probe — checkpoint+recipe NO-GO

- Run `32802357103`
- result `ops/vela-results/g7-rwkv7-1p5b-native-scale-probe-v1-latest.json`
- immediate W3-native accuracy `0.75`
- native stable `0/2`

Not a clean general scaling verdict.

## C. 2.9B v1 — infrastructure failure only

- Run `32810175701`
- scientific step exit `143`
- runner shutdown
- no authoritative scientific result

Do not use as model-performance evidence.

## D. 2.9B focused v2 — native blocker rescue PASS

- Run `32814109278`
- result `ops/vela-results/g7-rwkv7-2p9b-native-scale-probe-v2-latest.json`
- source/head `4cf22827cc5d9f012d42a5f38e40506b26f97ead`
- exact-SHA/wrapper/dependency hardening PASS
- `old_value_echoes` immediate qualification PASS
- telemetry/inventory native stable `2/2`
- 128/512/2048 expected accuracy `1.0`

## E. RWKV-7 2.9B FULL G7 — PASS

- Run `32835329386`
- source commit `db96d09fcb0893280590b449523a10db31c1ec4f`
- result snapshot commit `1bc953fe99a640d8047e5fbc580f1cb796f6ca99`
- authoritative result `ops/vela-results/g7-rwkv7-2p9b-full-selector-v3-v1-latest.json`

All jobs SUCCESS:
- reusable W2/W3 adaptation build
- eval shard A
- eval shard B
- aggregate
- record

Audited result:
- fixtures `4`
- cases `8`
- initial chain qualified `4/4`
- native stable `8/8`
- every fixture stable `2/2`
- migration-only excess failure `0`
- `native_coverage_gate=true`
- `migration_excess_gate=true`
- `formal_g7_2p9b_pass_candidate=true`

Selector-v3/provenance candidate trace is serialized in this result. W3-native/future trajectory remains evaluation-only and is not used for anchor selection.

Gate consequence: **G7 PASS in the currently defined synthetic scope.** This does not freeze the final backbone, selector rule, state schema, checkpoint interval, production hardware, or prove general deployed identity/robustness.

## F. Foundation consequence after G7 PASS

Private foundation now includes an explicit model-evidence port:
- `TargetFreeAnchorEvidence`
- `TargetFreeAnchorEvidenceProvider`
- `EvidenceBackedAnchorSelector`

The foundation consumes normalized target-free equivalent checkpoint IDs and preserves the generic `EquivalentCarriedAnchorGuardSelector` safety boundary. Non-target-free/oracle evidence is refused; full replay cannot be weakened. No RWKV tensor or score/threshold is embedded in core.

Latest tested private code commit:
- `7f369f6dbb11bf3fcd936232a776e64a071c2b5f`

Isolated public CI:
- Run `32874626696`
- `58/58 PASS`

## G. Next

1. Concrete RWKV provider + ReplayAdapter bridge to the foundation evidence port.
2. Integrated foundation regression: model evidence -> guarded anchor -> replay -> validate/fallback -> canonical commit/audit.
3. Then durable G8/application lifecycle work.
4. CPU inference backend/quantization/performance tuning remains deferred until correctness integration is complete.

No paid GPU, mainline merge, irreversible storage migration or final freeze without separate approval.
