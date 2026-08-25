# VELA Handoff — Live Runs 2026-08-25

Read after `docs/VELA_HANDOFF_CURRENT.md`. This file records work launched after the CURRENT handoff snapshot. It is continuity insurance, not a gate decision.

## A. G7 selector/replay v3 — PRIMARY / RUNNING

- Repo/branch: `jeonghun917/opened-arm@vela-experiment-infra`
- Workflow: `.github/workflows/vela-g7-rwkv7-0p4b-selector-v3.yml`
- Run ID: `32802306230`
- Last verified state: frozen dependency checks PASS; runtime install PASS; `Run selector v3 G7 diagnostic` in progress.
- Experiment source: `experiments/vela-active-state/g7-selector-v3/rwkv7_0p4b_g7_selector_v3.py`
- Planned authoritative result: `ops/vela-results/g7-rwkv7-0p4b-selector-v3-latest.json`

Policy under test:

- preserve chain-v2 target-free candidate detection and carried-W2 provenance;
- do not use W3-native or future trajectory outcomes for selection;
- if several carried W2-origin anchors are immediately functionally equivalent to the chain-v2 baseline result, choose the earliest such W2-origin anchor rather than the latest one;
- this is a conservative, threshold-free depth-guard diagnostic, not a frozen production rule.

Primary readout: on the native-stable G7 denominator, does `migration_only_excess_failure_count` fall from 4 to 0?

Even if it passes, do not mark final selector frozen. The policy is intentionally conservative and must later be optimized for replay cost/generalization.

## B. RWKV-7 1.5B native scale probe — SECONDARY/PARALLEL / RUNNING

- Workflow: `.github/workflows/vela-g7-rwkv7-1p5b-native-scale-probe-v1.yml`
- Run ID: `32802357103`
- Last verified state: frozen dependency checks PASS; runtime install PASS; focused 1.5B probe in progress.
- Experiment source: `experiments/vela-active-state/g7-scale-1p5b-v1/rwkv7_1p5b_g7_native_scale_probe.py`
- Public checkpoint: `BlinkDL/rwkv7-g1 / rwkv7-g1i-1.5b-20260805-ctx16384.pth`
- Planned authoritative result: `ops/vela-results/g7-rwkv7-1p5b-native-scale-probe-v1-latest.json`
- No paid GPU is used.

Scope is deliberately narrow: only `superseded_old_value_echoes`, the fixture that was W3-native unstable on both 0.4B future streams, using the same telemetry/inventory futures and 128/512/2048 horizons.

Decision rule:

- native stable `2/2`: strong evidence that 0.4B capacity/robustness is an important native-stability bottleneck; then expand to full 1.5B G7 qualification;
- `1/2`: partial scale benefit; do not freeze anything;
- `0/2`: no focused scale benefit; do not scale blindly.

## C. Why these two runs exist

G7 anchor-depth v3 already showed that all four native-stable migration failures are rescuable by some carried W2-origin bounded anchor, with no full-replay-only case. Therefore migration-side work should target unsafe late-anchor selection, while the separate `old_value_echoes` native instability is tested as a backbone-size/robustness question.

## D. Safety against false completion

Workflow start is not success. Do not change Gate Ledger, final backbone, final selector, checkpoint policy, canonical schema, or mainline until the result JSON is committed and independently checked.
