# VELA R2 Configurable State-Bank Benchmark v1

Reusable RunPod execution surface for state-bank experiments.

## Data-driven boundaries
- problem content, family, episode count, bank count/names and per-condition route targets come from the selected problem pack;
- routing condition names and route-field bindings come from `EXECUTION_CONFIG_V1.json`;
- state-mode names map to generic `input_state` / `commit` / `probe` semantics in config;
- exactly one model profile is selected per run from the pack model registry;
- current supported profiles are matched g1i 2.9B and 7.2B, but the runner does not require them to execute together;
- durable-memory retrieval is an explicit per-run variable.

The GitHub workflow defaults to contract-only. Paid RunPod execution is only entered by a manual workflow dispatch with `execute_paid=true`; one dispatch creates at most one pod and runs one selected model profile.
