# opened-arm

Public GitHub Actions runner for non-sensitive automation.

## Boundary

- `arm`: private control plane for sensitive access, private project state, and material that should not be published.
- `opened-arm`: public execution plane for workflows that benefit from standard public GitHub-hosted runners.
- Never commit API keys, model weights, source audio, private datasets, or private manifests here.
- Runtime credentials must be stored only as encrypted GitHub Actions secrets.

## Cori continuation status

The restart-safe Cori Matcha-TTS continuation controller advanced the accepted training state to:

- accepted epoch: **280**
- accepted global step: **140560**
- accepted checkpoint SHA-256: `081cf4012a4087f437b8bf2fa0a115da931c5aff26fe22a67acb4f25707cb7a9`

The next E280→E290 segment was stopped during an emergency halt. Recurring self-heal is now intentionally disabled: `.github/workflows/cori-e550-public-self-heal.yml` has no schedule and contains stop-only logic. **No workflow in this repository should be treated as authorization to resume paid GPU training.**

`ops/cori-selfheal-state.json` is the public non-secret operational state record for the last accepted checkpoint. A checkpoint is never promoted merely because it is newer; semantic epoch/global-step progress and SHA-256 are verified before acceptance.

Canonical asset locations, SHA-256 values, the mirrored Cori-adapted BigVGAN, and the E280 descriptive-preview status are recorded in `ops/cori-asset-registry.md`.

## Public-runner policy

Private-repository GitHub Actions minutes are exhausted for this project, so non-sensitive runner work should use `opened-arm` rather than the private development repository. Temporary evaluation/inspection workflows may live on short-lived branches or PRs, but model weights and listening audio must remain transient Actions artifacts or in private external storage and must never be committed.

Required Actions secrets for Lightning inspection/control:

- `LIGHTNING_USER_ID`
- `LIGHTNING_API_KEY`

Other provider credentials are task-specific and must not be assumed to exist in this repository.

## Security

This repository intentionally contains orchestration code and non-secret run metadata only. Cloud credentials can grant paid-compute or private-storage access, so rotate them if exposed and never place them in files, logs, issues, or pull requests.

## VELA experiment branch boundary

On `vela-experiment-infra`, VELA writes are restricted to `experiments/vela-active-state/`, `.github/workflows/vela-*.yml`, and `ops/vela-results/`. Existing Cori/dashboard project files and their workflows are out of scope for VELA experiment writes.
