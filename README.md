# opened-arm

Public GitHub Actions runner for non-sensitive automation.

## Boundary

- `arm`: private control plane for sensitive access, private project state, and material that should not be published.
- `opened-arm`: public execution plane for workflows that benefit from standard public GitHub-hosted runners.
- Never commit API keys, model weights, source audio, private datasets, or private manifests here.
- Runtime credentials must be stored only as encrypted GitHub Actions secrets.

## Cori continuation bridge

`opened-arm` currently contains the restart-safe Cori Matcha-TTS continuation controller targeting E550 in 10-epoch segments.

The controller inherits the last verified state from the prior private controller at E220 / global step 110,440 and knows about the already-submitted E230 job. It never promotes a checkpoint merely because it is newer: terminal job artifacts are scanned and accepted only when semantic epoch/global-step progress is verified.

Required repository Actions secrets:

- `LIGHTNING_USER_ID`
- `LIGHTNING_API_KEY`

If either secret is absent, the workflow exits as a successful no-op and launches no paid Lightning job.

## Security

This repository intentionally contains orchestration code and non-secret run metadata only. Lightning credentials grant paid-compute access, so rotate them if exposed and never place them in files, logs, issues, or pull requests.
