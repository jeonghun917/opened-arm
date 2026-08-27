# Pharos exact validation plane v0

## Purpose

This execution plane validates one exact commit from the fixed private repository `jeonghun917/pharos-orbis` while keeping public GitHub Actions compute in `jeonghun917/opened-arm`.

## Fixed route

- Capability: `pharos.exactValidation`
- Workflow: `.github/workflows/pharos-exact-validation.yml`
- Target repository: `jeonghun917/pharos-orbis`
- Input: one exact 40-character lowercase hexadecimal commit SHA
- Cost class: FREE
- Selection: EXACT
- Fallback: none
- Automatic retry: disabled

The workflow must never accept a repository name as an input.

## Validation recipe contract

The candidate commit itself must contain `ops/validation/recipe-v0.sh`.

The public validator:

1. resolves the exact requested SHA on a trusted preflight runner;
2. requires a Pharos-specific private checkout credential and publishes pending status before candidate code runs;
3. checks out only `jeonghun917/pharos-orbis` at that SHA with persisted credentials disabled on a read-only validation job;
4. proves `git rev-parse HEAD` equals the requested SHA and the checkout is clean;
5. requires a non-empty `ops/validation/recipe-v0.sh`;
6. runs `bash ops/validation/recipe-v0.sh` without write-capable publication credentials;
7. requires exit code 0 and a clean working tree afterward;
8. uses a separate fresh finalize runner to publish final status and write bounded evidence to `ops/pharos-exact-validation.json`.

A recipe change is a material Pharos validation-policy change and requires Pharos Primary review. The Platform execution plane executes the recipe but does not define product PASS semantics itself.

## Credential boundary

The required secret is `PHAROS_REPO_TOKEN`. It must be explicitly bound for private Pharos checkout/status publication. Dashboard credentials must not be reused by assumption.

Both public and private checkout use `persist-credentials: false`. The recipe executes in a dedicated read-only validation job and receives no publication credential environment variable. Final Pharos status publication and opened-arm evidence persistence happen only on a separate fresh runner after the validation job ends, so candidate recipe code cannot prepare runner-local hooks, environment files, PATH changes, or sibling-repository state that later executes with write-capable publication credentials.

Until the credential exists and a concrete recipe is present in the target Pharos commit, the route is `CONFIG_REQUIRED` and must fail closed.

## Evidence

`ops/pharos-exact-validation.json` records the target repository, target SHA, actual SHA, recipe version/path, step outcomes, run ID and run attempt. Evidence persistence does not convert a failed run into PASS.

## Authority boundary

This plane grants no general `opened-arm` mutation authority and no mutation authority over `jeonghun917/pharos-orbis`. It is a closed delegated validator only. It cannot start, merge or complete a Pharos product workstream.
