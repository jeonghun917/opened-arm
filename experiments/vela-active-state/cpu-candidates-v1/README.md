# VELA CPU candidate feasibility v1

Status: EXPERIMENT_ONLY / NO_WINNER / NO_ARCHITECTURE_FREEZE

Purpose: use free GitHub-hosted CPU runners to verify whether several non-GPU candidate families can be installed and run behind the VELA experiment boundary before paid GPU work.

Candidates:
- Z3: specialist constraint/verification organ. Smoke verifies satisfiable planning constraints and contradiction detection.
- ONA (OpenNARS for Applications): reasoning/goal/belief candidate. Smoke builds the real C implementation and runs its built-in tests plus a Narsese example.
- Soar 9.6.5: integrated cognitive-architecture candidate. First smoke verifies the published Python SML runtime can install/import on the runner; state semantics are a later test.
- Fast Downward: planner/search organ. Smoke builds the real planner and solves its official miconic test task with LM-cut A*.

Interpretation boundary:
- Build/run PASS means only that the candidate is practically connectable.
- Z3/Fast Downward are specialist organs, not evidence of a VELA subject by themselves.
- ONA/Soar passing does not prove continuity or single-subject integration.
- State/checkpoint semantics are tested separately after runtime feasibility.

The workflow writes a compact result snapshot to `ops/vela-results/cpu-candidates-latest.md` on the isolated `vela-experiment-infra` branch so results can be read without manually opening Actions logs.
