#!/usr/bin/env bash
set -euo pipefail

ROOT="${RUNNER_TEMP:-/tmp}/vela-fast-downward"
rm -rf "$ROOT"
git clone --depth 1 https://github.com/aibasel/downward.git "$ROOT"
cd "$ROOT"
./build.py
./fast-downward.py misc/tests/benchmarks/miconic/s1-0.pddl --search 'astar(lmcut())' | tee fd_smoke.log

grep -q 'Solution found' fd_smoke.log
printf '{"candidate":"Fast Downward","build":true,"official_smoke":true}\n'
