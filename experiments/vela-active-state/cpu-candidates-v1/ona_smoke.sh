#!/usr/bin/env bash
set -euo pipefail

ROOT="${RUNNER_TEMP:-/tmp}/vela-ona"
rm -rf "$ROOT"
git clone --depth 1 https://github.com/opennars/OpenNARS-for-Applications.git "$ROOT"
cd "$ROOT"
./build.sh

# Official built-in C tests. Successful exit is the primary feasibility signal.
./NAR > ona_selftest.log

test -x ./NAR

# Also verify the shell can consume a real bundled Narsese example.
set +e
timeout 20s ./NAR shell < ./examples/nal/example1.nal > ona_example.log 2>&1
shell_rc=$?
set -e
# EOF behavior can vary; timeout is accepted only if useful output was produced.
if [[ "$shell_rc" -ne 0 && "$shell_rc" -ne 124 ]]; then
  cat ona_example.log
  exit "$shell_rc"
fi
if [[ ! -s ona_example.log ]]; then
  echo "ONA shell produced no output"
  exit 1
fi

printf '{"candidate":"ONA","build":true,"selftest":true,"example_output":true,"shell_rc":%s}\n' "$shell_rc"
