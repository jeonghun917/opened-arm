#!/usr/bin/env bash
set -euo pipefail
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME once before first launch}"

HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$HERE/_launch"
rm -rf "$TMP"
mkdir -p "$TMP/shared"
cp "$HERE/kernel.py" "$TMP/kernel.py"
cp "$HERE/../shared/"*.py "$TMP/shared/"

python - "$HERE/kernel-metadata.template.json" "$TMP/kernel-metadata.json" "$KAGGLE_USERNAME" <<'PY'
import json, sys
src, dst, user = sys.argv[1:]
d = json.load(open(src))
d["id"] = d["id"].replace("__KAGGLE_USERNAME__", user)
json.dump(d, open(dst, "w"), indent=2)
PY

kaggle kernels push -p "$TMP"
echo "Submitted private Kaggle GPU kernel."
echo "status: kaggle kernels status ${KAGGLE_USERNAME}/vela-rwkv-state-smoke"
echo "output: kaggle kernels output ${KAGGLE_USERNAME}/vela-rwkv-state-smoke -p $HERE/output"
