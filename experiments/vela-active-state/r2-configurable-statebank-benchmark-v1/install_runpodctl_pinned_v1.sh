#!/usr/bin/env bash
set -euo pipefail
VERSION="2.14.0"
EXPECTED_SHA256="2e0fd370a52a0fc7e43a6434a209348a4f6836fcdf1ad2b093b609e938138be9"
URL="https://github.com/runpod/runpodctl/releases/download/v${VERSION}/runpodctl-linux-amd64"
OUT="${1:-/tmp/runpodctl}"
curl --retry 3 --retry-all-errors --retry-delay 1 --connect-timeout 10 -fsSL "$URL" -o "$OUT"
printf '%s  %s\n' "$EXPECTED_SHA256" "$OUT" | sha256sum -c -
chmod +x "$OUT"
"$OUT" version | tee /tmp/runpodctl-version.txt
grep -q 'runpodctl 2.14.0' /tmp/runpodctl-version.txt
