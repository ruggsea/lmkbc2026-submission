#!/usr/bin/env bash
# Regenerate the submitted predictions and check them against the file that was scored on the
# leaderboard. Needs Python with pandas (evaluate.py imports it) -- no GPU, no model, no network.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
EXPECT="aaa531a0242640823b56bfc488c863b8"
OUT="$(mktemp "${TMPDIR:-/tmp}/lmkbc_repro.XXXXXX").jsonl"

"$PY" "$HERE/src/reproduce.py" --base "$HERE/predictions/predictions.jsonl" --out "$OUT"

if command -v md5sum >/dev/null 2>&1; then GOT="$(md5sum "$OUT" | cut -d' ' -f1)"
else GOT="$(md5 -q "$OUT")"; fi

echo
echo "expected md5: $EXPECT"
echo "actual   md5: $GOT"
if [ "$GOT" = "$EXPECT" ]; then
  echo "MATCH - this repository regenerates the submitted predictions exactly."
else
  echo "MISMATCH - regenerated file differs from the submitted one."; exit 1
fi
