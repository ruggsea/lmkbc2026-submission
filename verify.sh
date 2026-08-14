#!/usr/bin/env bash
# Reproduce the reported validation score from the shipped predictions. No GPU needed.
#   bash verify.sh path/to/val.jsonl
# Official test labels are withheld, so the test score can only be reproduced by
# submitting predictions/predictions.jsonl to Codabench.
set -euo pipefail
GOLD="${1:-}"
if [ -z "$GOLD" ]; then
  echo "usage: bash verify.sh path/to/val.jsonl" >&2
  echo "get it from https://github.com/lm-kbc/dataset2026 (data/val.jsonl)" >&2
  exit 1
fi
python3 evaluate.py -p predictions/predictions_val.jsonl -g "$GOLD"
