#!/usr/bin/env bash
# Generate the eight hasArea elicitation channels, deterministically.
#
# Two sources of nondeterminism have to be pinned or the demo selection differs between runs:
#
#   PYTHONHASHSEED   the lead prompt picks its 100 demonstrations with
#                    random.Random(seed ^ hash(subject)), and Python randomises str hashing per
#                    process unless this is set. Verified: with it set, the prompts are byte-identical
#                    across runs; without it they are not.
#   --seed           the per-channel rotation. The channels shipped with this repo were generated with
#                    "--seed $RANDOM" and those values were never recorded, which is why the shipped
#                    pools cannot be regenerated. The seeds below are fixed so that any future run is
#                    reproducible.
#
# The geog family takes a fixed three-demo prefix and ignores the seed entirely, so those channels are
# deterministic regardless.
#
# Usage: PORT_LEAD=8010 PORT_GEOG=8011 bash src/gen/run_hasarea_channels.sh test
set -euo pipefail
SPLIT="${1:-test}"
N="${N:-20}"
TEMP="${TEMP:-0.8}"
PORT_LEAD="${PORT_LEAD:-8010}"
PORT_GEOG="${PORT_GEOG:-8011}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../../channels}"

export PYTHONHASHSEED=777
mkdir -p "$OUT"

# channel -> rotation seed. lead/geog are the base round; R5..R8 are the extra rotations.
run () {  # run <chan> <tag> <seed> <url>
  local chan="$1" tag="$2" seed="$3" url="$4"
  VLLM_URL="$url" python3 "$HERE/hasarea_channels.py" \
      --split "$SPLIT" --chan "$chan" --n "$N" --temp "$TEMP" --seed "$seed" --tag "$tag"
  local src="$HERE/ha2_${SPLIT}_${chan}${tag}_n${N}.jsonl"
  [ -f "$src" ] || src="$HERE/ha2_${SPLIT}_${chan}_n${N}.jsonl"
  mv "$src" "$OUT/hasarea_${chan}${tag}.jsonl"
}

run lead ""   0 "http://localhost:$PORT_LEAD"
run geog ""   0 "http://localhost:$PORT_GEOG"
run lead R5   5 "http://localhost:$PORT_LEAD"
run geog R5   5 "http://localhost:$PORT_GEOG"
run lead R6   6 "http://localhost:$PORT_LEAD"
run lead R7   7 "http://localhost:$PORT_LEAD"
run lead R8   8 "http://localhost:$PORT_LEAD"
run geog R8   8 "http://localhost:$PORT_GEOG"

echo "wrote 8 channels to $OUT"
