#!/usr/bin/env bash
# Run the whole pipeline against a live model and score the result.
#
# Regenerates every elicitation channel, applies the three post-processing stages, and writes a
# submission file. Use this to reproduce the system end to end; use ./verify.sh to check the
# pipeline alone against the submitted artefacts.
#
# Needs a vLLM server for gemma-4-31B. Two ports are used so the relations can run in parallel:
#   vllm serve google/gemma-4-31B --port 8010 --max-model-len 4096
#
# Usage: PORT_A=8010 PORT_B=8011 bash run_pipeline.sh [split] [outdir]
set -euo pipefail
SPLIT="${1:-test}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${2:-$HERE/run_$SPLIT}"
# The base predictions the three stages are applied to. predictions/predictions.jsonl is the test
# submission; any other split needs its own base file, so refuse rather than silently score the wrong
# split's predictions.
if [ -z "${BASE:-}" ]; then
  if [ "$SPLIT" = "test" ]; then BASE="$HERE/predictions/predictions.jsonl"
  else
    echo "error: --split $SPLIT needs BASE=<predictions for that split>" >&2; exit 2
  fi
fi
GEN="$HERE/src/gen"
PY="${PYTHON:-python3}"
A="http://localhost:${PORT_A:-8010}"
B="http://localhost:${PORT_B:-8011}"
mkdir -p "$OUT/channels"

echo "== hasArea: 8 elicitation channels =="
for s in 0 5 6 7; do
  ( cd "$GEN" && VLLM_URL=$A $PY hasarea_channels.py --split "$SPLIT" --chan lead --n 20 --temp 0.8 --seed $s --tag S$s >/dev/null ) &
  ( cd "$GEN" && VLLM_URL=$B $PY hasarea_channels.py --split "$SPLIT" --chan geog --n 20 --temp 0.8 --seed $s --tag S$s >/dev/null ) &
  wait %1 %2 || { echo "generation failed" >&2; exit 1; }
done
i=0
for s in 0 5 6 7; do
  for c in lead geog; do
    src="$GEN/ha2_${SPLIT}_${c}S${s}_n20.jsonl"
    [ -f "$src" ] && mv "$src" "$OUT/channels/hasarea_${c}$( [ $i -eq 0 ] || echo R$s ).jsonl"
  done
  i=$((i+1))
done

echo "== personHasCityOfDeath: 8 channels in two groups =="
( cd "$GEN" && VLLM_URL=$A $PY cd_channels.py --split "$SPLIT" --out "$OUT/channels/cdch_${SPLIT}_n30.json" >/dev/null ) &
p1=$!
( cd "$GEN" && VLLM_URL=$B $PY cdg_gen.py --split "$SPLIT" --out "$OUT/channels/cdg_${SPLIT}_n20.json" >/dev/null ) &
p2=$!
wait $p1 || { echo "cd_channels failed" >&2; exit 1; }
wait $p2 || { echo "cdg_gen failed" >&2; exit 1; }

echo "== awardWonBy: year-slot and cycle-consistency =="
( cd "$GEN" && VLLM_URL=$A $PY awyr_gen.py --split "$SPLIT" >/dev/null && mv "awyr_${SPLIT}_n8.json" "$OUT/channels/" ) &
p3=$!
( cd "$GEN" && VLLM_URL=$B $PY awr_cycle.py --split "$SPLIT" >/dev/null && mv "awr_cycle_${SPLIT}.json" "$OUT/channels/" ) &
p4=$!
wait $p3 || { echo "awyr_gen failed" >&2; exit 1; }
wait $p4 || { echo "awr_cycle failed" >&2; exit 1; }

# the award reranker needs the baseline list it reranks
cp "$HERE/channels/award_base_${SPLIT}.jsonl" "$OUT/channels/" 2>/dev/null || true

echo "== checking channels =="
"$PY" "$HERE/src/check_channels.py" --channels "$OUT/channels" --split "$SPLIT"

echo "== aggregation =="
"$PY" "$HERE/src/reproduce.py" --split "$SPLIT" --base "$BASE" \
      --channels "$OUT/channels" --out "$OUT/predictions.jsonl"
echo
echo "wrote $OUT/predictions.jsonl"
