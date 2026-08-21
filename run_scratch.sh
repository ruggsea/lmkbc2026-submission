#!/usr/bin/env bash
# Build a submission from nothing but a served model.
#
# Draws every relation's base pool, aggregates it into base predictions, draws the channels the
# three post-processing stages need, and applies them. No stored draws are read at any point, so
# this is the end-to-end path; ./verify.sh checks the aggregation alone against the submitted
# artefacts, and run_pipeline.sh sits in between (stages only, over the shipped base).
#
#   vllm serve google/gemma-4-31B --port 8010 --max-model-len 8192
#   vllm serve google/gemma-4-31B --port 8011 --max-model-len 8192
#   PORT_A=8010 PORT_B=8011 bash run_scratch.sh test out/
#
set -euo pipefail
SPLIT="${1:-test}"
OUT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scratch_$SPLIT}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac   # channel steps chdir, so OUT must be absolute
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/src/gen"
PY="${PYTHON:-python3}"
A="http://localhost:${PORT_A:-8010}"
B="http://localhost:${PORT_B:-8011}"
RAW="$OUT/raw"
CH="$OUT/channels"
mkdir -p "$RAW" "$CH"

G () { "$PY" "$GEN/base/gen_draws.py" --urls "$A/v1,$B/v1" --split "$SPLIT" "$@"; }

echo "== base pools =="
G --rel countryLandBordersCountry    --out "$RAW/borders_${SPLIT}_raw.jsonl"
G --rel hasCapacity                  --out "$RAW/hascap_${SPLIT}_raw.jsonl"
G --rel awardWonBy                   --out "$RAW/award_${SPLIT}_raw.jsonl"
# the city-of-death gate needs enough draws that its vote threshold is not a coin flip, so the
# three passes are drawn at four times the per-relation default
G --rel personHasCityOfDeath --pass-name gate  --n 80  --out "$RAW/cd_gate_${SPLIT}_raw.jsonl"
G --rel personHasCityOfDeath --pass-name prime --n 120 --out "$RAW/cd_prime_${SPLIT}_raw.jsonl"
G --rel personHasCityOfDeath --pass-name cot   --n 200 --out "$RAW/cd_cot_${SPLIT}_raw.jsonl"
"$PY" "$GEN/company_many64.py" --split "$SPLIT" --url "$A" \
      --out "$RAW/company_many64_${SPLIT}_raw.jsonl"

# the submitted base's hasArea recipe pools: k100 many-shot lead-sentence + geographic-entry
( cd "$HERE" && "$PY" src/g4_hasarea_manyshot_k100.py gen --split "$SPLIT" --url "$A" )
( cd "$HERE" && "$PY" src/g4_hasarea_geog_entry.py    gen --split "$SPLIT" --url "$B" )
mv "$HERE/data/g4_hasarea_manyshot_k100/${SPLIT}_raw.jsonl" "$RAW/hasarea_k100_${SPLIT}_raw.jsonl"
mv "$HERE/data/g4_hasarea_geog_entry/${SPLIT}_raw.jsonl"    "$RAW/hasarea_geog_${SPLIT}_raw.jsonl"

echo "== base predictions =="
"$PY" "$HERE/src/build_base.py" --raw "$RAW" --split "$SPLIT" --out "$OUT/base.jsonl"

echo "== hasArea: eight elicitation channels =="
# the lead prompt picks its demonstrations with random.Random(seed ^ hash(subject)), so str
# hashing has to be pinned or the prompts differ between runs
export PYTHONHASHSEED=777
ha () {  # chan tag seed url
  VLLM_URL="$4" "$PY" "$GEN/hasarea_channels.py" --split "$SPLIT" --chan "$1" \
      --n "${N:-20}" --temp "${TEMP:-0.8}" --seed "$3" --tag "$2" >/dev/null
  local src="$GEN/ha2_${SPLIT}_${1}${2}_n${N:-20}.jsonl"
  [ -f "$src" ] || src="$GEN/ha2_${SPLIT}_${1}_n${N:-20}.jsonl"
  mv "$src" "$CH/hasarea_${1}${2}.jsonl"
}
ha lead ""  0 "$A" & ha geog ""  0 "$B" & wait
ha lead R5  5 "$A" & ha geog R5  5 "$B" & wait
ha lead R6  6 "$A" & ha lead R7  7 "$B" & wait
ha lead R8  8 "$A" & ha geog R8  8 "$B" & wait

echo "== personHasCityOfDeath and awardWonBy channels =="
( cd "$GEN" && VLLM_URL=$A "$PY" cd_channels.py --split "$SPLIT" --out "$CH/cdch_${SPLIT}_n30.json" >/dev/null ) &
( cd "$GEN" && VLLM_URL=$B "$PY" cdg_gen.py     --split "$SPLIT" --out "$CH/cdg_${SPLIT}_n20.json" >/dev/null ) &
wait
( cd "$GEN" && VLLM_URL=$A "$PY" awyr_gen.py  --split "$SPLIT" >/dev/null && mv "awyr_${SPLIT}_n8.json" "$CH/" ) &
( cd "$GEN" && VLLM_URL=$B "$PY" awr_cycle.py --split "$SPLIT" >/dev/null && mv "awr_cycle_${SPLIT}.json" "$CH/" ) &
wait

# the award stage reranks the list this run produced, not a stored one
"$PY" - "$OUT/base.jsonl" "$CH/award_base_${SPLIT}.jsonl" <<'PY'
import json, sys
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for line in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(line)
        if r["Relation"] == "awardWonBy":
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
PY

echo "== checking channels =="
"$PY" "$HERE/src/check_channels.py" --channels "$CH" --split "$SPLIT"

echo "== aggregation =="
"$PY" "$HERE/src/reproduce.py" --split "$SPLIT" --base "$OUT/base.jsonl" \
      --channels "$CH" --out "$OUT/predictions.jsonl"
echo "wrote $OUT/predictions.jsonl"
