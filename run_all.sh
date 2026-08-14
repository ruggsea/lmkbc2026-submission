#!/usr/bin/env bash
# Regenerate the submitted predictions. Needs a vLLM server for gemma-4-31B on $PORT.
#   vllm serve google/gemma-4-31B --port 8000 --max-model-len 4096
set -euo pipefail
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
PORT="${PORT:-8000}"
DATA="${DATA:-data}"          # official train.jsonl / val.jsonl / test.jsonl
OUT="${OUT:-results}"
mkdir -p "$OUT"

python src/setrel_pt.py            --port "$PORT" --split test --n 30 --temp 0.7 --out "$OUT/borders.jsonl"
python src/g4_company_manyshot_k64.py    gen --split test --url "http://localhost:$PORT"
python src/g4_hasarea_manyshot_k100.py   gen --split test --url "http://localhost:$PORT"
python src/g4_hasarea_geog_entry.py      gen --split test --url "http://localhost:$PORT"
python src/citydeath_prime_pt.py   --port "$PORT" --split test --n 30 --temp 0.7 --out "$OUT/cd_prime.jsonl"
python src/citydeath_cot_sc.py     --port "$PORT" --split test --n 50 --temp 0.7 --out "$OUT/cd_cot.jsonl"
python src/cap_region_eval.py      --port "$PORT" --split test --n 30 --temp 0.8 --out "$OUT/hascapacity.jsonl"
python src/award_base.py           --port "$PORT" --split test --fmt list --n 20 --out "$OUT/award.jsonl"

python src/g4_stack_v7r.py --split test --out "$OUT/predictions.jsonl"

# Test labels are withheld: submit $OUT/predictions.jsonl to Codabench for a score.
# evaluate.py is included for scoring against any split whose gold you do have.
