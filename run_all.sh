#!/usr/bin/env bash
# Regenerate the submitted predictions. Needs a vLLM server for gemma-4-31B on $PORT.
#   vllm serve google/gemma-4-31B --port 8000 --max-model-len 4096
set -euo pipefail
PORT="${PORT:-8000}"
DATA="${DATA:-data}"          # official train.jsonl / val.jsonl / test.jsonl
OUT="${OUT:-results}"
mkdir -p "$OUT"

python src/setrel_pt.py            --port "$PORT" --split test --n 30 --temp 0.7 --out "$OUT/borders.jsonl"
python src/company_probe_pt.py     --port "$PORT" --split test --n 30 --temp 0.7 --out "$OUT/company.jsonl"
python src/hasarea_srcanchor.py    --port "$PORT" --split test --n 100 --temp 0.8 --out "$OUT/hasarea.jsonl"
python src/citydeath_prime_pt.py   --port "$PORT" --split test --n 30 --temp 0.7 --out "$OUT/cd_prime.jsonl"
python src/citydeath_cot_sc.py     --port "$PORT" --split test --n 50 --temp 0.7 --out "$OUT/cd_cot.jsonl"
python src/cap_region_eval.py      --port "$PORT" --split test --n 30 --temp 0.8 --out "$OUT/hascapacity.jsonl"
python src/award_base.py           --port "$PORT" --split test --fmt list --n 20 --out "$OUT/award.jsonl"

python src/assemble.py --in "$OUT" --out "$OUT/predictions.jsonl"
python evaluate.py -p "$OUT/predictions.jsonl" -g "$DATA/test.jsonl"
