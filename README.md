# LM-KBC 2026 submission (team CS2)

System description paper: to appear at AKBC @ EMNLP 2026.

Code and predictions for our entry to the [LM-KBC 2026 shared task](https://lm-kbc.github.io/challenge2026/).

Closed-book, one open-weight model, no training. `gemma-4-31B` base with a separate elicitation
and aggregation recipe per relation.

## Results

Official test (Codabench phase 29684), macro F1:

| relation | F1 |
|---|---|
| countryLandBordersCountry | 0.9753 |
| companyTradesAtStockExchange | 0.8470 |
| hasArea | 0.8500 |
| personHasCityOfDeath | 0.6000 |
| awardWonBy | 0.3609 |
| hasCapacity | 0.3265 |
| **All relations** | **0.6961** |

The submitted file is `predictions/predictions.jsonl` (475 rows, md5 `a6f64abb2a90b5c26e391fbf9ca677c0`).


## Method

One model, six recipes. Every relation samples `n` completions from the base model and aggregates
them with non-neural rules. Thresholds are selected on official train.

| relation | elicitation | draws | aggregation |
|---|---|---|---|
| countryLandBordersCountry | few-shot completion (3 hand-picked demonstrations), then collectivity injection | n=30, t=0.7 | keep neighbour if it appears in ≥0.35 of draws |
| companyTradesAtStockExchange | many-shot listing prompt with explicit abstention, k=64 demonstrations drawn from official train | n=100, t=0.7 | agreement ≥0.35 over scaffold-complete draws |
| hasArea | many-shot lead-sentence prompt, k=100 demonstrations log-stratified from a 331-entity bank, query removed | n=100 lead / n=50 geographic-entry, t=0.8 | cluster in log10 space, w=0.05, median of largest cluster; a second geographic-entry pool overrides when the two medians differ by more than w |
| personHasCityOfDeath | prime completion ∪ CoT self-consistency, yes/no gate | n=30 / 50 / 20, t=0.7 | modal city if vote share ≥ threshold |
| hasCapacity | REGION enumerate-then-pick | n=30, t=0.8 | numeric cluster vote, r=0.025 |
| awardWonBy | list completion | n=20, t=0.8 | keep name if it appears in ≥0.46 of draws |

The submitted file is assembled by `src/g4_stack_v7r.py`, which overlays the company, hasArea and
borders channels onto the preceding configuration.

The hasArea demonstration bank is built offline by `src/build_famous_area.py`. It excludes every
subject occurring in the official train, validation and test splits, and the query entity is removed
again at prompt-construction time; both checks match on Wikidata entity identity rather than surface
string. The bank is fixed before inference and identical for every query.

## Reproducing

Serve the model, then run the pipeline:

```bash
pip install -r requirements.txt
vllm serve google/gemma-4-31B --port 8000 --max-model-len 4096
PORT=8000 DATA=path/to/official/data bash run_all.sh
```

`run_all.sh` writes per-relation outputs and assembles them into `predictions.jsonl`. The task data comes from the [organisers' repo](https://github.com/lm-kbc/dataset2026):
`data/test.jsonl` is the submission set, `data/train.jsonl` supplies the demonstrations and
thresholds. `test.jsonl` changed on 2026-08-07 (15 subjects renamed, 2 rows dropped), so
re-download it rather than reusing an older copy. Test labels are withheld; scores come from
Codabench.

Sampling is stochastic, so a fresh run will not be byte-identical to the submitted file.

## Rules

- Closed-book at inference: no retrieval, no internet, no external corpora, no KB lookups.
- No fine-tuning, LoRA, probes or soft prompts.
- One model, 31B parameters, inside the 32B budget.
- `pseudoval/` builds an offline Wikidata-derived development set used for early threshold screening.
  It is not part of the inference path.

## Layout

```
src/            per-relation elicitation and aggregation, plus the assembler
pseudoval/      SPARQL construction of the auxiliary dev set
predictions/    the submitted file
evaluate.py     official scorer (from the organisers' repo)
run_all.sh      end-to-end regeneration
```
