"""g4 hasArea encyclopedia Geography-section channel (lit-refill 31x #2).

"Geography of X" subsection prose with explicit land-area sentence. Distinct from
LEAD (31w TIE/KILL), type-blurb (31u KILL), and factbook nested fields (31x #1).

Stage-0 kill-gate: val F1 > k100 champion +0.01 paired P(better)>0.55.

Usage:
  python tools/g4_hasarea_geog_entry.py gen --split val --url http://localhost:8000
  python tools/g4_hasarea_geog_entry.py analyze
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/pd2026"))
from evaluate import RELATION_TYPE, evaluate_per_sr_pair, read_jsonl_file
from hasarea_score import parse

REL = "hasArea"
OUT = REPO / "data/g4_hasarea_geog_entry"
URL = "http://localhost:8000"
CHAMP_K100_VAL = 0.7900

GEOG = """Geography of Tasmania
Tasmania is an island state of Australia south of the mainland. Total land area: 68,401 km².

Geography of Sardinia
Sardinia is a large island in the Mediterranean west of Italy. Total land area: 24,090 km².

Geography of Hokkaido
Hokkaido is the northernmost main island of Japan. Total land area: 83,424 km².

"""


def build_prompt(subject):
    return GEOG + f"Geography of {subject}\n{subject} is"


def gold_rows(split):
    # 31bk #2: leak-free 500-row/relation design split (tools/build_bigdesign.py).
    p = REPO / "data/g4_bigdesign/bigdesign.jsonl" if split == "bigdesign" else REPO / f"data/off_{split}.jsonl"
    return [r for r in read_jsonl_file(str(p)) if r["Relation"] == REL]


def gen_one(url, model, subject, n, temp, seed):
    r = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": build_prompt(subject),
            "n": n,
            "temperature": temp,
            "top_p": 0.95,
            "max_tokens": 96,
            "seed": seed,
            "stop": ["\n\n", "\nGeography of"],
        },
        timeout=600,
    )
    r.raise_for_status()
    return [c["text"] for c in r.json()["choices"]]


def cmd_gen(args):
    # 31bk #2: bigdesign draws land in their own dir so the certified train/val/test dumps
    # under OUT stay byte-identical.
    out_dir = REPO / "data/g4_bigdesign_draws/hasarea_geog" if args.split == "bigdesign" else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = gold_rows(args.split)
    out = out_dir / f"{args.split}_raw.jsonl"
    done = {json.loads(l)["subject"] for l in open(out)} if out.exists() else set()
    todo = [r["SubjectEntity"] for r in rows if r["SubjectEntity"] not in done]
    model = requests.get(f"{args.url}/v1/models").json()["data"][0]["id"]
    print(f"[gen geog-entry] {args.split}: {len(todo)} todo", flush=True)
    if not todo:
        return

    def work(s):
        return {"subject": s, "draws": gen_one(args.url, model, s, args.n, args.temp, args.seed)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(out, "a") as f:
        for i, o in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def load_parsed(split):
    out = {}
    for line in open(OUT / f"{split}_raw.jsonl"):
        d = json.loads(line)
        vals = [parse(t, True) for t in d["draws"]]
        out[d["subject"]] = [v for v in vals if v and v > 0]
    return out


def cluster_log(vals, w=0.05):
    vs = sorted(vals)
    cl = []
    for v in vs:
        for c in cl:
            if abs(v - c[0]) / c[0] <= w:
                c.append(v)
                break
        else:
            cl.append([v])
    if not cl:
        return None
    c = max(cl, key=len)
    return math.exp(sum(math.log(v) for v in c) / len(c))


def macro_f1(parsed, gold_rows_list):
    preds = []
    for g in gold_rows_list:
        s = g["SubjectEntity"]
        if s not in parsed:
            continue
        v = cluster_log(parsed[s])
        preds.append({"SubjectEntity": s, "Relation": REL, "ObjectEntities": [str(v)] if v else []})
    res = evaluate_per_sr_pair(preds, gold_rows_list, RELATION_TYPE)
    f1 = sum(r["f1"] for r in res) / len(res) if res else 0.0
    item = {r["SubjectEntity"]: r["f1"] for r in res}
    return f1, item


def paired_null(a, b, iters=10000):
    subs = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in subs]
    rng = random.Random(0)
    n = len(diffs)
    flips = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    return sum(diffs) / n, sum(1 for x in flips if x > 0) / iters


def cmd_ptest(_args):
    from evaluate import macro_average_per_relation, read_jsonl_file

    parsed = load_parsed("test")
    ha_map = {}
    for s, draws in parsed.items():
        v = cluster_log(draws)
        ha_map[s] = [str(v)] if v else []

    v5 = read_jsonl_file(str(sorted((REPO / "submissions").glob("g4_champion_best_v5_test*.jsonl"))[0]))
    rows = []
    for r in v5:
        s, rel = r["SubjectEntity"], r["Relation"]
        if rel == REL and s in ha_map:
            rows.append({**r, "ObjectEntities": ha_map[s]})
        else:
            rows.append(r)

    ptest_gold = Path(os.environ.get("LMKBC_PTEST_GOLD", REPO / "data/pseudo_test/gold_final.jsonl"))
    if not ptest_gold.exists():
        raise FileNotFoundError(ptest_gold)
    pg = read_jsonl_file(str(ptest_gold))
    sc = macro_average_per_relation(evaluate_per_sr_pair(rows, pg, RELATION_TYPE))
    sc_v5 = macro_average_per_relation(evaluate_per_sr_pair(v5, pg, RELATION_TYPE))
    g = sc["*** All Relations ***"]["macro-f1"]
    g0 = sc_v5["*** All Relations ***"]["macro-f1"]
    ha = sc.get(REL, {}).get("macro-f1", 0.0)
    lines = [
        "IDEA g4 hasArea GEOG-ENTRY ptest confirm (lit-refill 31y #1)",
        f"ptest GLOBAL geog-entry overlay: {g:.4f} (v5 ref {g0:.4f}, Δ{g - g0:+.4f})",
        f"ptest hasArea geog-entry: {ha:.4f}",
    ]
    deploy = "PASS" if g >= g0 + 0.005 else "NULL"
    lines.append(f"DEPLOY ptest: {deploy}")
    (OUT / "verdict_ptest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def cmd_analyze(_args):
    va = load_parsed("val")
    va_f1, va_item = macro_f1(va, gold_rows("val"))
    k100_item = {}
    kpath = REPO / "data/g4_hasarea_manyshot_k100/val_raw.jsonl"
    if kpath.exists():
        kparsed = {}
        for line in open(kpath):
            d = json.loads(line)
            vals = [parse(t, True) for t in d["draws"]]
            kparsed[d["subject"]] = [v for v in vals if v and v > 0]
        _, k100_item = macro_f1(kparsed, gold_rows("val"))
    delta, p_better = paired_null(va_item, k100_item) if k100_item else (0.0, 0.0)
    lines = [
        "IDEA g4 hasArea GEOG-ENTRY encyclopedia section (lit-refill 31x #2)",
        f"k100-champ val F1={CHAMP_K100_VAL:.4f} (ref)",
        f"geog-entry val F1={va_f1:.4f}",
        f"val paired geog-entry vs k100: Δ={delta:+.4f} P(better)={p_better:.4f}",
    ]
    verdict = "PASS" if va_f1 >= CHAMP_K100_VAL + 0.01 and p_better > 0.55 else "KILL"
    lines.append(f"STAGE-0 {verdict}")
    (OUT / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "analyze", "ptest"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "ptest":
        cmd_ptest(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
