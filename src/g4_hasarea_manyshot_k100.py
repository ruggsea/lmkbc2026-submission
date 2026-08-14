"""g4 systematic elicitation sweep — hasArea many-shot k100 LEAD + log-scale demo spread.

Many-shot Wikipedia-lead convention ICL (k=100 demos from famous-area bank, leave-one-out)
+ magnitude-anchor guard (stratified log-scale demo selection). Distinct from shipped
srcanchor champion (wikitext infobox continuation). Tests whether dense LEAD ICL lifts
hasArea F1 on gemma-4-31B vs srcanchor@cluster_log w=0.05.

Usage:
  python tools/g4_hasarea_manyshot_k100.py gen --split train --url http://localhost:8000
  python tools/g4_hasarea_manyshot_k100.py analyze
"""
import argparse, json, math, random, statistics, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/pd2026"))
from evaluate import RELATION_TYPE, evaluate_per_sr_pair, read_jsonl_file
from hasarea_score import parse

REL = "hasArea"
OUT = REPO / "data/g4_hasarea_manyshot_k100"
FAMOUS = REPO / "data/special/hasarea_famous.jsonl"
URL = "http://localhost:8000"
K = 100
AGG_W = 0.05
CHAMP_VAL = 0.770  # g4_v3 shipped reference

TYPE_BLURB = {
    "country": "is a country",
    "island": "is an island",
    "lake": "is a lake",
    "big city": "is a city",
    "city": "is a city",
    "mountain range": "is a mountain range",
}


def fmt_area(v):
    if v >= 1e6:
        return f"{v:.0e}".replace("e+0", "e+").replace("e+0", "e+")
    if v >= 1000:
        return str(int(round(v)))
    if v >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def load_famous():
    rows = []
    for line in open(FAMOUS, encoding="utf-8"):
        d = json.loads(line)
        gold = d["ObjectEntities"]
        while isinstance(gold, list):
            gold = gold[0] if gold else None
        try:
            area = float(str(gold).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if area <= 0:
            continue
        # _aliases (Wikidata label + altLabel set, lowercased) written by
        # tools/pd2026/build_famous_area.py; falls back to just the surface label for bank
        # rows built before the alias hardening (data/g4_demobank_guard/AUDIT, 2026-08-01).
        aliases = frozenset(a.lower() for a in d.get("_aliases", [d["SubjectEntity"]]))
        rows.append((d["SubjectEntity"], area, d.get("_class", ""), aliases))
    return rows


def pick_demos(famous, exclude, k=K, seed=0):
    # entity-identity LOO: drop any demo whose Wikidata alias set contains the query subject's
    # surface string, not just an exact-string match on the stored label. An exact-string-only
    # check misses same-entity aliasing (e.g. bank "Ireland" vs query "Island of Ireland" --
    # same QID, different surface form; data/g4_demobank_guard/verdict.txt).
    ex = exclude.strip().lower()
    pool = [(s, a, c, al) for s, a, c, al in famous if ex not in al]
    if len(pool) < k:
        return pool
    by_log = sorted(pool, key=lambda x: math.log10(x[1]))
    # stratified: evenly spaced indices across log-scale
    idxs = [int(i * (len(by_log) - 1) / (k - 1)) for i in range(k)]
    chosen = [by_log[i] for i in idxs]
    rng = random.Random(seed ^ hash(exclude) & 0xFFFFFFFF)
    rng.shuffle(chosen)
    return chosen


def lead_line(subject, area, cls):
    blurb = TYPE_BLURB.get(cls, "is a geographic entity")
    return f"{subject} {blurb}. It has an area of {fmt_area(area)} km2.\n\n"


def build_prompt(subject, famous, seed=0):
    demos = pick_demos(famous, subject, K, seed)
    header = "".join(lead_line(s, a, c) for s, a, c, al in demos)
    return header + f"{subject} is"


def gold_rows(split):
    pmap = {
        "train": "data/official/train.jsonl",
        "val": "data/official/val.jsonl",
        "test": "data/official/test.jsonl",
        # 31bk #2: leak-free 500-row/relation design split (tools/build_bigdesign.py).
        "bigdesign": "data/g4_bigdesign/bigdesign.jsonl",
    }
    p = REPO / pmap[split]
    return [r for r in read_jsonl_file(str(p)) if r["Relation"] == REL]


def gold_val(r):
    g = r["ObjectEntities"]
    while isinstance(g, list):
        g = g[0] if g else None
    return float(str(g).replace(",", ""))


def clusters(vals, w=AGG_W):
    vs = sorted(v for v in vals if v and v > 0)
    cl = []
    for v in vs:
        for c in cl:
            if abs(v - c[0]) / c[0] <= w:
                c.append(v)
                break
        else:
            cl.append([v])
    return cl


def cluster_log(vals):
    if not vals:
        return None
    cl = clusters(vals)
    if not cl:
        return None
    c = max(cl, key=len)
    return math.exp(sum(math.log(v) for v in c) / len(c))


def gen_one(url, model, subject, famous, n, temp, seed):
    prompt = build_prompt(subject, famous, seed)
    r = requests.post(f"{url}/v1/completions", json={
        "model": model, "prompt": prompt, "n": n, "temperature": temp,
        "top_p": 0.95, "max_tokens": 48, "seed": seed, "stop": ["\n\n", "\n"],
    }, timeout=600)
    r.raise_for_status()
    return [c["text"] for c in r.json()["choices"]]


def cmd_gen(args):
    # 31bk #2: bigdesign draws land in their own dir so the certified train/val/test dumps
    # under OUT stay byte-identical.
    out_dir = REPO / "data/g4_bigdesign_draws/hasarea_k100" if args.split == "bigdesign" else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    famous = load_famous()
    rows = gold_rows(args.split)
    if args.limit:
        rows = rows[: args.limit]
    out = out_dir / f"{args.split}_raw.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["subject"] for l in open(out)}
    todo = [r["SubjectEntity"] for r in rows if r["SubjectEntity"] not in done]
    model = requests.get(f"{args.url}/v1/models").json()["data"][0]["id"]
    print(f"[gen] {args.split}: {len(rows)} subjects, {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        return

    def work(s):
        return {"subject": s, "draws": gen_one(args.url, model, s, famous, args.n, args.temp, args.seed)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(out, "a") as f:
        for i, o in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"[gen] wrote {out}", flush=True)


def load_parsed(split):
    out = {}
    for line in open(OUT / f"{split}_raw.jsonl"):
        d = json.loads(line)
        vals = [parse(t, True) for t in d["draws"]]
        out[d["subject"]] = [v for v in vals if v and v > 0]
    return out


def macro_f1(parsed_by_subj, gold_rows):
    preds = []
    for g in gold_rows:
        s = g["SubjectEntity"]
        if s not in parsed_by_subj:
            continue
        v = cluster_log(parsed_by_subj[s])
        preds.append({"SubjectEntity": s, "Relation": REL,
                      "ObjectEntities": [str(v)] if v else []})
    res = evaluate_per_sr_pair(preds, gold_rows, RELATION_TYPE)
    f1 = sum(r["f1"] for r in res) / len(res) if res else 0.0
    prec = sum(r["p"] for r in res) / len(res) if res else 0.0
    item = {r["SubjectEntity"]: r["f1"] for r in res}
    return f1, prec, item


def paired_null(item_a, item_b, iters=10000, seed=0):
    subs = sorted(set(item_a) & set(item_b))
    diffs = [item_a[s] - item_b[s] for s in subs]
    rng = random.Random(seed)
    n = len(diffs)
    flips = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    mean = sum(diffs) / n
    p_better = sum(1 for x in flips if x > 0) / iters
    p_worse = sum(1 for x in flips if x < 0) / iters
    return mean, p_better, p_worse


def cmd_analyze(_args):
    train_gold = gold_rows("train")
    val_gold = gold_rows("val")
    tr = load_parsed("train")
    va = load_parsed("val")
    tr_f1, tr_p, tr_item = macro_f1(tr, train_gold)
    va_f1, va_p, va_item = macro_f1(va, val_gold)

    lines = [
        "IDEA g4 hasArea many-shot k100 LEAD (log-stratified demos, cluster_log w=0.05)",
        "",
        f"train F1={tr_f1:.4f} prec={tr_p:.4f}",
        f"val   F1={va_f1:.4f} prec={va_p:.4f}  (champ srcanchor val F1={CHAMP_VAL:.4f})",
        f"val Δ vs champ: {va_f1 - CHAMP_VAL:+.4f}",
    ]
    verdict = "PASS" if va_f1 > CHAMP_VAL + 0.01 else "KILL"
    lines.append(f"\nSTAGE-0 {verdict}: need val Δ>+0.01 vs srcanchor champion ({CHAMP_VAL:.3f}).")

    OUT.mkdir(parents=True, exist_ok=True)
    vt = OUT / "verdict.txt"
    vt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", vt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--split", choices=["train", "val", "test", "bigdesign"], required=True)
    g.add_argument("--url", default=URL)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--temp", type=float, default=0.8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("analyze")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
