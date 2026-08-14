"""g4 systematic elicitation sweep — company Kirill SENTINEL+k64 port (dv_many64 + span parse).

Many-shot MSCI prose template (k=64 fixed demos) + SENTINEL abstention parser from base_econ.
Distinct from company_probe scaffold (shipped champion). Tests whether g4's strong ICL on Kirill's
empty-side demos lifts company F1 vs company_probe@k=0.50.

Usage:
  python tools/g4_company_manyshot_k64.py gen --split train --url http://localhost:8000
  python tools/g4_company_manyshot_k64.py analyze
"""
import argparse, json, os, random, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate import RELATION_TYPE, evaluate_per_sr_pair, normalize_string, read_jsonl_file

# Inlined from inference.base_econ (avoid importing Engine/vLLM just for the parser).
SENTINEL = re.compile(
    r"not publicly listed|not listed|not admitted|not traded|does not trade|^none$|^\(none\)$",
    re.I,
)
SPAN_CUE = re.compile(
    r"(?:listed|registered|quoted|trades?|traded|trading)\s+on(?:\s+the)?\s+", re.I
)


def parse_span(text):
    first = text.strip().split("\n")[0].strip()
    if not first or SENTINEL.search(first):
        return []
    m = SPAN_CUE.search(first)
    seg = first[m.end():] if m else (first.split("—")[-1] if "—" in first else first)
    seg = seg.split(".")[0]
    parts = re.split(r"\s*(?:;|,|\band\b)\s*", seg)
    return [p.strip(' .;,"') for p in parts if p.strip(' .;,"') and not SENTINEL.search(p)]

REL = "companyTradesAtStockExchange"
OUT = REPO / "data/g4_company_manyshot_k64"
TPL = REPO / "tasks/companyTradesAtStockExchange/prompts_base/dv_many64.txt"
URL = "http://localhost:8000"
TAUS = [round(0.05 * i, 2) for i in range(2, 13)]  # 0.10 .. 0.60


def load_tpl():
    t = TPL.read_text()
    assert "{s}" in t
    return t


def gold_rows(split):
    paths = {
        "train": REPO / "data/official/train.jsonl",
        "val": REPO / "data/official/val.jsonl",
        "test": REPO / "data/official/test.jsonl",
        # 31bk #2: leak-free 500-row/relation design split (tools/build_bigdesign.py).
        "bigdesign": REPO / "data/g4_bigdesign/bigdesign.jsonl",
    }
    if split not in paths:
        raise ValueError(split)
    return [r for r in read_jsonl_file(str(paths[split])) if r["Relation"] == REL]


def gen_one(url, model, subject, tpl, n, temp, seed):
    prompt = tpl.replace("{s}", subject)
    r = requests.post(f"{url}/v1/completions", json={
        "model": model, "prompt": prompt, "n": n, "temperature": temp,
        "top_p": 0.95, "max_tokens": 96, "seed": seed, "stop": ["\n"],
    }, timeout=600)
    r.raise_for_status()
    return [c["text"] for c in r.json()["choices"]]


def cmd_gen(args):
    # 31bk #2: bigdesign draws land in their own dir so the certified train/val/test dumps
    # under OUT stay byte-identical.
    out_dir = REPO / "data/g4_bigdesign_draws/company" if args.split == "bigdesign" else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = load_tpl()
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
        return {"subject": s, "draws": gen_one(args.url, model, s, tpl, args.n, args.temp, args.seed)}

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(out, "a") as f:
        for i, o in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"[gen] wrote {out}", flush=True)


def parse_draws(raw_draws):
    return [parse_span(t) for t in raw_draws]


def vote_pred(parsed_draws, tau):
    n = len(parsed_draws)
    if not n:
        return []
    cnt, surf = Counter(), {}
    for cands in parsed_draws:
        seen = set()
        for c in cands:
            nn = normalize_string(c)
            if nn and nn not in seen:
                seen.add(nn)
                cnt[nn] += 1
                surf.setdefault(nn, c)
    return [surf[nn] for nn, v in cnt.items() if v / n >= tau - 1e-9]


def macro_f1(dump_by_subj, gold_rows, tau):
    subs = {g["SubjectEntity"] for g in gold_rows}
    preds = []
    for s in subs:
        if s not in dump_by_subj:
            continue
        preds.append({"SubjectEntity": s, "Relation": REL,
                        "ObjectEntities": vote_pred(dump_by_subj[s], tau)})
    res = evaluate_per_sr_pair(preds, [g for g in gold_rows if g["SubjectEntity"] in subs], RELATION_TYPE)
    f1 = sum(r["f1"] for r in res) / len(res) if res else 0.0
    prec = sum(r["p"] for r in res) / len(res) if res else 0.0
    item = {r["SubjectEntity"]: r["f1"] for r in res}
    return f1, prec, item


def paired_null(item_a, item_b, iters=10000, seed=0):
    subs = sorted(set(item_a) & set(item_b))
    diffs = [item_a[s] - item_b[s] for s in subs]
    rng = random.Random(seed)
    n = len(diffs)
    flips = []
    for _ in range(iters):
        flips.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    mean = sum(diffs) / n
    p_better = sum(1 for x in flips if x > 0) / iters
    p_worse = sum(1 for x in flips if x < 0) / iters
    return mean, p_better, p_worse, subs


def load_parsed(split):
    raw_path = OUT / f"{split}_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    out = {}
    for line in open(raw_path):
        d = json.loads(line)
        out[d["subject"]] = parse_draws(d["draws"])
    return out


def champ_parsed(split):
    """company_probe champion draws (k=0.50 reference)."""
    paths = {
        "train": OUT / "train_champdraws.jsonl",
        "val": REPO / "data/g4_company_listing/val_champdraws.jsonl",
    }
    p = paths[split]
    if not p.exists():
        return None
    sys.path.insert(0, str(REPO / "tools/pd2026"))
    from company_probe_pt import parse as probe_parse
    out = {}
    for line in open(p):
        d = json.loads(line)
        out[d["subject"]] = [probe_parse(t) if probe_parse(t) is not None else [] for t in d["draws"]]
    return out


def cmd_analyze(_args):
    train_gold = gold_rows("train")
    val_gold = gold_rows("val")
    ms_train = load_parsed("train")
    ms_val = load_parsed("val")

    grid = {t: macro_f1(ms_train, train_gold, t)[0] for t in TAUS}
    best_t = max(grid, key=grid.get)
    tr_f1, tr_p, tr_item = macro_f1(ms_train, train_gold, best_t)
    va_f1, va_p, va_item = macro_f1(ms_val, val_gold, best_t)

    champ_train = champ_parsed("train")
    champ_val = champ_parsed("val")
    champ_k = 0.50
    null_tr = null_va = None
    if champ_train:
        _, _, champ_tr_item = macro_f1(champ_train, train_gold, champ_k)
        d, pb, pw, _ = paired_null(tr_item, champ_tr_item)
        null_tr = (d, pb, pw)
    if champ_val:
        champ_va_f1, _, champ_va_item = macro_f1(champ_val, val_gold, champ_k)
        d, pb, pw, _ = paired_null(va_item, champ_va_item)
        null_va = (d, pb, pw, champ_va_f1)
    else:
        champ_va_f1 = 0.777  # g4_v3 shipped reference

    lines = [
        "IDEA g4 company Kirill SENTINEL+k64 (dv_many64 span+SENTINEL, gemma-4-31B)",
        "",
        f"train tau*={best_t} F1={tr_f1:.4f} prec={tr_p:.4f}  (grid={grid})",
        f"val   tau={best_t} F1={va_f1:.4f} prec={va_p:.4f}  (champ probe k=0.50 val F1={champ_va_f1:.4f})",
    ]
    if null_tr:
        lines.append(f"train paired vs champ: Δ={null_tr[0]:+.4f} P(better)={null_tr[1]:.4f} P(worse)={null_tr[2]:.4f}")
    if null_va:
        lines.append(f"val paired vs champ:   Δ={null_va[0]:+.4f} P(better)={null_va[1]:.4f} P(worse)={null_va[2]:.4f}")

    verdict = "PASS" if (null_tr and null_tr[0] > 0 and null_tr[1] < 0.05) or (va_f1 > champ_va_f1 + 0.01) else "KILL"
    lines.append(f"\nSTAGE-0 {verdict}: need train paired p<0.05 or val Δ>+0.01 vs company_probe@0.50.")

    OUT.mkdir(parents=True, exist_ok=True)
    vt = OUT / "verdict.txt"
    vt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", vt)


def cmd_assemble_v4(args):
    """Swap company rows in g4_v3 submission with k64 preds; score vs gold."""
    from datetime import date
    tau = args.tau
    split = args.split
    raw = load_parsed(split) if (OUT / f"{split}_raw.jsonl").exists() else None
    if raw is None:
        raise FileNotFoundError(f"missing {split}_raw.jsonl — run gen first")
    k64_map = {s: vote_pred(draws, tau) for s, draws in raw.items()}
    v3_pat = f"g4_champion_best_v3_{split}*.jsonl"
    v3 = sorted((REPO / "submissions").glob(v3_pat))
    if not v3:
        raise FileNotFoundError(v3_pat)
    rows = read_jsonl_file(str(v3[0]))
    rows_v4 = [
        {**r, "ObjectEntities": k64_map.get(r["SubjectEntity"], r["ObjectEntities"])}
        if r["Relation"] == REL else r
        for r in rows
    ]
    out = REPO / f"submissions/g4_champion_best_v4_{split}_{date.today().isoformat()}.jsonl"
    with open(out, "w") as f:
        for r in rows_v4:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gold_path = {
        "val": REPO / "data/official/val.jsonl",
        "test": REPO / "data/official/test.jsonl",
    }[split]
    gold = read_jsonl_file(str(gold_path))
    from evaluate import macro_average_per_relation
    sc = macro_average_per_relation(evaluate_per_sr_pair(rows_v4, gold, RELATION_TYPE))
    global_f1 = sc["*** All Relations ***"]["macro-f1"]
    comp = sc.get(REL, {}).get("macro-f1", 0)
    lines = [f"g4_champion_v4 {split}: GLOBAL={global_f1:.4f} company={comp:.4f} tau={tau}"]
    # ptest surface if available
    ptest_gold = Path(os.environ.get("LMKBC_PTEST_GOLD", REPO / "data/pseudo_test/gold_final.jsonl"))
    if split == "test" and ptest_gold.exists():
        pg = read_jsonl_file(str(ptest_gold))
        sc_pt = macro_average_per_relation(evaluate_per_sr_pair(rows_v4, pg, RELATION_TYPE))
        lines.append(f"  ptest GLOBAL={sc_pt['*** All Relations ***']['macro-f1']:.4f}")
        v3_sc = macro_average_per_relation(evaluate_per_sr_pair(rows, pg, RELATION_TYPE))
        d = sc_pt["*** All Relations ***"]["macro-f1"] - v3_sc["*** All Relations ***"]["macro-f1"]
        lines.append(f"  ptest Δ vs v3={d:+.4f}")
    txt = "\n".join(lines) + "\n"
    (OUT / f"{split}_v4_confirm.txt").write_text(txt)
    print(txt)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--split", choices=["train", "val", "test", "bigdesign"], required=True)
    g.add_argument("--url", default=URL)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--temp", type=float, default=0.7)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("analyze")
    a.set_defaults(func=cmd_analyze)

    v4 = sub.add_parser("assemble-v4")
    v4.add_argument("--split", choices=["val", "test"], required=True)
    v4.add_argument("--tau", type=float, default=0.3)
    v4.set_defaults(func=cmd_assemble_v4)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
