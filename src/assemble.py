"""Assemble g4 champion stacks by row-merging per-relation wins into g4_v3 base."""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate import macro_average_per_relation, evaluate_per_sr_pair, read_jsonl_file, RELATION_TYPE

COMPANY_REL = "companyTradesAtStockExchange"
HASAREA_REL = "hasArea"
COMPANY_TAU = 0.3


def load_company_k64(split):
    from tools.g4_company_manyshot_k64 import load_parsed, vote_pred, OUT

    raw = load_parsed(split)
    return {s: vote_pred(draws, COMPANY_TAU) for s, draws in raw.items()}


def load_hasarea_k100(split):
    from tools.g4_hasarea_manyshot_k100 import load_parsed, cluster_log

    raw = load_parsed(split)
    ha_map = {}
    for s, draws in raw.items():
        v = cluster_log(draws)
        ha_map[s] = [str(v)] if v else []
    return ha_map


def _pin_base(pattern):
    """Locate a chain base by glob. Taking sorted(...)[0] silently pins the OLDEST match --
    the exact bug class that laundered the SPM val-gold hardcode through the v6 artifact
    (31bm) and was fixed in g4_stack_micro_wins. Take the newest, fail loud on disagreement."""
    import hashlib
    cands = sorted((REPO / "submissions").glob(pattern))
    if not cands:
        raise FileNotFoundError(pattern)
    digests = {q: hashlib.md5(q.read_bytes()).hexdigest() for q in cands}
    if len(set(digests.values())) > 1:
        raise RuntimeError(
            f"ambiguous base for {pattern}: candidates differ in content, pin one explicitly\n"
            + "\n".join(f"  {q.name} {d}" for q, d in digests.items())
        )
    return cands[-1]


def assemble(split, company=False, hasarea=False):
    rows = read_jsonl_file(str(_pin_base(f"g4_champion_best_v3_{split}*.jsonl")))
    co_map = load_company_k64(split) if company else {}
    ha_map = load_hasarea_k100(split) if hasarea else {}

    out_rows = []
    for r in rows:
        s, rel = r["SubjectEntity"], r["Relation"]
        if company and rel == COMPANY_REL and s in co_map:
            out_rows.append({**r, "ObjectEntities": co_map[s]})
        elif hasarea and rel == HASAREA_REL and s in ha_map:
            out_rows.append({**r, "ObjectEntities": ha_map[s]})
        else:
            out_rows.append(r)

    tag = "v5" if company and hasarea else ("v4" if company else "v5ha")
    out = REPO / f"submissions/g4_champion_best_{tag}_{split}_{date.today().isoformat()}.jsonl"
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out, out_rows


def score(split, rows):
    gold_path = REPO / f"data/official/{split}.jsonl"
    gold = read_jsonl_file(str(gold_path))
    sc = macro_average_per_relation(evaluate_per_sr_pair(rows, gold, RELATION_TYPE))
    lines = [
        f"g4 stack {split}: GLOBAL={sc['*** All Relations ***']['macro-f1']:.4f}",
        f"  company={sc.get(COMPANY_REL, {}).get('macro-f1', 0):.4f}",
        f"  hasArea={sc.get(HASAREA_REL, {}).get('macro-f1', 0):.4f}",
    ]
    ptest_gold = Path(os.environ.get("LMKBC_PTEST_GOLD", REPO / "data/pseudo_test/gold_final.jsonl"))
    if split == "test" and ptest_gold.exists():
        pg = read_jsonl_file(str(ptest_gold))
        sc_pt = macro_average_per_relation(evaluate_per_sr_pair(rows, pg, RELATION_TYPE))
        v3 = read_jsonl_file(str(_pin_base("g4_champion_best_v3_test*.jsonl")))
        v3_sc = macro_average_per_relation(evaluate_per_sr_pair(v3, pg, RELATION_TYPE))
        d = sc_pt["*** All Relations ***"]["macro-f1"] - v3_sc["*** All Relations ***"]["macro-f1"]
        lines.append(f"  ptest GLOBAL={sc_pt['*** All Relations ***']['macro-f1']:.4f}")
        lines.append(f"  ptest Δ vs v3={d:+.4f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--company", action="store_true")
    ap.add_argument("--hasarea", action="store_true")
    ap.add_argument("--both", action="store_true", help="company k64 + hasArea k100")
    args = ap.parse_args()
    company = args.company or args.both
    hasarea = args.hasarea or args.both
    out, rows = assemble(args.split, company=company, hasarea=hasarea)
    txt = score(args.split, rows)
    confirm = REPO / "data/g4_stack_confirm" / f"{args.split}_confirm.txt"
    confirm.parent.mkdir(parents=True, exist_ok=True)
    confirm.write_text(txt)
    print(txt)
    print("wrote", out)
    print("wrote", confirm)


if __name__ == "__main__":
    main()
