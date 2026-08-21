"""Assemble base predictions from freshly drawn pools.

One aggregation rule per relation, each re-derived from the pool the submission was built from:

  countryLandBordersCountry   agreement k=0.35, then country-name canonicalisation   67/67
  companyTradesAtStockExchange span vote over many-shot draws, tau=0.30              99/99
  hasCapacity                 REGION enumerate-then-pick, numeric cluster vote       98/98
  awardWonBy                  list consensus, keep a name held by >=0.46 of draws      9/9
  personHasCityOfDeath        prime union CoT votes, gate yes-fraction >=0.70,
                              modal city needs >=58 of the pooled draws              93/93
  hasArea                     median of largest cluster (w=0.05) over the k100 many-shot
                              pool, geographic-entry override on >5% disagreement -- the
                              recipe of the submitted base; falls back to the source-anchor
                              pool when the two pools are absent (stage 1 replaces either
                              with the eight-channel estimate)

The counts are how many of the submitted rows each rule reproduces when run against the original
pool, which is what pins the thresholds. Output is a base file for src/reproduce.py.
"""
import argparse, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, f"{HERE}/gen/base")
sys.path.insert(0, f"{HERE}/gen")
from evaluate import normalize_string

BORDERS_K = 0.35
COMPANY_TAU = 0.30
AWARD_K = 0.46
CD_GATE, CD_VOTE = 0.70, 58

# country surfaces the model emits that name an entity the gold lists under another label
PRED_CANON = {
    "west bank": "Palestine", "palestinian territories": "Palestine",
    "palestinian territory": "Palestine", "palestinian authority": "Palestine",
    "gaza strip": "Palestine", "macedonia": "North Macedonia", "fyrom": "North Macedonia",
    "ivory coast": "Côte d'Ivoire", "cote divoire": "Côte d'Ivoire", "the gambia": "Gambia",
    "burma": "Myanmar", "czechia": "Czech Republic", "swaziland": "Eswatini",
    "east timor": "Timor-Leste", "timor leste": "Timor-Leste", "cape verde": "Cabo Verde",
    "holy see": "Vatican City", "vatican": "Vatican City", "the netherlands": "Netherlands",
    "republic of ireland": "Ireland", "uk": "United Kingdom", "great britain": "United Kingdom",
    "britain": "United Kingdom", "pr china": "China", "peoples republic of china": "China",
    "people's republic of china": "China",
}


def load_raw(path):
    return {json.loads(l)["subject"]: json.loads(l)["draws"] for l in open(path, encoding="utf-8")}


def dedup_canon(objs):
    seen, out = set(), []
    for o in objs:
        c = PRED_CANON.get(normalize_string(o), o)
        cn = normalize_string(c)
        if cn and cn not in seen:
            seen.add(cn)
            out.append(c)
    return out


def agg_borders(draws):
    import setrel_pt as B
    return dedup_canon(B.aggregate(draws, B.TEMPLATES["countryLandBordersCountry"]["none_words"],
                                   BORDERS_K))


def agg_company(draws):
    from company_many64 import parse_span
    n = len(draws)
    if not n:
        return []
    cnt, surf = Counter(), {}
    for t in draws:
        seen = set()
        for c in parse_span(t):
            nn = normalize_string(c)
            if nn and nn not in seen:
                seen.add(nn)
                cnt[nn] += 1
                surf.setdefault(nn, c)
    return [surf[nn] for nn, v in cnt.items() if v / n >= COMPANY_TAU - 1e-9]


def agg_hascap(draws):
    from cap_region_eval import last_cap, vote
    v = vote([last_cap(c) for c in draws])
    return [str(int(round(v)))] if v else []


def agg_award(draws):
    from award_base_score import parse_draw
    n = max(1, len(draws))
    cnt, surf = Counter(), {}
    for t in draws:
        seen = set()
        for name in parse_draw(t, "list"):
            nn = normalize_string(name)
            if nn and nn not in seen:
                seen.add(nn)
                cnt[nn] += 1
                surf.setdefault(nn, name)
    return [surf[nn] for nn, v in cnt.items() if v / n >= AWARD_K]


def agg_citydeath(prime, cot, gate):
    import citydeath_prime_pt as P
    cnt, surf = P.pooled(prime, cot)
    if P.yes_frac(gate) >= CD_GATE and cnt:
        top, v = cnt.most_common(1)[0]
        if v >= CD_VOTE:
            return [surf[top]]
    return []


def agg_hasarea(draws):
    from hasarea_score import parse
    from hasarea_agg_sweep import agg
    vals = [v for v in (parse(t, True) for t in draws) if v and v > 0]
    v = agg(vals, "cluster_log", 0.05)
    return ["%.6g" % v] if v else []


def median_cluster(vals, w=0.05):
    """Median of the largest relative-tolerance cluster (verbatim from the shipped v7r assembler)."""
    vs = sorted(v for v in vals if v and v > 0)
    if not vs:
        return None
    cl = []
    for v in vs:
        for c in cl:
            if abs(v - c[0]) / c[0] <= w:
                c.append(v)
                break
        else:
            cl.append([v])
    c = max(cl, key=len)
    s = sorted(c)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def hasarea_base_map(raw_dir, split):
    """The recipe the submitted base actually used: median-of-largest-cluster over the k100
    many-shot pool, overridden by the geographic-entry pool when the two medians disagree by
    more than 5% (g4_stack_v7r.hasarea_median_disagree_map). Values formatted with str(),
    matching the frozen file."""
    from hasarea_score import parse

    def pool_map(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            v = median_cluster([x for x in (parse(t, True) for t in d["draws"]) if x and x > 0])
            if v:
                out[d["subject"]] = v
        return out

    k100 = pool_map(f"{raw_dir}/hasarea_k100_{split}_raw.jsonl")
    geog = pool_map(f"{raw_dir}/hasarea_geog_{split}_raw.jsonl")
    out = {}
    for s in set(k100) | set(geog):
        ka, ga = k100.get(s), geog.get(s)
        if ka and ga and abs(ga - ka) / ka > 0.05:
            out[s] = [str(ga)]
        elif ka:
            out[s] = [str(ka)]
        elif ga:
            out[s] = [str(ga)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding the drawn pools")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = a.raw

    P = {
        "countryLandBordersCountry": {s: agg_borders(d)
                                      for s, d in load_raw(f"{D}/borders_{a.split}_raw.jsonl").items()},
        "companyTradesAtStockExchange": {s: agg_company(d)
                                         for s, d in load_raw(f"{D}/company_many64_{a.split}_raw.jsonl").items()},
        "hasCapacity": {s: agg_hascap(d) for s, d in load_raw(f"{D}/hascap_{a.split}_raw.jsonl").items()},
        "awardWonBy": {s: agg_award(d) for s, d in load_raw(f"{D}/award_{a.split}_raw.jsonl").items()},
        "hasArea": (hasarea_base_map(D, a.split)
                    if os.path.exists(f"{D}/hasarea_k100_{a.split}_raw.jsonl")
                    else {s: agg_hasarea(d)
                          for s, d in load_raw(f"{D}/hasarea_{a.split}_raw.jsonl").items()}),
    }
    pr = load_raw(f"{D}/cd_prime_{a.split}_raw.jsonl")
    co = load_raw(f"{D}/cd_cot_{a.split}_raw.jsonl")
    ga = load_raw(f"{D}/cd_gate_{a.split}_raw.jsonl")
    P["personHasCityOfDeath"] = {s: agg_citydeath(pr[s], co.get(s, []), ga.get(s, [])) for s in pr}

    missing = Counter()
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for l in open(f"{HERE}/gen/off_{a.split}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            s, rel = d["SubjectEntity"], d["Relation"]
            if s not in P.get(rel, {}):
                missing[rel] += 1
            f.write(json.dumps({"SubjectEntity": s, "Relation": rel,
                                "ObjectEntities": P.get(rel, {}).get(s, [])}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {a.out}: {n} rows")
    if missing:
        print("subjects with no draws:", dict(missing))


if __name__ == "__main__":
    main()
