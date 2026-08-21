"""Port borders + company to gemma-3-27b-pt (base): few-shot completion -> n draws ->
per-candidate agreement vote (candidate kept if it appears in >= k fraction of draws).
k swept on OFFICIAL TRAIN, official val is the go/no-go. One runner, per-relation templates.

Usage (pippi ~/lmkbc_work):
  python setrel_pt.py --relation countryLandBordersCountry \
      --train data/official_train.jsonl --val data/official_val_borders.jsonl --n 30
"""
import argparse, asyncio, json, re, sys
from collections import Counter
sys.path.insert(0, ".")
from evaluate import evaluate_per_sr_pair, RELATION_TYPE, normalize_string

TEMPLATES = {
    "countryLandBordersCountry": {
        # 3 shots incl. an island 'none' -> teaches the empty rule
        "fewshot": (
            "Country: France\n"
            "Land border countries: Belgium; Luxembourg; Germany; Switzerland; Italy; Monaco; Spain; Andorra\n\n"
            "Country: Japan\n"
            "Land border countries: none\n\n"
            "Country: Bolivia\n"
            "Land border countries: Brazil; Paraguay; Argentina; Chile; Peru\n\n"
            "Country: {name}\n"
            "Land border countries:"),
        "none_words": {"none", "n/a", "no land borders", "island nation", "-"},
    },
    "companyTradesAtStockExchange": {
        # incl. a private/not-listed 'none' shot
        "fewshot": (
            "Company: Apple Inc.\n"
            "Stock exchanges where its shares are listed: NASDAQ\n\n"
            "Company: IKEA\n"
            "Stock exchanges where its shares are listed: none\n\n"
            "Company: Toyota\n"
            "Stock exchanges where its shares are listed: Tokyo Stock Exchange; New York Stock Exchange; London Stock Exchange\n\n"
            "Company: Siemens\n"
            "Stock exchanges where its shares are listed: Frankfurt Stock Exchange\n\n"
            "Company: {name}\n"
            "Stock exchanges where its shares are listed:"),
        "none_words": {"none", "n/a", "not listed", "private", "privately held", "unlisted", "-"},
    },
}
STOP = ["\n\n", "\nCountry:", "\nCompany:"]


def parse(text, none_words):
    """one draw -> list of candidate strings (possibly empty)."""
    line = text.strip().split("\n")[0].strip()
    if not line or line.lower().strip(".") in none_words:
        return []
    out = []
    for part in re.split(r"[;]", line):
        c = part.strip().strip(".").strip()
        if c.lower() in none_words:
            continue
        if 1 < len(c) <= 60:
            out.append(c)
    return out


def aggregate(draws, none_words, k):
    """candidates appearing in >= k fraction of draws (dedup by normalize)."""
    n = len(draws)
    cnt, surf = Counter(), {}
    for t in draws:
        seen = set()
        for c in parse(t, none_words):
            nn = normalize_string(c)
            if nn and nn not in seen:
                seen.add(nn)
                cnt[nn] += 1
                surf.setdefault(nn, c)
    return [surf[nn] for nn, v in cnt.items() if v / n >= k]


def load(path, relation):
    return {json.loads(l)["SubjectEntity"]: json.loads(l)["ObjectEntities"]
            for l in open(path, encoding="utf-8") if json.loads(l)["Relation"] == relation}


def macroF1(gold, draws, none_words, k, relation):
    subs = list(gold)
    pr = [{"SubjectEntity": s, "Relation": relation,
           "ObjectEntities": aggregate(draws[s], none_words, k)} for s in subs]
    gt = [{"SubjectEntity": s, "Relation": relation, "ObjectEntities": gold[s]} for s in subs]
    res = evaluate_per_sr_pair(pr, gt, RELATION_TYPE)
    n = len(res)
    return (sum(r["f1"] for r in res) / n, sum(r["p"] for r in res) / n, sum(r["r"] for r in res) / n)


async def gen(subjects, fewshot, base_urls, model, n, temp, max_tokens, conc):
    from openai import AsyncOpenAI
    # timeout+retries: a single lost response must not hang the whole gather forever
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=180.0, max_retries=3)
               for u in base_urls]
    sems = [asyncio.Semaphore(conc) for _ in base_urls]
    res = {}
    done = 0

    async def one(i, s):
        nonlocal done
        async with sems[i % len(sems)]:
            r = await clients[i % len(clients)].completions.create(
                model=model, prompt=fewshot.format(name=s), n=n, temperature=temp,
                top_p=0.95, max_tokens=max_tokens, stop=STOP)
        res[s] = [c.text for c in r.choices]
        done += 1
        if done % 25 == 0 or done == len(subjects):
            print(f"  {done}/{len(subjects)}", file=sys.stderr, flush=True)

    await asyncio.gather(*(one(i, s) for i, s in enumerate(subjects)))
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", required=True, choices=list(TEMPLATES))
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--raw-prefix", default=None, help="save raw draws to <prefix>_{train,val}.jsonl")
    ap.add_argument("--base-urls", default="http://localhost:8001/v1,http://localhost:8002/v1")
    ap.add_argument("--model", default="gemma3-27b-pt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--conc", type=int, default=48)
    a = ap.parse_args()
    T = TEMPLATES[a.relation]
    base_urls = [u.strip() for u in a.base_urls.split(",") if u.strip()]
    tg, vg = load(a.train, a.relation), load(a.val, a.relation)
    print(f"{a.relation}: {len(tg)} train / {len(vg)} val; n={a.n}", file=sys.stderr)

    td = await gen(list(tg), T["fewshot"], base_urls, a.model, a.n, a.temp, a.max_tokens, a.conc)
    vd = await gen(list(vg), T["fewshot"], base_urls, a.model, a.n, a.temp, a.max_tokens, a.conc)
    if a.raw_prefix:
        for tag, d in (("train", td), ("val", vd)):
            with open(f"{a.raw_prefix}_{tag}.jsonl", "w", encoding="utf-8") as f:
                for s, dr in d.items():
                    f.write(json.dumps({"subject": s, "draws": dr}, ensure_ascii=False) + "\n")

    ks = [x / 100 for x in range(5, 100, 5)]
    print("\n  k   | train F1 (P/R)        | val F1 (P/R)")
    best = max(ks, key=lambda k: macroF1(tg, td, T["none_words"], k, a.relation)[0])
    for k in ks:
        tf, tp, tr = macroF1(tg, td, T["none_words"], k, a.relation)
        vf, vp, vr = macroF1(vg, vd, T["none_words"], k, a.relation)
        mark = "  <- tune-best" if k == best else ""
        print(f"  {k:.2f} | {tf:.3f} ({tp:.2f}/{tr:.2f}) | {vf:.3f} ({vp:.2f}/{vr:.2f}){mark}")
    tf = macroF1(tg, td, T["none_words"], best, a.relation)[0]
    vf = macroF1(vg, vd, T["none_words"], best, a.relation)[0]
    print(f"\n{a.relation} gemma-3-27b-pt: tune-best k={best:.2f}  train-F1={tf:.3f}  VAL-F1={vf:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
