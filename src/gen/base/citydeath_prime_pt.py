"""REPLICATE Ruggero's cityDeath win on gemma-3-pt: factual-priming UNION + permissive YES/NO gate.

His champion (gemma-2-it, val 0.623): recite ∪ factual-priming (structured anchors: nationality,
field, life-years, FINAL-YEARS RESIDENCE, burial) + obit variant, gated by an explicit
"deceased AND city recorded? YES/NO" probe with yes-fraction >= 0.30 (permissive).

Base port: (1) PRIMING scaffold as completion (n draws) -> city votes; (2) union with our saved
CoT draws (of_cot_{split}_raw.jsonl) -> pooled plurality; (3) YES/NO gate few-shot (n draws) ->
yes_frac. Answer pooled plurality iff yes_frac >= theta AND pooled votes >= tau; theta+tau tuned
on OFFICIAL TRAIN, oval blind. Compare: ours CoT-concentration 0.59, his 0.623.
"""
import argparse, asyncio, json, re, string, sys
from collections import Counter
sys.path.insert(0, ".")
from evaluate import evaluate_per_sr_pair, RELATION_TYPE, normalize_string
from citydeath_cot_sc import parse_draw as cot_parse

REL = "personHasCityOfDeath"
PRIME_FS = (
    "Person: Albert Einstein\n"
    "Nationality: German-born, later Swiss and American\n"
    "Field: theoretical physics\n"
    "Years: 1879-1955\n"
    "Final-years residence: Princeton, New Jersey\n"
    "Buried: cremated, ashes scattered near Princeton\n"
    "City of death: Princeton\n\n"
    "Person: Keanu Reeves\n"
    "Nationality: Canadian\n"
    "Field: acting\n"
    "Years: born 1964, still alive\n"
    "Final-years residence: Los Angeles\n"
    "Buried: N/A\n"
    "City of death: N/A\n\n"
    "Person: Marie Curie\n"
    "Nationality: Polish-French\n"
    "Field: physics and chemistry\n"
    "Years: 1867-1934\n"
    "Final-years residence: Paris; sanatorium in Passy in her last days\n"
    "Buried: Paris (Panthéon)\n"
    "City of death: Passy\n\n"
    "Person: {name}\n"
    "Nationality:")
PRIME_STOP = ["\n\n", "\nPerson:"]
GATE_FS = (
    "Q: Is Albert Einstein deceased, and is his specific city of death publicly recorded as an established fact? A: Yes\n\n"
    "Q: Is Keanu Reeves deceased, and is his specific city of death publicly recorded as an established fact? A: No\n\n"
    "Q: Is Marie Curie deceased, and is her specific city of death publicly recorded as an established fact? A: Yes\n\n"
    "Q: Is Cristiano Ronaldo deceased, and is his specific city of death publicly recorded as an established fact? A: No\n\n"
    "Q: Is {name} deceased, and is their specific city of death publicly recorded as an established fact? A:")
CITY_RE = re.compile(r"city of death\s*:\s*(.+)", re.I)
NA = {"", "n/a", "na", "none", "unknown", "-", "alive", "still alive"}


def prime_city(text):
    m = CITY_RE.search(text)
    if not m:
        return None
    c = m.group(1).strip().split("\n")[0].split(",")[0].strip(string.punctuation + " ")
    return None if c.lower() in NA else c


def load_gold(path):
    return {json.loads(l)["SubjectEntity"]: json.loads(l)["ObjectEntities"]
            for l in open(path, encoding="utf-8") if json.loads(l)["Relation"] == REL}


def load_raw(path):
    return {json.loads(l)["subject"]: json.loads(l)["draws"] for l in open(path, encoding="utf-8")}


async def gen(subjects, template, stop, base_urls, model, n, temp, max_tokens, conc, tag):
    from openai import AsyncOpenAI
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=180.0, max_retries=3) for u in base_urls]
    sems = [asyncio.Semaphore(conc) for _ in base_urls]
    res = {}
    done = 0

    async def one(i, s):
        nonlocal done
        async with sems[i % len(sems)]:
            r = await clients[i % len(clients)].completions.create(
                model=model, prompt=template.format(name=s), n=n, temperature=temp,
                top_p=0.95, max_tokens=max_tokens, stop=stop)
        res[s] = [c.text for c in r.choices]
        done += 1
        if done % 25 == 0:
            print(f"  {tag} {done}/{len(subjects)}", file=sys.stderr, flush=True)
    await asyncio.gather(*(one(i, s) for i, s in enumerate(subjects)))
    return res


def yes_frac(draws):
    ny = nn = 0
    for t in draws:
        h = t.strip().split("\n")[0].strip().lower()
        if h.startswith("yes"):
            ny += 1
        elif h.startswith("no"):
            nn += 1
    return ny / (ny + nn) if ny + nn else 0.0


def pooled(prime_draws, cot_draws):
    """union city votes from priming + CoT -> Counter(norm), surface map."""
    cnt, surf = Counter(), {}
    for t in prime_draws:
        c = prime_city(t)
        if c:
            nn = normalize_string(c)
            cnt[nn] += 1
            surf.setdefault(nn, c)
    for t in cot_draws:
        c = cot_parse(t)[1]
        if c:
            c0 = c.split(",")[0].strip(string.punctuation + " ")
            nn = normalize_string(c0)
            if nn:
                cnt[nn] += 1
                surf.setdefault(nn, c0)
    return cnt, surf


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/official_train.jsonl")
    ap.add_argument("--val", default="data/official_val_citydeath.jsonl")
    ap.add_argument("--base-urls", default="http://localhost:8001/v1,http://localhost:8002/v1")
    ap.add_argument("--model", default="gemma3-27b-pt")
    ap.add_argument("--n-prime", type=int, default=30)
    ap.add_argument("--n-gate", type=int, default=20)
    ap.add_argument("--conc", type=int, default=48)
    a = ap.parse_args()
    base_urls = [u.strip() for u in a.base_urls.split(",") if u.strip()]

    D = {}
    for split, path in (("train", a.train), ("val", a.val)):
        g = load_gold(path)
        subs = list(g)
        print(f"{split}: {len(subs)} subjects", file=sys.stderr)
        prime = await gen(subs, PRIME_FS, PRIME_STOP, base_urls, a.model, a.n_prime, 0.7, 150, a.conc, f"{split}-prime")
        gate = await gen(subs, GATE_FS, ["\n"], base_urls, a.model, a.n_gate, 0.7, 4, a.conc, f"{split}-gate")
        cot = load_raw(f"of_cot_{split}_raw.jsonl")
        feats = {}
        for s in subs:
            cnt, surf = pooled(prime[s], cot[s])
            feats[s] = {"cnt": cnt, "surf": surf, "yf": yes_frac(gate[s])}
        for tag, d in (("prime", prime), ("gate", gate)):
            with open(f"cd_prime_{split}_{tag}_raw.jsonl", "w", encoding="utf-8") as f:
                for s in subs:
                    f.write(json.dumps({"subject": s, "draws": d[s]}, ensure_ascii=False) + "\n")
        D[split] = (g, feats)

    def macroF1(split, theta, tau):
        g, feats = D[split]
        pr, gt = [], []
        for s in g:
            f = feats[s]
            pred = []
            if f["yf"] >= theta and f["cnt"]:
                top, v = f["cnt"].most_common(1)[0]
                if v >= tau:
                    pred = [f["surf"][top]]
            pr.append({"SubjectEntity": s, "Relation": REL, "ObjectEntities": pred})
            gt.append({"SubjectEntity": s, "Relation": REL, "ObjectEntities": g[s]})
        res = evaluate_per_sr_pair(pr, gt, RELATION_TYPE)
        return sum(r["f1"] for r in res) / len(res)

    thetas = [x / 10 for x in range(1, 10)]
    taus = list(range(1, 41, 2))
    best = max(((th, ta) for th in thetas for ta in taus), key=lambda p: macroF1("train", *p))
    th, ta = best
    print(f"\nPRIMING-UNION + YES/NO gate (pooled prime n={a.n_prime} + CoT n=50; gate n={a.n_gate})")
    print(f"  tune-best theta={th} tau={ta}: train={macroF1('train',th,ta):.3f}  VAL={macroF1('val',th,ta):.3f}")
    # ablations at train-best
    print(f"  [ablation] gate-only theta={th}, tau=1:      VAL={macroF1('val',th,1):.3f}")
    best_ta = max(taus, key=lambda t: macroF1("train", 0.0, t))
    print(f"  [ablation] union-vote only (no gate), tau={best_ta}: VAL={macroF1('val',0.0,best_ta):.3f}")
    print("  [ref] ours CoT-concentration VAL=0.590 ; his instruct champion 0.623")


if __name__ == "__main__":
    asyncio.run(main())
