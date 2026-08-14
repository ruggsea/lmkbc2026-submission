"""REPLICATE Ruggero's company win on gemma-3-pt: the ITSELF-vs-PARENT listing-status probe.

His champion (gemma-2-it persona τ=0.30, val 0.764) interrogates the subsidiary trap explicitly
before answering. Base-completion port: a scaffold that forces the model to state HQ, parent, and
listing status ('directly listed' / 'subsidiary, parent listed' / 'private') BEFORE the exchange
list. Compare against our incumbent flat template (setrel_pt.py, val 0.711). k tuned on OFFICIAL
TRAIN, oval blind.
"""
import argparse, asyncio, json, re, sys
from collections import Counter
sys.path.insert(0, ".")
from evaluate import evaluate_per_sr_pair, RELATION_TYPE, normalize_string

REL = "companyTradesAtStockExchange"
FEWSHOT = (
    "Company: Apple Inc.\n"
    "Headquarters: Cupertino, United States\n"
    "Parent company: none (independent)\n"
    "Listing status: directly listed\n"
    "Stock exchanges where its own shares are listed: NASDAQ\n\n"
    "Company: IKEA\n"
    "Headquarters: Delft, Netherlands\n"
    "Parent company: Inter IKEA Holding (private group)\n"
    "Listing status: private, not listed\n"
    "Stock exchanges where its own shares are listed: none\n\n"
    "Company: Audi\n"
    "Headquarters: Ingolstadt, Germany\n"
    "Parent company: Volkswagen Group\n"
    "Listing status: subsidiary, parent listed (its own shares are not separately traded)\n"
    "Stock exchanges where its own shares are listed: none\n\n"
    "Company: Toyota\n"
    "Headquarters: Toyota City, Japan\n"
    "Parent company: none (independent)\n"
    "Listing status: directly listed\n"
    "Stock exchanges where its own shares are listed: Tokyo Stock Exchange; New York Stock Exchange; London Stock Exchange\n\n"
    "Company: {name}\n"
    "Headquarters:")
STOP = ["\n\n", "\nCompany:"]
NONE_WORDS = {"none", "n/a", "not listed", "private", "privately held", "unlisted", "-"}
LIST_RE = re.compile(r"stock exchanges where its own shares are listed\s*:\s*(.+)", re.I)


def parse(text):
    m = LIST_RE.search("Headquarters:" + text)
    if not m:
        return None  # scaffold incomplete -> no vote
    line = m.group(1).strip().split("\n")[0].strip()
    if not line or line.lower().strip(".") in NONE_WORDS:
        return []
    out = []
    for part in line.split(";"):
        c = part.strip().strip(".").strip()
        if c.lower() in NONE_WORDS:
            continue
        if 1 < len(c) <= 60:
            out.append(c)
    return out


def aggregate(draws, k):
    parsed = [p for p in (parse(t) for t in draws) if p is not None]
    if not parsed:
        return []
    n = len(parsed)
    cnt, surf = Counter(), {}
    for cands in parsed:
        seen = set()
        for c in cands:
            nn = normalize_string(c)
            if nn and nn not in seen:
                seen.add(nn)
                cnt[nn] += 1
                surf.setdefault(nn, c)
    return [surf[nn] for nn, v in cnt.items() if v / n >= k]


def load(path):
    return {json.loads(l)["SubjectEntity"]: json.loads(l)["ObjectEntities"]
            for l in open(path, encoding="utf-8") if json.loads(l)["Relation"] == REL}


def macroF1(gold, draws, k):
    subs = list(gold)
    pr = [{"SubjectEntity": s, "Relation": REL, "ObjectEntities": aggregate(draws[s], k)} for s in subs]
    gt = [{"SubjectEntity": s, "Relation": REL, "ObjectEntities": gold[s]} for s in subs]
    res = evaluate_per_sr_pair(pr, gt, RELATION_TYPE)
    return sum(r["f1"] for r in res) / len(res)


async def gen(subjects, base_urls, model, n, temp, conc):
    from openai import AsyncOpenAI
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=180.0, max_retries=3) for u in base_urls]
    sems = [asyncio.Semaphore(conc) for _ in base_urls]
    res = {}
    done = 0

    async def one(i, s):
        nonlocal done
        async with sems[i % len(sems)]:
            r = await clients[i % len(clients)].completions.create(
                model=model, prompt=FEWSHOT.format(name=s), n=n, temperature=temp,
                top_p=0.95, max_tokens=160, stop=STOP)
        res[s] = [c.text for c in r.choices]
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(subjects)}", file=sys.stderr, flush=True)
    await asyncio.gather(*(one(i, s) for i, s in enumerate(subjects)))
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/official_train.jsonl")
    ap.add_argument("--val", default="data/official_val_company.jsonl")
    ap.add_argument("--base-urls", default="http://localhost:8001/v1,http://localhost:8002/v1")
    ap.add_argument("--model", default="gemma3-27b-pt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--conc", type=int, default=48)
    a = ap.parse_args()
    base_urls = [u.strip() for u in a.base_urls.split(",") if u.strip()]
    tg, vg = load(a.train), load(a.val)
    print(f"company-probe: {len(tg)} train / {len(vg)} val", file=sys.stderr)
    td = await gen(list(tg), base_urls, a.model, a.n, a.temp, a.conc)
    vd = await gen(list(vg), base_urls, a.model, a.n, a.temp, a.conc)
    for tag, d in (("train", td), ("val", vd)):
        with open(f"company_probe_{tag}_raw.jsonl", "w", encoding="utf-8") as f:
            for s, dr in d.items():
                f.write(json.dumps({"subject": s, "draws": dr}, ensure_ascii=False) + "\n")
    ks = [x / 100 for x in range(5, 100, 5)]
    best = max(ks, key=lambda k: macroF1(tg, td, k))
    print("\n  k   | train F1 | val F1")
    for k in ks:
        mark = " <- tune-best" if k == best else ""
        print(f"  {k:.2f} | {macroF1(tg,td,k):.3f}   | {macroF1(vg,vd,k):.3f}{mark}")
    print(f"\nPROBE scaffold: tune-best k={best:.2f}  train={macroF1(tg,td,best):.3f}  VAL={macroF1(vg,vd,best):.3f}")
    print("[ref] our incumbent flat template VAL=0.711 ; his instruct champion 0.764")


if __name__ == "__main__":
    asyncio.run(main())
