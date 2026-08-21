"""personHasCityOfDeath — base-model few-shot completion + self-consistency voting.

Strategy (Tim's "city of death" idea):
  1. Conditional fact-file prompt on a BASE model (gemma-3-27b-pt) driven via the
     vLLM /v1/completions endpoint (raw text, no chat template). The 4-shot template
     teaches the conditional rule: Deceased -> a city, Alive -> N/A.
  2. Sample n draws at temperature 0.7 (self-consistency).
  3. Per subject: regex-parse `Status:` and `City of Death:` from each draw.
       - majority Status == Alive           -> predict []  (abstain; "return nothing if alive")
       - else                               -> modal city over the Deceased draws
                                               (ties -> earliest-appearing draw).
  4. City "N/A"/blank is treated as an alive/unknown vote.

Runs async against the served endpoint. Score with evaluate.py.

Usage (repo root, server up on pippi:8001 reachable via ssh tunnel or on-host):
  python tools/pd2026/citydeath_cot_sc.py \
      --subjects data/pseudodev_2026/pdev_tune.jsonl \
      --gold     data/pseudodev_2026/pdev_tune.jsonl \
      --out      lab/citydeath_cot_sc_tune.jsonl \
      --base-url http://localhost:8001/v1 --n 50
"""
import argparse, asyncio, json, re, string, sys
from collections import Counter, defaultdict

REL = "personHasCityOfDeath"

# --- the conditional fact-file few-shot template (base-model completion) ---
FEWSHOT = """Fact: Albert Einstein
Status: Deceased
City of Death: Princeton

Fact: Abraham Lincoln
Status: Deceased
City of Death: Washington

Fact: Taylor Swift
Status: Alive
City of Death: N/A

Fact: Paul McCartney
Status: Alive
City of Death: N/A

Fact: {name}
Status:"""

STOP = ["\n\nFact:", "\nFact:"]

_STATUS_RE = re.compile(r"status\s*:?\s*(alive|deceased|dead)", re.I)
_CITY_RE = re.compile(r"city of death\s*:?\s*(.+)", re.I)
_NA = {"", "n/a", "na", "none", "unknown", "-", "alive"}


def parse_draw(text):
    """One completion (which begins right after 'Status:') -> (status, city|None).

    status in {'alive','dead',None}. city is a cleaned string or None."""
    # the completion starts after the literal 'Status:' we ended the prompt on,
    # so the first token(s) are the status value; re-attach for uniform parsing.
    blob = "Status:" + text
    status = None
    m = _STATUS_RE.search(blob)
    if m:
        status = "alive" if m.group(1).lower() == "alive" else "dead"
    city = None
    m = _CITY_RE.search(blob)
    if m:
        raw = m.group(1).strip().splitlines()[0].strip()
        raw = raw.strip(string.punctuation + " ")
        if raw.lower() not in _NA:
            city = raw
    return status, city


def aggregate(draws, gate="concentration", tau=35):
    """List of raw completion strings for one subject -> predicted list (0 or 1 city).

    Two gates (see LOGBOOK 2026-07-12):
      - "status" (original spec): abstain if majority of draws vote Alive; else modal city.
      - "concentration" (winner): ignore the alive/dead vote entirely; answer the modal
        city only if it is named in >= tau of the draws, else abstain. City *agreement*
        is a far better joint liveness+correctness signal than the alive/dead ratio,
        because the model surfaces a correct death-city in a MINORITY of draws for
        long-tail people (majority-vote-on-status throws that knowledge away)."""
    n_alive = n_dead = 0
    cities = []  # (city, order) over any draw that names a real city
    for i, t in enumerate(draws):
        st, city = parse_draw(t)
        if st == "alive":
            n_alive += 1
        elif st == "dead":
            n_dead += 1
        if city is not None:
            cities.append((city, i))
    info = {"n_alive": n_alive, "n_dead": n_dead, "n_city": len(cities)}
    if not cities:
        return [], info
    counts = Counter(c for c, _ in cities)
    first_seen = {}
    for c, i in cities:
        first_seen.setdefault(c, i)
    # modal city; tie -> earliest appearance
    top, votes = max(counts.items(), key=lambda kv: (kv[1], -first_seen[kv[0]]))
    info.update(vote=votes, city=top)
    if gate == "status":
        if n_alive > n_dead:
            return [], info
        return [top], info
    # concentration gate
    if votes >= tau:
        return [top], info
    return [], info


def load_subjects(path):
    subs = []
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d["Relation"] == REL:
            subs.append(d["SubjectEntity"])
    return subs


async def run(subjects, base_urls, model, n, temp, max_tokens, concurrency, fewshot=None):
    from openai import AsyncOpenAI
    template = fewshot or FEWSHOT
    # one client + its own semaphore per endpoint; subjects round-robin across them
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in base_urls]
    sems = [asyncio.Semaphore(concurrency) for _ in base_urls]
    results = {}

    async def one(name, k):
        client, sem = clients[k], sems[k]
        async with sem:
            prompt = template.format(name=name)
            r = await client.completions.create(
                model=model, prompt=prompt, n=n, temperature=temp,
                top_p=0.95, max_tokens=max_tokens, stop=STOP)
            results[name] = [c.text for c in r.choices]

    # progress ticker
    done = 0
    async def wrapped(name, k):
        nonlocal done
        await one(name, k)
        done += 1
        if done % 25 == 0 or done == len(subjects):
            print(f"  {done}/{len(subjects)} subjects", file=sys.stderr, flush=True)

    await asyncio.gather(*(wrapped(s, i % len(base_urls)) for i, s in enumerate(subjects)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--gold", default=None, help="if given, score against it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-out", default=None, help="dump per-subject draws for later reuse")
    ap.add_argument("--base-urls", default="http://localhost:8001/v1,http://localhost:8002/v1",
                    help="comma-separated vLLM endpoints; subjects round-robin across them")
    ap.add_argument("--model", default="gemma3-27b-pt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--gate", choices=["concentration", "status"], default="concentration",
                    help="concentration (winner): answer modal city if named in >=tau draws; "
                         "status: original spec, abstain if majority vote Alive")
    ap.add_argument("--tau", type=int, default=35, help="concentration-gate threshold (of n draws)")
    ap.add_argument("--fewshot-file", default=None, help="custom Fact/Status few-shot ending in 'Fact: {name}\\nStatus:'")
    a = ap.parse_args()

    subjects = load_subjects(a.subjects)
    fewshot = open(a.fewshot_file, encoding="utf-8").read() if a.fewshot_file else None
    print(f"{len(subjects)} {REL} subjects; n={a.n} temp={a.temp} "
          f"fewshot={'custom' if fewshot else 'default'}", file=sys.stderr)
    base_urls = [u.strip() for u in a.base_urls.split(",") if u.strip()]
    draws = asyncio.run(run(subjects, base_urls, a.model, a.n, a.temp,
                            a.max_tokens, a.concurrency, fewshot))

    if a.raw_out:
        with open(a.raw_out, "w", encoding="utf-8") as f:
            for s in subjects:
                f.write(json.dumps({"subject": s, "draws": draws[s]}, ensure_ascii=False) + "\n")

    preds = {}
    diag = {}
    for s in subjects:
        pred, info = aggregate(draws[s], gate=a.gate, tau=a.tau)
        preds[s] = pred
        diag[s] = info
    with open(a.out, "w", encoding="utf-8") as f:
        for s in subjects:
            f.write(json.dumps({"SubjectEntity": s, "Relation": REL,
                                "ObjectEntities": preds[s]}, ensure_ascii=False) + "\n")
    n_abstain = sum(1 for s in subjects if not preds[s])
    print(f"wrote {a.out}: {len(subjects)} rows, {n_abstain} abstained (empty), "
          f"{len(subjects)-n_abstain} answered", file=sys.stderr)

    if a.gold:
        sys.path.insert(0, ".")
        from evaluate import evaluate_per_sr_pair, RELATION_TYPE
        gold = {}
        for l in open(a.gold, encoding="utf-8"):
            d = json.loads(l)
            if d["Relation"] == REL:
                gold[d["SubjectEntity"]] = d["ObjectEntities"]
        pr = [{"SubjectEntity": s, "Relation": REL, "ObjectEntities": preds[s]} for s in subjects if s in gold]
        gt = [{"SubjectEntity": s, "Relation": REL, "ObjectEntities": gold[s]} for s in subjects if s in gold]
        res = evaluate_per_sr_pair(pr, gt, RELATION_TYPE)
        f1 = sum(r["f1"] for r in res) / len(res)
        p = sum(r["p"] for r in res) / len(res)
        rc = sum(r["r"] for r in res) / len(res)
        print(f"\n=== {REL} on {a.gold} (n={len(res)}) ===")
        print(f"macro-P={p:.4f}  macro-R={rc:.4f}  macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
