"""Draw the base elicitation pool for one relation on one split.

Imports the per-relation prompt templates from this directory verbatim, so the draws match the
ones the submission was built from. Resumable per subject; round-robins over several endpoints.
Writes {subject, draws} lines.

  python src/gen/base/gen_draws.py --rel hasCapacity --split test --out raw/hascap_test.jsonl
"""
import argparse, asyncio, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SPLIT_FILE = {"train": f"{GEN}/off_train.jsonl",
              "val": f"{GEN}/off_val.jsonl",
              "test": f"{GEN}/off_test.jsonl"}


def subjects_for(rel, split):
    return [json.loads(l)["SubjectEntity"] for l in open(SPLIT_FILE[split], encoding="utf-8")
            if json.loads(l)["Relation"] == rel]


def build_jobs(rel, pass_name, split):
    """-> subjects, prompt builder, n, temperature, max_tokens, stop"""
    subs = subjects_for(rel, split)
    if rel == "countryLandBordersCountry":
        from setrel_pt import TEMPLATES, STOP
        fs = TEMPLATES[rel]["fewshot"]
        return subs, (lambda s: fs.format(name=s)), 30, 0.7, 120, STOP
    if rel == "companyTradesAtStockExchange":
        from company_probe_pt import FEWSHOT, STOP
        return subs, (lambda s: FEWSHOT.format(name=s)), 30, 0.7, 160, STOP
    if rel == "personHasCityOfDeath":
        if pass_name == "prime":
            from citydeath_prime_pt import PRIME_FS, PRIME_STOP
            return subs, (lambda s: PRIME_FS.format(name=s)), 30, 0.7, 150, PRIME_STOP
        if pass_name == "gate":
            from citydeath_prime_pt import GATE_FS
            return subs, (lambda s: GATE_FS.format(name=s)), 20, 0.7, 4, ["\n"]
        if pass_name == "cot":
            from citydeath_cot_sc import FEWSHOT, STOP
            return subs, (lambda s: FEWSHOT.format(name=s)), 50, 0.7, 24, STOP
        raise ValueError(f"personHasCityOfDeath needs --pass-name prime|gate|cot, got {pass_name!r}")
    if rel == "awardWonBy":
        from award_base import fs_list
        fs = fs_list("recipients", "")
        return subs, (lambda s: fs + "List of recipients of the %s:\n" % s), 20, 0.8, 1500, \
               ["\n\n", "List of", "Date:"]
    if rel == "hasArea":
        from hasarea_srcanchor import FMT
        cfg = FMT["wikitext"]
        return subs, (lambda s: cfg["prompt"].format(s=s)), 100, 0.8, cfg["maxtok"], cfg["stop"]
    if rel == "hasCapacity":
        from cap_region_eval import FS_REGION, split_name
        def mk(s):
            name, _ = split_name(s)
            return FS_REGION + "Venue: %s\nVenues named %s:" % (s, name)
        return subs, mk, 30, 0.8, 120, ["\n\n", "Venue:"]
    raise ValueError(rel)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", required=True)
    ap.add_argument("--split", required=True, choices=list(SPLIT_FILE))
    ap.add_argument("--pass-name", default="")
    ap.add_argument("--urls", default="http://localhost:8010/v1,http://localhost:8011/v1")
    ap.add_argument("--model", default="gemma-4-31B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="override the per-relation draw count")
    ap.add_argument("--seed", type=int, default=0, help="pin the sampler seed (0 = unseeded)")
    a = ap.parse_args()

    urls = [u.strip() for u in a.urls.split(",") if u.strip()]
    subs, mk, n, temp, maxtok, stop = build_jobs(a.rel, a.pass_name, a.split)
    if a.n:
        n = a.n

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            done.add(json.loads(l)["subject"])
    todo = [s for s in subs if s not in done]
    print(f"{a.rel}/{a.pass_name or '-'} [{a.split}]: {len(subs)} subj, {len(todo)} todo, "
          f"n={n} temp={temp} maxtok={maxtok} urls={len(urls)}", file=sys.stderr, flush=True)

    from openai import AsyncOpenAI
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=600.0, max_retries=4) for u in urls]
    sem = asyncio.Semaphore(a.conc * len(urls))
    import threading
    lock = threading.Lock()
    prog = [0]
    rr = itertools.count()

    async def one(s):
        async with sem:
            ci = next(rr) % len(clients)
            for attempt in range(5):
                try:
                    r = await clients[ci].completions.create(
                        model=a.model, prompt=mk(s), n=n, temperature=temp,
                        top_p=0.95, max_tokens=maxtok, stop=stop,
                        extra_body=({"seed": a.seed} if a.seed else {}))
                    with lock:
                        with open(a.out, "a", encoding="utf-8") as fo:
                            fo.write(json.dumps({"subject": s, "draws": [c.text for c in r.choices]},
                                                ensure_ascii=False) + "\n")
                        prog[0] += 1
                        if prog[0] % 10 == 0:
                            print(f"  {prog[0]}/{len(todo)}", file=sys.stderr, flush=True)
                    return
                except Exception as e:
                    if attempt == 4:
                        print(f"FAIL {s}: {e}", file=sys.stderr, flush=True)
                    await asyncio.sleep(5 * (attempt + 1))

    await asyncio.gather(*(one(s) for s in todo))
    print(f"DONE {a.out}: {prog[0]} new subjects", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
