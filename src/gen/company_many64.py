"""companyTradesAtStockExchange many-shot k=64 elicitation (the shipped generator).
dv_many64 prose template + SENTINEL span parser; n=100 t=0.7 seed=0 maxtok=96 stop=["\n"]."""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import requests

REL = "companyTradesAtStockExchange"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

SENTINEL = re.compile(r"not publicly listed|not listed|not admitted|not traded|does not trade|^none$|^\(none\)$", re.I)
SPAN_CUE = re.compile(r"(?:listed|registered|quoted|trades?|traded|trading)\s+on(?:\s+the)?\s+", re.I)


def parse_span(text):
    first = text.strip().split("\n")[0].strip()
    if not first or SENTINEL.search(first):
        return []
    m = SPAN_CUE.search(first)
    seg = first[m.end():] if m else (first.split("—")[-1] if "—" in first else first)
    seg = seg.split(".")[0]
    parts = re.split(r"\s*(?:;|,|\band\b)\s*", seg)
    return [p.strip(' .;,"') for p in parts if p.strip(' .;,"') and not SENTINEL.search(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    tpl = open(f"{REPO}/tasks/companyTradesAtStockExchange/prompts_base/dv_many64.txt",
               encoding="utf-8").read()
    assert "{s}" in tpl
    subs = [json.loads(l)["SubjectEntity"] for l in open(f"{HERE}/off_{a.split}.jsonl", encoding="utf-8")
            if json.loads(l)["Relation"] == REL]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["subject"] for l in open(a.out, encoding="utf-8")}
    todo = [s for s in subs if s not in done]
    model = requests.get(f"{a.url}/v1/models").json()["data"][0]["id"]
    print(f"company many64 [{a.split}]: {len(subs)} subj, {len(todo)} todo, n={a.n} seed={a.seed}", flush=True)

    def work(s):
        r = requests.post(f"{a.url}/v1/completions", json={
            "model": model, "prompt": tpl.replace("{s}", s), "n": a.n, "temperature": a.temp,
            "top_p": 0.95, "max_tokens": 96, "seed": a.seed, "stop": ["\n"]}, timeout=900)
        r.raise_for_status()
        return {"subject": s, "draws": [c["text"] for c in r.json()["choices"]]}

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a", encoding="utf-8") as f:
        for i, o in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("COMPANY_DONE " + a.out, flush=True)


if __name__ == "__main__":
    main()
