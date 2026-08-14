"""Resolve award subject names -> Wikidata QIDs via ONE batched WDQS SPARQL query (local curl).
Matches each name to an entity by exact en label/altLabel, preferring entities that are instance/subclass
of 'award' (Q618779). Writes {qmap, meta}. Usage: resolve_qids_sparql.py <subjects.jsonl> <out_map.json>"""
import sys, json, subprocess, time, collections

SUBJ, OUT = sys.argv[1], sys.argv[2]
names = [json.loads(l)["SubjectEntity"] for l in open(SUBJ, encoding="utf-8")]


def esc(x): return x.replace("\\", "\\\\").replace('"', '\\"')


def run(batch):
    vals = " ".join(f'"{esc(n)}"' for n in batch)
    # award-type constraint FIRST (small candidate set) -> then label match: fast
    q = (f'SELECT ?name ?award (1 AS ?isAward) WHERE {{ '
         f'VALUES ?name {{ {vals} }} '
         f'?award wdt:P31/wdt:P279* wd:Q618779 . '
         f'?award rdfs:label|skos:altLabel ?lbl . FILTER(LANG(?lbl)="en" && STR(?lbl)=STR(?name)) }}')
    for attempt in range(4):
        p = subprocess.run(["curl", "-s", "-m", "90", "-H", "Accept: application/sparql-results+json",
                            "-H", "User-Agent: lmkbc/1.0", "--data-urlencode", "query=" + q,
                            "https://query.wikidata.org/sparql"], capture_output=True, text=True)
        if p.stdout.strip().startswith("{"):
            return json.loads(p.stdout)["results"]["bindings"]
        time.sleep(10)
    return []


cand = collections.defaultdict(list)  # name -> [(isAward, qid)]
for i in range(0, len(names), 8):
    for b in run(names[i:i + 8]):
        nm = b["name"]["value"]; qid = b["award"]["value"].rsplit("/", 1)[-1]
        cand[nm].append((int(b["isAward"]["value"]), qid))
    time.sleep(3)

qmap, meta = {}, {}
for n in names:
    opts = sorted(cand.get(n, []), reverse=True)  # award-typed first
    qmap[n] = opts[0][1] if opts else ""
    meta[n] = {"src": "eval", "isAward": bool(opts[0][0]) if opts else False, "n_candidates": len(opts)}
json.dump({"qmap": qmap, "meta": meta}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
got = sum(1 for q in qmap.values() if q.startswith("Q"))
aw = sum(1 for n in names if meta[n]["isAward"])
print(f"resolved {got}/{len(names)} ({aw} award-typed) -> {OUT}")
print("UNRESOLVED:", [n for n in names if not qmap[n].startswith("Q")][:20])
