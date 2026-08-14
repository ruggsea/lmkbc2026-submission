"""Build a FAMOUS hasArea tuning slice matching the official/test distribution (islands, lakes,
countries, mountains, admin regions) — the numeric analog of famous2/3 for award. Design-time
Wikidata (closed-book applies only at inference). Fetches P2046 area with unit, normalizes to km2,
keeps high-sitelink (famous) entities. Excludes any subject already in official train/val/test or
pseudo-dev hasArea -- BY ENTITY IDENTITY (Wikidata QID + full alias set), not just exact-lowercase
label match: a bank row is dropped if ITS Wikidata label-or-altLabel set intersects an official
subject's surface string, so aliasing (e.g. bank "Ireland" == official val "Island of Ireland",
same QID) is caught even though the two surface strings differ. Reading official test.jsonl here
only touches SubjectEntity (ObjectEntities is always [] in this repo -- no test gold exists to
leak); see data/g4_demobank_guard/verdict.txt for the audit that motivated this.
Usage: python build_famous_area.py --fetch   (writes data/special/hasarea_famous.jsonl)
"""
import sys, json, time, subprocess, os
sys.path.insert(0, ".")
from tools.wikidata import aliases_for
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "lmkbc-area-famous/1.0 (research)"
OUT = "data/special/hasarea_famous.jsonl"
# class QID -> friendly; each queried separately to spread load
CLASSES = {
    "Q23442": "island", "Q23397": "lake", "Q6256": "country", "Q8502": "mountain",
    "Q46831": "mountain range", "Q46169": "peninsula", "Q42523": "volcano",
    "Q1637706": "big city", "Q4022": "river-basin?", "Q22698": "park",
}
# unit QID -> km2 multiplier
UNIT = {"Q712226": 1.0, "Q25343": 1e-6, "Q35852": 0.01, "Q232291": 2.589988, "Q3841820": 1e-4}

def sparql(q):
    for attempt in range(6):
        p = subprocess.run(["curl", "-s", "-m", "150", "-H", "Accept: application/sparql-results+json",
                            "-H", "User-Agent: " + UA, "--data-urlencode", "query=" + q, ENDPOINT],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        txt = p.stdout or ""
        if txt.strip().startswith("{"): return json.loads(txt)
        time.sleep(65 if "429" in txt else 8 * (attempt + 1))
    return None

def fetch():
    exclude = set()
    # cover TRAIN + VAL + TEST + pseudodev -- the original builder omitted test, letting
    # official-test subjects (and their aliases) sit undetected in the shipped demo bank.
    for f in ["data/official/train.jsonl", "data/official/val.jsonl", "data/official/test.jsonl",
              "data/pseudodev_2026/hasArea.jsonl"]:
        if os.path.exists(f):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r.get("Relation", "hasArea") == "hasArea": exclude.add(r["SubjectEntity"].strip().lower())
    rows = {}
    for qid, name in CLASSES.items():
        q = ('SELECT ?item ?itemLabel ?area ?unit ?sl WHERE { '
             '?item wdt:P31/wdt:P279* wd:%s . ?item wdt:P2046 ?a . '
             '?item p:P2046/psv:P2046 ?vn . ?vn wikibase:quantityAmount ?area . ?vn wikibase:quantityUnit ?u . '
             'BIND(STRAFTER(STR(?u),"entity/") AS ?unit) '
             '?item wikibase:sitelinks ?sl . FILTER(?sl >= 40) '
             'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?sl) LIMIT 120' % qid)
        d = sparql(q)
        if not d: print("  FAIL", name); continue
        cnt = 0
        for b in d["results"]["bindings"]:
            lab = b.get("itemLabel", {}).get("value", "")
            if not lab or (lab.startswith("Q") and lab[1:].isdigit()): continue
            if lab.strip().lower() in exclude: continue
            u = b.get("unit", {}).get("value", "")
            mult = UNIT.get(u)
            if mult is None: continue
            try: km2 = float(b["area"]["value"]) * mult
            except Exception: continue
            if km2 <= 0: continue
            if lab in rows: continue
            item_qid = b["item"]["value"].rsplit("/", 1)[-1]
            rows[lab] = {"SubjectEntity": lab, "Relation": "hasArea",
                         "ObjectEntities": [[repr(round(km2, 3))]], "_sitelinks": int(b["sl"]["value"]),
                         "_class": name, "_qid": item_qid}
            cnt += 1
        print("  %-16s +%d (total %d)" % (name, cnt, len(rows)))
        time.sleep(3)

    # identity pass: drop any row whose Wikidata label/altLabel set intersects an official
    # subject string, even if the row's own SubjectEntity label already passed the string check
    # above (same-QID aliasing, e.g. "Ireland" the island vs "Island of Ireland" the val subject).
    print("resolving aliases for %d candidates to close the identity gap..." % len(rows))
    al_by_qid = aliases_for([r["_qid"] for r in rows.values()])
    clean = {}
    for lab, r in rows.items():
        al = set(a.lower() for a in al_by_qid.get(r["_qid"], [])) | {lab.strip().lower()}
        if al & exclude:
            print("  IDENTITY DROP: %r (%s) aliases-with official subject %r" % (lab, r["_qid"], sorted(al & exclude)))
            continue
        r["_aliases"] = sorted(al)
        clean[lab] = r
    rows = clean

    ranked = sorted(rows.values(), key=lambda r: -r["_sitelinks"])[:500]
    with open(OUT, "w", encoding="utf-8") as f:
        for r in ranked: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import statistics
    med = statistics.median(float(r["ObjectEntities"][0][0]) for r in ranked)
    print("wrote %d famous-area subjects, median %.1f km2 -> %s" % (len(ranked), med, OUT))

if __name__ == "__main__":
    if "--fetch" in sys.argv: fetch()
    else: print(__doc__)
