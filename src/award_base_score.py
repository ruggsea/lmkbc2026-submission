"""Score awardWonBy base-completion pools (awb_*.jsonl) with a consensus-threshold sweep.
Parser (from smoke-inspection, Entry-pending): per line strip year prefix, take the name field
before any second dash / " for <work>" credit, strip parentheticals, split multi-winner lists.
Consensus: keep names appearing in >= k of N draws; sweep k on TUNE, report eval/official at
tune-best k (and the full sweep for the record). Scoring = evaluate.py conventions (alias-aware).
Usage: python src/award_base_score.py --pool results/finaleval/awb_list.jsonl
"""
import json, re, sys, argparse
from collections import Counter
sys.path.insert(0, ".")
from evaluate import normalize_string, evaluate_per_sr_pair

GOLD={"tune":"lab/gold_award_tune_rebuilt.jsonl",
      "eval":"lab/gold_award_eval_rebuilt.jsonl",
      "official":"lab/gold_award_official.jsonl"}
RT={"awardWonBy":"string"}
YEAR=re.compile(r"^\s*(\d{3,4})(?:[/–\-]\d{2,4})?\s*[–—\-:.]\s*")
PAREN=re.compile(r"\([^)]*\)")
HTMLTAG=re.compile(r"</?[a-zA-Z]+[^>]*>")  # gemma-4 wiki-list draws sometimes wrap "not awarded" /
# work titles in <em>/<i>/<strong> -- untagged, JUNK's ^-anchored match misses "<em>not awarded</em>"
# entirely (31bj#4 parse audit). Currently self-corrected downstream: g4_award_nameval.validate_name
# rejects any candidate containing "<" anyway, so this is a defensive fix with 0 shipped impact today.
JUNK=re.compile(r"^(and|the|his|her|their|crew|others?|various|unknown|none|no (award|prize|medal)|not awarded|award not)\b",re.I)

def clean_name(t):
    t=HTMLTAG.sub("",t)
    t=PAREN.sub("",t).strip(" \t\"'“”.,;:*")
    t=re.sub(r"\s+"," ",t)
    if len(t)<3 or len(t.split())>8: return None
    if re.fullmatch(r"[\d\s–\-]+",t): return None
    if JUNK.match(t): return None
    if "crew" in t.lower(): return None
    return t

def parse_line(line):
    line=line.strip()
    if not line: return []
    if line.endswith(":") and len(line.split())<=6: return []   # section header
    line=YEAR.sub("",line)
    # keep name field: cut at a second dash (credit/work) or " for <work>"
    line=re.split(r"\s+[–—]\s+| – | for ",line)[0]
    parts=re.split(r";|,\s*(?:and\s+)?|\s+and\s+|\s*&\s*",line)
    out=[]
    for p in parts:
        c=clean_name(p)
        if c: out.append(c)
    return out

def parse_draw(text,fmt):
    names=[]
    if fmt=="flat":
        for p in re.split(r";|,",text):
            c=clean_name(p)
            if c: names.append(c)
    elif fmt=="article":
        # skip the generated lead; parse the list after "Recipients:" (fallback: year-lines only)
        if "Recipients:" in text:
            for line in text.split("Recipients:",1)[1].split("\n"): names.extend(parse_line(line))
        else:
            for line in text.split("\n"):
                if YEAR.match(line): names.extend(parse_line(line))
    elif fmt=="cat":
        for line in text.split("\n"):
            l=line.strip()
            if not l: continue
            if "pages are in this category" in l or l.startswith("The following"): continue
            if l.startswith("(") or l.lower().startswith("category"): continue
            c=clean_name(l)
            if c: names.append(c)
    else:
        for line in text.split("\n"): names.extend(parse_line(line))
    # dedup within draw by normalized form
    seen=set(); out=[]
    for n in names:
        k=normalize_string(n)
        if k not in seen: seen.add(k); out.append(n)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pool",required=True,help="pool jsonl; comma-separate to merge draws per subject")
    ap.add_argument("--show",type=int,default=0,help="print parsed names for first N subjects")
    a=ap.parse_args()
    pools={}
    for path in a.pool.split(","):
        for r in (json.loads(l) for l in open(path,encoding="utf-8")):
            if r["subject"] in pools:
                pools[r["subject"]][2].extend(r["draws"])
            else:
                pools[r["subject"]]=(r["split"],r["fmt"],list(r["draws"]))
    gold={s:[json.loads(l) for l in open(p,encoding="utf-8")] for s,p in GOLD.items()}
    # parse all pools -> per subject: Counter over normalized names + representative surface form
    parsed={}
    shown=0
    for subj,(split,fmt,draws) in pools.items():
        cnt=Counter(); surf={}
        for d in draws:
            for n in parse_draw(d,fmt):
                k=normalize_string(n)
                cnt[k]+=1
                if k not in surf: surf[k]=n
        parsed[subj]=(split,cnt,surf,len(draws))
        if shown<a.show:
            print("== %s (top15 of %d uniq):"%(subj,len(cnt)),
                  [(surf[k],c) for k,c in cnt.most_common(15)]); shown+=1
    N=max(n for _,_,_,n in parsed.values())
    print("subjects=%d N=%d"%(len(parsed),N))
    # sweep k per split
    res={}
    for split,grows in gold.items():
        subjects={r["SubjectEntity"] for r in grows}
        rows=[]
        for k in range(1,N+1):
            preds=[]
            for r in grows:
                s=r["SubjectEntity"]
                if s in parsed:
                    _,cnt,surf,_=parsed[s]
                    objs=[surf[key] for key,c in cnt.items() if c>=k]
                else: objs=[]
                preds.append({"SubjectEntity":s,"Relation":"awardWonBy","ObjectEntities":objs})
            sc=evaluate_per_sr_pair(preds,grows,RT)
            f1=sum(x["f1"] for x in sc)/len(sc)
            p=sum(x["p"] for x in sc)/len(sc); rr=sum(x["r"] for x in sc)/len(sc)
            npred=sum(x["total_pred"] for x in sc)/len(sc)
            rows.append((k,f1,p,rr,npred))
        res[split]=rows
    print("\n%-9s"%"k"+ "".join("%-26s"%s for s in res))
    for i in range(min(N,30)):
        line="%-9d"%rows[i][0] if False else "%-9d"%(i+1)
        for s in res:
            k,f1,p,r,np_=res[s][i]
            line+="F1 %.4f P %.3f R %.3f  "%(f1,p,r)
        print(line)
    for s in res:
        best=max(res[s],key=lambda x:x[1])
        print("BEST %-9s k=%-3d F1=%.4f P=%.3f R=%.3f preds/award=%.1f"%(s,best[0],best[1],best[2],best[3],best[4]))
    bt=max(res["tune"],key=lambda x:x[1])[0]
    print("\nAT TUNE-BEST k=%d:"%bt)
    for s in res:
        k,f1,p,r,np_=res[s][bt-1]
        print("  %-9s F1=%.4f P=%.3f R=%.3f preds/award=%.1f"%(s,f1,p,r,np_))
if __name__=="__main__": main()
