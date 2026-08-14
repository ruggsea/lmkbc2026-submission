"""Held-out hasCapacity eval on an arbitrary gold file (pseudo-dev / special). Runs the best protocols on the
served base model and reports within-5% hit-rate (== official evaluate.py macro-F1 for single numeric gold).
Handles both gold formats ([["x"]] and ["x"]) and both subject styles (bare name / "X in region").
Protocols: DIRECT, RECITE (region-agnostic), REGION (enumerate-then-pick the one in {region}; needs 'in').
Writes preds_<tag>_<proto>.jsonl (scorer format) for the strongest protocols.
Usage: python cap_region_eval.py --gold data/pseudodev_2026/pdev_eval.jsonl --tag g3_pdev --n 30
"""
import json, re, argparse, time, statistics, concurrent.futures as cf
from openai import OpenAI
FL="results/finaleval"
def goldnum(r):
    oe=r.get("ObjectEntities") or []
    if not oe: return None
    v=oe[0]
    while isinstance(v,list): v=v[0] if v else None
    try: return float(str(v).replace(",",""))
    except: return None
def within(v,g): return v is not None and g and abs(v-g)/g<=0.05
def split_name(s):
    m=re.match(r"(.*?)\s+in\s+(.+)$",s); return (m.group(1).strip(),m.group(2).strip()) if m else (s,"")
def parse_cap(t):
    m=re.search(r"([0-9][0-9,\.]{2,})",t)
    if not m: return None
    try: v=float(m.group(1).replace(",",""))
    except: return None
    return v if 100<=v<=300000 else None
def last_cap(t): return parse_cap(t.rsplit("Capacity:",1)[1]) if "Capacity:" in t else parse_cap(t)
def vote(caps):
    caps=[c for c in caps if c is not None]
    if not caps: return None
    caps=sorted(caps); cl=[]
    for c in caps:
        for g in cl:
            if abs(c-g[0])/g[0]<=0.025: g.append(c); break
        else: cl.append([c])
    return statistics.median(max(cl,key=len))
FS_DIRECT=("Venue: Camp Nou in Barcelona\nCapacity: 99,354\n\n"
 "Venue: Fenway Park in Boston\nCapacity: 37,755\n\n"
 "Venue: Goodwin Field in California\nCapacity: 3,500\n\n")
FS_RECITE=("Venue: Camp Nou in Barcelona\nDescription: Home of FC Barcelona, opened 1957, Barcelona, Spain.\nCapacity: 99,354\n\n"
 "Venue: Fenway Park in Boston\nDescription: Baseball park of the Boston Red Sox, opened 1912, Boston.\nCapacity: 37,755\n\n"
 "Venue: Goodwin Field in California\nDescription: Cal State Fullerton college baseball park, California.\nCapacity: 3,500\n\n")
FS_REGION=("Venue: Wembley Stadium in London\n"
 "Venues named Wembley Stadium: Wembley Stadium in London, England (national stadium); Wembley Stadium in Melbourne, Australia (minor arena).\n"
 "In region: Wembley Stadium in London, England.\nCapacity: 90,000\n\n"
 "Venue: Olympic Stadium in Berlin\n"
 "Venues named Olympic Stadium: Olympic Stadium in Berlin (Germany); Olympic Stadium in Montreal, Canada; Olympic Stadium in London.\n"
 "In region: Olympic Stadium in Berlin, Germany.\nCapacity: 74,475\n\n"
 "Venue: Goodwin Field in California\n"
 "Venues named Goodwin Field: Goodwin Field, the Cal State Fullerton baseball park in California.\n"
 "In region: Goodwin Field, Cal State Fullerton, California.\nCapacity: 3,500\n\n")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gold",required=True); ap.add_argument("--tag",required=True)
    ap.add_argument("--port",type=int,default=8000); ap.add_argument("--n",type=int,default=30)
    a=ap.parse_args(); cli=OpenAI(base_url="http://localhost:%d/v1"%a.port,api_key="none",timeout=600)
    model=cli.models.list().data[0].id
    subs=[]
    for r in (json.loads(l) for l in open(a.gold,encoding="utf-8")):
        if r["Relation"]=="hasCapacity":
            g=goldnum(r)
            if g: subs.append((r["SubjectEntity"],g))
    n_reg=sum(1 for s,_ in subs if " in " in s)
    print("model=%s tag=%s venues=%d (with 'in region': %d)"%(model,a.tag,len(subs),n_reg),flush=True)
    T0=time.time()
    def run(sg,proto):
        s,g=sg; name,region=split_name(s)
        if proto=="DIRECT": prompt=FS_DIRECT+"Venue: %s\nCapacity:"%s; mt=12; stop=["\n"]
        elif proto=="RECITE": prompt=FS_RECITE+"Venue: %s\nDescription:"%s; mt=70; stop=["\n\n","Venue:"]
        else: prompt=FS_REGION+"Venue: %s\nVenues named %s:"%(s,name); mt=120; stop=["\n\n","Venue:"]
        try:
            r=cli.completions.create(model=model,prompt=prompt,n=a.n,temperature=0.8,top_p=0.95,max_tokens=mt,stop=stop)
            return s,vote([last_cap(("Capacity:"+c.text) if proto=="DIRECT" else c.text) for c in r.choices])
        except Exception: return s,None
    res={}
    for proto in ["DIRECT","RECITE","REGION"]:
        prog=[0]; d={}
        def w(sg,proto=proto):
            s,v=run(sg,proto); prog[0]+=1
            if prog[0]%50==0: print("[+%4.0fs] %s %s %d/%d"%(time.time()-T0,a.tag,proto,prog[0],len(subs)),flush=True)
            return s,v
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            for s,v in ex.map(w,subs): d[s]=v
        res[proto]=d
        # write scorer-format preds
        with open(FL+"/preds_%s_%s.jsonl"%(a.tag,proto),"w",encoding="utf-8") as fo:
            for s,g in subs:
                p=d.get(s); oe=[str(int(round(p)))] if p else []
                fo.write(json.dumps({"SubjectEntity":s,"Relation":"hasCapacity","ObjectEntities":oe},ensure_ascii=False)+"\n")
    print("\n=== %s (%s), n=%d venues ==="%(a.tag,model,len(subs)))
    for proto in ["DIRECT","RECITE","REGION"]:
        d=res[proto]; hr=sum(within(d.get(s),g) for s,g in subs)/len(subs)
        print("  %-7s within-5%% (=macro-F1): %.3f"%(proto,hr))
    print("WROTE preds_%s_*.jsonl"%a.tag)
if __name__=="__main__": main()
