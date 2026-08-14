"""awardWonBy base-completion pools on gemma-3-27b-pt (capacity playbook, Phase 0).
Two few-shot formats, base completion (no chat template):
  A "list" : Wikipedia-style "List of recipients of the X:" with year-by-year lines
             (base-form of Entry 018's recite-year-by-year, the only positive award lever)
  B "flat" : "Award: X\nRecipients: a; b; c" flat semicolon list
Draws N completions per award at t0.8 and saves the FULL raw pool (aggregation is offline,
capacity lesson: pools are reusable). Resumable per subject. Shards across two vLLM ports.
Usage: python src/award_base.py --fmt list --n 20 [--smoke]
"""
import json, argparse, time, itertools, concurrent.futures as cf
from openai import OpenAI
FL="results/finaleval"
GOLD=[("tune","lab/gold_award_tune_rebuilt.jsonl"),
      ("eval","lab/gold_award_eval_rebuilt.jsonl"),
      ("official","lab/gold_award_official.jsonl"),
      ("otrain","lab/gold_award_otrain.jsonl"),
      ("oval","lab/gold_award_oval.jsonl"),
      ("special","lab/gold_award_special.jsonl"),
      ("famous2","lab/gold_award_famous2.jsonl")]

DEMO_PRITZKER=[(1979,"Philip Johnson"),(1980,"Luis Barragán"),(1981,"James Stirling"),
 (1982,"Kevin Roche"),(1983,"I. M. Pei"),(1984,"Richard Meier"),(1985,"Hans Hollein"),
 (1986,"Gottfried Böhm"),(1987,"Kenzo Tange"),(1988,"Gordon Bunshaft and Oscar Niemeyer"),
 (1989,"Frank Gehry"),(1990,"Aldo Rossi"),(1991,"Robert Venturi"),(1992,"Álvaro Siza"),
 (1993,"Fumihiko Maki"),(1994,"Christian de Portzamparc"),(1995,"Tadao Ando")]
DEMO_ABEL=[(2003,"Jean-Pierre Serre"),(2004,"Michael Atiyah and Isadore Singer"),
 (2005,"Peter Lax"),(2006,"Lennart Carleson"),(2007,"S. R. Srinivasa Varadhan"),
 (2008,"John G. Thompson and Jacques Tits"),(2009,"Mikhail Gromov"),(2010,"John Tate"),
 (2011,"John Milnor"),(2012,"Endre Szemerédi"),(2013,"Pierre Deligne"),(2014,"Yakov Sinai"),
 (2015,"John Nash and Louis Nirenberg"),(2016,"Andrew Wiles"),(2017,"Yves Meyer"),
 (2018,"Robert Langlands"),(2019,"Karen Uhlenbeck")]

def fs_list(word="recipients",date=""):
    d="Date: %s\n"%date if date else ""
    b=[]
    for name,rows in [("Pritzker Architecture Prize",DEMO_PRITZKER),("Abel Prize",DEMO_ABEL)]:
        b.append(d+"List of %s of the %s:\n"%(word,name)+"\n".join("%d – %s"%(y,n) for y,n in rows))
    return "\n\n".join(b)+"\n\n"

DEMO_PRITZKER_FULL=DEMO_PRITZKER+[(1996,"Rafael Moneo"),(1997,"Sverre Fehn"),(1998,"Renzo Piano"),
 (1999,"Norman Foster"),(2000,"Rem Koolhaas"),(2001,"Jacques Herzog and Pierre de Meuron"),
 (2002,"Glenn Murcutt"),(2003,"Jørn Utzon"),(2004,"Zaha Hadid"),(2005,"Thom Mayne"),
 (2006,"Paulo Mendes da Rocha"),(2007,"Richard Rogers"),(2008,"Jean Nouvel"),(2009,"Peter Zumthor"),
 (2010,"Kazuyo Sejima and Ryue Nishizawa"),(2011,"Eduardo Souto de Moura"),(2012,"Wang Shu"),
 (2013,"Toyo Ito"),(2014,"Shigeru Ban"),(2015,"Frei Otto"),(2016,"Alejandro Aravena"),
 (2017,"Rafael Aranda, Carme Pigem and Ramon Vilalta"),(2018,"Balkrishna Doshi"),
 (2019,"Arata Isozaki"),(2020,"Yvonne Farrell and Shelley McNamara"),
 (2021,"Anne Lacaton and Jean-Philippe Vassal"),(2022,"Diébédo Francis Kéré"),
 (2023,"David Chipperfield"),(2024,"Riken Yamamoto")]
DEMO_ABEL_FULL=DEMO_ABEL+[(2020,"Hillel Furstenberg and Gregory Margulis"),
 (2021,"László Lovász and Avi Wigderson"),(2022,"Dennis Sullivan"),(2023,"Luis Caffarelli"),
 (2024,"Michel Talagrand")]
def fs_long(word="recipients"):
    b=[]
    for name,rows in [("Pritzker Architecture Prize",DEMO_PRITZKER_FULL),("Abel Prize",DEMO_ABEL_FULL)]:
        b.append("List of %s of the %s:\n"%(word,name)+"\n".join("%d – %s"%(y,n) for y,n in rows))
    return "\n\n".join(b)+"\n\n"
LEAD_P=("The Pritzker Architecture Prize is an international architecture prize awarded annually "
 "since 1979 to honor a living architect whose built work demonstrates talent, vision and commitment. "
 "It was established by Jay Pritzker through the Hyatt Foundation and is often called the Nobel Prize of architecture.")
LEAD_A=("The Abel Prize is a prize awarded annually by the King of Norway to one or more outstanding "
 "mathematicians, named after Niels Henrik Abel and first awarded in 2003.")
def fs_article():
    b=[]
    for lead,name,rows in [(LEAD_P,"Pritzker Architecture Prize",DEMO_PRITZKER_FULL),
                           (LEAD_A,"Abel Prize",DEMO_ABEL_FULL)]:
        b.append(lead+"\nRecipients:\n"+"\n".join("%d – %s"%(y,n) for y,n in rows))
    return "\n\n".join(b)+"\n\n"
def fs_wart():
    # ARTICLE + explicit Wikipedia-page signal (extract header + title line), single-newline layout
    b=[]
    for lead,name,rows in [(LEAD_P,"Pritzker Architecture Prize",DEMO_PRITZKER_FULL),
                           (LEAD_A,"Abel Prize",DEMO_ABEL_FULL)]:
        b.append("From Wikipedia, the free encyclopedia\n%s\n"%name+lead+
                 "\nRecipients:\n"+"\n".join("%d – %s"%(y,n) for y,n in rows))
    return "\n\n".join(b)+"\n\n"
ERAS=[1930,1960,1980,1995,2010,2020]
def fs_count(word="recipients"):
    # demos declare their TRUE people-counts (53, 27); target declares --declared
    b=[]
    for name,cnt,rows in [("Pritzker Architecture Prize",53,DEMO_PRITZKER_FULL),
                          ("Abel Prize",27,DEMO_ABEL_FULL)]:
        b.append("List of %s of the %s (%d %s):\n"%(word,name,cnt,word)+
                 "\n".join("%d – %s"%(y,n) for y,n in rows))
    return "\n\n".join(b)+"\n\n"
def _decades(rows):
    out=[]; cur=None
    for y,n in rows:
        dec=y//10*10
        if dec!=cur: out.append("%ds:"%dec); cur=dec
        out.append("%d – %s"%(y,n))
    return "\n".join(out)
def fs_scaf(word="recipients"):
    b=[]
    for name,rows in [("Pritzker Architecture Prize",DEMO_PRITZKER_FULL),("Abel Prize",DEMO_ABEL_FULL)]:
        b.append("List of %s of the %s, by decade:\n"%(word,name)+_decades(rows))
    return "\n\n".join(b)+"\n\n"

CAT_PRITZKER=["Alejandro Aravena","Aldo Rossi","Álvaro Siza","Arata Isozaki","Balkrishna Doshi",
 "Christian de Portzamparc","Eduardo Souto de Moura","Frank Gehry","Fumihiko Maki","Glenn Murcutt",
 "Gottfried Böhm","Hans Hollein","I. M. Pei","Jacques Herzog","James Stirling","Jean Nouvel",
 "Kevin Roche","Luis Barragán","Norman Foster","Oscar Niemeyer","Philip Johnson","Rafael Moneo",
 "Rem Koolhaas","Renzo Piano","Richard Meier","Richard Rogers","Robert Venturi","Shigeru Ban",
 "Tadao Ando","Thom Mayne","Toyo Ito","Wang Shu","Zaha Hadid"]
CAT_ABEL=["Andrew Wiles","Avi Wigderson","Dennis Sullivan","Endre Szemerédi","Grigory Margulis",
 "Hillel Furstenberg","Isadore Singer","Jacques Tits","Jean-Pierre Serre","John Milnor",
 "John G. Thompson","John Nash","John Tate","Karen Uhlenbeck","László Lovász","Lennart Carleson",
 "Louis Nirenberg","Luis Caffarelli","Michael Atiyah","Mikhail Gromov","Peter Lax","Pierre Deligne",
 "Robert Langlands","S. R. Srinivasa Varadhan","Yakov Sinai","Yves Meyer"]
def fs_cat():
    b=[]
    for name,rows in [("Pritzker Architecture Prize",CAT_PRITZKER),("Abel Prize",CAT_ABEL)]:
        b.append('Pages in category "Recipients of the %s"\n'%name+
                 "The following %d pages are in this category:\n"%len(rows)+"\n".join(rows))
    return "\n\n".join(b)+"\n\n"
def fs_flat():
    b=[]
    for name,rows in [("Pritzker Architecture Prize",DEMO_PRITZKER),("Abel Prize",DEMO_ABEL)]:
        names=list(itertools.chain.from_iterable(n.split(" and ") for _,n in rows))
        b.append("Award: %s\nRecipients: %s"%(name,"; ".join(names)))
    return "\n\n".join(b)+"\n\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fmt",choices=["list","flat","cat","long","article","seed","count","scaf","cont","wart"],required=True)
    ap.add_argument("--declared",type=int,default=120,help="declared recipient count (fmt=count)")
    ap.add_argument("--seeds",default="",help="cont_seeds.jsonl from award_cont_prep (fmt=cont)")
    ap.add_argument("--date",default="",help="early-placement 'Date: <Month Year>' header (list fmt)")
    ap.add_argument("--n",type=int,default=20); ap.add_argument("--temp",type=float,default=0.8)
    ap.add_argument("--maxtok",type=int,default=1500)
    ap.add_argument("--ports",default="8000,8001"); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tag",default="",help="suffix for the output file (extension pools)")
    ap.add_argument("--chunk",type=int,default=0,help="split n draws into requests of this size (0=one request; avoids KV-thrash timeouts on all-maxtok stragglers)")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--word",default="recipients",help="list-header noun (label-register sweep)")
    a=ap.parse_args()
    clis=[OpenAI(base_url="http://localhost:%s/v1"%p,api_key="none",timeout=1200) for p in a.ports.split(",")]
    model=clis[0].models.list().data[0].id
    subs=[]; seen=set()
    for split,path in GOLD:
        for r in (json.loads(l) for l in open(path,encoding="utf-8")):
            s=r["SubjectEntity"]
            if s in seen: continue
            seen.add(s); subs.append((s,split))
    if a.smoke: subs=subs[:10]; a.n=min(a.n,5)
    out=FL+"/awb_%s%s%s.jsonl"%(a.fmt,a.tag,"_smoke" if a.smoke else "")
    done=set()   # resume strictly from this run's own output file
    try:
        for l in open(out,encoding="utf-8"): done.add(json.loads(l)["subject"])
    except FileNotFoundError: pass
    todo=[(i,s,sp) for i,(s,sp) in enumerate(subs) if s not in done]
    if a.fmt=="list": fs=fs_list(a.word,a.date)
    elif a.fmt=="cat": fs=fs_cat()
    elif a.fmt in ("long","seed"): fs=fs_long(a.word)
    elif a.fmt=="article": fs=fs_article()
    elif a.fmt=="wart": fs=fs_wart()
    elif a.fmt=="count": fs=fs_count(a.word)
    elif a.fmt=="scaf": fs=fs_scaf(a.word)
    elif a.fmt=="cont": fs=fs_long(a.word)
    else: fs=fs_flat()
    seeds={}
    if a.fmt=="cont":
        for r in (json.loads(l) for l in open(a.seeds,encoding="utf-8")):
            seeds[r["subject"]]=r["lines"]
    print("fmt=%s n=%d temp=%.1f subjects=%d todo=%d model=%s out=%s"%(
        a.fmt,a.n,a.temp,len(subs),len(todo),model,out),flush=True)
    if a.fmt in ("list","long","seed"):
        d="Date: %s\n"%a.date if a.date else ""
        mkprompt=lambda s: fs+d+"List of %s of the %s:\n"%(a.word,s)
        stops=["\n\n","List of","Date:"]
    elif a.fmt=="count":
        mkprompt=lambda s: fs+"List of %s of the %s (%d %s):\n"%(a.word,s,a.declared,a.word)
        stops=["\n\n","List of"]
    elif a.fmt=="scaf":
        mkprompt=lambda s: fs+"List of %s of the %s, by decade:\n"%(a.word,s)
        stops=["\n\n","List of"]
    elif a.fmt=="cont":
        # paste the pass-1 consensus list, let the model CONTINUE it
        mkprompt=lambda s: fs+"List of %s of the %s:\n"%(a.word,s)+ \
            ("\n".join(seeds.get(s,[]))+"\n" if seeds.get(s) else "")
        stops=["\n\n","List of"]
    elif a.fmt=="article":
        mkprompt=lambda s: fs+"The %s is"%s
        stops=["\n\n"]
    elif a.fmt=="wart":
        mkprompt=lambda s: fs+"From Wikipedia, the free encyclopedia\n%s\nThe %s is"%(s,s)
        stops=["\n\n","From Wikipedia"]
    elif a.fmt=="cat":
        mkprompt=lambda s: fs+'Pages in category "Recipients of the %s"\n'%s
        stops=["\n\n","Pages in","Category:"]
    else:
        mkprompt=lambda s: fs+"Award: %s\nRecipients:"%s
        stops=["\n\n","Award:"]
    T0=time.time(); nd=[0]
    import threading; lock=threading.Lock()
    def w(job):
        i,s,split=job
        cli=clis[i%len(clis)]
        try:
            if a.fmt=="seed":
                draws=[]
                for era in ERAS:
                    r=cli.completions.create(model=model,prompt=mkprompt(s)+"%d –"%era,n=a.n,
                                             temperature=a.temp,top_p=0.95,max_tokens=a.maxtok,stop=stops)
                    draws+=["%d – %s"%(era,c.text) for c in r.choices]
            else:
                draws=[]
                step=a.chunk if a.chunk>0 else a.n
                for off in range(0,a.n,step):
                    r=cli.completions.create(model=model,prompt=mkprompt(s),n=min(step,a.n-off),
                                             temperature=a.temp,top_p=0.95,max_tokens=a.maxtok,stop=stops)
                    draws+=[c.text for c in r.choices]
        except Exception as e:
            print("ERR %s: %s"%(s,e),flush=True); return
        with lock:
            with open(out,"a",encoding="utf-8") as fo:
                fo.write(json.dumps({"subject":s,"split":split,"fmt":a.fmt,"n":a.n,
                                     "temp":a.temp,"draws":draws})+"\n")
            nd[0]+=1
            if nd[0]%10==0 or a.smoke:
                el=time.time()-T0
                print("[%d/%d] %.1fmin ETA %.1fmin (last: %s)"%(nd[0],len(todo),el/60,
                      el/nd[0]*(len(todo)-nd[0])/60,s[:40]),flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(w,todo))
    print("AWB_DONE %s"%out,flush=True)
if __name__=="__main__": main()
