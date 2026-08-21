#!/usr/bin/env python3
"""Gate v4: same three-signal rule, but city votes come from the per-channel extractor
(cd_extract.city_from), which reads all 8 channels instead of the 2 the prefix-regex reached."""
import os as _o
_CH = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'channels')
def _p(name): return _o.path.join(_CH, name)

import json,re,os,collections,sys


from evaluate import normalize_string as Nz
from cd_extract import city_from
ALIVE=re.compile(r'still\s+(living|alive)|\balive\b|\bpresent\b',re.I)
YRS=re.compile(r'Years\s*:\s*(\d{3,4})\s*[-–]\s*(\d{3,4})')
BASE=lambda s: re.sub(r'\s*\([^)]*\)\s*$','',s).strip()
REL='personHasCityOfDeath'
A_MAX,DY_MIN,C_MIN=0.25,0.90,0.30
def signals(split, chdir=None):
    ch=collections.defaultdict(lambda: collections.defaultdict(list))
    import os as _o2
    _base = chdir or _CH
    for f in [_o2.path.join(_base, f'cdch_{split}_n30.json'), _o2.path.join(_base, f'cdg_{split}_n20.json')]:
        if not os.path.exists(f): continue
        for name,subs in json.load(open(f)).items():
            for s,ds in subs.items(): ch[BASE(s)][name]+=[str(x) for x in ds]
    out={}
    for s,chs in ch.items():
        alld=[x for v in chs.values() for x in v]
        if not alld: continue
        al=sum(1 for x in alld if ALIVE.search(x) and not YRS.search(x))/len(alld)
        sc=chs.get('scaffold',[])
        dy=(sum(1 for x in sc if YRS.search(x))/len(sc)) if sc else 0.0
        c=collections.Counter(); raw={}
        for name,ds in chs.items():
            for x in ds:
                v=city_from(name,x)
                if v: c[Nz(v)]+=1; raw.setdefault(Nz(v),v)
        tot=sum(c.values())
        if not tot: continue
        top,n=c.most_common(1)[0]
        out[s]=(al,dy,raw[top],n/tot)
    return out
def apply(rows,split,S=None,chdir=None):
    S=S or signals(split, chdir); out=[]; conv=[]
    for r in rows:
        if r['Relation']==REL and not r['ObjectEntities']:
            s=S.get(BASE(r['SubjectEntity']))
            if s and s[0]<=A_MAX and s[1]>=DY_MIN and s[3]>=C_MIN:
                r=dict(r,ObjectEntities=[s[2]]); conv.append((r['SubjectEntity'],s[2]))
        out.append(r)
    return out,conv
