#!/usr/bin/env python3
"""hasArea pool generator v2: bigger n + two new elicitation surfaces (wd, atlas).
Usage: python3 ha_gen2.py --split val --chan wd --n 60
Writes ha2_{split}_{chan}_n{n}.jsonl  (same schema as ha_*.jsonl)
"""
import os as _o5
_WORKERS = int(_o5.environ.get('GEN_WORKERS', '6'))  # tunnelled endpoints drop under high concurrency

import time
import argparse, json, math, os, random, sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request

URL = os.environ.get('VLLM_URL', 'http://localhost:8010'); MODEL = 'gemma-4-31B'
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
# repo layout: split files under data/, demo bank under data/special/.
# DATA_DIR overrides both; keeps the original single-directory behaviour when unset.
DATA = os.environ.get('DATA_DIR', os.path.join(D, '..', '..', 'data'))
def _d(name):
    for c in (os.path.join(DATA, name), os.path.join(DATA, 'special', name), os.path.join(D, name)):
        if os.path.exists(c): return c
    return os.path.join(DATA, name)
REL = 'hasArea'
K = 100
TYPE_BLURB = {"country": "is a country", "island": "is an island", "lake": "is a lake",
              "big city": "is a city", "city": "is a city", "mountain range": "is a mountain range"}

GEOG = """Geography of Tasmania
Tasmania is an island state of Australia south of the mainland. Total land area: 68,401 km2.

Geography of Sardinia
Sardinia is a large island in the Mediterranean west of Italy. Total land area: 24,090 km2.

Geography of Hokkaido
Hokkaido is the northernmost main island of Japan. Total land area: 83,424 km2.

"""
INFOBOX = """{{Infobox place
| name = Tasmania
| area_km2 = 68401
}}

{{Infobox place
| name = Sardinia
| area_km2 = 24090
}}

{{Infobox place
| name = Hokkaido
| area_km2 = 83424
}}

"""


def load(sp):
    f = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}[sp]
    return [json.loads(l) for l in open(_d(f))]


def fmt_area(v):
    if v >= 1e6: return f"{v:.0e}".replace("e+0", "e+")
    if v >= 1000: return str(int(round(v)))
    if v >= 10: return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fmt_comma(v):
    if v >= 100: return f"{int(round(v)):,}"
    if v >= 10: return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def load_famous():
    rows = []
    for line in open(_d('hasarea_famous.jsonl'), encoding='utf-8'):
        d = json.loads(line)
        g = d['ObjectEntities']
        while isinstance(g, list): g = g[0] if g else None
        try: area = float(str(g).replace(',', ''))
        except (TypeError, ValueError): continue
        if area <= 0: continue
        al = frozenset(a.lower() for a in d.get('_aliases', [d['SubjectEntity']]))
        rows.append((d['SubjectEntity'], area, d.get('_class', ''), al))
    return rows


def logspread(subject, famous, k, seed=0):
    ex = subject.strip().lower()
    pool = [x for x in famous if ex not in x[3]]
    if len(pool) >= k:
        by_log = sorted(pool, key=lambda x: math.log10(x[1]))
        idxs = [int(i * (len(by_log) - 1) / (k - 1)) for i in range(k)]
        chosen = [by_log[i] for i in idxs]
    else:
        chosen = pool
    rng = random.Random(seed ^ (hash(subject) & 0xFFFFFFFF)); rng.shuffle(chosen)
    return chosen


def lead_prompt(s, fam, seed=0):
    hdr = ''.join(f"{n} {TYPE_BLURB.get(c,'is a geographic entity')}. It has an area of {fmt_area(a)} km2.\n\n"
                  for n, a, c, al in logspread(s, fam, K, seed))
    return hdr + f"{s} is"


# --- NEW CHANNEL 1: Wikidata statement. The gold IS the Wikidata `area` (P2046) claim, so a surface
# that names the property targets the gold's scope convention (total incl. water, lake incl. swamps)
# rather than the prose convention (land area, lake proper).
def wd_prompt(s, fam, seed=0):
    hdr = ''.join(f"{n}\n  area (P2046): {fmt_comma(a)} square kilometre\n\n"
                  for n, a, c, al in logspread(s, fam, 60, seed + 7))
    return hdr + f"{s}\n  area (P2046): "


# --- NEW CHANNEL 2: gazetteer table row (pipe-delimited), a surface form unlike prose or wikitext.
def atlas_prompt(s, fam, seed=0):
    hdr = "NAME | TYPE | AREA_KM2\n" + ''.join(
        f"{n} | {c or 'geographic entity'} | {fmt_comma(a)}\n"
        for n, a, c, al in logspread(s, fam, 60, seed + 13))
    return hdr + f"{s} |"


PROMPTS = {
    'lead':    lead_prompt,
    'geog':    lambda s, fam, seed=0: GEOG + f"Geography of {s}\n{s} is",
    'infobox': lambda s, fam, seed=0: INFOBOX + "{{Infobox place\n| name = " + s + "\n| area_km2 = ",
    'wd':      wd_prompt,
    'atlas':   atlas_prompt,
}
STOPS = {'lead': ['\n\n'], 'geog': ['\n\n', '\nGeography of'], 'infobox': ['\n', '}}'],
         'wd': ['\n\n', '\n'], 'atlas': ['\n']}
MT = {'lead': 64, 'geog': 96, 'infobox': 16, 'wd': 24, 'atlas': 32}


def gen(prompt, n, temp, stop, mt):
    b = json.dumps({'model': MODEL, 'prompt': prompt, 'n': n, 'temperature': temp,
                    'top_p': 0.95, 'max_tokens': mt, 'stop': stop}).encode()
    rq = urllib.request.Request(URL + '/v1/completions', data=b, headers={'Content-Type': 'application/json'})
    # The endpoint may be a tunnelled forward that is rebuilt periodically. Retry across a
    # rebuild rather than losing the channel, and raise rather than return nothing.
    err = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(rq, timeout=3600) as r:
                return [c['text'] for c in json.load(r)['choices']]
        except Exception as e:
            err = e
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f'generation failed after retries: {err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True); ap.add_argument('--chan', required=True)
    ap.add_argument('--n', type=int, default=60); ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    rows = [r for r in load(a.split) if r['Relation'] == REL]
    fam = load_famous()
    pf, stop, mt = PROMPTS[a.chan], STOPS[a.chan], MT[a.chan]

    def one(r):
        s = r['SubjectEntity']
        return {'SubjectEntity': s, 'draws': gen(pf(s, fam, a.seed), a.n, a.temp, stop, mt)}
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        out = list(ex.map(one, rows))
    p = os.path.join(D, f'ha2_{a.split}_{a.chan}{a.tag}_n{a.n}.jsonl')
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + '\n')
    os.replace(tmp, p)
    print(f'GEN {a.split} {a.chan} n={a.n} -> {p} ({len(out)} rows)', flush=True)


main()
