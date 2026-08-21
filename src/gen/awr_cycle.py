import os as _o5
_WORKERS = int(_o5.environ.get('GEN_WORKERS', '6'))  # tunnelled endpoints drop under high concurrency

import os as _o3, sys as _s3
_REPO_ROOT = _o3.path.dirname(_o3.path.dirname(_o3.path.dirname(_o3.path.abspath(__file__))))
if _REPO_ROOT not in _s3.path: _s3.path.insert(0, _REPO_ROOT)
#!/usr/bin/env python3
"""Cycle consistency: for every (award, name) pair, generate NAME -> awards and store the raw
draws. A remembered winner should map back to the award; a confabulated one should not, because
the name alone does not carry the award context that produced it.

Writes awr_cycle_{split}.json: {award: {name: [draws]}}.
"""
import argparse, json, os, sys, time, collections
import urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)
from evaluate import normalize_string
REL = 'awardWonBy'
SPLITS = ['train', 'val', 'test']
_SPLITF = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}


def load_gold(sp):
    """Rows for one split. Used only by the demonstration-leak guard below."""
    import os as _o4
    p = _o4.path.join(_o4.path.dirname(_o4.path.abspath(__file__)), _SPLITF[sp])
    if not _o4.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]

URL = os.environ.get('VLLM_URL', 'http://localhost:8010')
MODEL = 'gemma-4-31B'

# Demo names must not collide with any gold object; demo award mentions must not be split
# subjects (asserted). Mix of people and an organization, each with several honors.
DEMOS = [
    ('Salman Rushdie', 'Booker Prize, Best of the Booker, Golden PEN Award'),
    ('Renzo Piano', 'Pritzker Architecture Prize, AIA Gold Medal, RIBA Royal Gold Medal'),
    ('Damien Chazelle', 'Academy Award for Best Director, Golden Globe Award for Best Director'),
    ('Apple Inc.', 'Red Dot Design Award, National Design Award'),
]
HDR = '\n\n'.join(f'Name: {n}\nAwards and honors: {a}' for n, a in DEMOS) + '\n\n'


def leak_checks():
    subs = {r['SubjectEntity'] for sp in SPLITS for r in load_gold(sp) if r['Relation'] == REL}
    for _, awards in DEMOS:
        for a in awards.split(', '):
            for s in subs:
                assert s not in a and a not in s, f'demo award leak: {a} vs {s}'
    goldnames = set()
    for sp in SPLITS:
        for r in load_gold(sp):
            if r['Relation'] != REL:
                continue
            for alts in r['ObjectEntities']:
                for x in (alts if isinstance(alts, list) else [alts]):
                    goldnames.add(normalize_string(x))
    leak = [n for n, _ in DEMOS if normalize_string(n) in goldnames]
    assert not leak, f'demo name leak into gold objects: {leak}'


def gen(name, n=6, temp=0.8):
    prompt = HDR + f'Name: {name}\nAwards and honors:'
    b = json.dumps({'model': MODEL, 'prompt': prompt, 'n': n, 'temperature': temp,
                    'top_p': 0.95, 'max_tokens': 64, 'stop': ['\nName:', '\n\n']}).encode()
    rq = urllib.request.Request(URL + '/v1/completions', data=b,
                                headers={'Content-Type': 'application/json'})
    err = None
    for att in range(10):
        time.sleep(min(att * 6, 30) if att else 0)
        try:
            with urllib.request.urlopen(rq, timeout=1800) as r:
                return [ch['text'] for ch in json.load(r)['choices']]
        except Exception as e:
            err = e
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    a = ap.parse_args()
    leak_checks()

    ver = json.load(open(f'awr_verify_{a.split}.json'))  # same name universe
    jobs = [(s, nm) for s, names in ver.items() for nm in names]
    # names can repeat across awards; generate once per unique name
    uniq = {}
    for s, nm in jobs:
        uniq.setdefault(nm, None)
    names = list(uniq)
    print(f'{a.split}: {len(jobs)} pairs, {len(names)} unique names', flush=True)

    draws = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for i, (nm, out) in enumerate(zip(names, ex.map(gen, names))):
            draws[nm] = out
            if i % 100 == 0:
                print(f'  {i}/{len(names)}', flush=True)
    result = {s: {nm: draws[nm] for nm in namesd} for s, namesd in ver.items()}
    tmp = f'awr_cycle_{a.split}.tmp'
    json.dump(result, open(tmp, 'w'), ensure_ascii=False)
    os.replace(tmp, f'awr_cycle_{a.split}.json')
    print(f'wrote awr_cycle_{a.split}.json', flush=True)


if __name__ == '__main__':
    main()
