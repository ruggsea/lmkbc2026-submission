import os as _o5
_WORKERS = int(_o5.environ.get('GEN_WORKERS', '6'))  # tunnelled endpoints drop under high concurrency

import os as _o3, sys as _s3
_REPO_ROOT = _o3.path.dirname(_o3.path.dirname(_o3.path.dirname(_o3.path.abspath(__file__))))
if _REPO_ROOT not in _s3.path: _s3.path.insert(0, _REPO_ROOT)
#!/usr/bin/env python3
"""Forced per-year slot generation for awardWonBy.

The award rows that score well are exactly the ones whose emission depth matches gold size (Nobel
115 preds / 121 gold -> f1 0.941; Turing 70/81 -> 0.914). The ones that fail are starved (Pulitzer
Biography 2 preds / 118 gold -> 0.033). So the problem on those rows is depth, and depth from the
existing pool is dead because free-form sampling keeps returning the same few famous winners.

An annual award has a natural index that free sampling does not exploit: the year. Asking
"<award>, <year>:" once per year FORCES coverage of the whole range instead of sampling it -- a
year the model would never volunteer still gets asked. That is the difference from awky_*, which
sampled free-form "year: name" pairs and hoped the tail came up.

Sweeps a wide range and lets consensus decide which years are real: years outside the award's actual
lifetime produce incoherent, low-agreement draws and are filtered on plurality share, so no separate
year-range query is needed.

Writes awyr_{split}_n{n}.json: {award: {year: [draws]}}.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)
from hade_common import gen

REL = 'awardWonBy'
SPLITF = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}
Y0, Y1 = 1901, 2024

# Demos span award kinds (science, film, literature, music) and deliberately include a year with a
# multi-person winner, so the format does not imply exactly one name.
DEMOS = [
    ('Academy Award for Best Director', 1994, 'Steven Spielberg'),
    ('Booker Prize', 1981, 'Salman Rushdie'),
    ('Abel Prize', 2004, 'Michael Atiyah, Isadore Singer'),
    ('Pritzker Architecture Prize', 1989, 'Frank Gehry'),
    ('Hugo Award for Best Novel', 1966, 'Frank Herbert'),
    ("Palme d'Or", 1994, 'Quentin Tarantino'),
]
HDR = '\n\n'.join(f'Award: {a}\nYear: {y}\nWinner: {w}' for a, y, w in DEMOS) + '\n\n'


def subjects(sp):
    return [r['SubjectEntity'] for r in map(json.loads, open(SPLITF[sp])) if r['Relation'] == REL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temp', type=float, default=0.7)
    a = ap.parse_args()

    subs = subjects(a.split)
    allsubs = {x for sp in SPLITF for x in subjects(sp)}
    leak = sorted({d[0] for d in DEMOS if d[0] in allsubs})
    assert not leak, f'demo leak (checked against all splits): {leak}'
    jobs = [(s, y) for s in subs for y in range(Y0, Y1 + 1)]
    print(f'{a.split}: {len(subs)} awards x {Y1 - Y0 + 1} years = {len(jobs)} prompts', flush=True)

    def run(j):
        s, y = j
        txt = gen(HDR + f'Award: {s}\nYear: {y}\nWinner:', a.n, a.temp, 24,
                  ['\n', 'Award:'])
        return s, y, txt

    out = {}
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for i, (s, y, txt) in enumerate(ex.map(run, jobs)):
            out.setdefault(s, {})[y] = txt
            if i % 250 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    tmp = f'awyr_{a.split}_n{a.n}.tmp'
    json.dump(out, open(tmp, 'w'), ensure_ascii=False)
    os.replace(tmp, f'awyr_{a.split}_n{a.n}.json')
    print(f'wrote awyr_{a.split}_n{a.n}.json', flush=True)


if __name__ == '__main__':
    main()
