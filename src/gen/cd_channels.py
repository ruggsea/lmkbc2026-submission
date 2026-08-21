import os as _o5
_WORKERS = int(_o5.environ.get('GEN_WORKERS', '6'))  # tunnelled endpoints drop under high concurrency

import os as _o3, sys as _s3
_REPO_ROOT = _o3.path.dirname(_o3.path.dirname(_o3.path.dirname(_o3.path.abspath(__file__))))
if _REPO_ROOT not in _s3.path: _s3.path.insert(0, _REPO_ROOT)
#!/usr/bin/env python3
'''cityOfDeath: generate RAW DRAWS for several independent prompt channels.
Saves draws only; combination rules are swept offline from the json.
Channels differ in framing, not in "ask the model which candidate is right".'''
import argparse, json, os, sys, collections
from concurrent.futures import ThreadPoolExecutor
import urllib.request

URL = os.environ.get('VLLM_URL', 'http://localhost:8010')
MODEL = 'gemma-4-31B'
D = os.path.dirname(os.path.abspath(__file__))
FILES = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}
REL = 'personHasCityOfDeath'


def load(sp):
    return [json.loads(l) for l in open(os.path.join(D, FILES[sp]), encoding='utf-8')]


def gen(prompt, n, temp, stop, mt):
    b = json.dumps({'model': MODEL, 'prompt': prompt, 'n': n, 'temperature': temp,
                    'top_p': 0.95, 'max_tokens': mt, 'stop': stop}).encode()
    rq = urllib.request.Request(URL + '/v1/completions', data=b,
                                headers={'Content-Type': 'application/json'})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(rq, timeout=900) as r:
                return [c['text'] for c in json.load(r)['choices']]
        except Exception:
            if attempt == 2:
                raise
    return []


# ---------------- channel definitions ----------------
# each: (name, prompt_builder(subject) -> str, stop, marker)

SCAFFOLD = ('Person: Albert Einstein\nNationality: German-born, later Swiss and American\nField: theoretical physics\n'
            'Years: 1879-1955\nFinal-years residence: Princeton, New Jersey\nBuried: cremated, ashes scattered near Princeton\nCity of death: Princeton\n\n'
            'Person: Keanu Reeves\nNationality: Canadian\nField: acting\nYears: born 1964, still alive\n'
            'Final-years residence: Los Angeles\nBuried: N/A\nCity of death: N/A\n\n'
            'Person: Marie Curie\nNationality: Polish-French\nField: physics and chemistry\nYears: 1867-1934\n'
            'Final-years residence: Paris; sanatorium in Passy in her last days\nBuried: Paris (Pantheon)\nCity of death: Passy\n\n')

PLAIN = ('Person: Albert Einstein\nCity of death: Princeton\n\n'
         'Person: Marie Curie\nCity of death: Passy\n\n'
         'Person: Keanu Reeves\nCity of death: N/A\n\n')

OBIT = ('Encyclopedia entry.\n\n'
        'Albert Einstein (1879-1955) was a German-born theoretical physicist. '
        'He died on 18 April 1955 in Princeton, New Jersey, United States.\n\n'
        'Marie Curie (1867-1934) was a Polish-French physicist and chemist. '
        'She died on 4 July 1934 in Passy, Haute-Savoie, France.\n\n'
        'Keanu Reeves (born 1964) is a Canadian actor. He is still alive.\n\n')

DEATHREC = ('Death register.\n\n'
            'Name: Albert Einstein | Date of death: 1955-04-18 | Place of death: Princeton\n'
            'Name: Marie Curie | Date of death: 1934-07-04 | Place of death: Passy\n'
            'Name: Keanu Reeves | Date of death: ALIVE | Place of death: N/A\n')


def build(ch, s):
    if ch == 'scaffold':
        return SCAFFOLD + f'Person: {s}\nNationality:', ['\n\n'], 60, 'City of death:'
    if ch == 'plain':
        return PLAIN + f'Person: {s}\nCity of death:', ['\n'], 12, None
    if ch == 'obit':
        return OBIT + f'{s} (', ['\n\n'], 60, ' in '
    if ch == 'deathrec':
        return DEATHREC + f'Name: {s} | Date of death:', ['\n'], 30, 'Place of death:'
    raise ValueError(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val')
    ap.add_argument('--channels', default='scaffold,plain,obit,deathrec')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    rows = [r for r in load(a.split) if r['Relation'] == REL][:a.limit]
    chans = a.channels.split(',')
    result = {}

    def one(job):
        ch, s = job
        prompt, stop, mt, marker = build(ch, s)
        ts = gen(prompt, a.n, a.temp, stop, mt)
        return ch, s, ts

    jobs = [(ch, r['SubjectEntity']) for ch in chans for r in rows]
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for ch, s, ts in ex.map(one, jobs):
            result.setdefault(ch, {})[s] = ts
    json.dump(result, open(os.path.join(D, a.out), 'w'), ensure_ascii=False)
    print('DONE', a.out, {k: len(v) for k, v in result.items()}, flush=True)


main()
