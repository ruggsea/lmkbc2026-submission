import os as _o5
_WORKERS = int(_o5.environ.get('GEN_WORKERS', '6'))  # tunnelled endpoints drop under high concurrency

import os as _o3, sys as _s3
_REPO_ROOT = _o3.path.dirname(_o3.path.dirname(_o3.path.dirname(_o3.path.abspath(__file__))))
if _REPO_ROOT not in _s3.path: _s3.path.insert(0, _REPO_ROOT)
#!/usr/bin/env python3
'''cityOfDeath: generate raw draws for NEW generation framings (burial, hospital/institution,
date-anchored, small-place anti-default). Diversity comes from prompt FRAMING, not aggregation --
each framing is closed-book, base-completion style, demos hand-written (none are TRAIN/VAL/TEST
subjects for personHasCityOfDeath, checked at run time).'''
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request

URL = os.environ.get('VLLM_URL', 'http://localhost:8011')
MODEL = 'gemma-4-31B'
D = os.path.dirname(os.path.abspath(__file__))
FILES = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}
REL = 'personHasCityOfDeath'

# ---------------- framings ----------------
# each demo bank ends right where the model must produce the target field, so
# n draws are cheap (short max_tokens, stop at newline).

BURIAL = ('Burial and death register.\n\n'
    'Person: Napoleon Bonaparte\nBuried at: Paris, Les Invalides (after 1840 repatriation)\n'
    'Died at (city): Longwood, Saint Helena\n\n'
    'Person: Franz Kafka\nBuried at: Prague, New Jewish Cemetery\n'
    'Died at (city): Kierling\n\n'
    'Person: Marie Curie\nBuried at: Paris, Pantheon\n'
    'Died at (city): Passy\n\n'
    'Person: Elvis Presley\nBuried at: Memphis, Graceland\n'
    'Died at (city): Memphis\n\n'
    'Person: Keanu Reeves\nBuried at: N/A, still alive\n'
    'Died at (city): N/A\n\n')

HOSPITAL = ('Admission and death record.\n\n'
    'Person: Albert Einstein\nAdmitted to (institution): Princeton Hospital, Princeton, New Jersey\n'
    'City of death (municipality of that institution): Princeton\n\n'
    'Person: Winston Churchill\nAdmitted to (institution): his home, Hyde Park Gate, London\n'
    'City of death (municipality of that institution): London\n\n'
    'Person: Franz Kafka\nAdmitted to (institution): Hoffmann Sanatorium, Kierling, near Vienna\n'
    'City of death (municipality of that institution): Kierling\n\n'
    'Person: Napoleon Bonaparte\nAdmitted to (institution): Longwood House, Saint Helena (exile residence)\n'
    'City of death (municipality of that institution): Longwood\n\n'
    'Person: Keanu Reeves\nAdmitted to (institution): N/A, still alive\n'
    'City of death (municipality of that institution): N/A\n\n')

DATEANCHOR = ('Death chronology.\n\n'
    'Person: Wolfgang Amadeus Mozart\nYear of death: 1791\nCity of death (as of the year of death): Vienna\n\n'
    'Person: Leon Trotsky\nYear of death: 1940\nCity of death (as of the year of death): Mexico City\n\n'
    'Person: Napoleon Bonaparte\nYear of death: 1821\nCity of death (as of the year of death): Longwood\n\n'
    'Person: Freddie Mercury\nYear of death: 1991\nCity of death (as of the year of death): London\n\n'
    'Person: Keanu Reeves\nYear of death: N/A, still alive\nCity of death (as of the year of death): N/A\n\n')

# deliberately NO capitals / famous metros in the demo bank -- pushes against the
# big-city default failure mode (e.g. "London" 25/25 for someone who died elsewhere).
SMALLPLACE = (
    'Person: Franz Kafka\nCity of death: Kierling\n\n'
    'Person: Vincent van Gogh\nCity of death: Auvers-sur-Oise\n\n'
    'Person: Napoleon Bonaparte\nCity of death: Longwood\n\n'
    'Person: Robert Louis Stevenson\nCity of death: Vailima\n\n'
    'Person: Katherine Mansfield\nCity of death: Fontainebleau\n\n'
    'Person: Keanu Reeves\nCity of death: N/A\n\n')

FRAMINGS = {
    'burial':     (BURIAL,     'Buried at:',       'Died at (city):'),
    'hospital':   (HOSPITAL,   'Admitted to (institution):', 'City of death (municipality of that institution):'),
    'dateanchor': (DATEANCHOR, 'Year of death:',    'City of death (as of the year of death):'),
    'smallplace': (SMALLPLACE, None,                'City of death:'),
}


def load(sp):
    return [json.loads(l) for l in open(os.path.join(D, FILES[sp]), encoding='utf-8')]


def build(fr, s):
    bank, mid, tail = FRAMINGS[fr]
    if fr == 'burial':
        return bank + f'Person: {s}\nBuried at:', ['\n\n'], 60
    if fr == 'hospital':
        return bank + f'Person: {s}\nAdmitted to (institution):', ['\n\n'], 60
    if fr == 'dateanchor':
        return bank + f'Person: {s}\nYear of death:', ['\n\n'], 40
    if fr == 'smallplace':
        return bank + f'Person: {s}\nCity of death:', ['\n'], 16
    raise ValueError(fr)


def gen(prompt, n, temp, stop, mt):
    b = json.dumps({'model': MODEL, 'prompt': prompt, 'n': n, 'temperature': temp,
                    'top_p': 0.95, 'max_tokens': mt, 'stop': stop}).encode()
    rq = urllib.request.Request(URL + '/v1/completions', data=b,
                                headers={'Content-Type': 'application/json'})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(rq, timeout=900) as r:
                return [c['text'] for c in json.load(r)['choices']]
        except Exception as e:
            # A dropped connection must not look like "the model returned nothing": an empty pool
            # flows through aggregation into a plausible but wrong submission. Back off, then fail.
            err = e
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f'generation failed after retries: {err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--framings', default='burial,hospital,dateanchor,smallplace')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    rows = [r for r in load(a.split) if r['Relation'] == REL]
    subs = {r['SubjectEntity'].lower() for r in rows}
    all_demo_names = set()
    for bank, _, _ in FRAMINGS.values():
        for line in bank.split('\n'):
            if line.startswith('Person: '):
                all_demo_names.add(line[len('Person: '):].lower())
    leak = [n for n in all_demo_names if n in subs]
    assert not leak, f'demo leak into {a.split}: {leak}'

    frs = a.framings.split(',')
    result = {}

    def one(job):
        fr, s = job
        prompt, stop, mt = build(fr, s)
        ts = gen(prompt, a.n, a.temp, stop, mt)
        return fr, s, ts

    jobs = [(fr, r['SubjectEntity']) for fr in frs for r in rows]
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for fr, s, ts in ex.map(one, jobs):
            result.setdefault(fr, {})[s] = ts

    p = os.path.join(D, a.out)
    tmp = p + '.tmp'
    json.dump(result, open(tmp, 'w'), ensure_ascii=False)
    os.replace(tmp, p)
    print('DONE', a.out, {k: len(v) for k, v in result.items()}, flush=True)


main()
