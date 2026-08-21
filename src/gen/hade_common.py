import os as _o3, sys as _s3
_REPO_ROOT = _o3.path.dirname(_o3.path.dirname(_o3.path.dirname(_o3.path.abspath(__file__))))
if _REPO_ROOT not in _s3.path: _s3.path.insert(0, _REPO_ROOT)
#!/usr/bin/env python3
"""Shared helpers for the hade_* (derived-quantity hasArea) channels.
Completion-style calls against the gemma-4-31B BASE model, same pattern as cg_sense3.py.
"""
import json, time, os, re, sys
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
URL = os.environ.get('VLLM_URL', 'http://localhost:8011')  # 8010 is used by other agents this session
MODEL = 'gemma-4-31B'
REL = 'hasArea'


def load_split(sp):
    f = {'train': 'off_train.jsonl', 'val': 'off_val.jsonl', 'test': 'off_test.jsonl'}[sp]
    rows = [json.loads(l) for l in open(os.path.join(D, f))]
    return [r for r in rows if r['Relation'] == REL]


def assert_no_leak(demo_names):
    """Same guard as cg_sense3.py: demo entities must never be a subject in off_train/val/test."""
    subs = set()
    for sp in ('train', 'val', 'test'):
        subs |= {r['SubjectEntity'].lower() for r in load_split(sp)}
    leak = [d for d in demo_names if d.lower() in subs]
    assert not leak, f'demo leak: {leak}'


def gen(prompt, n, temp, max_tokens, stop):
    b = json.dumps({'model': MODEL, 'prompt': prompt, 'n': n, 'temperature': temp,
                    'top_p': 0.95, 'max_tokens': max_tokens, 'stop': stop}).encode()
    rq = urllib.request.Request(URL + '/v1/completions', data=b,
                                 headers={'Content-Type': 'application/json'})
    err = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(rq, timeout=1800) as r:
                return [ch['text'] for ch in json.load(r)['choices']]
        except Exception as e:
            # See cdg_gen.py: returning [] here turns a network blip into silent data loss.
            err = e
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f'generation failed after retries: {err}')


# ---------- numeric parsing ----------
# number with an optional word/letter magnitude suffix directly after it ("9.6 million", "9.6M",
# "19,300"). The \b after the suffix group keeps "587 km" from matching the "k" in "km".
_QTY = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(billion|bn|million|mn|thousand|k)?\b", re.I)
_MULT = {'b': 1e9, 'm': 1e6, 't': 1e3, 'k': 1e3}


def parse_qty(text):
    """First number in text, applying a million/billion/thousand suffix if present. None if no
    digit found or the value is <= 0 (covers 'N/A', 'unknown', empty draws)."""
    if not text:
        return None
    if re.search(r'\bn\s*/\s*a\b', text, re.I) and not re.search(r'\d', text):
        return None
    m = _QTY.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(',', ''))
    except ValueError:
        return None
    if m.group(2):
        v *= _MULT[m.group(2)[0].lower()]
    return v if v > 0 else None


def golds(r):
    """All accepted gold numeric values for a row (handles alias-expanded multi-value rows)."""
    o = []
    for e in r['ObjectEntities']:
        for a in (e if isinstance(e, list) else [e]):
            try:
                o.append(float(str(a).replace(',', '')))
            except (ValueError, TypeError):
                pass
    return o
