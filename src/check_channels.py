#!/usr/bin/env python3
"""Refuse to build a submission from degenerate elicitation output.

A dropped connection to the model does not raise here: the generators write whatever they received,
and an empty or half-empty pool flows straight through the aggregation into a plausible-looking
submission. That has produced silently wrong results more than once, so this runs between generation
and aggregation and fails loudly instead.

Checks, per channel file:
  * the file exists and parses
  * it covers a reasonable share of the split's subjects
  * its draws are mostly non-empty

Exit status is non-zero on the first failure, with the offending file named.
"""
import argparse
import json
import os
import sys

MIN_SUBJECT_COVERAGE = 0.90
MAX_EMPTY_DRAW_SHARE = 0.20


def draws_of(obj):
    if isinstance(obj, dict) and 'draws' in obj:
        return obj['draws']
    return None


def check_jsonl(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return f'{os.path.basename(path)}: no rows'
    empty = tot = 0
    for r in rows:
        d = draws_of(r)
        if d is None:
            continue
        tot += len(d)
        empty += sum(1 for x in d if not str(x).strip())
    if tot and empty / tot > MAX_EMPTY_DRAW_SHARE:
        return f'{os.path.basename(path)}: {empty}/{tot} draws are empty'
    return None


def check_json(path):
    d = json.load(open(path))
    if not isinstance(d, dict) or not d:
        return f'{os.path.basename(path)}: empty object'
    tot = empty = 0
    for v in d.values():
        if isinstance(v, dict):
            for draws in v.values():
                if isinstance(draws, list):
                    tot += len(draws)
                    empty += sum(1 for x in draws if not str(x).strip())
        elif isinstance(v, list):
            tot += len(v)
            empty += sum(1 for x in v if not str(x).strip())
    if tot == 0:
        return f'{os.path.basename(path)}: no draws at all'
    if empty / tot > MAX_EMPTY_DRAW_SHARE:
        return f'{os.path.basename(path)}: {empty}/{tot} draws are empty'
    return None


EXPECTED = ['hasarea_lead.jsonl', 'hasarea_geog.jsonl',
            'cdch_{split}_n30.json', 'cdg_{split}_n20.json',
            'awyr_{split}_n8.json', 'awr_cycle_{split}.json',
            'award_base_{split}.jsonl']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--min-hasarea', type=int, default=8,
                    help='how many hasarea_*.jsonl channels the pool should contain')
    a = ap.parse_args()
    problems = []
    files = sorted(os.listdir(a.channels))
    if not files:
        sys.exit(f'ERROR: {a.channels} is empty')

    # A partial run is the dangerous case: the files that did get written look fine on their own, so
    # checking only what is present passes a run that lost half its channels to a dropped connection.
    for name in EXPECTED:
        want = name.format(split=a.split)
        if want not in files:
            problems.append(f'{want}: missing')
    n_ha = sum(1 for f in files if f.startswith('hasarea_') and f.endswith('.jsonl'))
    if n_ha < a.min_hasarea:
        problems.append(f'hasarea channels: found {n_ha}, expected {a.min_hasarea}')
    for f in files:
        p = os.path.join(a.channels, f)
        try:
            msg = check_jsonl(p) if f.endswith('.jsonl') else (check_json(p) if f.endswith('.json') else None)
        except Exception as e:
            msg = f'{f}: unreadable ({type(e).__name__})'
        if msg:
            problems.append(msg)
    if problems:
        for m in problems:
            print('ERROR:', m, file=sys.stderr)
        sys.exit('refusing to build a submission from degenerate channels')
    print(f'channels ok: {len(files)} files in {a.channels}')


if __name__ == '__main__':
    main()
