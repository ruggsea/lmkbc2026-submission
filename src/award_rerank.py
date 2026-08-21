#!/usr/bin/env python3
"""Rerank each awardWonBy list using a second, differently shaped elicitation.

The baseline framing asks the model to continue a year-by-year recipient table for the query award.
Its failure mode is that a confabulated name reads exactly as fluently as a remembered one, and asking
the model to judge its own names does not separate them: discriminative yes/no verification of these
candidates is near chance on this base model.

What does separate them is asking a differently shaped question and checking whether the answer
survives.

  year-slot framing   "who received <award> in <year>?" -- one year at a time instead of a list.
                      A name's ATTESTATION is the best single-year rate at which it appears here.
  cycle consistency   generate from a NAME back to the awards it won and check whether the target
                      award is mentioned. A remembered winner round-trips; a confabulated one usually
                      does not.

Two rules follow, both acting only on rows the baseline framing is unsure about. Row confidence is the
fraction of a row's baseline names that the year-slot framing attests at all:

  PRUNE  rows with rowconf < 0.5: drop baseline names the year-slot framing never produced.
  ADD    rows with rowconf < 0.7: add year-slot candidates whose cycle score is at least 1/3.

Neither rule reads gold, and the thresholds are fixed constants applied identically to every split.

Inputs, all model outputs under channels/:
  award_base_<split>.jsonl   baseline award predictions to rerank
  awyr_<split>_n8.json       year-slot draws, 8 per award-year
  awr_cycle_<split>.json     reverse draws, 6 per candidate name
"""
import argparse
import collections
import json
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SRC))
from evaluate import normalize_string  # noqa: E402

REL = 'awardWonBy'
ROWCONF_PRUNE = 0.5
ROWCONF_ADD = 0.7
ATTEST_MIN = 0.25
CYCLE_MIN = 1.0 / 3.0

_BAD = re.compile(r'^(unknown|n/?a|none|not awarded|no award|award not given|-+|\?+)$', re.I)
_STOP = {'the', 'of', 'for', 'a', 'an', 'and', 'in', 'award', 'prize', 'medal', 'honor',
         'honors', 'award)', '(the'}


def names_in(draw):
    """Split one year-slot completion into candidate person names."""
    t = str(draw).strip().strip('.').strip()
    t = re.split(r'\s*\((?:for|with)\b', t)[0]
    out = []
    for part in re.split(r',| and | & |;', t):
        p = part.strip().strip('.,;').strip()
        if not p or _BAD.match(p) or len(p) < 3 or len(p) > 60:
            continue
        if not re.search(r'[A-Za-z]', p) or re.match(r'^\d', p):
            continue
        out.append(p)
    return out


def attestation(yearslot):
    """award -> {normalised name: best single-year appearance rate}."""
    out = {}
    for award, years in yearslot.items():
        acc = {}
        for draws in years.values():
            counts = collections.Counter()
            total = 0
            for d in draws:
                found = names_in(d)
                if not found:
                    continue
                total += 1
                for nm in found:
                    counts[nm] += 1
            if not total:
                continue
            for nm, k in counts.items():
                key = normalize_string(nm)
                acc[key] = max(acc.get(key, 0.0), k / total)
        out[award] = acc
    return out


def cycle_score(award, draws):
    """Share of reverse draws mentioning enough of the award's distinctive tokens."""
    toks = [t for t in re.findall(r'[a-z0-9]+', award.lower()) if t not in _STOP]
    if not toks or not draws:
        return 0.0
    hits = sum(1 for d in draws
               if sum(1 for t in toks if t in str(d).lower()) / len(toks) >= 0.6)
    return hits / len(draws)


def rerank(base, yearslot, cycle):
    """base: {award: [names]} -> reranked {award: [names]}."""
    att = attestation(yearslot)
    out = {}
    for award, preds in base.items():
        a = att.get(award, {})
        cyc = {nm: cycle_score(award, draws) for nm, draws in cycle.get(award, {}).items()}
        rowconf = (sum(1 for p in preds if a.get(normalize_string(p), 0.0) >= ATTEST_MIN)
                   / len(preds)) if preds else 0.0
        keep = list(preds)
        if rowconf < ROWCONF_PRUNE:
            keep = [p for p in preds if a.get(normalize_string(p), 0.0) > 0]
        if rowconf < ROWCONF_ADD:
            seen = {normalize_string(p) for p in preds} | {normalize_string(p) for p in keep}
            keep = keep + [nm for nm, f in cyc.items()
                           if normalize_string(nm) not in seen and f >= CYCLE_MIN]
        out[award] = keep
    return out


def load(channels, split):
    ch = lambda n: os.path.join(channels, n)  # noqa: E731
    base = {r['SubjectEntity']: list(r['ObjectEntities'])
            for r in map(json.loads, open(ch(f'award_base_{split}.jsonl')))
            if r['Relation'] == REL}
    return base, json.load(open(ch(f'awyr_{split}_n8.json'))), json.load(open(ch(f'awr_cycle_{split}.json')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--channels', default=os.path.join(os.path.dirname(SRC), 'channels'))
    ap.add_argument('--out')
    a = ap.parse_args()
    base, yearslot, cycle = load(a.channels, a.split)
    out = rerank(base, yearslot, cycle)
    print(f'awardWonBy: {sum(1 for k in out if out[k] != base[k])} of {len(out)} rows reranked')
    if a.out:
        with open(a.out, 'w') as f:
            for s, v in out.items():
                f.write(json.dumps({'SubjectEntity': s, 'Relation': REL,
                                    'ObjectEntities': v}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
