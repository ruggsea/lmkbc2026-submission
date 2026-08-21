#!/usr/bin/env python3
"""Regenerate the submitted predictions from checked-in model outputs. No manual corrections.

Three stages run on top of the base predictions, each one a rule applied to raw elicitation output:

  1. hasArea      pool eight replicate elicitation channels and take a stabbing-interval estimate
                  (src/rebuild_hasarea.py)
  2. cityOfDeath  convert abstained rows only where three independent signals agree: the subject is
                  not described as living, a scaffold framing supplies death years, and one city
                  carries enough of the channel vote (src/cd_gate4.py)
  3. awardWonBy   prune baseline names unattested in a forced year-slot framing, and add year-slot
                  candidates that survive a cycle-consistency check (src/award_rerank.py)

Every input is a model output checked into channels/. Nothing here consults gold, and no per-row
value is hardcoded.
"""
import argparse, json, os, subprocess, sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(ROOT, 'predictions', 'predictions.jsonl'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--channels', default=os.path.join(ROOT, 'channels'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    tmp = a.out + '.stage1'
    subprocess.check_call([sys.executable, os.path.join(SRC, 'rebuild_hasarea.py'),
                           '--channels', a.channels,
                           '--base', a.base, '--out', tmp])

    rows = [json.loads(l) for l in open(tmp)]

    import cd_gate4 as G4
    rows, conv = G4.apply(rows, a.split, chdir=a.channels)
    print(f'cityOfDeath: {len(conv)} abstained rows converted by the three-signal gate')

    import award_rerank as AR
    abase, ayearslot, acycle = AR.load(a.channels, a.split)
    pm = AR.rerank(abase, ayearslot, acycle)
    n = 0
    for r in rows:
        if r['Relation'] == 'awardWonBy' and r['SubjectEntity'] in pm:
            if list(r['ObjectEntities']) != list(pm[r['SubjectEntity']]):
                n += 1
            r['ObjectEntities'] = list(pm[r['SubjectEntity']])
    print(f'awardWonBy: {n} rows changed by the prune+cycle rule')

    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.remove(tmp)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
