#!/usr/bin/env python3
"""Rebuild the hasArea block by pooling replicate elicitation channels.

hasArea is elicited several times under two prompt framings -- a lead-sentence framing ("X is a ...
It has an area of N km2") and a geographic-entity framing -- and each framing is repeated as
independent replicate rounds. Pooling the draws from several rounds and taking a single robust point
estimate is worth more than any one round, because the model's per-round spread is much wider than
the 5% relative tolerance the task scores against.

The estimator is `est_stab_mid`: it places a +-5% relative interval around every pooled draw, finds
the region of the number line covered by the most intervals, and returns the midpoint of that region.
Compared with a plain median it is insensitive to a minority of wildly wrong draws, which is the
dominant failure mode here.

Channels pooled (produced by src/gen/hasarea_channels.py, one file per framing per round):
    lead, geog, leadR5, geogR5, leadR6, leadR7, leadR8, geogR8

Values are formatted with '%.6g', matching the submitted file.
"""
import argparse, json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hasarea_score import parse

CHANNELS = ['lead', 'geog', 'leadR5', 'geogR5', 'leadR6', 'leadR7', 'leadR8', 'geogR8']
TOL = 0.05


def est_stab_mid(vals, tol=TOL):
    """Stabbing-interval estimator, midpoint variant.

    Place a +-tol relative interval around every pooled draw and sweep the number line to find the
    regions covered by the maximal number of intervals. Among those maximal regions take the one with
    the largest hi/lo ratio (the widest agreement band in relative terms) and return its geometric
    midpoint. Falls back to the median when no region is found.
    """
    if not vals:
        return None
    ev = []
    for u in vals:
        ev.append((u * (1 - tol), 1))
        ev.append((u * (1 + tol), -1))
    ev.sort(key=lambda e: (e[0], -e[1]))
    best_n = 0
    cur = 0
    regions = []
    start = None
    for x, d in ev:
        if d == 1:
            cur += 1
            if cur > best_n:
                best_n = cur
                regions = []
                start = x
            elif cur == best_n:
                start = x
        else:
            if cur == best_n and start is not None:
                regions.append((start, x))
                start = None
            cur -= 1
    if not regions:
        return statistics.median(vals)
    lo, hi = max(regions, key=lambda r: r[1] / r[0])
    return math.sqrt(lo * hi)


def load_channel(chan_dir, chan):
    """subject -> list of parsed positive areas from one channel file"""
    path = os.path.join(chan_dir, f'hasarea_{chan}.jsonl')
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        o = json.loads(line)
        vals = [v for v in (parse(t, True) for t in o['draws']) if v and v > 0]
        if vals:
            out[o['SubjectEntity']] = vals
    return out


def discover(chan_dir):
    """Every hasarea_*.jsonl channel present, in a stable order.

    The pool is whatever elicitation channels the run produced; nothing is selected here. Using the
    directory contents rather than a fixed list means a regenerated pool is used exactly as generated.
    """
    import glob
    names = []
    for p in sorted(glob.glob(os.path.join(chan_dir, 'hasarea_*.jsonl'))):
        names.append(os.path.basename(p)[len('hasarea_'):-len('.jsonl')])
    return names


def bridge(base_subjects, pool_subjects):
    """Map base-file subject names onto channel subject names where they differ.

    Some prediction files carry short subject strings where the official split carries a
    disambiguated one ("Mainland" against "Mainland, Orkney"). Without a bridge those rows silently
    keep their base value and the relation is scored with the pooling partly disabled.

    Only unambiguous matches are accepted: one side must contain the other, case-insensitively, and
    the candidate must be unique. Anything else is left alone rather than guessed at.
    """
    out = {}
    pool = list(pool_subjects)
    for b in base_subjects:
        if b in pool_subjects:
            continue
        bl = b.casefold()
        cand = [p for p in pool if bl in p.casefold() or p.casefold() in bl]
        if len(cand) == 1:
            out[b] = cand[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', required=True, help='directory holding hasarea_<channel>.jsonl')
    ap.add_argument('--base', required=True, help='predictions file supplying rows for empty pools')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    chans = discover(a.channels) or CHANNELS
    pools = [load_channel(a.channels, c) for c in chans]
    missing = [c for c, p in zip(chans, pools) if not p]
    if missing:
        sys.exit(f'ERROR: no draws parsed for channels {missing} in {a.channels} -- refusing to '
                 f'write a silently degraded block')

    rows = [json.loads(l) for l in open(a.base)]
    pool_subjects = set()
    for p in pools:
        pool_subjects |= set(p)
    names = bridge([r['SubjectEntity'] for r in rows if r['Relation'] == 'hasArea'], pool_subjects)
    if names:
        print(f'hasArea: bridged {len(names)} subject names that differ between the base file and the '
              f'channels, e.g. {list(names.items())[0]}')
    n_pooled = n_empty = 0
    for r in rows:
        if r['Relation'] != 'hasArea':
            continue
        key = names.get(r['SubjectEntity'], r['SubjectEntity'])
        vals = []
        for p in pools:
            vals += p.get(key, [])
        if not vals:
            n_empty += 1
            continue
        e = est_stab_mid(vals)
        if e is not None:
            r['ObjectEntities'] = ['%.6g' % e]
            n_pooled += 1
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'hasArea: {n_pooled} rows estimated from {len(chans)} pooled channels, '
          f'{n_empty} left at base value -> {a.out}')


if __name__ == '__main__':
    main()
