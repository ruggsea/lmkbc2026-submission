"""hasArea aggregation retune for gemma-4 (offline, from saved LEAD pools).
g4 recall 0.95 == g3, but cluster@2.5%-vote converts 0.75 vs g3 0.82 -> the aggregator, not the
model, regressed. Sweep: cluster width x aggregator (biggest-cluster median / log-median / trimmed
mean / weighted-cluster), select on OTRAIN pool, report OVAL. Usage:
  python hasarea_agg_sweep.py --train ha_g4_otrain.jsonl --val ha_g4_oval.jsonl
"""
import argparse, json, math, statistics, sys
sys.path.insert(0, ".")
from hasarea_score import parse  # parse(draw, unit_aware) -> value_km2 or None


def load(path):
    out = []
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        gold = d.get("gold")
        try:
            g = float(str(gold).replace(",", ""))
        except (TypeError, ValueError):
            continue
        vals = [parse(t, True) for t in d["draws"]]
        vals = [v for v in vals if v and v > 0]
        out.append((d["subject"], g, vals))
    return out


def within(v, g):
    return v is not None and abs(v - g) / g <= 0.05


def clusters(vals, w):
    vs = sorted(vals)
    cl = []
    for v in vs:
        for c in cl:
            if abs(v - c[0]) / c[0] <= w:
                c.append(v)
                break
        else:
            cl.append([v])
    return cl


def agg(vals, method, w):
    if not vals:
        return None
    if method == "logmedian":
        return math.exp(statistics.median(math.log(v) for v in vals))
    if method == "trimmed":
        vs = sorted(vals)
        k = max(1, len(vs) // 10)
        core = vs[k:-k] or vs
        return math.exp(sum(math.log(v) for v in core) / len(core))
    cl = clusters(vals, w)
    if method == "cluster":       # biggest cluster, median
        return statistics.median(max(cl, key=len))
    if method == "cluster_log":   # biggest cluster, log-mean
        c = max(cl, key=len)
        return math.exp(sum(math.log(v) for v in c) / len(c))
    if method == "cluster2":      # biggest cluster but require >=2, else logmedian fallback
        c = max(cl, key=len)
        if len(c) >= 2:
            return statistics.median(c)
        return math.exp(statistics.median(math.log(v) for v in vals))
    return None


def f1(pool, method, w):
    return sum(within(agg(vals, method, w), g) for _, g, vals in pool) / len(pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    a = ap.parse_args()
    T, V = load(a.train), load(a.val)
    print(f"pools: train {len(T)} / val {len(V)}")
    methods = ["cluster", "cluster_log", "cluster2", "logmedian", "trimmed"]
    widths = [0.01, 0.02, 0.025, 0.03, 0.05, 0.075]
    best = None
    print("method       width | train  val")
    for m in methods:
        for w in (widths if m.startswith("cluster") else [0.025]):
            tf, vf = f1(T, m, w), f1(V, m, w)
            print(f"  {m:10s} {w:.3f} | {tf:.3f}  {vf:.3f}")
            if best is None or tf > best[0]:
                best = (tf, m, w)
    tf, m, w = best
    print(f"\ntrain-best: {m} w={w}  train={tf:.3f}  OVAL={f1(V,m,w):.3f}")
    # oracle
    orc = sum(any(within(v, g) for v in vals) for _, g, vals in V) / len(V)
    print(f"oval oracle (any draw within 5%): {orc:.3f}")


if __name__ == "__main__":
    main()
