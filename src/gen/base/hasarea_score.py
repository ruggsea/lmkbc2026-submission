"""Offline scorer for hasArea base pools. Reusable pools (capacity/award lesson).
Parses area value + UNIT from each draw; converts to km2. Compares:
  - naive parse (number only, assume km2) vs unit-aware (trust+convert stated unit)
Reports per pool: parse-fail rate, unit-mix, recall(best-of-N within 5%), and realized tol5-F1 under
several consensus votes (cluster@2.5% linear, log-median, mode@2-sig). Also a unit-oracle recall (best
recoverable if we knew the true unit) to size the unit-confusion problem.
Usage: python hasarea_score.py <pool1.jsonl> [pool2.jsonl ...]
"""
import json, sys, re, statistics, math

# unit -> km2 multiplier
UNITS = [
    (r"km2|km²|km\^2|square kilomet|sq\.?\s*km", 1.0),
    (r"sq\.?\s*mi|square mile|mi2|mi²", 2.589988),
    (r"hectare|\bha\b", 0.01),
    (r"acre", 0.00404686),
    (r"m2|m²|square met|sq\.?\s*m(?!i)", 1e-6),
]
# exponent branch is load-bearing: gemma-4 emits "1.1e+6 km2" for large areas (194 train / 0 val /
# 288 test draws). Without it "1e+6" parsed as 1.0 -- invisible on val, live on the split we ship to.
NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")
# European full-number format "103.029,251 km2" (dot=thousands, comma=decimal) -- rare but confirmed
# live in g4_hasarea_atlas/scope_icl draws (0 instances in the shipped K100/geog_entry dumps as of the
# 31bj#4 parse audit, but the general parser is shared by ~67 tools so the branch is defensive).
EURO_NUM = re.compile(r"-?\d{1,3}(?:\.\d{3})+,\d+")
# ordinal-ranking numbers ("the 33rd largest island") must not be mistaken for the area figure that
# follows later in the same draw -- 31bj#4 parse audit found NUM.search grabbing "33" instead of the
# "11,067" in "the 33rd largest island ... Total land area: 11,067 km2." (live in g4_hasarea_geog_entry,
# the shipped GEOG_DIR overlay; 9 test / 9 train / 2 val draws -- currently absorbed by cluster-median
# aggregation with 0 shipped-subject impact, but a single-draw consumer would be hit directly).
ORDINAL_SUFFIX = re.compile(r"^(?:st|nd|rd|th)\b", re.I)

def parse(draw, unit_aware):
    """Return km2 value from a draw, or None. unit_aware: trust+convert the stated unit."""
    t = draw.replace("\n", " ")
    em = EURO_NUM.search(t)
    if em:
        try:
            v = float(em.group(0).replace(".", "").replace(",", "."))
        except ValueError:
            return None
        m = em  # for the unit-window scan below
    else:
        m = None
        for cand in NUM.finditer(t):
            if ORDINAL_SUFFIX.match(t[cand.end():cand.end() + 3]):
                continue  # "33rd" -- skip, keep scanning for the real quantity
            m = cand
            break
        if not m: return None
        try: v = float(m.group(0).replace(",", ""))
        except ValueError: return None
    if v <= 0: return None
    if unit_aware:
        tail = t[m.end():m.end()+18].lower()
        for pat, mult in UNITS:
            if re.search(pat, tail):
                return v * mult
        # also scan a small window before the number (e.g. "area of ... ")
    return v

def within(v, g): return v is not None and g and abs(v - g) / g <= 0.05

def cluster_vote(vals, r=0.025):
    vals = sorted(v for v in vals if v and v > 0)
    if not vals: return None
    cl = []
    for c in vals:
        for g in cl:
            if abs(c - g[0]) / g[0] <= r: g.append(c); break
        else: cl.append([c])
    return statistics.median(max(cl, key=len))

def log_median(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals: return None
    return math.exp(statistics.median([math.log(v) for v in vals]))

def maxcov(vals, r=0.05):
    """capacity maxcov: value whose ±r window covers the most draws; return that cluster's median."""
    vals = [v for v in vals if v and v > 0]
    if not vals: return None
    best = None; bn = -1
    for v in vals:
        clu = [u for u in vals if abs(u - v) / v <= r]
        if len(clu) > bn: bn = len(clu); best = statistics.median(clu)
    return best

def mode_sig(vals, sig=2):
    from collections import Counter
    def r2(v):
        if v <= 0: return None
        d = sig - 1 - math.floor(math.log10(abs(v)))
        return round(v, d)
    c = Counter(r2(v) for v in vals if v and v > 0)
    c.pop(None, None)
    return c.most_common(1)[0][0] if c else None

def unit_oracle(vals_by_unit, g):
    # best recall if we could pick the right global unit multiplier
    for mult in [1.0, 2.589988, 0.01, 0.00404686, 100.0, 1/2.589988, 1e-6]:
        if any(within(v * mult, g) for v in vals_by_unit): return mult
    return None

def score_pool(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    stats = {}
    for mode in ("naive", "unitaware"):
        recall = f_clu = f_log = f_mode = pf = 0
        unit_recover = 0
        for r in rows:
            g = r["gold"]
            raw_nums = [parse(d, unit_aware=False) for d in r["draws"]]
            vals = [parse(d, unit_aware=(mode == "unitaware")) for d in r["draws"]]
            vals = [v for v in vals if v is not None]
            if not vals: pf += 1; continue
            if any(within(v, g) for v in vals): recall += 1
            if within(cluster_vote(vals), g): f_clu += 1
            if within(log_median(vals), g): f_log += 1
            if within(maxcov(vals), g): f_mode += 1  # reuse the f_mode slot for maxcov@5%
            if mode == "naive":
                rn = [v for v in raw_nums if v]
                if rn and not any(within(v, g) for v in rn) and unit_oracle(rn, g) not in (None, 1.0):
                    unit_recover += 1
        stats[mode] = dict(recall=recall/n, f_clu=f_clu/n, f_log=f_log/n, f_mode=f_mode/n,
                           parsefail=pf/n, unit_recover=unit_recover/n)
    fmt = rows[0].get("fmt", "?")
    print("== %s (fmt=%s, n=%d) ==" % (path.split("/")[-1], fmt, n))
    for mode in ("naive", "unitaware"):
        s = stats[mode]
        print("  %-9s recall %.3f | F1 clu %.3f  log %.3f  maxcov %.3f | parsefail %.3f%s" % (
            mode, s["recall"], s["f_clu"], s["f_log"], s["f_mode"], s["parsefail"],
            (" | unit-recoverable %.3f" % s["unit_recover"]) if mode == "naive" else ""))

if __name__ == "__main__":
    for p in sys.argv[1:]: score_pool(p)
