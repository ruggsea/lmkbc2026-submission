"""g4 v7r = v7 with company vote λ=0 (ORDER-INVARIANT). Cell 31bi #4.

v7 (tools/g4_stack_v9.py) weights company draw i by exp(-λ·i/(n-1)) where i is the index in the
vLLM `choices` array — an index that carries no information. v7r sets λ=0; everything else
(hasArea median-cluster override, borders collectivity inject, the v6 base) is identical.

Measures v6 / v7 / v7r paired on the frozen dumps, plus an independent order-permutation check.

Usage: .venv/bin/python tools/g4_stack_v7r.py analyze
       .venv/bin/python tools/g4_stack_v7r.py build   # write v7r predictions
"""
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate import RELATION_TYPE, evaluate_per_sr_pair, macro_average_per_relation, normalize_string, read_jsonl_file
from tools.g4_borders_collectivity_inject import inject_borders
from tools.pd2026.hasarea_score import parse

OUT = REPO / "data/g4_v7r_orderinvariant"
V7_LAM = 2.0        # shipped v7 rank-decay
V7R_LAM = 0.0       # order-invariant
RANK_TAU = 0.3
COMP_REL = "companyTradesAtStockExchange"
HASAREA_REL = "hasArea"
BORDERS_REL = "countryLandBordersCountry"
AGG_W = 0.05
K100_DIR = REPO / "data/g4_hasarea_manyshot_k100"
GEOG_DIR = REPO / "data/g4_hasarea_geog_entry"
N_SHUFFLE = 300
N_BOOT = 10000


# ---- verbatim from tools/g4_stack_v9.py (the shipped v7 assembler) ----------------------------
def vote_decay(draws, tau, lam):
    n = len(draws)
    scores, surf = Counter(), {}
    draw_w = 0.0
    for i, cands in enumerate(draws):
        w = math.exp(-lam * i / max(1, n - 1)) if n > 1 else 1.0
        draw_w += w
        seen = set()
        for c in cands:
            nn = normalize_string(c)
            if nn and nn not in seen:
                seen.add(nn)
                scores[nn] += w
                surf.setdefault(nn, c)
    denom = draw_w or 1.0
    return [surf[nn] for nn, s in scores.items() if s / denom >= tau - 1e-9]


def median_cluster(vals):
    vs = sorted(v for v in vals if v and v > 0)
    if not vs:
        return None
    cl = []
    for v in vs:
        for c in cl:
            if abs(v - c[0]) / c[0] <= AGG_W:
                c.append(v)
                break
        else:
            cl.append([v])
    c = max(cl, key=len)
    s = sorted(c)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def load_area_map(split, ddir):
    out = {}
    raw = ddir / f"{split}_raw.jsonl"
    if not raw.exists():
        return out
    for line in open(raw):
        d = json.loads(line)
        vals = [parse(t, True) for t in d["draws"]]
        vals = [v for v in vals if v and v > 0]
        v = median_cluster(vals)
        if v:
            out[d["subject"]] = v
    return out


def hasarea_median_disagree_map(split):
    k100 = load_area_map(split, K100_DIR)
    geog = load_area_map(split, GEOG_DIR)
    out = {}
    for s in set(k100) | set(geog):
        ka, ga = k100.get(s), geog.get(s)
        if ka and ga and abs(ga - ka) / ka > AGG_W:
            out[s] = [str(ga)]
        elif ka:
            out[s] = [str(ka)]
        elif ga:
            out[s] = [str(ga)]
    return out


def _pin_base(pattern):
    """Locate a chain base by glob. Several dated files can match; picking one by glob
    order silently changes the deliverable -- this is the bug class that laundered the SPM
    val-gold hardcode through the v6 artifact (31bm), and the contaminated v6 files were
    still glob-reachable until 31bo. Take the newest and FAIL LOUD if candidates disagree."""
    cands = sorted((REPO / "submissions").glob(pattern))
    if not cands:
        raise FileNotFoundError(pattern)
    digests = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in cands}
    if len(set(digests.values())) > 1:
        raise RuntimeError(
            f"ambiguous base for {pattern}: candidates differ in content, pin one explicitly\n"
            + "\n".join(f"  {p.name} {d}" for p, d in digests.items())
        )
    return cands[-1]


def champion_submission(split):
    return read_jsonl_file(str(_pin_base(f"g4_champion_best_v6_{split}*.jsonl")))


def gold_for(split):
    if split == "test":
        ptest = Path(os.environ.get("LMKBC_PTEST_GOLD", REPO / "data/pseudo_test/gold_final.jsonl"))
        if ptest.exists():
            return read_jsonl_file(str(ptest))
    return read_jsonl_file(str(REPO / f"data/official/{split}.jsonl"))


def company_parsed(split):
    from tools.g4_company_manyshot_k64 import load_parsed

    return load_parsed(split)


def overlay(split, lam):
    """v9's overlay, with the company decay λ exposed. lam=2.0 -> v7, lam=0.0 -> v7r."""
    rows = champion_submission(split)
    parsed = company_parsed(split)
    comp = {s: vote_decay(d, RANK_TAU, lam) for s, d in parsed.items()}
    ha = hasarea_median_disagree_map(split) if split != "test" or (K100_DIR / "test_raw.jsonl").exists() else {}
    out = []
    for r in rows:
        s, rel = r["SubjectEntity"], r["Relation"]
        if rel == COMP_REL and s in comp:
            out.append({**r, "ObjectEntities": comp[s]})
        elif rel == HASAREA_REL and s in ha:
            out.append({**r, "ObjectEntities": ha[s]})
        elif rel == BORDERS_REL:
            out.append({**r, "ObjectEntities": inject_borders(s, r.get("ObjectEntities", []))})
        else:
            out.append(r)
    return out


# ---- measurement ------------------------------------------------------------------------------
def per_row(rows, gold):
    return {(d["SubjectEntity"], d["Relation"]): d["f1"] for d in evaluate_per_sr_pair(rows, gold, RELATION_TYPE)}


def table(rows, gold):
    return macro_average_per_relation(evaluate_per_sr_pair(rows, gold, RELATION_TYPE))


def global_from_rowf1(f1_by_key, keys_by_rel):
    return sum(sum(f1_by_key[k] for k in ks) / len(ks) for ks in keys_by_rel.values()) / len(keys_by_rel)


def paired_bootstrap(a_f1, b_f1, keys, n=N_BOOT, seed=0, strata=None):
    """Paired row bootstrap of GLOBAL(a) - GLOBAL(b).

    evaluate.py's '*** All Relations ***' macro-f1 is a PLAIN MEAN over all (subject, relation)
    rows -- NOT a mean of per-relation means -- so the bootstrap resamples rows i.i.d. over the
    whole row set (strata=<rel->keys> optionally fixes the per-relation row counts instead).
    """
    rng = random.Random(seed)
    d = [a_f1[k] - b_f1[k] for k in keys]
    N = len(d)
    deltas = []
    if strata is None:
        for _ in range(n):
            deltas.append(sum(d[rng.randrange(N)] for _ in range(N)) / N)
    else:
        blocks = [[a_f1[k] - b_f1[k] for k in ks] for ks in strata.values()]
        for _ in range(n):
            deltas.append(sum(sum(b[rng.randrange(len(b))] for _ in b) for b in blocks) / N)
    deltas.sort()
    lo = deltas[int(0.025 * n)]
    hi = deltas[int(0.975 * n) - 1]
    p_le0 = sum(1 for x in deltas if x <= 0) / n
    return lo, hi, p_le0, sum(deltas) / n


def company_f1(parsed, gold, lam, order=None):
    """company relation macro-f1 for a given draw ordering. order: dict subj -> permutation."""
    preds = {}
    for s, d in parsed.items():
        dd = [d[i] for i in order[s]] if order else d
        preds[s] = vote_decay(dd, RANK_TAU, lam)
    rows = [{"SubjectEntity": s, "Relation": COMP_REL, "ObjectEntities": p} for s, p in preds.items()]
    g = [r for r in gold if r["Relation"] == COMP_REL]
    sc = macro_average_per_relation(evaluate_per_sr_pair(rows, g, RELATION_TYPE))
    return sc[COMP_REL]["macro-f1"]


def shuffle_study(split, out_lines):
    parsed = company_parsed(split)
    gold = gold_for(split)
    w = sum(1 for r in gold if r["Relation"] == COMP_REL) / len(gold)  # company's GLOBAL row weight
    shipped = company_f1(parsed, gold, V7_LAM)
    plain = company_f1(parsed, gold, V7R_LAM)
    rev = company_f1(parsed, gold, V7_LAM, {s: list(range(len(d)))[::-1] for s, d in parsed.items()})
    rng = random.Random(12345)
    vals = []
    for _ in range(N_SHUFFLE):
        order = {}
        for s, d in parsed.items():
            p = list(range(len(d)))
            rng.shuffle(p)
            order[s] = p
        vals.append(company_f1(parsed, gold, V7_LAM, order))
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    p_ge = sum(1 for v in vals if v >= shipped - 1e-12) / len(vals)
    sv = sorted(vals)
    out_lines += [
        f"  company macro-f1 [{split}] shipped draw order (λ=2.0): {shipped:.4f}",
        f"  company macro-f1 [{split}] λ=0 (order-invariant)      : {plain:.4f}",
        f"  company macro-f1 [{split}] REVERSED order (λ=2.0)     : {rev:.4f}",
        f"  company macro-f1 [{split}] {N_SHUFFLE} random orders (λ=2.0): mean {mean:.4f} sd {sd:.4f} "
        f"min {sv[0]:.4f} p2.5 {sv[int(0.025*len(sv))]:.4f} p97.5 {sv[int(0.975*len(sv))-1]:.4f} max {sv[-1]:.4f}",
        f"  P(random order >= shipped order) = {p_ge:.3f}   order-averaged λ=2.0 minus λ=0 = {mean - plain:+.4f}",
        f"  => honest ΔGLOBAL of rank-decay (order-averaged λ=2.0 vs λ=0), company row weight {w:.3f}: "
        f"{(mean - plain) * w:+.4f}   [the booked, order-lucky ΔGLOBAL was {(shipped - plain) * w:+.4f}]",
    ]
    return {"shipped": shipped, "plain": plain, "rev": rev, "mean": mean, "sd": sd, "p_ge": p_ge}


def draw_index_quality(split, out_lines, n_null=200):
    """Direct test of the premise: does the vLLM `choices` index correlate with draw quality?

    Regress single-draw F1 on the draw index; null = shuffle the index labels within each subject.
    """
    from evaluate import f1_score, string_true_positives

    parsed = company_parsed(split)
    gold = {r["SubjectEntity"]: r["ObjectEntities"] for r in gold_for(split) if r["Relation"] == COMP_REL}

    def one(cands, gts):
        preds = list({normalize_string(c) for c in cands if isinstance(c, str) and normalize_string(c)})
        tp = string_true_positives(preds, gts)
        return f1_score(tp / len(preds) if preds else 1.0, tp / len(gts) if gts else 1.0)

    mat = []
    for s, draws in parsed.items():
        g = gold.get(s)
        if g is None:
            continue
        g = [[x] for x in g] if g and isinstance(g[0], str) else g
        mat.append([one(c, g) for c in draws])
    n = len(mat[0])
    means = [sum(row[i] for row in mat) / len(mat) for i in range(n)]
    xs = list(range(n))
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)

    def slope(m):
        my = sum(m) / n
        return sum((x - mx) * (v - my) for x, v in zip(xs, m)) / sxx

    obs = slope(means)
    rng = random.Random(7)
    null = []
    for _ in range(n_null):
        acc = [0.0] * n
        for row in mat:
            p = list(row)
            rng.shuffle(p)
            for i, v in enumerate(p):
                acc[i] += v
        null.append(slope([a / len(mat) for a in acc]))
    p = sum(1 for v in null if v <= obs) / n_null
    dec = [sum(means[k * n // 10:(k + 1) * n // 10]) / (n // 10) for k in range(10)]
    out_lines += [
        f"  [{split}] mean single-draw company F1 by draw-index decile: " + " ".join(f"{d:.4f}" for d in dec),
        f"         slope {obs:+.6f} F1/draw (decile1 {dec[0]:.4f} -> decile10 {dec[-1]:.4f}, "
        f"Δ{dec[-1]-dec[0]:+.4f});  P(null slope <= observed) = {p:.3f} over {n_null} within-subject shuffles",
    ]
    return obs, p


def cmd_build():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        rows = overlay(split, V7R_LAM)
        p = OUT / f"g4_v7r_{split}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p} ({len(rows)} rows)", flush=True)
    # train: no v6 base submission exists, so only the company arm (the sole v7/v7r difference)
    parsed = company_parsed("train")
    for tag, lam in (("v7", V7_LAM), ("v7r", V7R_LAM)):
        p = OUT / f"company_train_{tag}.jsonl"
        with open(p, "w") as f:
            for s, d in parsed.items():
                f.write(json.dumps({"SubjectEntity": s, "Relation": COMP_REL,
                                    "ObjectEntities": vote_decay(d, RANK_TAU, lam)}, ensure_ascii=False) + "\n")
        print(f"wrote {p}", flush=True)


def cmd_analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["31bi #4 — ORDER-INVARIANT DELIVERABLE v7r (company λ=0). Fixed-config re-measurement, no tuning.",
         f"v7 = λ={V7_LAM} τ={RANK_TAU} rank-decay on the vLLM choices index + hasArea median-cluster + v6 base",
         f"v7r = identical, λ={V7R_LAM} (plain vote). hasArea median-cluster is provably order-invariant (sorts first).",
         ""]

    # ---- 1. fidelity -------------------------------------------------------------------------
    L.append("== 1. FIDELITY CHECK (reproduce the booked v7 from frozen dumps) ==")
    fid_ok = True
    scores = {}
    # booked constants updated 31bo (de-leak follow-up to 31bm/31bn). The ORIGINAL constants
    # (val 0.7018, test 0.6936) were PRE-DE-LEAK: val 0.7018 included the Saint-Pierre-and-
    # Miquelon official-val gold hardcode (tools/g4_borders_collectivity_inject.py, removed in
    # 31bm) baked into the v6 base this overlay sits on. SPM is a val-ONLY subject (0 train /
    # 0 test rows), so test's constant was never contaminated and is unchanged. The new val
    # constant (0.6999) was derived by directly recomputing overlay("val", V7_LAM) against the
    # CLEAN (2026-08-01) v6 base -- not copied from an ad hoc script print, computed once here
    # and pinned so this remains a real assertion, not a vacuous self-check. The -0.0019 move
    # from the old constant matches the already-documented SPM contamination size (1 val row's
    # F1 flips 1.0->0.0, -0.002092 GLOBAL from borders alone, partly offset by +0.000185 from an
    # unrelated award parser fix that landed in the same rebuild -- data/g4_leak_sweep/AUDIT.txt C4).
    # ★ RE-ANCHORED 31bx (2026-08-01), MAIN LOOP. The val constant above (0.6999) was pinned
    # under the STALE official gold and this script had been hard-aborting with
    # "val: recomputed 0.6897 vs booked 0.6999 -> MISMATCH | FIDELITY: FAIL -- STOP" ever since
    # the 31bt upstream-gold resync. It was flagged in the driver prompt as a live CODE DEFECT.
    # It is NOT a code defect: 0.6999 - 0.6897 = 0.0102, which is EXACTLY the documented
    # gold-resync shift for this repo ("the evaluator change is worth ~0.000; the gold change is
    # the entire -0.0102", submissions/G4_CHAMPION_MANIFEST.md). Corroborated by test still
    # matching its own constant to 4 dp (0.6936) -- ptest is scored against LOCAL ptest gold and
    # was untouched by the resync, so a genuine overlay bug would have moved BOTH splits.
    # Fix = re-pin val to the upstream-gold recomputation. The assertion stays real (the constant
    # is still an independently computed value that the overlay must reproduce), it is simply
    # anchored to the correct gold now.
    for split, booked in (("val", 0.6897), ("test", 0.6936)):
        gold = gold_for(split)
        v7 = overlay(split, V7_LAM)
        g = table(v7, gold)["*** All Relations ***"]["macro-f1"]
        ok = abs(g - booked) < 1e-4  # booked figures are quoted to 4 dp
        if split == "val":
            # val's 2026-07-31 shipped file carried the SPM hardcode and is quarantined
            # (submissions/quarantine_contaminated/, 31bo) -- do not read it back in. No clean
            # persisted v7 (lam=2.0) val file exists (v7 was retracted in favour of v7r before a
            # post-de-leak version was ever generated), so fidelity for val is asserted by
            # recomputation against the pinned constant only.
            fid_ok &= ok
            L.append(f"  {split}: recomputed v7 GLOBAL {g:.4f} vs booked {booked:.4f} -> {'MATCH' if ok else 'MISMATCH'}"
                     f" | no clean persisted shipped file exists for this split (quarantined, 31bo)")
        else:
            # test's 2026-07-31 shipped file was never quarantined (SPM never touches test rows)
            # and remains a legitimate independent cross-check.
            shipped = read_jsonl_file(str(REPO / f"submissions/g4_champion_best_v7_{split}_2026-07-31.jsonl"))
            ident = all(a["SubjectEntity"] == b["SubjectEntity"] and a["Relation"] == b["Relation"]
                        and [normalize_string(x) for x in a["ObjectEntities"]] == [normalize_string(x) for x in b["ObjectEntities"]]
                        for a, b in zip(v7, shipped)) and len(v7) == len(shipped)
            gs = table(shipped, gold)["*** All Relations ***"]["macro-f1"]
            fid_ok &= ok and abs(gs - booked) < 1e-4
            L.append(f"  {split}: recomputed v7 GLOBAL {g:.4f} vs booked {booked:.4f} -> {'MATCH' if ok else 'MISMATCH'}"
                     f" | shipped-file GLOBAL {gs:.4f} | recomputed rows identical to shipped file: {ident}")
        scores[split] = gold
    L.append(f"  FIDELITY: {'PASS — reproduction is exact, deltas below are trustworthy' if fid_ok else 'FAIL — STOP'}")
    L.append("")
    if not fid_ok:
        (OUT / "verdict.txt").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    # ---- 2. 3x3 table ------------------------------------------------------------------------
    sysrows, rowf1, keys_by_rel = {}, {}, {}
    for split in ("val", "test"):
        gold = gold_for(split)
        sysrows[("v6", split)] = champion_submission(split)
        sysrows[("v7", split)] = overlay(split, V7_LAM)
        sysrows[("v7r", split)] = overlay(split, V7R_LAM)
        kbr = {}
        for r in gold:
            kbr.setdefault(r["Relation"], []).append((r["SubjectEntity"], r["Relation"]))
        keys_by_rel[split] = kbr
        for s in ("v6", "v7", "v7r"):
            rowf1[(s, split)] = per_row(sysrows[(s, split)], gold)

    L.append("== 2. GLOBAL (macro-of-macro-f1), system x split ==")
    L.append("  split      |    v6    |    v7    |   v7r    | v7r-v7  | v7r-v6")
    for split, name in (("val", "val (official)"), ("test", "ptest (pseudo-test)")):
        g = {s: table(sysrows[(s, split)], gold_for(split))["*** All Relations ***"]["macro-f1"] for s in ("v6", "v7", "v7r")}
        L.append(f"  {name:20s} | {g['v6']:.4f} | {g['v7']:.4f} | {g['v7r']:.4f} | {g['v7r']-g['v7']:+.4f} | {g['v7r']-g['v6']:+.4f}")
    L.append("  train: NOT PRODUCIBLE — no g4_champion_best_v6_train_*.jsonl base submission exists in submissions/;")
    L.append("         the v6 assembly was only ever materialised for val and ptest. Train is covered below by the")
    L.append("         company arm alone, which is the ONLY component that differs between v7 and v7r.")
    tg = gold_for("train")
    pt = company_parsed("train")
    ctrain = {tag: company_f1(pt, tg, lam) for tag, lam in (("v7", V7_LAM), ("v7r", V7R_LAM))}
    wtr = sum(1 for r in tg if r["Relation"] == COMP_REL) / len(tg)
    L.append(f"  train company macro-f1: v7 {ctrain['v7']:.4f} · v7r {ctrain['v7r']:.4f} "
             f"(Δ{ctrain['v7r']-ctrain['v7']:+.4f}; GLOBAL-equivalent Δ{(ctrain['v7r']-ctrain['v7'])*wtr:+.4f} "
             f"at company row weight {wtr:.3f})")
    L.append("")

    # ---- 3. per-relation ---------------------------------------------------------------------
    L.append("== 3. PER-RELATION macro-f1 ==")
    L.append("  (GLOBAL is a plain row-mean, so it is NOT the average of the six relation rows below:")
    L.append("   awardWonBy has 10 rows and borders 68, vs 100 each for the other four.)")
    for split, name in (("val", "val"), ("test", "ptest")):
        gold = gold_for(split)
        t = {s: table(sysrows[(s, split)], gold) for s in ("v6", "v7", "v7r")}
        L.append(f"  [{name}]  relation                        v6       v7      v7r    v7r-v7")
        for rel in sorted(RELATION_TYPE):
            if rel not in t["v6"]:
                continue
            a, b, c = (t[s][rel]["macro-f1"] for s in ("v6", "v7", "v7r"))
            L.append(f"           {rel:30s} {a:.4f}  {b:.4f}  {c:.4f}  {c-b:+.4f}")
        a, b, c = (t[s]["*** All Relations ***"]["macro-f1"] for s in ("v6", "v7", "v7r"))
        L.append(f"           {'*** GLOBAL ***':30s} {a:.4f}  {b:.4f}  {c:.4f}  {c-b:+.4f}")
    L.append("")

    # ---- 4. paired bootstrap -----------------------------------------------------------------
    L.append(f"== 4. PAIRED ROW-LEVEL COMPARISON (paired row bootstrap, {N_BOOT} resamples, 95% CI) ==")
    L.append("  NOTE: GLOBAL = plain mean of row-F1 over all rows (evaluate.py '*** All Relations ***'),")
    L.append("  so a company-only change moves GLOBAL by Δcompany × 100/478 (val) = 0.209, NOT by Δ/6.")
    for split, name in (("val", "val"), ("test", "ptest")):
        kbr = keys_by_rel[split]
        keys = [k for ks in kbr.values() for k in ks]
        for ref in ("v7", "v6"):
            a, b = rowf1[("v7r", split)], rowf1[(ref, split)]
            ndiff = sum(1 for k in a if abs(a[k] - b[k]) > 1e-12)
            changed = [(r_a["SubjectEntity"], r_a["Relation"]) for r_a, r_b
                       in zip(sysrows[("v7r", split)], sysrows[(ref, split)])
                       if sorted(normalize_string(x) for x in r_a["ObjectEntities"])
                       != sorted(normalize_string(x) for x in r_b["ObjectEntities"])]
            byrel = Counter(r for _, r in changed)
            lo, hi, p_le0, mean = paired_bootstrap(a, b, keys)
            slo, shi, sp, smean = paired_bootstrap(a, b, keys, strata=kbr)
            L.append(f"  [{name}] v7r - {ref}: ΔGLOBAL {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ<=0) = {p_le0:.3f}"
                     f"   [rel-stratified: {smean:+.4f} CI [{slo:+.4f},{shi:+.4f}] P(Δ<=0)={sp:.3f}]")
            L.append(f"         rows whose PREDICTION differs: {len(changed)}/{len(sysrows[('v7r', split)])} "
                     f"{dict(byrel)}; rows whose row-F1 differs: {ndiff}/{len(a)}")
    L.append("  v6 and v7r produce BYTE-EQUIVALENT company predictions (v6's company arm already is the")
    L.append("  plain λ=0 k64 vote at τ=0.3). So v7r == v6 + hasArea median-cluster, exactly; the only")
    L.append("  v7r-vs-v6 rows that move are hasArea (median-cluster rewrites all 100 values, but only 1")
    L.append("  crosses the 5% numeric tolerance).")
    L.append("")

    # ---- 5. shuffles -------------------------------------------------------------------------
    L.append(f"== 5. INDEPENDENT ORDER-PERMUTATION CHECK ({N_SHUFFLE} shuffles of the frozen company pool) ==")
    st = {}
    for split, name in (("train", "train (the legitimate tuning split)"), ("val", "val"), ("test", "ptest")):
        L.append(f"  --- {name} ---")
        st[split] = shuffle_study(split, L)
    L.append("  Reproduces the prior audit's val numbers within Monte-Carlo error (audit: shipped 0.8769,")
    L.append("  λ=0 0.8569, reversed 0.8436, 300-shuffle mean 0.8581 sd 0.0099, P=0.067).")
    L.append("  BUT NOTE: the native order beats its own shuffle null on ALL THREE splits (P = 0.027 /")
    L.append("  0.073 / 0.027 on train / val / ptest; Fisher-combined p ~ 0.003). That is not luck.")
    L.append("")

    L.append("== 5b. IS THE vLLM `choices` INDEX REALLY INFORMATION-FREE? (direct test — it is NOT) ==")
    L.append("  The retraction's stated premise was that the choices index 'carries no information'.")
    L.append("  Tested directly: regress single-draw company F1 on the draw index (null = shuffle the")
    L.append("  index labels within each subject, which destroys only the index-quality link).")
    dq = {}
    for split in ("train", "val", "test"):
        dq[split] = draw_index_quality(split, L)
    L += [
        "  VERDICT on the premise: FALSE. Draw quality DECLINES monotonically-ish with the choices index",
        "  on all three independent splits, at P <= 0.010 each. vLLM returns the n completions in an order",
        "  that is weakly but genuinely correlated with quality (plausibly finish-order/length, which",
        "  correlates with confidence). So rank-decay was picking up a REAL signal, not a lottery ticket.",
        "  What survives of the retraction: the signal is TINY and its exploitation is FRAGILE. The",
        "  `choices` ordering is an undocumented vLLM implementation detail (scheduler / batch-state /",
        "  version dependent), not an API guarantee. Under any regeneration that permutes it, the λ=2.0",
        "  gain collapses to the order-averaged value: +0.0004 (val) / -0.0003 (ptest) / -0.0005 (train)",
        "  GLOBAL. And the native-order gain itself (+0.0042 val / +0.0025 ptest / +0.0007 train) is below",
        "  the 0.0072 fresh-generation GLOBAL sd on every split, even though its sign is consistent 3/3.",
        "",
    ]

    # ---- 6. recommendation -------------------------------------------------------------------
    gv = {s: table(sysrows[(s, "val")], gold_for("val"))["*** All Relations ***"]["macro-f1"] for s in ("v6", "v7", "v7r")}
    gt = {s: table(sysrows[(s, "test")], gold_for("test"))["*** All Relations ***"]["macro-f1"] for s in ("v6", "v7", "v7r")}
    wv = 100 / len(gold_for("val"))
    wt = 100 / len(gold_for("test"))
    oa_v = gv["v7r"] + (st["val"]["mean"] - st["val"]["plain"]) * wv
    oa_t = gt["v7r"] + (st["test"]["mean"] - st["test"]["plain"]) * wt
    L += [
        "== 6. RECOMMENDATION ==",
        f"  v7r val GLOBAL {gv['v7r']:.4f} (v7 {gv['v7']:.4f}, v6 {gv['v6']:.4f})",
        f"  v7r ptest GLOBAL {gt['v7r']:.4f} (v7 {gt['v7']:.4f}, v6 {gt['v6']:.4f})",
        f"  Nominal cost of going v7 -> v7r: val {gv['v7r']-gv['v7']:+.4f}, ptest {gt['v7r']-gt['v7']:+.4f}.",
        f"  EXPECTED cost, i.e. v7r vs the order-AVERAGED v7 (the only reproducible reading of v7):",
        f"    val {gv['v7r']-oa_v:+.4f} (order-averaged v7 = {oa_v:.4f}), "
        f"ptest {gt['v7r']-oa_t:+.4f} (order-averaged v7 = {oa_t:.4f}).",
        "    Both are ~1/20th of the fresh-generation GLOBAL sd (0.0072) -- indistinguishable from zero.",
        "  RECOMMENDATION: YES, v7r is the honest deliverable. It is order-invariant by construction",
        "  (plain vote + a median-cluster that sorts before aggregating), it reproduces bit-for-bit from",
        "  the frozen dumps, and it costs ~0 in expectation once the draw order is not held fixed.",
        "  Concretely v7r == v6 + the hasArea median-cluster override, which is the half of v7 that was",
        "  never in dispute (+0.0021 GLOBAL on both splits).",
        "  CAVEAT I OWE THE LOOP (§5b): the retraction's REASON was wrong even though its CONCLUSION holds.",
        "  The choices index is NOT information-free -- draw quality declines with it at P<=0.01 on all",
        "  three splits, and the native order beats its shuffle null 3/3 (Fisher p~0.003). v7 is therefore",
        "  reproducible *given the same generation*; what it is not is portable across vLLM versions /",
        "  batch states. Shipping v7 means betting a +0.004 val edge on an undocumented library detail,",
        "  with an expected value of ~0 the moment that detail changes. v7r removes the bet at no cost.",
        "  If the loop wants that signal back, the right move is a PRINCIPLED confidence proxy (sequence",
        "  logprob / length), not the array index -- the index is only a proxy for whatever that is.",
        # 31bo: this line used to hardcode pre-de-leak figures (v6 0.6955, v7 0.7018, v7r 0.6976 --
        # all included the SPM val-gold hardcode via the v6 base). Now reads directly off gv/gt,
        # the SAME honestly-recomputed dicts used throughout SS2-4 above, so it cannot go stale again.
        f"  NONE of v6 ({gv['v6']:.4f}/{gt['v6']:.4f}), v7 ({gv['v7']:.4f}/{gt['v7']:.4f}) or v7r "
        f"({gv['v7r']:.4f}/{gt['v7r']:.4f}) reaches the 0.70 GLOBAL",
        f"  target honestly. v7's {gv['v7']:.4f} rests on a fragile, environment-dependent draw ordering; its",
        f"  order-averaged (portable) value is {oa_v:.4f} on val — below 0.70. The 0.70 target is NOT met.",
    ]
    (OUT / "verdict.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        cmd_build()
    else:
        cmd_analyze()
