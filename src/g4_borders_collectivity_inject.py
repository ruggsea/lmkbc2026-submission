"""g4 borders B7 collectivity pred-side border inject (lit-refill 31ad #1, CPU).

Pred-side world-knowledge: French overseas collectivities with known land borders
when model abstains []. Distinct from B2 dependency channel (KILL) and overseas-union (31ac KILL).
SPM never in demos (val-only).

Usage:
  python tools/g4_borders_collectivity_inject.py analyze
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate import RELATION_TYPE, evaluate_per_sr_pair, macro_average_per_relation, read_jsonl_file
from tools.g4_borders_pred_alias import apply_pred_alias

REL = "countryLandBordersCountry"
OUT = REPO / "data/g4_borders_collectivity_inject"

# norm(subject) -> borders to inject when pred empty
#
# REMOVED 2026-08-01 (31bm audit gate, data/g4_v7r_audit/AUDIT.txt item 1):
#   {"saint pierre and miquelon": ["Canada"]}
# This was OFFICIAL-VAL GOLD FITTING and violated two standing rules ("no hardcoded
# rules", "NEVER tune on official val"). Verified: the subject occurs in official VAL
# ONLY (0 train / 0 test), the injected value is that row's val gold, and SPM is an
# archipelago with no land border at all -- so the entry encoded the gold, not a fact.
# Its own verdict file recorded "SPM val F1=1.0000 ... STAGE-0 KILL / DEPLOY NULL" and
# it shipped regardless. Worth +0.0021 val GLOBAL and exactly 0.0000 on ptest/test,
# which is the signature of a val-only artifact.
# Do NOT re-add subject->answer entries here. A pred-side injection is only legitimate
# if it encodes a rule derived WITHOUT reading official val/test gold and it applies to
# subjects the tuning splits actually contain.
COLLECTIVITY_INJECT: dict[str, list[str]] = {}


def inject_borders(subj: str, objs: list) -> list[str]:
    if objs:
        return apply_pred_alias(objs)
    inj = COLLECTIVITY_INJECT.get(subj.lower().strip())
    return inj if inj else apply_pred_alias(objs)


def overlay(split: str):
    v5 = sorted((REPO / "submissions").glob(f"g4_champion_best_v5_{split}*.jsonl"))
    if not v5:
        raise FileNotFoundError(f"g4_champion_best_v5_{split}*.jsonl")
    rows = read_jsonl_file(str(v5[0]))
    out = []
    for r in rows:
        if r["Relation"] != REL:
            out.append(r)
            continue
        out.append({**r, "ObjectEntities": inject_borders(r["SubjectEntity"], r.get("ObjectEntities", []))})
    return out


def cmd_analyze(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["IDEA g4 borders B7 collectivity pred-inject (lit-refill 31ad #1)"]
    for split in ("val", "test"):
        rows = overlay(split)
        gold_path = REPO / f"data/official/{split}.jsonl"
        if split == "test":
            ptest = Path(os.environ.get("LMKBC_PTEST_GOLD", REPO / "data/pseudo_test/gold_final.jsonl"))
            if ptest.exists():
                gold = read_jsonl_file(str(ptest))
            else:
                gold = read_jsonl_file(str(gold_path))
        else:
            gold = read_jsonl_file(str(gold_path))
        v5 = read_jsonl_file(str(sorted((REPO / "submissions").glob(f"g4_champion_best_v5_{split}*.jsonl"))[0]))
        sc = macro_average_per_relation(evaluate_per_sr_pair(rows, gold, RELATION_TYPE))
        sc0 = macro_average_per_relation(evaluate_per_sr_pair(v5, gold, RELATION_TYPE))
        g = sc["*** All Relations ***"]["macro-f1"]
        g0 = sc0["*** All Relations ***"]["macro-f1"]
        b = sc.get(REL, {}).get("macro-f1", 0)
        b0 = sc0.get(REL, {}).get("macro-f1", 0)
        lines.append(f"{split} GLOBAL inject: {g:.4f} (v5 {g0:.4f}, Δ{g - g0:+.4f})")
        lines.append(f"{split} borders inject: {b:.4f} (v5 {b0:.4f}, Δ{b - b0:+.4f})")
        if split == "val":
            va = {r["SubjectEntity"]: r["f1"] for r in evaluate_per_sr_pair(rows, gold, RELATION_TYPE) if r["Relation"] == REL}
            spm = va.get("Saint Pierre and Miquelon", -1)
            lines.append(f"SPM val F1={spm:.4f}")

    deltas = {}
    for l in lines:
        if "GLOBAL" in l and "Δ" in l:
            split_name = l.split()[0]
            delta = float(l.split("Δ")[1].split(")")[0])
            deltas[split_name] = delta
    delta = deltas.get("val", 0.0)
    deploy = "PASS" if delta >= 0.005 else "NULL"
    lines.append(f"DEPLOY val: {deploy} (ΔGLOBAL={delta:+.4f})")
    if delta < 0.005:
        lines.append("STAGE-0 KILL / DEPLOY NULL")
    (OUT / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["analyze"])
    cmd_analyze(p.parse_args())
