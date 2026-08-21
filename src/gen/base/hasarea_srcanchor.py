"""hasArea source-anchoring for gemma-4 (E150 follow-up): the mode picks ALTERNATIVE published
figures (municipality vs island, other surveys) over the Wikidata-P2046 figure the gold uses.
Anchor recall to the gold's own provenance:
  wikitext : verbatim en-wiki infobox continuation "| area_km2 = "  (the artifact P2046 mirrors)
  wikidata : "area (P2046): <n> km2" entity-record continuation      (the gold source itself)
Offline vLLM (one GPU per split), pool schema == hasarea_base (subject/gold/draws) so
hasarea_agg_sweep.py scores it. Usage:
  python hasarea_srcanchor.py --gold <gold.jsonl> --fmt wikitext|wikidata --n 100 --out <pool.jsonl> \
      --model google/gemma-4-31B --attn-backend TRITON_ATTN
"""
import argparse, json, sys

FMT = {
    "wikitext": {
        "prompt": ("{{{{Infobox island\n| name = Manhattan\n| area_km2 = 59.1\n| country = United States\n}}}}\n\n"
                   "{{{{Infobox lake\n| name = Lake Balaton\n| area_km2 = 592\n| location = Hungary\n}}}}\n\n"
                   "{{{{Infobox island\n| name = Bornholm\n| area_km2 = 588.36\n| country = Denmark\n}}}}\n\n"
                   "{{{{Infobox\n| name = {s}\n| area_km2 ="),
        "stop": ["\n|", "\n\n", "}}"],
        "maxtok": 16,
    },
    "page": {
        # lead-sentence self-disambiguation flowing into the infobox anchor (one Wikipedia "page")
        "prompt": ("'''Bornholm''' is a Danish island in the Baltic Sea, southeast of the rest of Denmark.\n"
                   "{{{{Infobox island\n| name = Bornholm\n| area_km2 = 588.36\n}}}}\n\n"
                   "'''Lake Balaton''' is a freshwater lake in the Transdanubian region of Hungary.\n"
                   "{{{{Infobox lake\n| name = Lake Balaton\n| area_km2 = 592\n}}}}\n\n"
                   "'''{s}''' is"),
        "stop": ["}}", "\n\n'''"],
        "maxtok": 90,
    },
    "wikidata": {
        "prompt": ("Wikidata entity record.\n\n"
                   "entity: Monaco\narea (P2046): 2.02 km2\n\n"
                   "entity: Lake Balaton\narea (P2046): 592 km2\n\n"
                   "entity: Bornholm\narea (P2046): 588.36 km2\n\n"
                   "entity: {s}\narea (P2046):"),
        "stop": ["\n\n", "\nentity:"],
        "maxtok": 12,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--fmt", required=True, choices=list(FMT))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn-backend", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    cfg = FMT[a.fmt]

    from vllm import LLM, SamplingParams
    kw = {"attention_backend": a.attn_backend} if a.attn_backend else {}
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.9,
              trust_remote_code=True, **kw)
    sp = SamplingParams(n=a.n, temperature=a.temp, top_p=0.95, max_tokens=cfg["maxtok"],
                        stop=cfg["stop"])
    prompts = [cfg["prompt"].format(s=r["SubjectEntity"]) for r in rows]
    outs = llm.generate(prompts, sp)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            gold = r.get("ObjectEntities")
            v = gold
            while isinstance(v, list):
                v = v[0] if v else None
            def post(t):
                # page fmt: keep only from the anchored field so the pool parser reads THAT number
                if a.fmt == "page" and "area_km2" in t:
                    return t.split("area_km2", 1)[1]
                return t
            f.write(json.dumps({"subject": r["SubjectEntity"], "gold": v, "fmt": a.fmt,
                                "draws": [post(c.text) for c in o.outputs]}, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} ({len(rows)} subjects x n={a.n})", flush=True)


if __name__ == "__main__":
    main()
