"""Stage 4b — ship the steering vector with full provenance.

Copies vector_generation.py's layer-20 output to vectors/apps_v_code.pt and
writes apps_v_code.meta.json recording everything PLAN.md demands: model,
layer, SIGN CONVENTION (vector = H_RT − H_E; apply with coef −1.0 — never
positive expecting less reflection), keyword lists, split + problem indices
of both pools, trace/activation counts, and the full generation config.
"""
import argparse
import datetime
import json
import os
import shutil

import torch

import thought_tags
from thought_tags import (REFLECT_WORDS, REFLECT_PREFIXES,
                          TRANSITION_WORDS, TRANSITION_PREFIXES)


def load_selection(data_dir, prefix):
    path = os.path.join(data_dir, f"hidden_{prefix}", "selection.json")
    with open(path) as f:
        sel = json.load(f)
    # extraction-time provenance must match the pointer NOW, else this meta
    # would describe lists/dtype that did not produce the vector
    if sel["keyword_set"] != thought_tags.keyword_set:
        raise SystemExit(
            f"[package] {prefix}: extracted with keyword_set="
            f"'{sel['keyword_set']}' but thought_tags.keyword_set is now "
            f"'{thought_tags.keyword_set}' — re-run hidden_analysis or flip "
            "the pointer back before packaging.")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--save_prefix", default="500_500")
    ap.add_argument("--prefixs", nargs="+",
                    default=["correct_0_500", "incorrect_0_500"])
    ap.add_argument("--out", default="vectors/apps_v_code.pt")
    args = ap.parse_args()

    vec_path = os.path.join(args.data_dir, f"vector_{args.save_prefix}",
                            f"layer_{args.layer}_transition_reflection_steervec.pt")
    vec = torch.load(vec_path, weights_only=False)

    with open(os.path.join(args.data_dir, "gen_config.json")) as f:
        gen_config = json.load(f)

    pools = {}
    n_rt, n_steps = 0, 0
    extraction_dtype = None
    for prefix in args.prefixs:
        sel = load_selection(args.data_dir, prefix)
        extraction_dtype = sel.get("dtype", extraction_dtype)
        items = sel["items"]
        pool_rt = sum(s["n_check"] + s["n_switch"] for s in items)
        pool_steps = sum(s["n_steps"] for s in items)
        n_rt += pool_rt
        n_steps += pool_steps
        pools[prefix] = {
            "n_traces": len(items),
            "problem_ids": [s["problem_id"] for s in items],
            "difficulty_counts": _count(s["difficulty"] for s in items),
            "n_activations_total": pool_steps,
            "n_activations_reflection_transition": pool_rt,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copyfile(vec_path, args.out)
    meta = {
        "name": "apps_v_code",
        "created": datetime.date.today().isoformat(),
        "model": gen_config["model_name_or_path"],
        "dataset": "codeparrot/apps (parquet branch)",
        "split": gen_config["split"],
        "layer": args.layer,
        "extraction_dtype": extraction_dtype,
        "sign_convention": ("vector = mean(H_reflection ∪ H_transition) − mean(H_execution); "
                            "apply by ADDING coef * vector with coef = -1.0 "
                            "(SEAL code convention, opposite of the paper's formula)"),
        "apply_coef": -1.0,
        "boundary_token": ("any vocab token containing 'ĊĊ' (\\n\\n)"
                           + (", inside <think> only"
                              if "deepseek" in gen_config["model_name_or_path"].lower()
                              else " — WARNING: whole sequence (non-deepseek model, "
                                   "hidden_analysis think_only=False)")),
        "keyword_lists": {"set": thought_tags.keyword_set,
                          "reflection": REFLECT_WORDS,
                          "reflection_prefixes": REFLECT_PREFIXES,
                          "transition": TRANSITION_WORDS,
                          "transition_prefixes": TRANSITION_PREFIXES,
                          "priority": "reflection checked before transition"},
        "selection_rule": (("train split in file order, greedy n=1; first "
                            f"{gen_config['target']} correct + first "
                            f"{gen_config['target']} incorrect (--start 0 --sample "
                            f"{gen_config['target']}, deterministic prefix)")
                           if gen_config.get("reached_target", True)
                           else ("WARNING: split exhausted before target "
                                 f"({gen_config.get('n_correct')} correct / "
                                 f"{gen_config.get('n_incorrect')} incorrect of "
                                 f"{gen_config['target']}+{gen_config['target']}); "
                                 "pools hold fewer traces than the 0_500 names imply")),
        "generation": gen_config,
        "pools": pools,
        "activations": {"total": n_steps,
                        "reflection_transition": n_rt,
                        "execution": n_steps - n_rt},
        "vector_dim": int(vec.numel()),
        "vector_norm": float(vec.float().norm()),
        "source_vector": vec_path,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[package] vector -> {args.out} (dim={vec.numel()}, "
          f"norm={meta['vector_norm']:.3f})")
    print(f"[package] activations: RT={n_rt} E={n_steps - n_rt} total={n_steps}")
    print(f"[package] meta -> {meta_path}")


def _count(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


if __name__ == "__main__":
    main()
