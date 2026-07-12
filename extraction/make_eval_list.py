"""Emit a seeded-RANDOM eval task list (jsonl) from the APPS test split.

The extraction pools are a deterministic greedy prefix (first-500-correct +
first-500-incorrect over train in file order — SEAL's design, never random).
The EVAL task list is the opposite: a seeded random, difficulty-stratified
sample of the *test* split (disjoint from extraction by construction), fixed
once at extraction time so every later eval run scores the same problems.

Rows carry everything needed to prompt at eval time (question, starter_code,
fn_name); test cases are reloaded from the dataset by problem_id when scoring
(apps_scoring.py), keeping the list small. Membership is random (seeded);
rows are ordered by difficulty tier then problem_id for stable diffs.

  <out>.jsonl       {problem_id, split, difficulty, kind, fn_name, n_tests,
                     question, starter_code}
  <out>.meta.json   {seed, per_difficulty, counts, filtered, method}
"""
import argparse
import json
import os
import random

from apps_data import load_apps, parse_tests

TIER_ORDER = ["introductory", "interview", "competition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per_difficulty", type=int, default=100,
                    help="problems sampled per difficulty tier (PLAN.md: n≈100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="output path; default eval_tasks/apps_<split>_n<per>_seed<seed>.jsonl")
    args = ap.parse_args()
    out = args.out or os.path.join(
        "eval_tasks", f"apps_{args.split}_n{args.per_difficulty}_seed{args.seed}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ds = load_apps(args.split)
    tiers = {t: [] for t in TIER_ORDER}
    n_filtered = 0
    for i in range(len(ds)):
        row = ds[i]
        tests = parse_tests(row)
        if tests is None:  # same filter as extraction: unusable tests are not evaluable
            n_filtered += 1
            continue
        tiers.setdefault(row["difficulty"], []).append((i, tests))

    rng = random.Random(args.seed)
    picked = []
    counts = {}
    for tier in TIER_ORDER:
        pool = tiers.get(tier, [])
        take = min(args.per_difficulty, len(pool))
        sample = rng.sample(pool, take)
        sample.sort(key=lambda it: ds[it[0]]["problem_id"])  # stable order within tier
        counts[tier] = {"usable": len(pool), "sampled": take}
        for i, tests in sample:
            row = ds[i]
            picked.append({
                "problem_id": row["problem_id"], "split": args.split,
                "difficulty": row["difficulty"],
                "kind": "call" if tests["fn_name"] else "stdio",
                "fn_name": tests["fn_name"], "n_tests": tests["n_tests"],
                "question": row["question"],
                "starter_code": row.get("starter_code") or "",
            })

    with open(out, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    meta = {"dataset": "codeparrot/apps (parquet branch)", "split": args.split,
            "seed": args.seed, "per_difficulty": args.per_difficulty,
            "method": ("filter problems with missing/broken tests, group by "
                       "difficulty, random.Random(seed).sample per tier, "
                       "order by tier then problem_id"),
            "counts": counts, "n_rows": len(picked),
            "n_filtered_no_tests": n_filtered,
            "tier_order": TIER_ORDER}
    with open(os.path.splitext(out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[eval_list] {len(picked)} tasks -> {out}")
    print(f"[eval_list] per tier: " + ", ".join(
        f"{t}={counts[t]['sampled']}/{counts[t]['usable']}" for t in TIER_ORDER))
    print(f"[eval_list] filtered (no/broken tests): {n_filtered}")


if __name__ == "__main__":
    main()
