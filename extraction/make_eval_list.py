"""Emit a seeded-RANDOM eval task list (jsonl) from the APPS TRAIN split.

The extraction pools are a deterministic greedy prefix (first-500-correct +
first-500-incorrect over train in file order — SEAL's design, never random).
The EVAL task list is the opposite: a seeded random sample of the SAME train
split, drawn uniformly across the whole usable pool — difficulty is random
(NOT stratified per tier), so the realized difficulty mix follows the pool.

Because extraction consumes a file-order prefix of train, an eval sample from
train can overlap the extraction problems. Pass --exclude_from with the
extraction run's math_eval.jsonl to sample only from problems extraction never
touched (run_extraction.sh does this — generation has finished by then). The
overlap is always quantifiable afterwards: both files record problem_id.

Rows carry everything needed to prompt at eval time (question, starter_code,
fn_name); test cases are reloaded from the dataset by problem_id when scoring
(apps_scoring.py). Membership is random (seeded); rows are ordered by
problem_id for stable diffs.

  <out>.jsonl       {problem_id, split, difficulty, kind, fn_name, n_tests,
                     question, starter_code}
  <out>.meta.json   {seed, n_total, counts, excluded, method}
"""
import argparse
import json
import os
import random

from apps_data import load_apps, parse_tests
import train_registry

TIER_ORDER = ["introductory", "interview", "competition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_total", type=int, default=300,
                    help="problems sampled uniformly across the usable pool "
                         "(difficulty random, not stratified)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_from", default=None,
                    help="optional math_eval.jsonl whose problem_ids are "
                         "excluded (problems already consumed by extraction)")
    ap.add_argument("--out", default=None,
                    help="output path; default eval_tasks/apps_<split>_n<n>_seed<seed>[_excl].jsonl")
    args = ap.parse_args()
    if args.split == "train":
        # eval tasks come from the ACTIVE train set (the `train` pointer);
        # this script implements the code set (APPS)
        train_registry.require("train_code", "make_eval_list")
    suffix = "_excl" if args.exclude_from else ""
    out = args.out or os.path.join(
        "eval_tasks", f"apps_{args.split}_n{args.n_total}_seed{args.seed}{suffix}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    excluded_ids = set()
    if args.exclude_from:
        with open(args.exclude_from) as f:
            excluded_ids = {json.loads(ln)["problem_id"] for ln in f}

    ds = load_apps(args.split)
    pool, n_filtered, n_excluded = [], 0, 0
    for i in range(len(ds)):
        row = ds[i]
        tests = parse_tests(row)
        if tests is None:  # same filter as extraction: unusable tests are not evaluable
            n_filtered += 1
            continue
        if row["problem_id"] in excluded_ids:
            n_excluded += 1
            continue
        pool.append((i, tests))

    rng = random.Random(args.seed)
    take = min(args.n_total, len(pool))
    sample = rng.sample(pool, take)
    sample.sort(key=lambda it: ds[it[0]]["problem_id"])  # stable order, random membership

    picked, diff_counts = [], {t: 0 for t in TIER_ORDER}
    for i, tests in sample:
        row = ds[i]
        diff_counts[row["difficulty"]] = diff_counts.get(row["difficulty"], 0) + 1
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
            "train_set": train_registry.TRAIN_NAME,
            "seed": args.seed, "n_total_requested": args.n_total,
            "method": ("filter problems with missing/broken tests, exclude "
                       "extraction-used problem_ids if --exclude_from given, "
                       "random.Random(seed).sample over the WHOLE remaining "
                       "pool (difficulty random, not stratified), order by "
                       "problem_id"),
            "n_rows": len(picked),
            "difficulty_counts_realized": diff_counts,
            "pool_usable": len(pool),
            "n_filtered_no_tests": n_filtered,
            "excluded": {"source": args.exclude_from, "n_excluded": n_excluded}}
    with open(os.path.splitext(out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[eval_list] {len(picked)} tasks -> {out}")
    print("[eval_list] difficulty mix (random, not stratified): "
          + ", ".join(f"{t}={diff_counts.get(t, 0)}" for t in TIER_ORDER))
    print(f"[eval_list] pool={len(pool)} usable | filtered={n_filtered} | "
          f"excluded (extraction overlap)={n_excluded}")


if __name__ == "__main__":
    main()
