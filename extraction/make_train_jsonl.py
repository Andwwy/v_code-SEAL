"""Materialize data/APPS/train_code.jsonl — the code analog of SEAL's
data/MATH/train_math.jsonl.

APPS train split from the parquet branch, in FILE ORDER (the order the
extraction scans), one row per problem with SEAL-compatible core fields
(problem, level — what SEAL reads from its train file) plus provenance:

  {problem, level, problem_id, split, difficulty, kind, fn_name, n_tests,
   starter_code, has_tests}

Test cases are NOT embedded (they would add ~100MB); scoring always reloads
input_output from the dataset by problem_id. has_tests=false marks the known
APPS defect rows that the pipeline filters (and logs) before generation.

NOTE: the per-run data.jsonl written by gen_apps_vllm.py remains what Stage 3
consumes — it is the subset of THIS file that was actually generated, aligned
row-for-row with math_eval.jsonl (SEAL's zip contract). This file is the
dataset-level artifact the pointer in train_registry.py resolves to.
"""
import argparse
import json
import os

from apps_data import load_apps, parse_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None,
                    help="default: data/APPS/train_code.jsonl (for split=train)")
    args = ap.parse_args()
    out = args.out or os.path.join("data", "APPS",
                                   "train_code.jsonl" if args.split == "train"
                                   else f"{args.split}_code.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ds = load_apps(args.split)
    n_no_tests = 0
    with open(out, "w") as f:
        for i in range(len(ds)):
            row = ds[i]
            tests = parse_tests(row)
            if tests is None:
                n_no_tests += 1
            f.write(json.dumps({
                "problem": row["question"],
                "level": row["difficulty"],
                "problem_id": row["problem_id"],
                "split": args.split,
                "difficulty": row["difficulty"],
                "kind": (None if tests is None
                         else ("call" if tests["fn_name"] else "stdio")),
                "fn_name": tests["fn_name"] if tests else None,
                "n_tests": tests["n_tests"] if tests else 0,
                "starter_code": row.get("starter_code") or "",
                "has_tests": tests is not None,
            }) + "\n")
    print(f"[train_jsonl] {len(ds)} rows -> {out} "
          f"({n_no_tests} rows with has_tests=false)")


if __name__ == "__main__":
    main()
