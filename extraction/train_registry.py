"""Train-set pointer — single switch for which train set the pipeline runs on.

The repo carries both train sets in SEAL's data/ convention:

  train_math : data/MATH/train_math.jsonl  (SEAL's own MATH train file,
               renamed from data/MATH/train.jsonl; 7500 rows:
               {problem, level, type, solution})
  train_code : data/APPS/train_code.jsonl  (APPS train split materialized in
               FILE ORDER from the codeparrot/apps parquet branch by
               make_train_jsonl.py; 5000 rows: {problem, level, + provenance};
               test cases stay in the dataset, reloaded by problem_id)

Every entry point that consumes "the train set" (gen_apps_vllm.py,
make_eval_list.py, run_extraction.sh) resolves it through the `train`
variable below and refuses to run if it points at a set the script does not
implement — flip the pointer here to switch domains, nothing else references
a concrete set name.

NOTE: this repo's stages 1-4 implement the CODE pipeline (APPS generation +
test-based scoring). train_math is carried for provenance and for math-domain
runs through SEAL's original scripts, which score by boxed-answer match.
"""
import argparse
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_SETS = {
    "train_math": {
        "domain": "math",
        "path": os.path.join(_ROOT, "data", "MATH", "train_math.jsonl"),
        "source": "VITA-Group/SEAL data/MATH/train.jsonl (renamed)",
    },
    "train_code": {
        "domain": "code",
        "path": os.path.join(_ROOT, "data", "APPS", "train_code.jsonl"),
        "source": "codeparrot/apps parquet branch, train split, file order "
                  "(make_train_jsonl.py)",
    },
}

# POINTER — the active train set. All other dependencies go through this.
train = "train_code"

TRAIN_NAME = train
TRAIN_PATH = TRAIN_SETS[train]["path"]
TRAIN_DOMAIN = TRAIN_SETS[train]["domain"]


def require(expected, script):
    """Guard for entry points: fail loudly if the pointer selects a set this
    script does not implement."""
    if train != expected:
        raise SystemExit(
            f"[{script}] train points at '{train}' but this script implements "
            f"'{expected}'. Flip `train` in extraction/train_registry.py "
            f"(or use the pipeline that implements {train}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="store_true", help="print active set name")
    ap.add_argument("--path", action="store_true", help="print active set path")
    args = ap.parse_args()
    if args.name:
        print(TRAIN_NAME)
    elif args.path:
        print(TRAIN_PATH)
    else:
        print(f"train = {TRAIN_NAME} ({TRAIN_DOMAIN}) -> {TRAIN_PATH}")
