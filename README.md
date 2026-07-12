# v_code extraction 2.0 — SEAL-faithful APPS steering vector (500+500 greedy)

Rebuilds the code-domain reasoning steering vector at full SEAL scale on APPS
train (real train/test separation + difficulty axis, unlike v1's MBPP).
Design rationale and risks: [PLAN.md](PLAN.md).

## Pipeline (extraction/)

```
train_registry.py                the `train` POINTER: train = train_code selects
                                 the active train set; all entry points resolve
                                 through it and refuse a set they don't implement
make_train_jsonl.py              materializes data/APPS/train_code.jsonl
gen_apps_vllm.py      Stage 1+2  vLLM greedy traces over the active train set in
                                 file order, scored live by apps_scoring.py,
                                 stops at 500+500
apps_data.py                     APPS parquet-branch loader + test filtering + prompts
apps_scoring.py       Stage 2    stdin/stdout + call-based subprocess harness
                                 (reused at eval time for pass@1)
hidden_analysis.py    Stage 3    SEAL's hidden_analysis, APPS-adapted: \n\n ("ĊĊ")
                                 boundary hidden states inside <think>, tagged by
                                 thought_tags.py keyword lists
vector_generation.py  Stage 4    SEAL VERBATIM: layer 20, S = H_RT − H_E
package_vector.py     Stage 4b   vectors/apps_v_code.pt + full-provenance meta.json
make_eval_list.py     Stage 4c   seeded-RANDOM eval task list from APPS *train*,
                                 difficulty random (NOT stratified; jsonl + meta)
                                 — fixed once so every eval run scores the same
                                 problems
run_extraction.sh                orchestrates all of the above
```

Selection design, explicitly: **extraction pools are greedy first-filled**
(first 500 correct + first 500 incorrect over train in file order — SEAL's
`--start 0 --sample 500`, deterministic, never random); the **eval task list
is seeded-random from the same train split, difficulty random** (uniform over
the usable pool, so the realized mix follows the pool). Because extraction
consumes a file-order prefix of train, the orchestrator regenerates the list
after generation with `--exclude_from` so eval never scores a problem whose
trace built the vector (the v1 lesson); the frozen no-exclusion list ships in
`eval_tasks/` and overlap is always quantifiable from recorded problem_ids.

Run on a GPU pod (24GB is enough; 1.5B model):

```bash
pip install -r requirements.txt
bash extraction/run_extraction.sh 0            # gpu index; env vars override knobs
```

Interrupted generation resumes for free (`--resume` skips finished problem_ids;
correct/incorrect prefixes are deterministic, so resuming never changes the
selection).

## Train sets and the `train` pointer (data/)

SEAL's repo keeps its train data at `data/MATH/train.jsonl`. This repo carries
both domains in that convention, renamed per set:

| set | file | contents |
|---|---|---|
| `train_math` | `data/MATH/train_math.jsonl` | SEAL's own MATH train file (7,500 rows: problem, level, type, solution), renamed |
| `train_code` | `data/APPS/train_code.jsonl` | APPS train split in file order (5,000 rows: problem, level + provenance; `has_tests` marks the 550 defect rows), built by `make_train_jsonl.py` |

The active set is a single variable in
[train_registry.py](extraction/train_registry.py):

```python
train = "train_code"   # POINTER — flip to "train_math" for the math set
```

No other file names a concrete set: `gen_apps_vllm.py`, `make_eval_list.py`,
and `run_extraction.sh` all resolve the pointer (and hard-fail if it selects a
set they don't implement — this repo's stages implement `train_code`; math
runs go through SEAL's original boxed-answer pipeline). The active set name is
recorded in `gen_config.json` and the eval-list meta, and a resumed run
refuses a changed pointer. Test cases are not embedded in `train_code.jsonl`
(~100MB); scoring reloads them from the dataset by `problem_id`.

## What "SEAL-faithful" pins down

| knob | value | source |
|---|---|---|
| decoding | greedy, n=1, `max_tokens 10000`, single BOS | `eval_MATH_vllm.py`, `generate_vector.sh` |
| trace order | train split file order, no shuffling | their `random.seed(42)` is never used |
| selection | first 500 correct + first 500 incorrect (`--start 0 --sample 500`) | deterministic prefix, not a random sample |
| boundaries | every vocab token containing `ĊĊ`, inside `<think>` only | `hidden_analysis.py` |
| tagging | reflection checked before transition, else execution; code-adapted contains-lists | v1 `thought_tags.py` |
| vector | layer 20, `mean(refl∪trans) − mean(exec)` | `vector_generation.py` (verbatim) |
| sign | vector = `H_RT − H_E` → **apply with coef −1.0** | code convention, opposite of the paper's formula |

Deviations (all mechanical, none affect the math): APPS test-runner replaces
the MATH answer-checker, with grading semantics matching SEAL's own
`code_evaluation/testing_util.py` (audited — see below); problems with
missing/broken tests are filtered and logged (`skipped.jsonl`); batched
forward passes + optional layer subsetting + logits trimming in Stage 3
(bit-identical hidden states, just memory/disk).

## Outputs

- `results/train_code/<model>/baseline_10000/` — traces (`math_eval.jsonl`),
  pools (`hidden_{correct,incorrect}_0_500/`), skips, per-pool `selection.json`
  (the results dir is keyed by the active `train` pointer)
- `vectors/apps_v_code.pt` + `vectors/apps_v_code.meta.json` — the vector and
  its provenance (model, layer, sign, keyword lists, split + problem ids,
  activation counts, full gen config)
- `eval_tasks/apps_train_n300_seed42.jsonl` + `.meta.json` — the frozen random
  eval task list (train split, difficulty random; seed, realized difficulty
  mix, and filter/exclusion stats recorded in the meta). The pipeline run adds
  `apps_train_n300_seed42_excl.jsonl`, the same sampling with extraction-used
  problems excluded.

## Measured on real APPS train data (local validation, 2026-07-11)

- 5,000 train rows load in file order; **4,450 usable** (550 filtered for
  missing/empty tests — the known APPS defect). Difficulty counts match the
  plan table exactly. Kind split: **2,714 call-based / 1,736 stdio** (61% of
  train is LeetCode-style `fn_name` — both kinds are handled).
- Tests/problem: mean 5.8, median 3, max 1440.
- Frozen eval list (train, seed 42, n=300, difficulty random): realized mix
  145 introductory / 117 interview / 38 competition — proportional to the
  4,450-problem usable pool; deterministic across runs. (For reference, the
  test split has all 5,000 problems with usable tests — the missing-tests
  defect is train-only.)
- Harness vs ground-truth solutions (first py3-parseable solution per problem,
  100 per kind): **stdio 87/100, call-based 100/100**. All 13 stdio failures
  are inherent to exact grading and fail identically under SEAL's own
  `testing_util`: 10 special-judge problems ("print any valid answer") plus 3
  float-precision problems (ground truth rounded differently than the
  solution prints). Consequence: the correct pool stays clean (exact pass ⇒
  genuinely correct); the incorrect pool is mildly diluted with
  correct-reasoning traces. `fail` is recorded per trace in math_eval.jsonl
  so this can be quantified later.
- APPS quirks handled: 9000+-digit integers in test data (Python ≥3.11
  int↔str cap lifted in parent + children), double-encoded string args in
  call-based problems (ground-truth outputs assume the RAW quoted string —
  args are passed as-is; expected accepted raw or json-decoded, and
  1-element-list-wrapped, per the original Hendrycks harness), list-of-lines
  stdin inputs, non-UTF-8 output, programs that `sys.exit(1)` after printing
  (SystemExit swallowed, like the reference).

## Adversarial fidelity audit (2026-07-11)

A 12-agent audit (5 review dimensions, each critical/major finding
independently re-verified against the code) compared this pipeline to the
SEAL reference:

- **Faithful as-is**: generation (greedy n=1 / 10k / single BOS / file order;
  early-stop proven equivalent to prefix selection), 500+500 selection path,
  hidden extraction (`vector_generation.py` byte-identical, keyword lists
  identical, logits-trimming verified to not touch hidden states).
- **7 confirmed findings, all fixed**: grading now mirrors `testing_util`
  exactly — per-line all-or-nothing Decimal rule with NO numeric tolerance,
  SEAL's `import_string` prepended to BOTH stdio and call-based solutions,
  call-based comparison is plain `==` after top-level tuple→list (the earlier
  `isclose` would have passed off-by-one integers at ≥1e6 magnitude); re-runs
  rebuild the vector (`--overwrite`); interrupted runs self-repair the
  two-file output alignment on resume; resume refuses silently-mixed scoring
  configs; an exhausted split hard-fails instead of building under-filled
  pools with `0_500` names.

## Expectations / gotchas

- 1.5B accuracy on APPS is low; the **correct pool is the bottleneck** — expect
  to scan a few thousand train problems (competition tier ≈ 0% correct; the
  correct pool skews easy, mirroring SEAL's own correct-skews-easy effect).
- Scoring runs model code in subprocesses with rlimits — that is *not* a
  security sandbox (same caveat as SEAL's `reliability_guard`).
- Eval (Stage 5) must keep `max_tokens 10000` and single-BOS (both v1 bugs),
  and apply the vector with **coef −1.0** at layer 20 on `\n\n` positions
  inside `<think>`.
