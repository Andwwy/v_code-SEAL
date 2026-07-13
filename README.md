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

Pod gotcha: images that preinstall vllm>=0.11 crash at `import vllm` with
`RuntimeError: Could not load libtorchcodec` / `libnvrtc.so.13: cannot open
shared object file` — newer vllm imports torchcodec (video support this
pipeline never uses) at startup, and pod images lack the CUDA-13 NVRTC lib
and FFmpeg it needs. Fix: `pip install -r requirements.txt` (pins vllm<0.11,
which pulls its own compatible torch and skips torchcodec entirely), or keep
the preinstalled stack and run
`pip install nvidia-cuda-nvrtc-cu13 && apt-get update && apt-get install -y ffmpeg`.

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

Deviations from upstream, the complete list:

1. **Dataset + scorer** (the point of the project): APPS train via the
   `train_code` pointer; test-based scoring with semantics matching SEAL's own
   `code_evaluation/testing_util.py`; problems with missing/broken tests
   filtered and logged (`skipped.jsonl`) — a test-based scorer cannot label
   them. Skipping never reorders the remainder.
2. **Keyword lists**: code-adapted by default (`keyword_set = "code"` in
   `thought_tags.py`); flip to `"seal_math"` for upstream's verbatim lists
   (incl. the `Wait`/`Alternatively` prefixes and the `"think differenly"`
   typo) — verified to reproduce upstream tagging exactly.
3. **Prompt-length handling** (forced by APPS, absent upstream): upstream's
   `max_model_len = max_tokens + 2000` assumed MATH-sized prompts (its own
   data never hit the cap). APPS has 6/4,450 usable prompts over 2,000 tokens
   (one is 13,644 tokens — the verbatim formula would *crash* on it). We give
   every prompt ≤ 4,096 tokens the full 10,000-token generation budget
   (`max_model_len = 10000 + 4096 + 16`) and skip longer ones (logged). Under
   the upstream formula those 6 problems would get truncated budgets instead.
4. **Mechanical only** (proven not to change results — see validation):
   batched left-padded forwards with pad-aware position_ids, `--dtype`
   (default float32 = upstream-era numerics), `--keep_layers`, logits trimmed
   to the last position, chunked generation with early stop (≡ prefix
   selection), resume/repair, TP=1, `trim_output` applied exactly as upstream
   (generations truncated at upstream's three markers before scoring and
   extraction).

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

## Executable validation against upstream VITA-Group/SEAL (2026-07-12)

A fresh clone of https://github.com/VITA-Group/SEAL (not the locally-modified
fork) was compared and *executed* against this pipeline:

- **Vector calculation**: upstream `vector_generation.py` and ours produce
  **bitwise-identical** steering vectors on identical synthetic 29-layer
  pools (layer 20, irregular pools incl. empty reflection/transition sets),
  and both match the hand-computed
  `mean(H_reflection ∪ H_transition) − mean(H_execution)`. Apply with
  coef −1.0, per upstream `steering.sh`.
- **Hidden states**: with the real DeepSeek-R1-Distill-Qwen-1.5B (fp32, CPU),
  our pipeline reproduces upstream's per-trace hidden states **bitwise at
  batch_size=1** across all 29 layers; batch_size=2 differs by ~1.8e-6
  relative (GEMM reduction-order noise). Skeptic agents additionally verified
  the left-padded position_ids recipe ≡ unpadded forwards at 2e-17 in fp64
  and no attention leak across pad positions.
- **Boundary discovery + tagging**: `generate_index` machinery byte-identical
  in outputs given the same lists (real tokenizer, 7 edge-case traces, both
  think_only modes); step boundaries never differ; `keyword_set="seal_math"`
  reproduces upstream tagging verbatim.
- **Recipe**: greedy n=1 / temp 0 / max_tokens 10000 / single BOS / file
  order / `--start 0 --sample 500` prefix / `hidden_{type}_0_500` naming /
  stage arguments — all verified equal to upstream `generate_vector.sh` +
  `eval_MATH_vllm.py`; upstream's `trim_output` is applied verbatim.
- **transformers-v5 gotcha found during validation**: under transformers 5.x,
  `from_pretrained` without a dtype loads the checkpoint dtype (bf16) — so
  upstream's own script re-run today would silently change numerics. Our
  explicit `--dtype float32` (plus a post-load dtype assert) pins the
  published SEAL-era behavior.

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
