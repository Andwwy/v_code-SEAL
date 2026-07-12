# v_code extraction 2.0 — SEAL-faithful code steering vector from APPS

**Goal.** Rebuild the code-derived reasoning steering vector at full SEAL scale, fixing the two
weaknesses of v1 (MBPP): (a) extraction likely came from the same MBPP *test* split the eval reads,
(b) MBPP is too small (374 train problems) to match SEAL's 500+500 recipe (~40–60k activations).

**Why APPS.** Same authors/year as MATH (Hendrycks et al., 2021) — it is MATH's coding counterpart:
real train/test separation and a difficulty axis, which MBPP lacks entirely.

| difficulty | train | test |
|---|---|---|
| introductory | 2,639 | 1,000 |
| interview | 2,000 | 3,000 |
| competition | 361 | 1,000 |
| **total** | **5,000** | **5,000** |

Load via the **parquet branch** of `codeparrot/apps` — the script loader is dead on modern `datasets`:
`load_dataset("parquet", data_files="https://huggingface.co/datasets/codeparrot/apps/resolve/refs%2Fconvert%2Fparquet/all/<split>/0000.parquet")`
(test split has shards `0000` + `0001`).

## SEAL's exact recipe (what "faithful" means)

Verified against `VITA-Group/SEAL` code, not just the paper:

1. **Generate**: vLLM, greedy (temp 0, n=1), `max_tokens 10000`, `--remove_bos` (single BOS),
   over the **train split in file order** — no shuffling (their `random.seed(42)` is never used).
2. **Score** each trace correct/incorrect.
3. **Select**: first **500 correct + first 500 incorrect** in order (`--start 0 --sample 500`).
   Deterministic prefix, not a random sample. (Their MATH train file is subject-grouped, so their
   1,000 traces came from ~2 of 7 subjects — we replicate the *mechanism*, and can optionally add a
   seeded-random variant as an ablation.)
4. **Extract**: HF forward pass per trace; keep hidden states of every `\n\n` token (`ĊĊ` vocab match)
   **inside `<think>` only**; tag thoughts by keywords (reflection checked before transition, else
   execution). We use the code-adapted lists from v1 `extraction/thought_tags.py`.
5. **Vector**: `S = mean(reflection∪transition) − mean(execution)` at **layer 20**
   (code convention `H_RT − H_E`, *opposite* of the paper's formula; apply with **coef −1.0**).

## Pipeline

### Stage 1 — trace generation (GPU, vLLM)
- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; greedy, 10k max tokens, single-BOS.
- Iterate APPS **train** in file order; stop once both pools have 500 (expect low accuracy at 1.5B —
  correct pool is the bottleneck; may need 2,000–5,000 problems, ~10–25M tokens, a few hours on 24GB).
- Save per-problem: prompt, trace, difficulty, problem idx, split. **Record split + idx in metadata**
  (the v1 lesson).

### Stage 2 — scoring harness (CPU)
- APPS is **stdin/stdout**: run generated program against `input_output` pairs, compare stdout
  (whitespace-normalized), per-test timeout, subprocess sandbox. This replaces the MBPP assert-runner.
- **Filter problems with missing/empty test cases** (known APPS defect); log test counts per problem.
- Reused at eval time for pass@1.

### Stage 3 — hidden-state extraction (GPU, HF)
- SEAL's `hidden_analysis.py` flow: one forward pass per selected trace, all-layer hidden states at
  `\n\n` positions, correct + incorrect pools kept separate (`hidden_correct_0_500`, `hidden_incorrect_0_500`).
- Disk: all 29 layers ≈ ~8MB/trace → **~8GB total**; trim to layers 15–25 if tight.

### Stage 4 — vector generation (CPU, seconds)
- SEAL's `vector_generation.py` verbatim: layer 20, `--prefixs correct_0_500 incorrect_0_500`.
- Ship as `vectors/apps_v_code.pt` + meta.json recording: model, layer, sign convention
  (`H_RT − H_E`, apply coef −1.0), keyword list version, **split + problem indices**, trace/activation counts.

### Stage 5 — eval (GPU HF + CPU harness)
- Steered decode: hook adds `coef · S` to layer-20 residual at each `\n\n` inside `<think>`;
  baseline vs `coef −1.0`; greedy; **max_tokens 10000** (not 3072 — v1 truncation bug); single-BOS.
- Benchmarks:
  - **APPS test** (in-domain, disjoint by construction): n≈100 per difficulty tier → replicates
    SEAL's Fig. 2 difficulty-stratified analysis in the code domain.
  - **GSM8K test** (math transfer), **LogiQA 2.0 test** (logic transfer) — reuse v1 harness.
- Report per benchmark: accuracy, mean tokens, Δ both; plus thought-type counts baseline vs steered.

## Requirements

| resource | need |
|---|---|
| GPU | 1× 24GB (4090/A5000, RunPod as in v1) — 1.5B fits everywhere |
| GPU time | ~10–20 h total (generation few h, extraction ~2 h, eval several h/bench) |
| CPU | scoring harness, parallel, ~1–2 h |
| disk | ~10 GB (hidden states dominate; traces ~100s MB) |
| deps | vLLM (gen), transformers (extraction+eval), datasets (parquet branch), no training |

## Known risks
- 1.5B on APPS competition tier ≈ 0% correct → correct pool fills from introductory/interview only
  (mirrors SEAL's correct-skews-easy effect; document it, don't fight it).
- APPS broken test cases → filter, log.
- Sign convention: vector is `H_RT − H_E`; **never** apply with positive coef expecting less reflection.
- Keep eval max_tokens at 10000 and single-BOS everywhere (both v1 bugs).

## Deliverables
1. `extraction/` — gen + scoring + hidden-analysis scripts (APPS-adapted)
2. `vectors/apps_v_code.pt` + full-provenance meta.json
3. `eval/` — steered eval on APPS-test / GSM8K / LogiQA
4. `results/` — per-task JSON + summary, difficulty-stratified for APPS
