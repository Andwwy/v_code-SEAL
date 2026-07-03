# v_code — porting SEAL's reasoning-steering to code

**Goal:** reproduce SEAL's reasoning-efficiency steering on *code* tasks. SEAL builds a steering
vector from the difference between "reflection / transition" thoughts and "execution" thoughts in a
reasoning model, then subtracts it during decoding to cut redundant reasoning while preserving
accuracy. This repo ports that to code with a **code-adapted thought taxonomy** and evaluates it on
three benchmarks with `DeepSeek-R1-Distill-Qwen-1.5B`.

## Layout

| dir | what |
|---|---|
| `extraction/` | the code-adapted thought taxonomy (`thought_tags.py`) + probes/evidence that justify it |
| `vectors/` | the shipped steering vector `mbpp_v_code.pt` (+ `.meta.json` provenance) |
| `eval/` | 3-benchmark steering eval (MBPP / GSM8K / LogiQA) — `run_eval.py`, `steering.py`, `benchmarks.py` |
| `results/` | eval outputs — per-task correctness + token counts (`<bench>.json`) and `summary.json` |

## Pipeline

1. **Taxonomy** (`extraction/`) — split each reasoning trace on `\n\n` and classify every thought as
   execution / reflection / transition using **code-adapted** keyword lists (promoted `wait` /
   `alternatively` from prefix-only → `contains`, added code cues like `edge case`, `double-check`;
   see `extraction/README.md`). This is the part that makes it a *code* steering vector rather than a
   math one.
2. **Vector** (`vectors/mbpp_v_code.pt`) — built with **SEAL's own pipeline** (fork script
   `scripts/generate_vector_mbpp.sh`: `gen_mbpp_vllm.py` vLLM generation → `hidden_analysis.py` HF
   forward pass → `vector_generation.py`), driven by the taxonomy above. Convention
   **`S = H_RT − H_E`** (reflection∪transition − execution), layer 20, ~120 traces/class
   (~1/10 SEAL scale), extracted on clean single-BOS activations. Provenance in
   `vectors/mbpp_v_code.pt.meta.json`.
3. **Steering eval** (`eval/`) — add `coef · S` to the layer-20 residual at each `\n\n` inside the
   `<think>` block. With `S = H_RT − H_E`, **`coef = −1.0`** pushes toward execution (less reflection).
   Batched greedy decode, baseline vs steered, scored per benchmark.

## Results (n = 100 each, layer 20, coef −1.0)

| benchmark | base | steered | Δacc | tokens |
|---|---|---|---|---|
| MBPP | 37% | 45% | **+8** | 2311 → 2048 (−11%) |
| GSM8K | 64% | 67% | **+3** | 423 → 423 (−0%) |
| LogiQA | 36% | 44% | **+8** | 1620 → 1276 (−21%) |

Steering shortens reasoning and slightly improves accuracy — the SEAL effect, at small scale.

> **Read the deltas carefully.** These runs used `max_tokens = 3072` (SEAL production is 10000). On
> MBPP/LogiQA the longer *baseline* is truncated more often, and a truncated trace never reaches its
> answer (scored wrong) — this deflates the baseline and inflates the +8 deltas. GSM8K is the only
> truncation-free benchmark (short traces) and shows **+3**, the honest estimate of the true effect.
> Re-run at `max_tokens = 10000` for un-truncated numbers.

## Reproduce

**Eval** (needs the vector; runs in the SEAL env, HF only — no vLLM):

```bash
cd eval
VECTOR=../vectors/mbpp_v_code.pt bash run_eval.sh    # coef -1.0, layer 20, MBPP/GSM8K/LogiQA, n=100
```

**Vector** (built with the SEAL fork, not this repo): `scripts/generate_vector_mbpp.sh` in the SEAL
fork runs `gen_mbpp_vllm.py --remove_bos → hidden_analysis.py → vector_generation.py`, using the
keyword lists in `extraction/thought_tags.py`.

## Convention — don't flip the sign

`vectors/mbpp_v_code.pt` is **`S = H_RT − H_E`**; apply with **`coef −1.0`**. A `mean(execution) −
mean(reflection)` vector would be the *opposite* sign and would steer backward at `coef −1.0`. The
convention is recorded in `vectors/mbpp_v_code.pt.meta.json`; `eval/` defaults to `coef −1.0`.
