# v_code — does a code-derived reasoning-steering vector generalize?

**Thesis.** SEAL builds a reasoning-efficiency steering vector on **math** (MATH train), using keyword
rules tuned for math reasoning, then subtracts it during decoding to cut redundant reasoning while
holding accuracy. This project asks a different question: build the vector from a **code** task
(MBPP), using the **same three SEAL thought-types** (execution / reflection / transition) but keyword
rules **adapted for code**, and test whether that code-derived vector **generalizes** across domains.

The method and the tags are SEAL's; only the *task* (code, not math) and the *keyword adaptation*
change. We then apply the single code vector to three domains with `DeepSeek-R1-Distill-Qwen-1.5B` —
**MBPP (code, in-domain)**, **GSM8K (math)**, **LogiQA (logic)** — where GSM8K and LogiQA test
cross-domain **transfer**.

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

| benchmark | domain | base | steered | Δacc | tokens |
|---|---|---|---|---|---|
| MBPP | code (in-domain) | 37% | 45% | **+8** | 2311 → 2048 (−11%) |
| GSM8K | math (transfer) | 64% | 67% | **+3** | 423 → 423 (−0%) |
| LogiQA | logic (transfer) | 36% | 44% | **+8** | 1620 → 1276 (−21%) |

> **⚠️ These numbers are from a buggy run — re-run before citing.** They were produced with a
> **double-BOS** tokenization bug and `max_tokens = 3072` (SEAL uses single-BOS `--remove_bos` and
> `max_tokens = 10000`). Both are now fixed in `eval/`, but `results/` predates the fix.
> - **MBPP / LogiQA:** the longer *baseline* was truncated more than the steered run (a truncated
>   trace never reaches its answer → scored wrong), which deflates the baseline and **inflates** the
>   +8 deltas.
> - **GSM8K was hit hardest.** The double-BOS prefix collapsed the model out of its reflective mode
>   into a terse "First, I need to…" mode — baseline averaged **423 tokens with zero reflection** vs
>   SEAL's faithful GSM8K baseline of **~2000 tokens**. With no reflection to suppress, steering did
>   nothing (−0% tokens, +3 noise). This is the *most* corrupted result, not a clean control.
>
> **Faithful reference** — SEAL's *in-domain* run (math vector on GSM8K, `--remove_bos`, 10000 tok):
> **73.8% → 81.2% (+7.4), 1997 → 1028 tokens (−49%)**. Our code vector is evaluated cross-domain, so
> expect a *smaller* transfer effect — but re-running with the fixed harness should restore the
> ~2000-token reflective baseline and a real (non-zero) effect. **Re-run all three; GSM8K first.**

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
