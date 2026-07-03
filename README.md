# v_code — porting SEAL's steering to code

Goal: build a code steering vector (`v_code`) the SEAL way. First we had to check
whether R1-Distill's **code** reasoning decomposes into SEAL's three thought types
(execution / reflection / transition) — and if the math keyword rules transfer.

## How we tested it
Ran R1-Distill-Qwen-1.5B on a few problems, split each reasoning trace on `\n\n`,
classified every thought with SEAL's **original** keywords, and eyeballed the labels:
- **5 MBPP** problems — `probe_mbpp_thoughts.py`
- **5 hard LiveCodeBench** problems — `probe_lcb_thoughts.py` (and `lcb_hard_probe_colab.ipynb` for Colab)
- tag counts + missed-cue analysis — `analyze_tags.py`

## What we found
| Dataset | execution | reflection | transition |
|---|---|---|---|
| MBPP (5) | 123 | 23 | **0** |
| LiveCodeBench hard (3, partial) | 253 | 170 | **16** |

- **All 3 tags map to code** — but transition only shows up on **hard** problems (MBPP is too easy → the model rechecks rather than switches strategy).
- SEAL's **prefix-only** rule for `Wait`/`Alternatively` **undercounts** on code: reflection/switching happens *mid-thought* ("…but wait", "…alternatively"), landing in execution.
- Noise to avoid: bare `another` ("another test case" = execution) and `perhaps` (ubiquitous hedging).

## Decision: keyword lists we use (code-adapted, all `contains`, case-insensitive)
Reflection is checked before transition; anything matching neither = execution. Source of truth: `thought_tags.py`.

| Tag | Keywords |
|---|---|
| **Reflection** | `wait`, `but wait`, `verify`, `make sure`, `hold on`, `think again`, `'s correct`, `'s incorrect`, `let me check`, `seems right`, `hmm`, `what if`, `double-check`, `recheck`, `edge case` |
| **Transition** | `alternatively`, `another way`, `another approach`, `another method`, `another solution`, `another strategy`, `another technique`, `think differently`, `instead`, `a better way`, `rethink`, `start over`, `on second thought` |
| **Execution** | — (default) |

Changes vs SEAL original: promoted `wait`/`alternatively` from prefix-only → `contains`; fixed the dead `think differenly` typo; added code cues (`but wait`, `hmm`, `what if`, `double-check`, `recheck`, `edge case`, `instead`, …). `hmm`/`edge case`/`instead` are still provisional pending the full LiveCodeBench traces.

## Evidence
- `evidence/mbpp_probe_traces.jsonl` — 5 MBPP traces, full text
- `evidence/lcb_hard_partial.json` — 3 hard LiveCodeBench traces (salvaged from the interrupted run; thoughts truncated to 180 chars)

## Pipeline files
- `thought_tags.py` — classifier + `\n\n`-boundary extraction (our keywords)
- `probe_*.py`, `lcb_hard_probe_colab.ipynb`, `analyze_tags.py` — the tests above

## Building the steering vector
The `v_code` MBPP steering vector is built with **SEAL's own pipeline** (vLLM generation →
HF forward pass → `vector_generation.py`), driven by the code-adapted keywords above — not a
standalone script in this repo. This repo holds the keyword taxonomy + probing that the
extraction consumes; the SEAL convention is `S = H_RT − H_E` (apply with `coef −1.0` in the eval).
