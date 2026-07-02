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
- `extract_vector.py` — SEAL extraction on MBPP → `S = mean(execution) − mean(reflection∪transition)`; `--balance` labels each trace correct/incorrect (runs MBPP `test_list`) and builds a 50/50 pool to a target activation budget
- `probe_*.py`, `lcb_hard_probe_colab.ipynb`, `analyze_tags.py` — the tests above

## Run the extraction on RunPod
Use a **PyTorch CUDA base image** + a `/workspace` volume. Upload this folder (or clone it), then:
```bash
cd /workspace/v_code
cp .env.example .env          # optional: add HF_TOKEN
bash setup_runpod.sh          # installs transformers/datasets/accelerate, sanity-checks GPU
bash run_extract.sh           # balanced 50/50 extraction -> vectors/mbpp_v_code_steervec.pt
```
`run_extract.sh` defaults to `TARGET_ACTIVATIONS=10500` (≈1/10 of SEAL), `LAYER=20`, `MAX_TOKENS=3000`,
scanning up to all 374 MBPP-`full` train tasks. Override via env, e.g. `TARGET_ACTIVATIONS=21000 bash run_extract.sh`.
Output: `vectors/mbpp_v_code_steervec.pt` + `.meta.json` (pool sizes, pass rate, tag counts, norm); log in `extract_run.log`.
Runtime ≈ 1.5–2 hr on an A100 (1.5B model, ~275 traces, HF generate — needed for hidden states).
