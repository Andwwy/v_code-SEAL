#!/usr/bin/env python3
"""
Probe: does R1-Distill's *code* reasoning (on MBPP) decompose into SEAL's three
thought types (execution / reflection / transition)?

For N MBPP problems: generate a reasoning trace, split the <think> block on blank
lines into thoughts, and classify each with SEAL's EXACT keyword rules (copied from
hidden_analysis.generate_index). Prints every thought labeled + per-trace and total
counts, and saves full traces to jsonl so we can eyeball whether the 3-tag scheme
maps cleanly to code — or whether code needs new/renamed categories.

Runs on Mac (MPS/CPU) for small N; no vLLM needed.
  python probe_mbpp_thoughts.py --n 5 --max_tokens 3000
"""
import argparse, json, os, re

# --- SEAL's exact thought-classification keywords (hidden_analysis.generate_index) ---
CHECK_WORDS  = ["verify", "make sure", "hold on", "think again", "'s correct",
                "'s incorrect", "let me check", "seems right"]
CHECK_PREFIX = ["wait"]
SWITCH_WORDS = ["think differenly", "another way", "another approach", "another method",
                "another solution", "another strategy", "another technique"]
SWITCH_PREFIX = ["alternatively"]


def classify(step_text):
    s = step_text.strip().lower()
    if any(s.startswith(p) for p in CHECK_PREFIX) or any(w in s for w in CHECK_WORDS):
        return "reflection"
    if any(s.startswith(p) for p in SWITCH_PREFIX) or any(w in s for w in SWITCH_WORDS):
        return "transition"
    return "execution"


def split_thoughts(gen):
    body = gen
    if "<think>" in body:
        body = body.split("<think>", 1)[1]
    if "</think>" in body:
        body = body.split("</think>", 1)[0]
    return [t for t in re.split(r"\n\s*\n", body) if t.strip()]


def build_prompt(tok, problem, tests):
    instr = ("Please solve the following Python programming problem. "
             "Think step-by-step, then give the final function.\n\n"
             f"Problem: {problem}\n\nYour function should pass these tests:\n"
             + "\n".join(tests))
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--out", default="v_code/mbpp_probe_traces.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dataset_repo", default="google-research-datasets/mbpp",
                    help="HF dataset id (new hub needs namespace/name, not bare 'mbpp')")
    ap.add_argument("--config", default="sanitized")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[probe] device={device} dtype={args.dtype} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    ds = load_dataset(args.dataset_repo, args.config, split="test")
    rows = ds.select(range(args.n))

    results, totals = [], {"execution": 0, "reflection": 0, "transition": 0}
    for r in rows:
        problem = r.get("prompt") or r.get("text")
        tests = r.get("test_list", [])
        prompt = build_prompt(tok, problem, tests)
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, do_sample=False, max_new_tokens=args.max_tokens)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        thoughts = split_thoughts(gen)
        labeled = [(classify(t), t) for t in thoughts]
        per = {k: sum(l == k for l, _ in labeled) for k in totals}
        for k in totals: totals[k] += per[k]
        results.append({"task_id": r.get("task_id"), "problem": problem,
                        "generation": gen,
                        "thoughts": [{"type": l, "text": t} for l, t in labeled]})
        closed = "</think>" in gen
        print(f"\n===== task {r.get('task_id')} | {len(thoughts)} thoughts "
              f"| exec {per['execution']} refl {per['reflection']} trans {per['transition']} "
              f"| think_closed={closed} =====")
        for l, t in labeled:
            print(f"  [{l:10}] {t.strip()[:180].replace(chr(10),' ')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[probe] totals over {args.n} traces: {totals}")
    print(f"[probe] saved full traces -> {args.out}")


if __name__ == "__main__":
    main()
