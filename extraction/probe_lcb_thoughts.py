#!/usr/bin/env python3
"""
Probe R1-Distill's reasoning on HARD LiveCodeBench problems, to see whether
*transition* (strategy-switching) thoughts appear — MBPP was too easy and showed
zero. Same segmentation + SEAL keyword classifier as the MBPP probe, so results
are comparable. Prints every thought labeled, and (crucially) dumps full traces to
jsonl so we can read the actual switching vocabulary and design better regex.

  python probe_lcb_thoughts.py --n 5 --difficulty hard --max_tokens 6000
"""
import argparse, json, os, re

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


def build_prompt(tok, q_content, starter):
    instr = ("You will be given a competitive programming problem. "
             "Solve it in Python. Think step-by-step, then give the final solution.\n\n"
             + q_content)
    if starter and starter.strip():
        instr += f"\n\nUse this starter code:\n```python\n{starter}\n```"
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--max_tokens", type=int, default=6000)
    ap.add_argument("--out", default="evidence/lcb_probe_traces.jsonl")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--scan_cap", type=int, default=400, help="max rows to scan for difficulty match")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[probe-lcb] device={device} dtype={args.dtype} difficulty={args.difficulty}")

    # stream + project columns so we don't download the huge private_test_cases
    ds = load_dataset("livecodebench/code_generation", "default", split="test", streaming=True)
    ds = ds.select_columns(["question_title", "question_content", "difficulty", "starter_code"])
    picked, seen = [], 0
    for r in ds:
        seen += 1
        if r.get("difficulty") == args.difficulty:
            picked.append(r)
            if len(picked) >= args.n:
                break
        if seen >= args.scan_cap:
            break
    print(f"[probe-lcb] scanned {seen} rows, picked {len(picked)} '{args.difficulty}' problems")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    results, totals = [], {"execution": 0, "reflection": 0, "transition": 0}
    for r in picked:
        prompt = build_prompt(tok, r["question_content"], r.get("starter_code", ""))
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=8000).to(device)
        with torch.no_grad():
            out = model.generate(**ids, do_sample=False, max_new_tokens=args.max_tokens)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        thoughts = split_thoughts(gen)
        labeled = [(classify(t), t) for t in thoughts]
        per = {k: sum(l == k for l, _ in labeled) for k in totals}
        for k in totals: totals[k] += per[k]
        results.append({"title": r["question_title"], "difficulty": r["difficulty"],
                        "generation": gen,
                        "thoughts": [{"type": l, "text": t} for l, t in labeled]})
        print(f"\n===== {r['question_title'][:60]} | {len(thoughts)} thoughts "
              f"| exec {per['execution']} refl {per['reflection']} trans {per['transition']} "
              f"| think_closed={'</think>' in gen} =====")
        for l, t in labeled:
            print(f"  [{l:10}] {t.strip()[:180].replace(chr(10),' ')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[probe-lcb] totals over {len(picked)} traces: {totals}")
    print(f"[probe-lcb] saved full traces -> {args.out}")


if __name__ == "__main__":
    main()
