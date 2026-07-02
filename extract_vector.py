#!/usr/bin/env python3
"""v_code steering-vector extraction on code (MBPP) — SEAL pipeline with our
code-adapted keyword classifier (thought_tags.py).

Two modes:
  plain     : take the first --n tasks (no correctness labeling)
  --balance : SEAL-style — label each trace correct/incorrect by running MBPP's
              test_list, collect a 50/50 pool until it reaches --target_activations
              (default 10500 ≈ 1/10 of SEAL's activation budget).

Per trace: generate (greedy) -> forward-pass for hidden states -> at each `\\n\\n`
boundary take the layer-L vector tagged by the following thought -> pool into
execution vs reflection∪transition -> S = mean(exec) - mean(refl∪trans).

Runs on Mac (MPS) for tiny tests; use Colab/GPU for the full ~275-trace balanced run.
  python extract_vector.py --balance --target_activations 10500 --layer 20
"""
import argparse, json, os, re, subprocess, sys
import torch
from thought_tags import boundaries_from_ids, REFLECT_WORDS, TRANSITION_WORDS


def build_prompt(tok, problem, tests):
    instr = ("Please solve the following Python programming problem. "
             "Think step-by-step, then give the final function.\n\n"
             f"Problem: {problem}\n\nYour function should pass these tests:\n" + "\n".join(tests))
    return tok.apply_chat_template([{"role": "user", "content": instr}],
                                   tokenize=False, add_generation_prompt=True)


def extract_code(gen_text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", gen_text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    if "</think>" in gen_text:
        return gen_text.split("</think>", 1)[1].strip()
    return gen_text.strip()


def passes_tests(gen_text, tests, setup="", timeout=6):
    """Run the generated code + MBPP asserts in a sandboxed subprocess. True = correct."""
    code = extract_code(gen_text)
    if not code or not tests:
        return False
    script = (setup + "\n" + code + "\n" + "\n".join(tests))
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False


def process_row(model, tok, row, layer, max_tokens, device):
    problem = row.get("prompt") or row.get("text")
    tests = row.get("test_list", []) or []
    setup = row.get("test_setup_code", "") or ""
    pin = tok(build_prompt(tok, problem, tests), return_tensors="pt").to(device)
    plen = pin["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**pin, do_sample=False, max_new_tokens=max_tokens,
                             pad_token_id=tok.eos_token_id)
    full = gen[0]
    with torch.no_grad():
        out = model(full.unsqueeze(0), output_hidden_states=True)
    h = out.hidden_states[layer][0].float().cpu()          # [seq, dim]
    positions, labels = boundaries_from_ids(full, plen, tok, think_only=True)
    dim = h.shape[1]
    sE, nE = torch.zeros(dim), 0
    sRT, nRT = torch.zeros(dim), 0
    cnt = {"execution": 0, "reflection": 0, "transition": 0}
    for pos, lab in zip(positions, labels):
        cnt[lab] += 1
        if lab == "execution":
            sE += h[pos]; nE += 1
        else:
            sRT += h[pos]; nRT += 1
    gen_text = tok.decode(full[plen:], skip_special_tokens=False)
    del out, h
    return {"sE": sE, "nE": nE, "sRT": sRT, "nRT": nRT, "cnt": cnt,
            "gen": gen_text, "tests": tests, "setup": setup}


def build_vector(selected):
    sE = sum(t["sE"] for t in selected); nE = sum(t["nE"] for t in selected)
    sRT = sum(t["sRT"] for t in selected); nRT = sum(t["nRT"] for t in selected)
    assert nE > 0 and nRT > 0, f"need both poles (E={nE}, RT={nRT})"
    return (sE / nE) - (sRT / nRT), nE, nRT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", action="store_true", help="SEAL-style 50/50 correct/incorrect")
    ap.add_argument("--target_activations", type=int, default=10500, help="stop when balanced pool hits this (~1/10 SEAL)")
    ap.add_argument("--n", type=int, default=200, help="plain mode: number of tasks")
    ap.add_argument("--max_scan", type=int, default=374, help="balance mode: max train tasks to try")
    ap.add_argument("--dataset_repo", default="google-research-datasets/mbpp")
    ap.add_argument("--config", default="full")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default="vectors/mbpp_v_code_steervec.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"[extract] device={device} dtype={args.dtype} layer={args.layer} balance={args.balance}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    ds = load_dataset(args.dataset_repo, args.config, split=args.split)

    counts = {"execution": 0, "reflection": 0, "transition": 0}
    if args.balance:
        correct, incorrect = [], []
        for k in range(min(args.max_scan, len(ds))):
            tr = process_row(model, tok, ds[k], args.layer, args.max_tokens, device)
            ok = passes_tests(tr["gen"], tr["tests"], tr["setup"])
            (correct if ok else incorrect).append(tr)
            for kk in counts: counts[kk] += tr["cnt"][kk]
            m = min(len(correct), len(incorrect))
            bal_acts = sum(t["nE"] + t["nRT"] for t in correct[:m] + incorrect[:m])
            print(f"[{k+1}] {'PASS' if ok else 'FAIL':4} | correct={len(correct)} incorrect={len(incorrect)} "
                  f"| balanced_activations={bal_acts}")
            if m > 0 and bal_acts >= args.target_activations:
                break
        m = min(len(correct), len(incorrect))
        selected = correct[:m] + incorrect[:m]
        print(f"\n[balance] pools: correct={len(correct)} incorrect={len(incorrect)} -> using {m}/class")
    else:
        n = min(args.n, len(ds))
        selected = []
        for k in range(n):
            tr = process_row(model, tok, ds[k], args.layer, args.max_tokens, device)
            selected.append(tr)
            for kk in counts: counts[kk] += tr["cnt"][kk]
            print(f"[{k+1}/{n}] running {counts}")

    S, nE, nRT = build_vector(selected)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(S, args.out)
    meta = {"model": args.model, "layer": args.layer, "mode": "balanced" if args.balance else "plain",
            "n_traces": len(selected), "n_execution": nE, "n_reflection_transition": nRT,
            "total_activations": nE + nRT, "tag_counts": counts,
            "per_class": (len(selected) // 2 if args.balance else None),
            "reflect_words": REFLECT_WORDS, "transition_words": TRANSITION_WORDS,
            "vector_dim": int(S.numel()), "vector_norm": float(S.norm())}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)
    print(f"\n[extract] traces={len(selected)} activations: |E|={nE} |RT|={nRT} (total {nE+nRT})")
    print(f"[extract] saved vector -> {args.out}  (dim={S.numel()}, norm={S.norm():.3f})")
    print(f"[extract] meta -> {args.out}.meta.json")


if __name__ == "__main__":
    main()
