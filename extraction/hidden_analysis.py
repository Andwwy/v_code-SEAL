"""Stage 3 — hidden-state extraction at `\\n\\n` boundaries (SEAL's
hidden_analysis.py, APPS-adapted).

Faithful to VITA-Group/SEAL: one HF forward pass per selected trace, hidden
states kept at every token containing `ĊĊ` (the byte-level "\\n\\n" marker)
INSIDE `<think>` only; each boundary tagged by the thought that follows it —
reflection checked before transition, everything else execution. Selection is
`--start 0 --sample 500` per pool: the first 500 correct / first 500 incorrect
traces in file order (deterministic prefix, exactly SEAL's
`hidden_correct_0_500` / `hidden_incorrect_0_500`).

Differences from the SEAL original, none of which change the extracted states:
  - keyword lists imported from thought_tags.py (the v1 code-adapted lists —
    single source of truth, identical matching semantics),
  - batched left-padded forward passes with pad-aware position_ids (v1 speedup),
  - --keep_layers to store a subset of layers (disk),
  - logits trimmed to the last position when transformers supports it (memory
    only — hidden states are unaffected; lets batch_size>1 fit 10k-token traces
    on 24GB),
  - a selection.json provenance record per pool (problem ids + tag counts).
"""
import json
import os
import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

import thought_tags
from thought_tags import (REFLECT_WORDS, REFLECT_PREFIXES,
                          TRANSITION_WORDS, TRANSITION_PREFIXES)


def generate_math_data(data_dir, data_path):
    """Split traces into correct/incorrect pools, in file order (SEAL verbatim,
    plus provenance passthrough). With greedy n=1 each problem lands in exactly
    one pool, so 'first 500 of each pool' is a stable prefix."""
    correct, incorrect = [], []
    with open(data_path) as f:
        data = [json.loads(line) for line in f.readlines()]
    with open(f"{data_dir}/math_eval.jsonl") as f:
        eval = [json.loads(line) for line in f.readlines()]

    data = data[:len(eval)]
    for d, e in zip(data, eval):
        local_correct, local_incorrect = [], []
        prompt = e["prompt"]
        assert d["problem"] == e["problem"]
        for o, c in zip(e["model_generation"], e["all_eval"]):
            item = {"prompt": prompt, "response": o, "level": d["level"],
                    "gt": e.get("answer", ""),
                    "problem_id": e.get("problem_id"),
                    "difficulty": e.get("difficulty", d["level"])}
            if c:
                local_correct.append(item)
            else:
                local_incorrect.append(item)
        correct.extend(local_correct)
        incorrect.extend(local_incorrect)
    return correct, incorrect


def generate_index(text, tokenizer, split_id, think_only=True):
    # SEAL's generate_index; keyword lists/prefixes come from the keyword_set
    # pointer in thought_tags.py (default: v_code code-adapted, all-contains;
    # flip to seal_math for upstream's exact lists incl. Wait/Alternatively
    # prefixes). Variable names kept from SEAL.
    check_words = REFLECT_WORDS
    check_prefix = REFLECT_PREFIXES
    swicth_words = TRANSITION_WORDS
    switch_prefix = TRANSITION_PREFIXES

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return [], [], []

        start = tokens.index(think_begin_id) + 1
        if think_end_id not in tokens[start:]:
            end = len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = []
    check_index = []
    switch_index = []

    for i in range(len(index) - 1):
        step_index.append(index[i] + start)
        step = think_tokens[index[i] + 1:index[i + 1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        if any([step.lower().startswith(p.lower()) for p in check_prefix]) or any([w.lower() in step.lower() for w in check_words]):
            check_index.append(i)
        elif any([step.lower().startswith(p.lower()) for p in switch_prefix]) or any([w.lower() in step.lower() for w in swicth_words]):
            switch_index.append(i)
    return step_index, check_index, switch_index


def generate(model_path, data, save_dir, keep_layers=None, batch_size=4,
             dtype="float32"):
    think_only = "deepseek" in model_path.lower()
    # UPSTREAM default is fp32 (no torch_dtype passed); bfloat16 is an opt-in
    # speed/memory mode that perturbs hidden values at floating-point-noise level
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype)
    # transformers v5 deprecated torch_dtype= and changed the no-dtype default
    # to checkpoint dtype (bf16 for this model) — fail loud if a future version
    # silently ignores the kwarg instead of drifting numerics
    loaded = next(model.parameters()).dtype
    assert loaded == torch_dtype, f"requested {torch_dtype} but model loaded as {loaded}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"] + d["response"] for d in data]

    layer_num = model.config.num_hidden_layers + 1
    # Only the steering layer is needed downstream; storing all layers makes
    # hidden.pt ~num_layers x larger. keep_layers=None preserves original behavior.
    if keep_layers is None:
        keep_layers = list(range(layer_num))
    hidden_dict = [{} for _ in range(layer_num)]

    # Memory, not math: computing logits for all 10k positions of a trace costs
    # ~3.6GB/sequence at this vocab size. Trimming to the last position leaves
    # every hidden state identical. transformers renamed the kwarg, so detect it.
    fwd_params = inspect.signature(model.forward).parameters
    logits_kw = next((k for k in ("logits_to_keep", "num_logits_to_keep")
                      if k in fwd_params), None)
    logits_opt = {logits_kw: 1} if logits_kw else {}

    # Batched forward passes (left-padded). For left padding we pass position_ids
    # that skip pad tokens so RoPE matches the unpadded/batch-1 forward, and shift
    # each sequence's step indices by its left-pad count.
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    selection = []
    for b0 in tqdm(range(0, len(prompts), batch_size), total=n_batches):
        batch = prompts[b0:b0 + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        enc = {kk: v.to(model.device) for kk, v in enc.items()}
        attn = enc["attention_mask"]
        position_ids = attn.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn == 0, 1)
        with torch.no_grad():
            output = model(input_ids=enc["input_ids"], attention_mask=attn,
                           position_ids=position_ids, output_hidden_states=True,
                           **logits_opt)
            hs = {i: output.hidden_states[i].detach().cpu() for i in keep_layers}
        max_len = enc["input_ids"].shape[1]
        for j, p in enumerate(batch):
            k = b0 + j
            pad = max_len - int(attn[j].sum().item())              # left-pad count for this row
            step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=think_only)
            step_index = torch.LongTensor(step_index) + pad        # shift into padded coords
            check_index = torch.LongTensor(check_index)
            switch_index = torch.LongTensor(switch_index)
            for i in keep_layers:
                step_h = hs[i][j][step_index]
                hidden_dict[i][k] = {"step": step_h, "check_index": check_index, "switch_index": switch_index}
            selection.append({"k": k, "problem_id": data[k].get("problem_id"),
                              "difficulty": data[k].get("difficulty"),
                              "n_steps": len(step_index),
                              "n_check": len(check_index),
                              "n_switch": len(switch_index)})
        del output, hs
    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    json.dump(prompts, open(f"{save_dir}/prompts.json", "w"))
    # keyword_set/dtype recorded at EXTRACTION time so packaging can verify the
    # pointer wasn't flipped mid-pipeline (meta must describe what actually ran)
    json.dump({"keyword_set": thought_tags.keyword_set, "dtype": dtype,
               "items": selection},
              open(f"{save_dir}/selection.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--keep_layers", type=int, nargs="+", default=None,
                        help="Only extract/save these hidden-layer indices (default: all). "
                             "Pass the steering layer to shrink hidden.pt ~num_layers x.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Forward-pass batch size (left-padded). Results match "
                             "batch_size=1 up to floating-point error. With 10k-token "
                             "traces on 24GB: fp32 -> 2, bf16 -> 4 (logits are trimmed).")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="float32 = UPSTREAM-exact numerics (SEAL passes no "
                             "torch_dtype); bfloat16 halves memory/time at "
                             "noise-level hidden-state differences.")
    args = parser.parse_args()
    correct, incorrect = generate_math_data(data_dir=args.data_dir, data_path=args.data_path)
    if args.type == "correct":
        data = correct
    else:
        data = incorrect
    save_dir = f"{args.data_dir}/hidden_{args.type}"
    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start+args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    print(save_dir)
    print(f"[hidden] {args.type}: {len(data)} traces (dtype={args.dtype})")
    generate(args.model_path, data, save_dir, keep_layers=args.keep_layers,
             batch_size=args.batch_size, dtype=args.dtype)
