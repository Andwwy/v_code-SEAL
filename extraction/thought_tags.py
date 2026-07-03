"""v_code thought classifier — code-adapted keyword lists (all 'contains' matching,
per the evidence analysis in evidence/) + boundary extraction for steering-vector
building. Keep this as the single source of truth for the tags.
"""
import re

# --- code-adapted keyword lists (ours) ---
REFLECT_WORDS = [
    "wait", "but wait", "verify", "make sure", "hold on", "think again",
    "'s correct", "'s incorrect", "let me check", "seems right", "hmm",
    "what if", "double-check", "recheck", "edge case",
]
TRANSITION_WORDS = [
    "alternatively", "another way", "another approach", "another method",
    "another solution", "another strategy", "another technique",
    "think differently", "instead", "a better way", "rethink",
    "start over", "on second thought",
]


def classify(step_text):
    """execution / reflection / transition — reflection checked first."""
    s = step_text.strip().lower()
    if any(w in s for w in REFLECT_WORDS):
        return "reflection"
    if any(w in s for w in TRANSITION_WORDS):
        return "transition"
    return "execution"


def split_thoughts(gen):
    """Thoughts inside <think>…</think>, split on blank lines (for eyeballing traces)."""
    body = gen
    if "<think>" in body:
        body = body.split("<think>", 1)[1]
    if "</think>" in body:
        body = body.split("</think>", 1)[0]
    return [t for t in re.split(r"\n\s*\n", body) if t.strip()]


def _split_token_ids(tokenizer):
    # tokens that contain the "\n\n" marker (GPT2/Qwen byte-level "ĊĊ")
    return {i for tok, i in tokenizer.get_vocab().items() if "ĊĊ" in tok}


def boundaries_from_ids(full_ids, prompt_len, tokenizer, think_only=True):
    """For each `\\n\\n` boundary token inside the think region, return its index into
    full_ids and the tag of the thought that FOLLOWS it (mirrors SEAL's generate_index,
    but works in token space to avoid re-tokenization mismatch, and uses classify())."""
    ids = full_ids.tolist() if hasattr(full_ids, "tolist") else list(full_ids)
    split_ids = _split_token_ids(tokenizer)
    if think_only:
        tb = tokenizer.encode("<think>", add_special_tokens=False)[0]
        te = tokenizer.encode("</think>", add_special_tokens=False)[0]
        start = ids.index(tb) + 1 if tb in ids else prompt_len
        end = ids.index(te, start) if te in ids[start:] else len(ids)
    else:
        start, end = prompt_len, len(ids)
    region = ids[start:end]
    marks = [i for i, t in enumerate(region) if t in split_ids] + [len(region)]
    positions, labels = [], []
    for i in range(len(marks) - 1):
        positions.append(marks[i] + start)                     # index into full_ids
        seg = tokenizer.decode(region[marks[i] + 1: marks[i + 1]])  # thought AFTER this boundary
        labels.append(classify(seg))
    return positions, labels
