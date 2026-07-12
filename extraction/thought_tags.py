"""v_code thought classifier — keyword lists + boundary extraction for
steering-vector building. Single source of truth for the tags.

Two keyword sets, selected by the `keyword_set` pointer (same pattern as the
`train` pointer in train_registry.py):

  code      : the v1 code-adapted lists (all 'contains' matching, per the
              evidence analysis in the v1 repo) — wait/alternatively promoted
              from prefix->contains, upstream's "differenly" typo fixed, code
              cues added. THE DEFAULT, per PLAN.md.
  seal_math : VITA-Group/SEAL's original MATH lists VERBATIM (including the
              "think differenly" typo and the Wait/Alternatively PREFIX
              matching) — flip the pointer for upstream-exact tagging.

This is the ONE semantic deviation from upstream beyond the dataset/scorer;
everything around it (boundary discovery, think-region slicing, tag priority)
is executable-verified identical to upstream hidden_analysis.generate_index.
"""
import re

KEYWORD_SETS = {
    "code": {
        "reflect_words": [
            "wait", "but wait", "verify", "make sure", "hold on", "think again",
            "'s correct", "'s incorrect", "let me check", "seems right", "hmm",
            "what if", "double-check", "recheck", "edge case",
        ],
        "reflect_prefixes": [],
        "transition_words": [
            "alternatively", "another way", "another approach", "another method",
            "another solution", "another strategy", "another technique",
            "think differently", "instead", "a better way", "rethink",
            "start over", "on second thought",
        ],
        "transition_prefixes": [],
    },
    "seal_math": {  # upstream VITA-Group/SEAL hidden_analysis.py, verbatim
        "reflect_words": [
            "verify", "make sure", "hold on", "think again",
            "'s correct", "'s incorrect", "Let me check", "seems right",
        ],
        "reflect_prefixes": ["Wait"],
        "transition_words": [
            "think differenly", "another way", "another approach",
            "another method", "another solution", "another strategy",
            "another technique",
        ],
        "transition_prefixes": ["Alternatively"],
    },
}

# POINTER — the active keyword set (matching is case-insensitive either way).
keyword_set = "code"

REFLECT_WORDS = KEYWORD_SETS[keyword_set]["reflect_words"]
REFLECT_PREFIXES = KEYWORD_SETS[keyword_set]["reflect_prefixes"]
TRANSITION_WORDS = KEYWORD_SETS[keyword_set]["transition_words"]
TRANSITION_PREFIXES = KEYWORD_SETS[keyword_set]["transition_prefixes"]


def classify(step_text):
    """execution / reflection / transition — reflection checked first
    (same priority as upstream generate_index)."""
    s = step_text.strip().lower()
    if any(s.startswith(p.lower()) for p in REFLECT_PREFIXES) or \
       any(w.lower() in s for w in REFLECT_WORDS):
        return "reflection"
    if any(s.startswith(p.lower()) for p in TRANSITION_PREFIXES) or \
       any(w.lower() in s for w in TRANSITION_WORDS):
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
