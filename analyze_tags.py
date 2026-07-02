#!/usr/bin/env python3
"""Gather probe evidence into v_code/evidence/ and summarize what we have per tag,
surfacing candidate keywords the current classifier MISSES (mid-thought cues that
landed in 'execution'). Pure CPU, no model."""
import json, os, re, shutil
from collections import Counter

EV = "evidence"
os.makedirs(EV, exist_ok=True)

# 1) collect MBPP full traces (move into evidence/)
if os.path.exists("mbpp_probe_traces.jsonl"):
    shutil.copy("mbpp_probe_traces.jsonl", f"{EV}/mbpp_probe_traces.jsonl")
mbpp = [json.loads(l) for l in open(f"{EV}/mbpp_probe_traces.jsonl")] if os.path.exists(f"{EV}/mbpp_probe_traces.jsonl") else []

# 2) salvage the interrupted LCB run from its log -> evidence/lcb_hard_partial.json
lcb = []
if os.path.exists("lcb_probe.log"):
    cur = None
    for line in open("lcb_probe.log"):
        h = re.match(r"===== (.+?) \| (\d+) thoughts", line)
        if h:
            cur = {"title": h.group(1).strip(), "thoughts": []}
            lcb.append(cur)
            continue
        m = re.match(r"\s*\[(execution|reflection|transition)\s*\]\s?(.*)", line.rstrip("\n"))
        if m and cur is not None:
            cur["thoughts"].append({"type": m.group(1), "text": m.group(2)})
    json.dump(lcb, open(f"{EV}/lcb_hard_partial.json", "w"), indent=2)

def thoughts(traces, src):
    for t in traces:
        for th in t["thoughts"]:
            yield src, th["type"], th["text"]

allt = list(thoughts(mbpp, "mbpp")) + list(thoughts(lcb, "lcb"))

print("=== counts by source x tag ===")
for k, v in sorted(Counter((s, t) for s, t, _ in allt).items()):
    print(f"  {k}: {v}")
print("  TOTAL:", dict(Counter(t for _, t, _ in allt)))

print("\n=== ALL transition thoughts (the switching vocabulary we caught) ===")
for s, t, x in allt:
    if t == "transition":
        print(f"  [{s}] {x[:150]}")

print("\n=== reflection sample (first 10) ===")
for s, t, x in [z for z in allt if z[1] == "reflection"][:10]:
    print(f"  [{s}] {x[:120]}")

# candidate cues currently NOT in the keyword lists — how often they hide in 'execution'
CAND_REFLECT = ["wait", "let me test", "let me verify", "let me re-examine", "edge case",
                "what if", "actually", "hmm", "double-check", "recheck", "is that right",
                "that's wrong", "mistake", "but wait", "let me re-read"]
CAND_SWITCH  = ["instead", "alternatively", "another", "different approach", "different way",
                "better way", "rethink", "start over", "let me think again", "maybe i should",
                "on second thought", "perhaps", "let me try a"]

def missed(cues):
    c = Counter()
    for s, t, x in allt:
        if t == "execution":
            low = x.lower()
            for cue in cues:
                if cue in low:
                    c[cue] += 1
    return c

print("\n=== reflect-cues hiding in EXECUTION thoughts (undercounted reflection) ===")
for cue, n in missed(CAND_REFLECT).most_common():
    print(f"  '{cue}': {n}")
print("\n=== switch-cues hiding in EXECUTION thoughts (undercounted transition) ===")
for cue, n in missed(CAND_SWITCH).most_common():
    print(f"  '{cue}': {n}")

print(f"\n[evidence] folder: {EV}/  ->", os.listdir(EV))
