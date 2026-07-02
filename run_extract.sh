#!/usr/bin/env bash
# Run the balanced (50/50 correct/incorrect) MBPP steering-vector extraction on RunPod.
# Override any knob via env, e.g.:  TARGET_ACTIVATIONS=10500 bash run_extract.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"
[[ -f .env ]] && { set -a; source .env; set +a; }
: "${HF_HOME:=/workspace/hf_cache}"; export HF_HOME
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

: "${TARGET_ACTIVATIONS:=10500}"   # ~1/10 of SEAL's activation budget
: "${LAYER:=20}"
: "${MAX_TOKENS:=3000}"
: "${MAX_SCAN:=374}"               # all of MBPP-full train
: "${OUT:=vectors/mbpp_v_code_steervec.pt}"

echo "=================================================================="
echo " v_code balanced MBPP extraction"
echo " target_activations=$TARGET_ACTIVATIONS layer=$LAYER max_tokens=$MAX_TOKENS"
echo " out=$OUT"
echo "=================================================================="

python extract_vector.py --balance \
    --target_activations "$TARGET_ACTIVATIONS" \
    --layer "$LAYER" \
    --max_tokens "$MAX_TOKENS" \
    --max_scan "$MAX_SCAN" \
    --out "$OUT" 2>&1 | tee extract_run.log

echo "== done: vector at $OUT (+ .meta.json). See extract_run.log for pass rate / pools =="
