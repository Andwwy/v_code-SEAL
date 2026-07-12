#!/usr/bin/env bash
# v_code 2.0 — full APPS extraction pipeline, SEAL-faithful (500+500 greedy).
#   gen_apps_vllm.py (vLLM greedy 10k) -> hidden_analysis.py x2 (HF forward)
#   -> vector_generation.py (layer 20)  -> package_vector.py (vector + meta)
# Run from the repo root in the SEAL env (vllm+transformers). ~10-20h on 24GB.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=${1:-0}
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
: "${MAX_TOKENS:=10000}"         # SEAL's trace budget — NOT 3072 (v1 truncation bug)
: "${TARGET:=500}"               # traces per pool (SEAL: 500+500)
: "${CHUNK:=250}"                # problems per vLLM generate+score round
: "${TIMEOUT:=10}"               # seconds per test case
: "${MAX_TESTS:=0}"              # 0 = run every test case (logged if capped)
: "${BATCH_SIZE:=4}"             # hidden_analysis forward batch (10k traces, 24GB)
: "${KEEP_LAYERS:=}"             # empty = all layers (~9GB, faithful); "20" if disk-tight
: "${LAYER:=20}"                 # steering layer (SEAL)
: "${EVAL_N:=300}"               # random eval task list: total problems (difficulty random)
: "${EVAL_SEED:=42}"             # seed for the eval task sample
TAG=$(basename "$MODEL")
DIR="results/APPS_train/${TAG}/baseline_${MAX_TOKENS}"

echo "[1/6] vLLM generate + score APPS train (stop at ${TARGET}+${TARGET}) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u extraction/gen_apps_vllm.py \
    --model_name_or_path "$MODEL" --save_dir "$DIR" --split train \
    --max_tokens "$MAX_TOKENS" --target "$TARGET" --chunk_size "$CHUNK" \
    --timeout "$TIMEOUT" --max_tests "$MAX_TESTS" --resume

KL=()
[[ -n "$KEEP_LAYERS" ]] && KL=(--keep_layers $KEEP_LAYERS)

echo "[2/6] hidden states — incorrect (first $TARGET) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u extraction/hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type incorrect --start 0 --sample "$TARGET" --batch_size "$BATCH_SIZE" "${KL[@]}"

echo "[3/6] hidden states — correct (first $TARGET) ..."
CUDA_VISIBLE_DEVICES=$gpu python -u extraction/hidden_analysis.py \
    --model_path "$MODEL" --data_path "$DIR/data.jsonl" --data_dir "$DIR" \
    --type correct --start 0 --sample "$TARGET" --batch_size "$BATCH_SIZE" "${KL[@]}"

echo "[4/6] build steering vector (layer $LAYER) ..."
# --overwrite: stages 2/3 recompute hidden.pt on every run, so a re-run must
# rebuild the vector too (without it, vector_generation.py silently keeps a
# stale .pt and package_vector would ship it with fresh provenance)
python -u extraction/vector_generation.py \
    --data_dir "$DIR" --prefixs "correct_0_${TARGET}" "incorrect_0_${TARGET}" \
    --layers "$LAYER" --save_prefix "${TARGET}_${TARGET}" --overwrite

echo "[5/6] package vector + provenance meta ..."
python -u extraction/package_vector.py \
    --data_dir "$DIR" --layer "$LAYER" --save_prefix "${TARGET}_${TARGET}" \
    --prefixs "correct_0_${TARGET}" "incorrect_0_${TARGET}" \
    --out vectors/apps_v_code.pt

echo "== done: vectors/apps_v_code.pt + vectors/apps_v_code.meta.json =="
echo "   (SEAL convention: vector = H_RT - H_E; apply with coef -1.0 in the eval)"

echo "[6/6] random eval task list (APPS train, random difficulty, seeded) ..."
# --exclude_from: generation is done, so sample only from train problems the
# extraction never consumed (the v1 lesson: never eval on the extraction data)
python -u extraction/make_eval_list.py \
    --split train --n_total "$EVAL_N" --seed "$EVAL_SEED" \
    --exclude_from "$DIR/math_eval.jsonl"

echo "== eval list: eval_tasks/apps_train_n${EVAL_N}_seed${EVAL_SEED}_excl.jsonl =="
