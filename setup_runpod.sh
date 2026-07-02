#!/usr/bin/env bash
# One-time env setup for the v_code extraction on a RunPod GPU pod.
# Assumes a PyTorch CUDA base image (torch already installed).
#   cp .env.example .env   # optional: add HF_TOKEN
#   bash setup_runpod.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$DIR"
[[ -f .env ]] && { set -a; source .env; set +a; echo "[setup] loaded .env"; }
: "${HF_HOME:=/workspace/hf_cache}"; export HF_HOME; mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

pip install -q --upgrade pip
echo "[setup] installing requirements..."
pip install -q -r requirements.txt

if [[ -n "${HF_TOKEN:-}" ]]; then
  python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("[setup] HF login OK")
PY
else
  echo "[setup] no HF_TOKEN (fine — model + MBPP are public)"
fi

echo "[setup] sanity check..."
python - <<'PY'
import torch, transformers, datasets
assert torch.cuda.is_available(), "no CUDA — pick a GPU pod / PyTorch base image"
print(f"[setup] torch {torch.__version__} CUDA={torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0)})")
print(f"[setup] transformers {transformers.__version__} datasets {datasets.__version__}")
import thought_tags
print(f"[setup] thought_tags OK — {len(thought_tags.REFLECT_WORDS)} reflect / "
      f"{len(thought_tags.TRANSITION_WORDS)} transition keywords")
PY
echo "[setup] done -> bash run_extract.sh"
