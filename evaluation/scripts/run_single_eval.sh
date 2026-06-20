#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run one (model, dataset) evaluation through evaluate.py.
#
# All paths are taken from environment variables so this script is portable
# across clusters; nothing is hard-coded. See the top-level README.md for the
# variable reference and example invocations.
#
# Required env vars:
#   MODEL          one of: latent_policy_guard | dynaguard_hf | qwen_hf | qwen_vllm
#   DATASET        one of: dynabench | guardset_x | wildguard | policyguardbench | harmbench
#   DATASET_PATH   local file OR HF dataset id (e.g. allenai/wildguardmix)
#
# Required for latent_policy_guard:
#   MODEL_PATH     base Qwen3-4B (or compatible) checkpoint dir
#   CKPT_DIR       LPG checkpoint dir containing model.safetensors
#
# Required for dynaguard_hf / qwen_hf:
#   MODEL_PATH     HF model dir
#
# Optional:
#   GPU            CUDA device id (default 0)
#   OUTPUT         result JSON path (default results/<model>_<dataset>.json)
#   MAX_SAMPLES    cap on samples (default: full split)
#   MAX_NEW_TOKENS default 1024 for dynaguard, 160 for others
#   THINK          nothink | free | structured (Qwen vLLM only)
#   VLLM_URL       e.g. http://localhost:8000 (qwen_vllm only)
# ---------------------------------------------------------------------------
set -euo pipefail

CDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CDIR"

: "${MODEL:?set MODEL}"
: "${DATASET:?set DATASET}"
: "${DATASET_PATH:?set DATASET_PATH}"

GPU="${GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT="${OUTPUT:-results/${MODEL}_${DATASET}.json}"
mkdir -p "$(dirname "$OUTPUT")" logs

# Per-model defaults
case "$MODEL" in
    dynaguard_hf)
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
        ;;
    *)
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
        ;;
esac

# Model-specific args
COMMON_ARGS=(
    --model "$MODEL"
    --dataset "$DATASET"
    --dataset_path "$DATASET_PATH"
    --output "$OUTPUT"
    --max_new_tokens "$MAX_NEW_TOKENS"
)
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")

case "$MODEL" in
    latent_policy_guard)
        : "${MODEL_PATH:?set MODEL_PATH (base model dir)}"
        : "${CKPT_DIR:?set CKPT_DIR (LPG checkpoint dir)}"
        EXTRA_ARGS=(
            --model_path "$MODEL_PATH"
            --ckpt_dir "$CKPT_DIR"
            --no_system_prompt
            --num_latent_per_stage "${NUM_LATENT_PER_STAGE:-4,6}"
            --stage_names "${STAGE_NAMES:-intent,risk}"
            --lora_r "${LORA_R:-128}"
            --lora_alpha "${LORA_ALPHA:-32}"
            --use_prj "${USE_PRJ:-True}"
            --prj_dim "${PRJ_DIM:-3072}"
            --greedy "${GREEDY:-True}"
            --remove_eos "${REMOVE_EOS:-True}"
            --model_max_length "${MODEL_MAX_LENGTH:-1024}"
        )
        ;;
    dynaguard_hf)
        : "${MODEL_PATH:?set MODEL_PATH (DynaGuard HF dir)}"
        EXTRA_ARGS=(--model_path "$MODEL_PATH")
        ;;
    qwen_hf)
        : "${MODEL_PATH:?set MODEL_PATH (Qwen HF dir)}"
        EXTRA_ARGS=(--model_path "$MODEL_PATH" --temperature 0.0)
        ;;
    qwen_vllm)
        : "${VLLM_URL:?set VLLM_URL (e.g. http://localhost:8000)}"
        EXTRA_ARGS=(
            --model_path "${MODEL_PATH:-Qwen/Qwen3-4B}"
            --vllm_url "$VLLM_URL"
            --temperature 0.0
        )
        [ -n "${VLLM_MODEL_NAME:-}" ] && EXTRA_ARGS+=(--vllm_model_name "$VLLM_MODEL_NAME")
        [ -n "${THINK:-}" ] && EXTRA_ARGS+=(--think "$THINK")
        ;;
    *)
        echo "Unsupported MODEL=$MODEL" >&2
        exit 2
        ;;
esac

LOG="logs/$(basename "$OUTPUT" .json).log"
echo "[$(date +%H:%M:%S)] GPU=$GPU MODEL=$MODEL DATASET=$DATASET -> $OUTPUT"
echo "  log: $LOG"

CUDA_VISIBLE_DEVICES="$GPU" python evaluate.py \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"

echo "[$(date +%H:%M:%S)] DONE -> $OUTPUT"
