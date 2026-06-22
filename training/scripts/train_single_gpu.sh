#!/usr/bin/env bash
# Single-GPU launch of LPG training (Colab / one A100 or L4).
# Differences from train.sh: no torchrun, no DeepSpeed, single process.
# Smoke-test defaults are overridable via env vars.
#
# Required env vars:
#   MODEL_PATH   Base model path or HF id (e.g. microsoft/Phi-4-mini-instruct).
#   DATA_PATH    Training JSONL.
#
# Optional env vars (with smoke-test defaults):
#   SAVE_DIR, LR, EPOCHS, BSZ, GRAD_ACCUM, MAX_TOKENS, SAVE_STRATEGY, LOG_STEPS

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH (e.g. microsoft/Phi-4-mini-instruct)}"
: "${DATA_PATH:?set DATA_PATH (path to training JSONL)}"

SAVE_DIR=${SAVE_DIR:-./outputs}
SUMMARY_CACHE_PATH=${SUMMARY_CACHE_PATH:-${DATA_PATH}}

# Tunables (smoke-test defaults; override for the full run)
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
BSZ=${BSZ:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_TOKENS=${MAX_TOKENS:-800}
SAVE_STRATEGY=${SAVE_STRATEGY:-no}
LOG_STEPS=${LOG_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-120}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}

mkdir -p "$SAVE_DIR"

python train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name lpg_phi4_single_gpu \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps "$LOG_STEPS" \
    --logging_strategy steps \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --summary_cache_path "$SUMMARY_CACHE_PATH" \
    --seed 11 \
    --model_max_length 1024 \
    --per_device_train_batch_size "$BSZ" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --bf16 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "${SAVE_STEPS:-120}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-1}" \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type linear \
    --do_train \
    --report_to none \
    --num_latent_per_stage "4,6" \
    --stage_names "intent,risk" \
    --use_prj True \
    --prj_dim 3072 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --stage_align_loss_factor 1.0 \
    --remove_eos True \
    --ce_loss_factor 2.0 \
    --distill_loss_factor 10 \
    --max_token_num "$MAX_TOKENS" \
    --require_teacher_summaries False \
    --use_decoder True \
    --use_baselm_explain True \
    --explain_proj_layers 3 \
    --explain_loss_factor 0.5 \
    --summary_max_target_length 192 \
    --remove_unused_columns False