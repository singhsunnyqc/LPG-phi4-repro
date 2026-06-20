#!/usr/bin/env bash
# Train Latent Policy Guard (LPG) on top of Qwen3-4B.
#
# This launches `train.py` with the configuration reported in the paper:
#   - 2 latent reasoning stages: intent (m1=4) + risk (m2=6)
#   - LoRA on all Qwen attention + MLP projections (r=128, alpha=32)
#   - SmoothL1 distillation against teacher hidden states + boundary alignment
#   - Compact verdict supervision ("safe" / "unsafe, policy N1, N2, ...")
#
# Required env vars:
#   MODEL_PATH   Path or HF id of the base model (e.g. Qwen/Qwen3-4B).
#   DATA_PATH    JSONL file with `annotation_input`, `generated_reasoning`,
#                and `teacher_summaries`. See ../README.md for the schema.
#
# Optional env vars:
#   NUM_GPUS     GPUs for torchrun (default: 2).
#   SAVE_DIR     Output directory for checkpoints (default: ./outputs).
#   MASTER_PORT  torchrun rendezvous port  (default: 29502).
#
# Example:
#   MODEL_PATH=Qwen/Qwen3-4B \
#   DATA_PATH=data/train_data.jsonl \
#   NUM_GPUS=4 \
#   bash scripts/train.sh

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH (base model dir or HF id, e.g. Qwen/Qwen3-4B)}"
: "${DATA_PATH:?set DATA_PATH (path to training JSONL)}"

SAVE_DIR=${SAVE_DIR:-./outputs}
NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29502}
SUMMARY_CACHE_PATH=${SUMMARY_CACHE_PATH:-${DATA_PATH}}

mkdir -p "$SAVE_DIR"

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name lpg_qwen3_4b \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --summary_cache_path "$SUMMARY_CACHE_PATH" \
    --seed 11 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 10 \
    --save_safetensors False \
    --weight_decay 0.1 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type linear \
    --do_train \
    --report_to tensorboard \
    --num_latent_per_stage "4,6" \
    --stage_names "intent,risk" \
    --logging_strategy steps \
    --use_prj True \
    --prj_dim 3072 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --stage_align_loss_factor 1.0 \
    --remove_eos True \
    --ce_loss_factor 2.0 \
    --distill_loss_factor 10 \
    --max_token_num 800 \
    --require_teacher_summaries False \
    --use_decoder True \
    --use_baselm_explain True \
    --explain_proj_layers 3 \
    --explain_loss_factor 0.5 \
    --summary_max_target_length 192 \
    --remove_unused_columns False \
    --deepspeed scripts/ds_config_zero2.json
