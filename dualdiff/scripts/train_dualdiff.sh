#!/usr/bin/env bash
set -euo pipefail

# DualDiff repair-model training configuration used for the released model.
# Effective batch size is NUM_PROCESSES * TRAIN_BATCH_SIZE (default 8 x 1).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUALDIFF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DUALDIFF_DIR"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-576}"
TRAIN_WIDTH="${TRAIN_WIDTH:-1024}"
TIMESTEP="${TIMESTEP:-199}"
LAMBDA_GRAM="${LAMBDA_GRAM:-1.0}"
GRAM_WARMUP="${GRAM_WARMUP:-2000}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATASET_PATH="${DATASET_PATH:-$DUALDIFF_DIR/data/DL3DV_enhance_nearest_meta.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$DUALDIFF_DIR/outputs/dualdiff}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
RUN_NAME="${RUN_NAME:-dualdiff}"
REPORT_TO="${REPORT_TO:-}"

TRACKER_ARGS=()
if [ -n "$REPORT_TO" ]; then
    TRACKER_ARGS=(--report_to "$REPORT_TO")
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset JSON not found: $DATASET_PATH" >&2
    echo "Build it with:" >&2
    echo "  python scripts/prepare_data.py -r <rendered_root> -g <gt_root> --view_counts 24" >&2
    exit 1
fi

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"

accelerate launch \
    --mixed_precision="$MIXED_PRECISION" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --multi_gpu \
    --num_machines 1 \
    --num_processes "$NUM_PROCESSES" \
    finetune.py \
    --output_dir="$OUTPUT_DIR" \
    --dataset_path="$DATASET_PATH" \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --height="$TRAIN_HEIGHT" \
    --width="$TRAIN_WIDTH" \
    --learning_rate "$LEARNING_RATE" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2000 \
    --eval_freq 1000 \
    --viz_freq 1000 \
    --lambda_lpips 1.0 \
    --lambda_l2 1.0 \
    --lambda_gram "$LAMBDA_GRAM" \
    --gram_loss_warmup_steps "$GRAM_WARMUP" \
    "${TRACKER_ARGS[@]}" \
    --tracker_project_name "${TRACKER_PROJECT_NAME:-dualdiff}" \
    --tracker_run_name "$RUN_NAME" \
    --timestep "$TIMESTEP"
