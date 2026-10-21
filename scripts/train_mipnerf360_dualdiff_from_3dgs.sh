#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

GPU_ID="${GPU_ID:-0}"
DATASET_PATH="${DATASET_PATH:-$CODE_DIR/data/mipnerf360}"
INIT_PCD_ROOT="${INIT_PCD_ROOT:-$CODE_DIR/data/mipnerf360_preprocessed_pcs}"
SCENES_STR="${SCENES_STR:-bicycle bonsai counter garden kitchen room stump}"

N_VIEWS="${N_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-4}"
GS_ITERATIONS="${GS_ITERATIONS:-30000}"
ITERATIONS="${ITERATIONS:-60000}"
GS_ROOT="${GS_ROOT:-$CODE_DIR/output/mipnerf360_baselines/3dgs_MipNeRF360_${N_VIEWS}view_${RESOLUTION}x}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/output/mipnerf360_baselines/dualdiff_MipNeRF360_${N_VIEWS}view_${RESOLUTION}x}"
FIX_ITERATIONS="${FIX_ITERATIONS:-30001 32001 34001 36001 38001 40001 42001 44001 46001 48001 50001 52001 54001 56001 58001}"
DUALDIFF_CKPT="${DUALDIFF_CKPT:-}"
DUALDIFF_MODEL_ID="${DUALDIFF_MODEL_ID:-Akaneqwq/DualDiff}"
DUALDIFF_MODEL_FILENAME="${DUALDIFF_MODEL_FILENAME:-model.pkl}"
SD_TURBO_PATH="${SD_TURBO_PATH:-$CODE_DIR/pretrained/sd-turbo}"
RELIABLE_PRESET="${RELIABLE_PRESET:-normal}"
NOVEL_LOSS_WEIGHT="${NOVEL_LOSS_WEIGHT:-0.3}"
RUN_RENDER="${RUN_RENDER:-1}"
PORT_BASE="${PORT_BASE:-6200}"

if [ -n "$DUALDIFF_CKPT" ]; then
    [ -f "$DUALDIFF_CKPT" ] || { echo "Missing DualDiff checkpoint: $DUALDIFF_CKPT" >&2; exit 1; }
    DUALDIFF_ARGS=(--dualdiff_ckpt "$DUALDIFF_CKPT")
else
    DUALDIFF_ARGS=(
        --dualdiff_model_id "$DUALDIFF_MODEL_ID"
        --dualdiff_model_filename "$DUALDIFF_MODEL_FILENAME"
    )
fi

case "$N_VIEWS" in
    3|6|9|12|24) ;;
    *) echo "N_VIEWS must be 3, 6, 9, 12, or 24; got: $N_VIEWS" >&2; exit 1 ;;
esac

read -r -a SCENES <<< "$SCENES_STR"
if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "SCENES_STR must contain at least one scene" >&2
    exit 1
fi

for scene in "${SCENES[@]}"; do
    source_scene="$DATASET_PATH/$scene"
    init_pcd="$INIT_PCD_ROOT/$scene/${N_VIEWS}_views/dense/fused.ply"
    gs_ckpt="$GS_ROOT/$scene/chkpnt${GS_ITERATIONS}.pth"
    dualdiff_model="$OUTPUT_ROOT/$scene"

    [ -d "$source_scene/sparse/0" ] || { echo "Missing COLMAP scene: $source_scene" >&2; exit 1; }
    [ -f "$init_pcd" ] || { echo "Missing initial point cloud: $init_pcd" >&2; exit 1; }
    [ -f "$gs_ckpt" ] || { echo "Missing 3DGS checkpoint: $gs_ckpt" >&2; exit 1; }
    [ -d "$SD_TURBO_PATH" ] || { echo "Missing SD-Turbo model directory: $SD_TURBO_PATH" >&2; exit 1; }

    echo ">>> DualDiff: scene=$scene views=${N_VIEWS} resolution=${RESOLUTION}x"
    SD_TURBO_PATH="$SD_TURBO_PATH" CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py dualdiff-reliable \
        -s "$source_scene" \
        -m "$dualdiff_model" \
        --init_pcd_path "$init_pcd" \
        --start_checkpoint "$gs_ckpt" \
        --eval \
        -r "$RESOLUTION" \
        --iterations "$ITERATIONS" \
        --n_views "$N_VIEWS" \
        --fix \
        --fix_iterations $FIX_ITERATIONS \
        "${DUALDIFF_ARGS[@]}" \
        --reliable_preset "$RELIABLE_PRESET" \
        --novel_loss_weight "$NOVEL_LOSS_WEIGHT" \
        --save_iterations "$ITERATIONS" \
        --checkpoint_iterations "$ITERATIONS" \
        --port "$((PORT_BASE + GPU_ID))"

    if [ "$RUN_RENDER" = "1" ]; then
        echo ">>> DualDiff render: scene=$scene"
        SD_TURBO_PATH="$SD_TURBO_PATH" CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dualdiff3d.render dualdiff \
            -m "$dualdiff_model" \
            -r "$RESOLUTION" \
            "${DUALDIFF_ARGS[@]}"
    fi
done

if [ "$RUN_RENDER" = "1" ]; then
    python -m dualdiff3d.metrics --path "$OUTPUT_ROOT" --iteration "$ITERATIONS"
fi

echo ">>> Mip-NeRF 360 DualDiff enhancement completed"
echo "Outputs: $OUTPUT_ROOT"
