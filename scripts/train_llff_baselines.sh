#!/usr/bin/env bash
set -euo pipefail

# Single-GPU LLFF baseline pipeline:
#   1) train 3DGS for GS_ITERATIONS and save a checkpoint;
#   2) render/evaluate the 3DGS model and aggregate eight-scene averages.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

GPU_ID="${GPU_ID:-0}"
DATASET_PATH="${DATASET_PATH:-$CODE_DIR/data/nerf_llff_data}"
INIT_PCD_ROOT="${INIT_PCD_ROOT:-$CODE_DIR/data/nerf_llff_data_preprocessed_pcs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/output/llff_baselines}"
SCENES_STR="${SCENES_STR:-fern flower fortress horns leaves orchids room trex}"

N_VIEWS="${N_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-4}"
GS_ITERATIONS="${GS_ITERATIONS:-30000}"
GS_EXP_NAME="${GS_EXP_NAME:-3dgs_LLFF_${N_VIEWS}view_${RESOLUTION}x}"
RUN_RENDER="${RUN_RENDER:-1}"

read -r -a SCENES <<< "$SCENES_STR"

if [ "${#SCENES[@]}" -ne 8 ]; then
    echo "Expected 8 LLFF scenes, got ${#SCENES[@]}: ${SCENES[*]}" >&2
    exit 1
fi

for scene in "${SCENES[@]}"; do
    source_scene="$DATASET_PATH/$scene"
    init_pcd="$INIT_PCD_ROOT/$scene/${N_VIEWS}_views/dense/fused.ply"
    gs_model="$OUTPUT_ROOT/$GS_EXP_NAME/$scene"
    gs_ckpt="$gs_model/chkpnt${GS_ITERATIONS}.pth"

    [ -d "$source_scene/sparse/0" ] || { echo "Missing COLMAP scene: $source_scene" >&2; exit 1; }
    [ -f "$init_pcd" ] || { echo "Missing initial point cloud: $init_pcd" >&2; exit 1; }

    echo ">>> 3DGS: scene=$scene views=${N_VIEWS} resolution=${RESOLUTION}x"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py 3dgs \
        -s "$source_scene" \
        -m "$gs_model" \
        --init_pcd_path "$init_pcd" \
        --eval \
        -r "$RESOLUTION" \
        --iterations "$GS_ITERATIONS" \
        --n_views "$N_VIEWS" \
        --save_iterations "$GS_ITERATIONS" \
        --checkpoint_iterations "$GS_ITERATIONS" \
        --port "$((6000 + GPU_ID))"

    [ -f "$gs_ckpt" ] || { echo "3DGS checkpoint was not created: $gs_ckpt" >&2; exit 1; }

    if [ "$RUN_RENDER" = "1" ]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" python -m dualdiff3d.render 3dgs \
            -m "$gs_model" -r "$RESOLUTION"
    fi

done

if [ "$RUN_RENDER" = "1" ]; then
    python -m dualdiff3d.metrics --path "$OUTPUT_ROOT/$GS_EXP_NAME" --iteration "$GS_ITERATIONS"
fi

echo ">>> LLFF baseline pipeline completed"
echo "3DGS averages:  $OUTPUT_ROOT/$GS_EXP_NAME/metrics_mean.txt"
