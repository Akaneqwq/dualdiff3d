#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"

GPU_ID="${GPU_ID:-0}"
DATASET_PATH="${DATASET_PATH:-$CODE_DIR/data/mipnerf360}"
INIT_PCD_ROOT="${INIT_PCD_ROOT:-$CODE_DIR/data/mipnerf360_preprocessed_pcs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/output/mipnerf360_baselines}"
SCENES_STR="${SCENES_STR:-bicycle bonsai counter garden kitchen room stump}"

N_VIEWS="${N_VIEWS:-6}"
RESOLUTION="${RESOLUTION:-4}"
GS_ITERATIONS="${GS_ITERATIONS:-30000}"
GS_EXP_NAME="${GS_EXP_NAME:-3dgs_MipNeRF360_${N_VIEWS}view_${RESOLUTION}x}"
RUN_RENDER="${RUN_RENDER:-1}"

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

echo ">>> Mip-NeRF 360 baseline pipeline completed"
echo "3DGS outputs: $OUTPUT_ROOT/$GS_EXP_NAME"
