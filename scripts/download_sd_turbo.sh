#!/usr/bin/env bash
set -euo pipefail

# Download the Diffusers-format SD-Turbo files required by DualDiff.
# Default source is Hugging Face. In regions where huggingface.co is slow,
# set HF_ENDPOINT=https://hf-mirror.com before running this script.
# Downloads are written to pretrained/sd-turbo unless SD_TURBO_DIR is set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${SD_TURBO_DIR:-$REPO_ROOT/pretrained/sd-turbo}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
REPO="stabilityai/sd-turbo"
REVISION="${SD_TURBO_REVISION:-main}"

FILES=(
    "model_index.json"
    "scheduler/scheduler_config.json"
    "tokenizer/merges.txt"
    "tokenizer/special_tokens_map.json"
    "tokenizer/tokenizer_config.json"
    "tokenizer/vocab.json"
    "text_encoder/config.json"
    "text_encoder/model.safetensors"
    "unet/config.json"
    "unet/diffusion_pytorch_model.safetensors"
    "vae/config.json"
    "vae/diffusion_pytorch_model.safetensors"
)

download_file() {
    local relative_path="$1"
    local output_path="$MODEL_DIR/$relative_path"
    local part_path="$output_path.part"
    local url="$HF_ENDPOINT/$REPO/resolve/$REVISION/$relative_path"

    mkdir -p "$(dirname "$output_path")"
    if [ -f "$output_path" ]; then
        echo "Already exists: $relative_path"
        return
    fi

    echo "Downloading: $relative_path"
    # Continue an interrupted download. If a stale partial file causes a
    # range failure, retry that one file from scratch.
    if ! curl --fail --location --show-error --retry 5 --retry-delay 3 \
        --continue-at - --output "$part_path" "$url"; then
        echo "Resume failed; retrying from scratch: $relative_path" >&2
        rm -f "$part_path"
        curl --fail --location --show-error --retry 5 --retry-delay 3 \
            --output "$part_path" "$url"
    fi
    mv "$part_path" "$output_path"
}

echo "SD-Turbo source: $HF_ENDPOINT/$REPO (revision=$REVISION)"
echo "SD-Turbo target: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

for file in "${FILES[@]}"; do
    download_file "$file"
done

echo
echo "SD-Turbo download completed."
echo "Use this local model with:"
echo "  export SD_TURBO_PATH=$MODEL_DIR"
