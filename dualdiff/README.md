# DualDiff

DualDiff is the novel-view **repair** model used by DualDiff3D. It keeps a structure branch on the degraded rendering and an appearance branch on a nearby reference view, then fuses them with Structure-Appearance Attention (SAA).

## Layout

| File | Role |
| --- | --- |
| `model.py` | The `DualDiff` module: dual SD-Turbo U-Nets, shared frozen VAE encoder, decoder with zero-init skips and LoRA |
| `attention.py` | `ReferenceAttentionControl`, the SAA mechanism that writes appearance hidden states and reads them in the structure branch |
| `dataset.py` | `PairedDataset` over the training JSON |
| `loss.py` | VGG Gram-matrix loss |
| `finetune.py` | Training entry point |
| `infer.py` | Inference on an image pair or a directory |
| `infer_json.py` | Inference over a JSON test split |
| `scripts/` | Dataset construction and the training launcher |

DualDiff3D reconstruction imports this package from the repository root as `from dualdiff.model import DualDiff`.

## Environment

Create the unified environment from the repository root:

```bash
conda env create -f environment.yml
conda activate dualdiff3d
```

The same environment supports standalone repair, repair-model training, and DualDiff3D reconstruction. The 3DGS CUDA extensions are not needed for standalone repair or repair-model training.

Download SD-Turbo weights from the repository root (optional local cache):

```bash
bash scripts/download_sd_turbo.sh
export SD_TURBO_PATH="$PWD/pretrained/sd-turbo"
```

If `huggingface.co` is slow, set `HF_ENDPOINT=https://hf-mirror.com` before the download script.

The released DualDiff checkpoint is downloaded automatically from
[`Akaneqwq/DualDiff`](https://huggingface.co/Akaneqwq/DualDiff) and stored
in the standard Hugging Face cache. Prefetch it with:

```bash
python -m dualdiff.hub
```

Set `DUALDIFF_MODEL_ID`, `DUALDIFF_MODEL_FILENAME`, or
`DUALDIFF_MODEL_REVISION` to select another Hub artifact. A local
`--model_path` / `--dualdiff_ckpt` always takes precedence.

## Training data preparation

Before training, prepare degraded sparse-view 3DGS renders and their
ground-truth DL3DV images. `prepare_data.py` only builds metadata; it does not
download images or generate the degraded renders.

For the released 24-view configuration, the input roots must have this layout:

```text
rendered_root/
└── <scene>/lr/24/train_24/
    ├── frame_00001_ref.png
    ├── frame_00002.png
    └── ...
gt_root/
└── <scene>/gaussian_splat/images_4/
    ├── frame_00001.png
    ├── frame_00002.png
    └── ...
```

Every rendered scene must contain at least one image whose stem includes
`_ref`. For example, `frame_00001_ref.png` is paired with the ground-truth
`frame_00001.png`; other renders use the nearest `_ref` entry in sorted frame
order.

From the repository root, build the metadata after both roots are ready:

```bash
python dualdiff/scripts/prepare_data.py \
  --rendered_path /path/to/rendered_root \
  --gt_path /path/to/gt_root \
  --view_counts 24
```

The command validates the image pairs and writes
`dualdiff/data/DL3DV_enhance_nearest_meta.json`. Scenes are split 90/10 into
train/test with a fixed seed. Training and JSON inference read entries of the
following form:

```json
{
  "train": {
    "train_0": {
      "image": "/path/to/rendered_novel.png",
      "target_image": "/path/to/ground_truth.png",
      "ref_image": "/path/to/nearest_train_view.png",
      "prompt": "remove degradation"
    }
  },
  "test": {}
}
```

## Training

Run training only after the metadata and all referenced images are available.
From `dualdiff/`:

```bash
bash scripts/train_dualdiff.sh
```
The launcher uses 8 GPU processes with batch size 1 per process by default.
Set `NUM_PROCESSES` and `TRAIN_BATCH_SIZE` to override the effective batch size,
and `TRAIN_HEIGHT` / `TRAIN_WIDTH` to override the training resolution.
External tracking is disabled by default; set `REPORT_TO=wandb` to enable Weights & Biases logging.

Training starts from SD-Turbo with DIFIX-style skip connections. Pass `--pretrained_path` to `finetune.py` to continue from an existing DualDiff checkpoint instead.

## Inference

Single pair (or two directories of matching length):

```bash
python dualdiff/infer.py \
  --input_image assets/example_input.png \
  --ref_image assets/example_ref.png \
  --output_dir outputs/repair
```

Defaults use the training resolution `576×1024` and timestep `199`. Override with `--height` / `--width` / `--timestep` if needed.

JSON test set:

```bash
python dualdiff/infer_json.py \
  --testset_json_path assets/example_dataset.json \
  --save_path results/dualdiff_test
```

To score refinement quality, reuse the reconstruction-side tooling from the repository root: `python -m dualdiff3d.render dualdiff` writes both raw and refined renders with their metrics, and `python -m dualdiff3d.metrics` averages them across scenes.
