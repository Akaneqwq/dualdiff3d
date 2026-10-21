# Experiment Scripts

Run them from the repository root.

## LLFF

```bash
N_VIEWS=6 bash scripts/train_llff_baselines.sh
N_VIEWS=6 bash scripts/train_llff_dualdiff_from_3dgs.sh
```

`train_llff_baselines.sh` trains all eight LLFF scenes for `GS_ITERATIONS` (default 30000) and saves a checkpoint per scene. `train_llff_dualdiff_from_3dgs.sh` loads `chkpnt30000.pth` and continues to 60000 iterations, refining novel views at `FIX_ITERATIONS` (default `30001 32001 ... 58001`).

## Mip-NeRF 360

```bash
N_VIEWS=6 bash scripts/train_mipnerf360_baselines.sh
N_VIEWS=6 bash scripts/train_mipnerf360_dualdiff_from_3dgs.sh
```

## Environment overrides

All scripts read these variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU_ID` | `0` | GPU passed to `CUDA_VISIBLE_DEVICES` |
| `DATASET_PATH` | `data/<dataset>` | COLMAP scenes (cameras and images) |
| `INIT_PCD_ROOT` | `data/<dataset>_preprocessed_pcs` | Initial point clouds at `<scene>/<N>_views/dense/fused.ply` |
| `OUTPUT_ROOT` | `output/...` | Where models, renders and metrics are written |
| `SCENES_STR` | dataset scene list | Space-separated scene names |
| `N_VIEWS` | `6` | Number of sparse input views |
| `RESOLUTION` | `4` | Downsampling factor, `4` uses `images_4` |
| `GS_ITERATIONS` | `30000` | Baseline 3DGS iterations |
| `RUN_RENDER` | `1` | Set to `0` to skip rendering and metrics |

The DualDiff scripts additionally read:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DUALDIFF_MODEL_ID` | `Akaneqwq/DualDiff` | Hugging Face model repository |
| `DUALDIFF_MODEL_FILENAME` | `model.pkl` | Checkpoint filename on the Hub |
| `DUALDIFF_CKPT` | unset | Optional local checkpoint overriding the Hub |
| `SD_TURBO_PATH` | `pretrained/sd-turbo` | Local SD-Turbo directory |
| `GS_ROOT` | baseline output dir | Where to find `chkpnt<GS_ITERATIONS>.pth` |
| `ITERATIONS` | `60000` | Total iterations after resuming |
| `FIX_ITERATIONS` | `30001 … 58001` | Iterations that trigger a refine step |
| `RELIABLE_PRESET` | `normal` | Preset from `arguments/reliable.py` |
| `NOVEL_LOSS_WEIGHT` | `0.3` | Weight of the confidence-masked novel-view loss |

## Point cloud initialization

By default, these scripts initialize Gaussians from the preprocessed `fused.ply` point cloud via `--init_pcd_path`. To initialize from a random point cloud, run `train.py` with `--rand_pcd` and omit `--init_pcd_path`.
