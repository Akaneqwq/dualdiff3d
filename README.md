<div align="center">

<h3>[ECCV 2026] DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting</h3>

<p>
<a href="https://akaneqwq.github.io/">Qian Wang</a><sup>&ast;</sup>,
<a href="https://github.com/ArisWayne">Yu Wang</a><sup>&ast;</sup>,
<a href="https://github.com/lwq20020127/">Weiqi Li</a>,
<a href="https://cxh0519.github.io/">Xinhua Cheng</a>,
Xiandong Meng, Ronggang Wang,
<a href="https://jianzhang.tech/">Jian Zhang</a>
</p>

<p><sup>&ast;</sup> Equal contribution.</p>

Peking University, Pengcheng Laboratory, Guangdong Provincial Key Laboratory of UHD Immersive Media Technology

[![Home Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://akaneqwq.github.io/dualdiff3d)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01516-b31b1b.svg)](https://arxiv.org/abs/2609.01516)

This repository is the official implementation of **DualDiff3D**, which decouples structure and appearance through dual diffusion priors with a reliability-enhanced Render–Refine–Optimize loop for sparse-view 3D Gaussian Splatting.

</div>

## Overview

- **DualDiff** is a standalone image repair model. It combines the structure of a degraded rendering with the appearance of a nearby reference view through Structure-Appearance Attention (SAA).
- **DualDiff3D** places frozen DualDiff in a reliability-enhanced Render–Refine–Optimize (RRO) loop for sparse-view 3DGS.

Original DualDiff3D code is licensed under Apache-2.0. Gaussian
Splatting-derived components remain subject to their original non-commercial
terms. Both licenses and their scopes are documented in [`LICENSE.md`](LICENSE.md);
other bundled dependencies retain the licenses included in their source directories.

## Installation

The same environment supports repair and reconstruction. It was tested with Python 3.10.21, PyTorch 2.9.1, and CUDA 13.0.

```bash
conda env create -f environment.yml
conda activate dualdiff3d

bash scripts/download_sd_turbo.sh
export SD_TURBO_PATH="$PWD/pretrained/sd-turbo"
```

DualDiff is downloaded automatically from
[`Akaneqwq/DualDiff`](https://huggingface.co/Akaneqwq/DualDiff) on first use
and cached by `huggingface_hub`. Use
`HF_ENDPOINT=https://hf-mirror.com bash scripts/download_sd_turbo.sh` if Hugging Face is
unavailable.

Verify the environment and prefetch the checkpoint before running inference:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m dualdiff.hub
```

## Part I: DualDiff image repair

No 3DGS installation is required. Repair an image using a degraded rendering and a nearby reference:

```bash
python dualdiff/infer.py \
  --input_image assets/example_input.png \
  --ref_image assets/example_ref.png \
  --output_dir outputs/repair
```

The two image arguments may also be directories with matching sorted file
lists. Pass `--model_path /path/to/checkpoint.pkl` to override the Hub model.
JSON inference, dataset preparation, and repair-model training are documented
in [dualdiff/README.md](dualdiff/README.md).

```bash
# Train the released DualDiff configuration
bash dualdiff/scripts/train_dualdiff.sh
```

## Part II: DualDiff3D reconstruction

### Setup

Install the vendored 3DGS CUDA extensions in the same environment:
This step requires a CUDA toolkit with `nvcc` compatible with the installed
PyTorch CUDA version.

```bash
nvcc --version
python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation -e submodules/fused-ssim
python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim; print('3DGS extensions OK')"
```

### Data

We use the [DropGaussian](https://github.com/DCVL-3D/DropGaussian_release) sparse-view layout. Download its [preprocessed point clouds](https://drive.google.com/drive/folders/1P3I9m_HU0jF50qwxIIhXhegOVk-kihdI?usp=sharing), or pass `--rand_pcd` for random point cloud initialization.

```text
data/
├── nerf_llff_data/<scene>/{images,sparse/0}
├── nerf_llff_data_preprocessed_pcs/<scene>/<N>_views/dense/fused.ply
├── mipnerf360/<scene>/{images,sparse/0}
└── mipnerf360_preprocessed_pcs/<scene>/<N>_views/dense/fused.ply
```

COLMAP conversion: `tools/colmap_llff.py` and `tools/colmap_360.py`.

### Train

Run vanilla 3DGS first, then resume from its 30k checkpoint with DualDiff3D:

```bash
# LLFF (default: 6 views)
N_VIEWS=6 bash scripts/train_llff_baselines.sh
N_VIEWS=6 bash scripts/train_llff_dualdiff_from_3dgs.sh

# Mip-NeRF 360 (default: 6 views; paper settings: 3, 6, or 9)
N_VIEWS=6 bash scripts/train_mipnerf360_baselines.sh
N_VIEWS=6 bash scripts/train_mipnerf360_dualdiff_from_3dgs.sh
```

For custom scenes and script options, see [scripts/README.md](scripts/README.md). 

### Render and evaluate

```bash
python -m dualdiff3d.render dualdiff -m output/dualdiff_scene -r 4
python -m dualdiff3d.metrics --path output/dualdiff_scene --iteration 60000
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wang2026dualdiff3d,
  title={DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting},
  author={Wang, Qian and Wang, Yu and Li, Weiqi and Cheng, Xinhua and Meng, Xiandong and Wang, Ronggang and Zhang, Jian},
  booktitle={European Conference on Computer Vision},
  pages={286--304},
  year={2026},
  organization={Springer}
}
```

## Acknowledgement

We thank the authors of [DIFIX3D+](https://github.com/nv-tlabs/Difix3D) for releasing their code and inspiring this work.

```bibtex
@inproceedings{wu2025difix3d+,
  title        = {Difix3d+: Improving 3d reconstructions with single-step diffusion models},
  author       = {Wu, Jay Zhangjie and Zhang, Yuxuan and Turki, Haithem and Ren, Xuanchi and Gao, Jun and Shou, Mike Zheng and Fidler, Sanja and Gojcic, Zan and Ling, Huan},
  booktitle    = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {26024--26035},
  year         = {2025},
  organization = {IEEE}
}
```