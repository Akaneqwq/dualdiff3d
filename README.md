<div align="center">

<h3>[ECCV 2026] DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting</h3>

[Qian Wang](https://akaneqwq.github.io/), [Yu Wang](https://github.com/ArisWayne), [Weiqi Li](https://github.com/lwq20020127/), [Xinhua Cheng](https://cxh0519.github.io/), Xiandong Meng, Ronggang Wang, [Jian Zhang](https://jianzhang.tech/)

Peking University, Pengcheng Laboratory, Guangdong Provincial Key Laboratory of UHD Immersive Media Technology

[![Home Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://Akaneqwq.github.io/dualdiff3d)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01516-b31b1b.svg)](https://arxiv.org/abs/2609.01516)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

This repository is the official implementation of **DualDiff3D**, which decouples structure and appearance through dual diffusion priors with a reliability-enhanced Render-Refine-Optimize loop for sparse-view 3D Gaussian Splatting.

</div>

## TODO
- [x] Release arXiv paper
- [x] Release project page
- [ ] Release training / inference code
- [ ] Release pretrained models


## Coming Soon

Stay tuned! The official code implementation is under preparation and will be released here.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wang2026dualdiff3d,
  title     = {DualDiff3D: Dual Structure-Appearance Diffusion
               Priors for Reliability-Enhanced 3D Gaussian Splatting},
  author    = {Qian Wang and Yu Wang and Weiqi Li and Xinhua Cheng
               and Xiandong Meng and Ronggang Wang and Jian Zhang},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  month     = {jun}
}
```

## Acknowledgement

We thank the authors of [DIFIX3D+](https://github.com/nv-tlabs/Difix3D) for releasing their code and inspiring this work. 

```bibtex
@inproceedings{wu2025difix3d+,
  title={Difix3d+: Improving 3d reconstructions with single-step diffusion models},
  author={Wu, Jay Zhangjie and Zhang, Yuxuan and Turki, Haithem and Ren, Xuanchi and Gao, Jun and Shou, Mike Zheng and Fidler, Sanja and Gojcic, Zan and Ling, Huan},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={26024--26035},
  year={2025},
  organization={IEEE}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
