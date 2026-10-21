# train_image_3D test is run on a brain MRI scan extracted from the 2nd channel of the first volume in the BraTS2020 dataset:
# [1] B. H. Menze, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", 
#       IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
# [2] S. Bakas, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", 
#       Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
# [3] S. Bakas, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", 
#       arXiv preprint arXiv:1811.02629 (2018)

import torch
import numpy as np
import os
from PIL import Image
import tifffile
from fused_ssim import fused_ssim3d

# GPU Device Detection and Configuration
# Note: 3D fused SSIM is only available on CUDA for now.

if not torch.cuda.is_available():
    raise RuntimeError("3D fused SSIM is only implemented for CUDA devices.")

gpu = torch.cuda.get_device_name()
fused_ssim_device = "cuda"

# Load ground truth volume and normalize to [0, 1]
gt_volume_np = tifffile.imread(os.path.join("..", "images", "3D_brain_mri.tiff")).astype(np.float32)/255
gt_volume = torch.from_numpy(gt_volume_np).to(device=fused_ssim_device).unsqueeze(0).unsqueeze(0)

# Initialize predicted volume with random values (to be optimized)
pred_volume = torch.nn.Parameter(torch.rand_like(gt_volume))

# Calculate initial SSIM value
with torch.no_grad():
    ssim_value = fused_ssim3d(pred_volume, gt_volume, train=False)
    print("Starting with 3D SSIM value:", ssim_value.item())

# Setup optimizer for training
optimizer = torch.optim.Adam([pred_volume])

# Training loop: Optimize predicted volume to match ground truth using SSIM loss
while ssim_value < 0.9999:
    optimizer.zero_grad()
    loss = 1.0 - fused_ssim3d(pred_volume, gt_volume)
    loss.backward()
    optimizer.step()

    # Evaluate current SSIM value
    with torch.no_grad():
        ssim_value = fused_ssim3d(pred_volume, gt_volume, train=False)
        print("SSIM value:", ssim_value.item())

images_dir = os.path.join("..", "images")
os.makedirs(images_dir, exist_ok=True)
gpu_name = gpu.lower().replace(' ', '-')

pred_volume_cpu = pred_volume.detach().cpu().squeeze(0).squeeze(0).numpy()
pred_volume_cpu = (np.clip(pred_volume_cpu, 0.0, 1.0) * 255.0).astype(np.uint8)
gt_volume_np = (gt_volume_np* 255.0).astype(np.uint8)

input_center_slice = gt_volume_np[:, :, gt_volume_np.shape[2] // 2]
pred_center_slice = pred_volume_cpu[:, :, pred_volume_cpu.shape[2] // 2]

# Save the central XY slices for both input and generated volumes
Image.fromarray(input_center_slice).save(os.path.join(images_dir, f"3D_brain_mri_center_slice.jpg"))
Image.fromarray(pred_center_slice).save(os.path.join(images_dir, f"3D_predicted_brain_mri_center_slice-{gpu_name}.jpg"))

# Save the learned volume as a TIFF
tifffile.imwrite( os.path.join(images_dir, f"3D_predicted_brain_mri-{gpu_name}.tiff"), pred_volume_cpu)