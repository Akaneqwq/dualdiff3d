import numpy as np
from scipy.spatial.transform import Rotation
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def _validate_rotation_matrix(rotation):
    """Return a validated 3x3 rotation matrix as a NumPy array."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("rotation must have determinant +1")
    return rotation


def camera_to_c2w_pose(rotation_cw, translation_wc):
    """Convert camera ``(R_cw, t_wc)`` fields to a c2w pose."""
    rotation_cw = _validate_rotation_matrix(rotation_cw)
    translation_wc = np.asarray(translation_wc, dtype=np.float64).reshape(-1)
    if translation_wc.shape != (3,):
        raise ValueError(
            f"translation must have shape (3,), got {translation_wc.shape}"
        )
    if not np.all(np.isfinite(translation_wc)):
        raise ValueError("translation must contain only finite values")

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_cw
    pose[:3, 3] = -rotation_cw @ translation_wc
    return pose


def c2w_pose_to_extrinsics(pose):
    """Convert a c2w pose ``[R_cw | C_world]`` to camera ``(R, T)`` fields."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain only finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("pose must be a homogeneous 4x4 transform")

    rotation_cw = _validate_rotation_matrix(pose[:3, :3])
    camera_center = pose[:3, 3]
    translation_wc = -rotation_cw.T @ camera_center
    return rotation_cw, translation_wc


def apply_c2w_pose_to_camera(camera, pose):
    """Write a c2w pose to a Camera and refresh its cached extrinsics."""
    camera.R, camera.T = c2w_pose_to_extrinsics(pose)
    camera.update_extrinsics()

def visualize_fix_results(I_low_bchw, I_ref_bchw, I_rep_mean,
                          C_var, C_diff, C_proj, M,
                          save_dir, idx, prefix="fix"):
    os.makedirs(save_dir, exist_ok=True)
    H, W = M.shape[-2:]
    to_pil = T.ToPILImage()

    # -------------------------------
    # Rendered / repaired / reference image grid.
    # -------------------------------
    grid_img = torch.cat([I_low_bchw, I_rep_mean, I_ref_bchw], dim=0)
    grid = vutils.make_grid(grid_img, nrow=3, padding=5)
    vis_img_pil = to_pil(grid.cpu())

    draw = ImageDraw.Draw(vis_img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    labels = ["Rendered", "Fixed", "Reference"]
    w_unit = vis_img_pil.width // 3
    for i, label in enumerate(labels):
        draw.text((10 + i * w_unit, 10), label, fill="white", font=font, stroke_width=2, stroke_fill="black")

    vis_img_path = os.path.join(save_dir, f"{prefix}_img_{idx:04d}.png")
    vis_img_pil.save(vis_img_path)

    # -------------------------------
    # Confidence component visualization.
    conf_maps = {
        "Variance": C_var,
        "Discrepancy": C_diff,
        "Reprojection": C_proj,
        "Fused": M
    }

    conf_pils = []
    for key, conf in conf_maps.items():
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        conf_np = (conf_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
        conf_color = cv2.applyColorMap(conf_np, cv2.COLORMAP_JET)
        conf_color = cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)
        conf_pil = Image.fromarray(conf_color).resize((W, H))
        conf_pils.append((key, conf_pil))

    # Compose a 2x2 confidence-map collage.
    collage = Image.new("RGB", (W * 2, H * 2))
    positions = [(0, 0), (W, 0), (0, H), (W, H)]
    for (key, img), pos in zip(conf_pils, positions):
        collage.paste(img, pos)
        draw = ImageDraw.Draw(collage)
        draw.text((pos[0] + 10, pos[1] + 10), key, fill="white", font=font, stroke_width=2, stroke_fill="black")

    conf_path = os.path.join(save_dir, f"{prefix}_conf_{idx:04d}.png")
    collage.save(conf_path)


def visualize_reprojection_debug(novel_img, ref_img, ref_sampled, valid_mask, e_proj,
                                 depth_hw, save_dir, idx, prefix="reproj"):
    os.makedirs(save_dir, exist_ok=True)
    to_pil = T.ToPILImage()

    novel_vis = novel_img.permute(2, 0, 1).detach().cpu().clamp(0, 1)
    ref_vis = ref_img.permute(2, 0, 1).detach().cpu().clamp(0, 1)
    sampled_vis = ref_sampled.permute(2, 0, 1).detach().cpu().clamp(0, 1)
    valid_vis = valid_mask.detach().float().cpu().unsqueeze(0).repeat(3, 1, 1)

    depth = depth_hw.detach().cpu()
    depth_valid = depth > 0
    if depth_valid.any():
        depth_min = depth[depth_valid].min()
        depth_max = depth[depth_valid].max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    else:
        depth_norm = torch.zeros_like(depth)
    depth_vis = depth_norm.unsqueeze(0).repeat(3, 1, 1)

    err = e_proj.detach().cpu()
    err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
    err_np = (err_norm.numpy() * 255).astype(np.uint8)
    err_color = cv2.applyColorMap(err_np, cv2.COLORMAP_JET)
    err_color = cv2.cvtColor(err_color, cv2.COLOR_BGR2RGB)
    err_vis = torch.from_numpy(err_color).permute(2, 0, 1).float() / 255.0

    grid = vutils.make_grid(
        torch.stack([novel_vis, ref_vis, sampled_vis, valid_vis, depth_vis, err_vis], dim=0),
        nrow=3,
        padding=5
    )
    debug_pil = to_pil(grid)

    draw = ImageDraw.Draw(debug_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    labels = ["Novel", "Reference", "RefSampled", "ValidMask", "Depth", "ReprojErr"]
    cell_w = debug_pil.width // 3
    cell_h = debug_pil.height // 2
    for i, label in enumerate(labels):
        row = i // 3
        col = i % 3
        draw.text(
            (10 + col * cell_w, 10 + row * cell_h),
            label,
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black"
        )

    debug_path = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")
    debug_pil.save(debug_path)


@torch.no_grad()
def _compute_confidences_from_repairs(I_low_bchw, I_ref_bchw, repairs_bchw,
                                      alpha=50.0, beta=10.0, gamma=10.0,
                                      lambdas=(0.34,0.33,0.33),
                                      depth_hw=None, novel_cam=None, ref_cam=None,
                                      debug_save_dir=None, debug_idx=None,
                                      use_reprojection_conf=True,
                                      rendered_alpha=None, rendered_depth_valid=None,
                                      alpha_min=1.0 / 255.0):
    """
    I_low_bchw:   [1,3,H,W], 0..1
    I_ref_bchw:   [1,3,H,W], 0..1
    repairs_bchw: [K,3,H,W], 0..1  
    depth_hw:     [H,W] or None
    """
    device = I_low_bchw.device
    if repairs_bchw.ndim != 4 or repairs_bchw.shape[0] < 1:
        raise ValueError("repairs_bchw must have shape [K,3,H,W] with K >= 1")

    lam1, lam2, lam3 = [float(x) for x in lambdas]
    lam_sum = lam1 + lam2 + lam3
    if lam_sum <= 0:
        raise ValueError("lambda confidence weights must have a positive sum")
    lam1, lam2, lam3 = lam1 / lam_sum, lam2 / lam_sum, lam3 / lam_sum
    # Variance confidence C_var.
    var_rgb = torch.var(repairs_bchw, dim=0, keepdim=True, unbiased=False)      # [1,3,H,W]
    sigma2 = var_rgb.mean(dim=1, keepdim=True)                  # [1,1,H,W]
    C_var = torch.exp(-alpha * sigma2).clamp(0, 1)

    # Supervision image: mean of K repaired samples.
    I_rep_mean = repairs_bchw.mean(dim=0, keepdim=True)         # [1,3,H,W]

    d = (I_rep_mean - I_low_bchw).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
    C_diff = torch.exp(-beta * d).clamp(0, 1)

    # Reprojection confidence C_proj.
    if use_reprojection_conf and (depth_hw is not None) and (novel_cam is not None) and (ref_cam is not None):
        if depth_hw.ndim == 4:  # [B,1,H,W]
            depth_hw = depth_hw[0,0]
        elif depth_hw.ndim == 3:  # [1,H,W]
            depth_hw = depth_hw[0]
        H, W = depth_hw.shape
        novel_img = I_rep_mean.squeeze(0).permute(1,2,0)    # [H,W,3]
        ref_img   = I_ref_bchw.squeeze(0).permute(1,2,0)    # [H,W,3]
        e_proj, valid_mask, debug_info = compute_reprojection_error(
            novel_img, ref_img, depth_hw, novel_cam, ref_cam,
            return_valid=True, return_debug=True
        )  # [H,W]
        # Unified geometry validity: finite positive depth, finite alpha above cutoff,
        # rasterizer validity flag, and valid reprojection.
        geom_valid = torch.isfinite(depth_hw) & (depth_hw > 0)
        if rendered_alpha is not None:
            if rendered_alpha.ndim == 4: rendered_alpha = rendered_alpha[0, 0]
            elif rendered_alpha.ndim == 3: rendered_alpha = rendered_alpha[0]
            geom_valid = geom_valid & torch.isfinite(rendered_alpha) & (rendered_alpha >= float(alpha_min))
        if rendered_depth_valid is not None:
            if rendered_depth_valid.ndim == 4: rendered_depth_valid = rendered_depth_valid[0, 0]
            elif rendered_depth_valid.ndim == 3: rendered_depth_valid = rendered_depth_valid[0]
            geom_valid = geom_valid & rendered_depth_valid.to(dtype=torch.bool)
        valid_mask = valid_mask & geom_valid
        e_proj = e_proj.unsqueeze(0).unsqueeze(0)           # [1,1,H,W]
        C_proj = torch.exp(-gamma * e_proj).clamp(0, 1)
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
        # Suppress invalid pixels in C_proj so they don't contribute there.
        C_proj = C_proj * valid_mask
        if debug_save_dir is not None and debug_idx is not None:
            visualize_reprojection_debug(
                novel_img=novel_img,
                ref_img=ref_img,
                ref_sampled=debug_info["ref_sampled"],
                valid_mask=debug_info["valid_mask"],
                e_proj=debug_info["error_map"],
                depth_hw=debug_info["depth_map"],
                save_dir=debug_save_dir,
                idx=debug_idx
            )
    else:
        C_proj = torch.zeros_like(C_var, device=device)

    if use_reprojection_conf:
        M = (lam1*C_var + lam2*C_diff + lam3*C_proj).clamp(0, 1)
        if (depth_hw is not None) and (novel_cam is not None) and (ref_cam is not None):
            # For invalid regions, ignore C_proj and renormalize lam1/lam2 so weights sum to 1.
            lam12 = lam1 + lam2
            if lam12 == 0:
                w1, w2 = 0.5, 0.5
            else:
                w1, w2 = lam1 / lam12, lam2 / lam12
            M_no_proj = (w1 * C_var + w2 * C_diff).clamp(0, 1)
            M = torch.where(valid_mask, M, M_no_proj)
            if rendered_alpha is not None or rendered_depth_valid is not None:
                M = torch.where(geom_valid.unsqueeze(0).unsqueeze(0), M, torch.zeros_like(M))
            if not valid_mask.any() and (rendered_alpha is not None or rendered_depth_valid is not None):
                M = torch.zeros_like(M)
    else:
        lam12 = lam1 + lam2
        if lam12 == 0:
            w1, w2 = 0.5, 0.5
        else:
            w1, w2 = lam1 / lam12, lam2 / lam12
        M = (w1 * C_var + w2 * C_diff).clamp(0, 1)

    # Apply the geometry-validity mask to the final confidence map.
    if depth_hw is not None:
        depth_for_mask = depth_hw
        if depth_for_mask.ndim == 4: depth_for_mask = depth_for_mask[0, 0]
        elif depth_for_mask.ndim == 3: depth_for_mask = depth_for_mask[0]
        geom_valid = torch.isfinite(depth_for_mask) & (depth_for_mask > 0)
        if rendered_alpha is not None:
            a = rendered_alpha
            if a.ndim == 4: a = a[0, 0]
            elif a.ndim == 3: a = a[0]
            geom_valid = geom_valid & torch.isfinite(a) & (a >= float(alpha_min))
        if rendered_depth_valid is not None:
            v = rendered_depth_valid
            if v.ndim == 4: v = v[0, 0]
            elif v.ndim == 3: v = v[0]
            geom_valid = geom_valid & v.to(dtype=torch.bool)
        M = M * geom_valid.unsqueeze(0).unsqueeze(0)
    return I_rep_mean, M, C_var, C_diff, C_proj

def visualize_camera_poses(poses, colors=None, title="Camera Poses", save_path="poses.png", scale=0.1):
    """
    Visualize camera positions and viewing directions.
    Args:
        poses: (N, 4, 4) c2w poses ``[R_cw | C_world]``
        colors: list of str (e.g., 'b', 'g', 'r', ...) of length N, or None for default
        title: Figure title.
        save_path: Output image path.
        scale: Direction-axis length.
    """
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), "poses must be (N, 4, 4)"
    N = poses.shape[0]
    if colors is None:
        colors = ['b'] * N  # default: all blue
    assert len(colors) == N, "colors length must match number of poses"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        pose = poses[i]
        pos = pose[:3, 3]
        R = pose[:3, :3]
        forward = -R[:, 2]  # Camera -Z forward direction.
        ax.scatter(pos[0], pos[1], pos[2], c=colors[i], marker='o')
        ax.quiver(pos[0], pos[1], pos[2],
                  forward[0], forward[1], forward[2],
                  length=scale, color=colors[i], linewidth=0.5, arrow_length_ratio=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.grid(True)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

class CameraPoseInterpolator:
    """
    A system for interpolating between sets of camera poses with visualization capabilities.
    """
    
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        """
        Initialize the interpolator with weights for pose distance computation.
        
        Args:
            rotation_weight: Weight for rotational distance in pose matching
            translation_weight: Weight for translational distance in pose matching
        """
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def compute_pose_distance(self, pose1, pose2):
        """
        Compute weighted distance between two camera poses.
        
        Args:
            pose1, pose2: 4x4 transformation matrices
            
        Returns:
            Combined weighted distance between poses
        """
        # Translation distance (Euclidean)
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)
        
        # Rotation distance (angular distance between quaternions)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1 = R1.as_quat()
        q2 = R2.as_quat()
        
        # Ensure quaternions are in the same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        rotation_dist = np.arccos(2 * np.dot(q1, q2)**2 - 1)
        
        return (self.translation_weight * translation_dist + 
                self.rotation_weight * rotation_dist)
    
    def find_nearest_assignments(self, training_poses, testing_poses):
        """
        Find the nearest training camera pose for each testing camera pose.
        
        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            
        Returns:
            assignments: list of closest training pose indices for each testing pose
        """
        M = len(testing_poses)
        assignments = []

        for j in range(M):
            # Compute distance from each training pose to this testing pose
            distances = [self.compute_pose_distance(training_pose, testing_poses[j])
                         for training_pose in training_poses]
            # Find the index of the nearest training pose
            nearest_index = np.argmin(distances)
            assignments.append(nearest_index)
        
        return assignments
    
    def interpolate_rotation(self, R1, R2, t):
        """
        Interpolate between two rotation matrices using SLERP.
        """
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        # Clamp dot product to avoid invalid values in arccos
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        theta = np.arccos(dot_product)
        
        if np.abs(theta) < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1-t)*theta) * q1 + np.sin(t*theta) * q2) / np.sin(theta)
        
        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix()

    def interpolate_pose(self, pose1, pose2, alpha):
        pose = np.eye(4)
        pose[:3, :3] = self.interpolate_rotation(pose1[:3, :3], pose2[:3, :3], alpha)
        pose[:3, 3] = (1.0 - alpha) * pose1[:3, 3] + alpha * pose2[:3, 3]
        return pose

    def order_poses_by_pca(self, poses):
        centers = poses[:, :3, 3]
        if len(centers) <= 2:
            return np.arange(len(poses))
        centered = centers - centers.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        scores = centered @ axis
        return np.argsort(scores)

    def _filter_pose_candidates(self, candidates, train_poses, min_pose_dist=0.03, max_pose_dist=0.75):
        filtered = []
        for pose in candidates:
            nearest_dist = min(self.compute_pose_distance(train_pose, pose) for train_pose in train_poses)
            if nearest_dist >= min_pose_dist and nearest_dist <= max_pose_dist:
                filtered.append(pose)
        return filtered

    def sample_progressive_poses(
        self,
        train_poses,
        fix_index,
        num_fix_stages,
        n_views=2,
        min_interp_alpha=0.15,
        max_interp_alpha=0.5,
        max_extrap_alpha=0.15,
        min_pose_dist=0.03,
        max_pose_dist=0.75,
    ):
        """Sample novel poses from training poses via progressive interpolation and extrapolation."""
        train_poses = np.asarray(train_poses)
        if len(train_poses) < 2:
            return train_poses.copy()

        progress = 0.0
        if num_fix_stages > 1:
            progress = float(fix_index) / float(num_fix_stages - 1)
        interp_alpha = (1.0 - progress) * min_interp_alpha + progress * max_interp_alpha
        extrap_alpha = max_extrap_alpha * max(progress, 1.0)

        order = self.order_poses_by_pca(train_poses)
        ordered = train_poses[order]

        interp_candidates = []
        for i in range(len(ordered) - 1):
            left = ordered[i]
            right = ordered[i + 1]
            interp_candidates.append(self.interpolate_pose(left, right, interp_alpha))
            if interp_alpha != 0.5:
                interp_candidates.append(self.interpolate_pose(left, right, 1.0 - interp_alpha))

        extrap_candidates = []
        if extrap_alpha > 1e-6 and len(ordered) >= 2:
            left, left_next = ordered[0], ordered[1]
            right_prev, right = ordered[-2], ordered[-1]

            left_extra = left.copy()
            left_extra[:3, 3] = left[:3, 3] + extrap_alpha * (left[:3, 3] - left_next[:3, 3])
            left_extra[:3, :3] = left[:3, :3]
            extrap_candidates.append(left_extra)

            right_extra = right.copy()
            right_extra[:3, 3] = right[:3, 3] + extrap_alpha * (right[:3, 3] - right_prev[:3, 3])
            right_extra[:3, :3] = right[:3, :3]
            extrap_candidates.append(right_extra)

        n_total = max(1, int(n_views) * len(train_poses))

        interp_candidates = self._filter_pose_candidates(interp_candidates, train_poses, min_pose_dist, max_pose_dist)
        extrap_candidates = self._filter_pose_candidates(extrap_candidates, train_poses, min_pose_dist, max_pose_dist)

        # Interpolation is the primary source; endpoint extrapolations are
        # additional candidates. Deduplicate by pose distance before selection.
        candidates = interp_candidates + extrap_candidates
        selected = []
        for pose in candidates:
            if all(self.compute_pose_distance(pose, existing) > 1e-8 for existing in selected):
                selected.append(pose)
            if len(selected) >= n_total:
                break
        if not selected:
            selected = [self.interpolate_pose(ordered[i], ordered[i + 1], 0.5) for i in range(len(ordered) - 1)]
        # Preserve the fixed budget where filtering leaves too few candidates.
        fallback = selected[:]
        while len(selected) < n_total and fallback:
            selected.append(fallback[len(selected) % len(fallback)])
        return np.asarray(selected[:n_total])

    def get_poses(self, camera_list):
        """Convert cameras into c2w pose matrices ``[R_cw | C_world]`` of shape (N, 4, 4)."""
        poses = []
        for cam in camera_list:
            poses.append(camera_to_c2w_pose(cam.R, cam.T))
        return np.asarray(poses, dtype=np.float64).reshape((-1, 4, 4))


@torch.no_grad()
def compute_reprojection_error(novel_img, ref_img, novel_depth, novel_cam, ref_cam,
                               return_valid=False, return_debug=False):
    """
    Compute pixel-level novel-to-reference reprojection L1 error after bilinear sampling.
    Project novel-view pixels into the reference view, sample reference colors, and compare with novel colors.
    Args:
        novel_img: [H,W,3], 0..1, float32/float16
        ref_img:   [H,W,3], 0..1
        novel_depth: [H,W], in camera-depth units.
        novel_cam: novel camera with K, R, T.
        ref_cam: reference camera with K, R, T.

    Returns:
        e_proj: [H,W] reprojection error, meaningful on valid pixels.
        (optional) debug_info: dict
    """
    device = novel_depth.device
    dtype  = novel_depth.dtype
    H, W   = novel_depth.shape

    # Prepare camera parameters on the same device and dtype.
    Kn = novel_cam.K.to(device=device, dtype=dtype)
    Rn = torch.as_tensor(novel_cam.R, device=device, dtype=dtype)
    Tn = torch.as_tensor(novel_cam.T, device=device, dtype=dtype).view(3)

    Kr = ref_cam.K.to(device=device, dtype=dtype)
    Rr = torch.as_tensor(ref_cam.R, device=device, dtype=dtype)
    Tr = torch.as_tensor(ref_cam.T, device=device, dtype=dtype).view(3)

    fx, fy = Kn[0,0], Kn[1,1]
    cx, cy = Kn[0,2], Kn[1,2]

    # Pixel grid aligned with depth indexing.
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )  # v: [H,W] rows, u: [H,W] columns
    source_xy = torch.stack([u, v], dim=-1)

    z = novel_depth          # [H,W]
    # Positive depth values are valid.
    valid_z = torch.isfinite(z) & (z > 0)

    # Novel pixel coordinates -> novel camera coordinates.
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    Z = z
    pts_cam = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # [HW,3]

    # Transform only valid-depth points to save memory.
    if valid_z.any():
        mask_flat = valid_z.reshape(-1)
        pts_cam_valid = pts_cam[mask_flat]                                # [N,3]

        # --- 3) novel camera -> world ---
        # Camera.R is c2w rotation, Camera.T is w2c translation:
        # X_cam = R.T @ X_world + T, so X_world = R @ (X_cam - T).
        pts_world_valid = (Rn @ (pts_cam_valid - Tn.view(1,3)).T).T      # [N,3]

        # --- 4) world -> reference camera ---
        pts_ref_cam_valid = (Rr.T @ pts_world_valid.T).T + Tr.view(1,3)  # [N,3]

        # Perspective projection into reference pixels.
        Xp = pts_ref_cam_valid[:, 0]
        Yp = pts_ref_cam_valid[:, 1]
        Zp = pts_ref_cam_valid[:, 2]

        eps = torch.finfo(dtype).eps
        Zp_safe = torch.clamp(Zp, min=eps)

        xr = (Xp / Zp_safe) * Kr[0,0] + Kr[0,2]
        yr = (Yp / Zp_safe) * Kr[1,1] + Kr[1,2]

        # Keep only points in front of the reference camera.
        valid_ref_z = torch.isfinite(Xp) & torch.isfinite(Yp) & torch.isfinite(Zp) & (Zp > 0)

        # Normalize to grid_sample coordinates in [-1, 1].
        x_norm = (xr / (W - 1) * 2) - 1
        y_norm = (yr / (H - 1) * 2) - 1

        # Keep only valid points that project inside the image.
        in_bounds = (x_norm >= -1) & (x_norm <= 1) & (y_norm >= -1) & (y_norm <= 1)
        valid_final = valid_ref_z & torch.isfinite(xr) & torch.isfinite(yr) & in_bounds

        # Build the full sampling grid; invalid positions are masked later.
        grid = torch.zeros((H*W, 2), device=device, dtype=dtype)
        grid[mask_flat, 0] = x_norm
        grid[mask_flat, 1] = y_norm
        grid = grid.view(1, H, W, 2)

        # Prepare ref_img as NCHW.
        ref_nchw = ref_img.permute(2,0,1).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]
        ref_sampled = F.grid_sample(ref_nchw, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)  # [1,3,H,W]
        ref_sampled = ref_sampled.squeeze(0).permute(1,2,0)  # [H,W,3]

        # L1 reprojection error.
        e = (novel_img.to(device=device, dtype=dtype) - ref_sampled).abs().mean(dim=-1)  # [H,W]

        # Zero reprojection error outside the valid projected region.
        valid_mask = torch.zeros((H*W,), device=device, dtype=torch.bool)
        valid_mask[mask_flat] = valid_final
        valid_mask = valid_mask.view(H, W)

        e = torch.where(valid_mask, e, torch.zeros_like(e))

    else:
        e = torch.zeros((H,W), device=device, dtype=dtype)
        valid_mask = torch.zeros((H,W), device=device, dtype=torch.bool)
        ref_sampled = torch.zeros_like(novel_img.to(device=device, dtype=dtype))
        xr = torch.zeros((0,), device=device, dtype=dtype)
        yr = torch.zeros((0,), device=device, dtype=dtype)

    if return_debug:
        projected_xy = torch.zeros((H, W, 2), device=device, dtype=dtype)
        if valid_z.any():
            projected_xy[..., 0].view(-1)[mask_flat] = xr
            projected_xy[..., 1].view(-1)[mask_flat] = yr
        debug_info = {
            "ref_sampled": ref_sampled,
            "valid_mask": valid_mask,
            "error_map": e,
            "depth_map": novel_depth,
            "projected_xy": projected_xy,
            "source_xy": source_xy
        }
        if return_valid:
            return e, valid_mask, debug_info
        return e, debug_info

    return (e, valid_mask) if return_valid else e

