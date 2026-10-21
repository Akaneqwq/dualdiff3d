import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T

from utils.fix_utils import (
    _compute_confidences_from_repairs,
    apply_c2w_pose_to_camera,
    visualize_fix_results,
)

TO_PIL = T.ToPILImage()
TO_TENSOR = T.ToTensor()


def has_valid_render_geometry(depth, alpha=None, rendered_depth_valid=None, alpha_min=1.0 / 255.0):
    """Return whether any pixel has finite, positive depth and valid raster support."""
    depth = torch.as_tensor(depth)
    mask = torch.isfinite(depth) & (depth > 0)
    if alpha is not None:
        alpha = torch.as_tensor(alpha, device=depth.device)
        mask = mask & torch.isfinite(alpha) & (alpha >= float(alpha_min))
    if rendered_depth_valid is not None:
        mask = mask & torch.as_tensor(rendered_depth_valid, device=depth.device, dtype=torch.bool)
    return bool(mask.any().item())


@dataclass
class ReliableReconstructionConfig:
    k_repair: int
    alpha_var: float
    beta_diff: float
    gamma_proj: float
    lambda_conf: tuple
    conf_threshold: float
    repair_ratio: float
    rollback_patience: int
    cproj_warmup_until: int
    cproj_pose_dist_thresh: float
    rollback_eval_interval: int
    profile_fix: bool
    novel_pool_max_size: int
    novel_pool_dedup_pose_dist: float
    novel_loss_weight: float = 0.3
    alpha_min: float = 1.0 / 255.0

    @classmethod
    def from_args(cls, args):
        k_repair = getattr(args, "k_repair", 3)
        if k_repair < 1:
            raise ValueError("--k_repair must be >= 1")
        novel_pool_max_size = getattr(args, "novel_pool_max_size", 0)
        novel_pool_dedup_pose_dist = getattr(args, "novel_pool_dedup_pose_dist", 0.03)
        if novel_pool_max_size < 0:
            raise ValueError("--novel_pool_max_size must be >= 0")
        if novel_pool_dedup_pose_dist < 0:
            raise ValueError("--novel_pool_dedup_pose_dist must be >= 0")
        alpha_min = getattr(args, "alpha_min", 1.0 / 255.0)
        if alpha_min < 0:
            raise ValueError("--alpha_min must be >= 0")

        return cls(
            k_repair=k_repair,
            alpha_var=getattr(args, "alpha_var", 500.0),
            beta_diff=getattr(args, "beta_diff", 10.0),
            gamma_proj=getattr(args, "gamma_proj", 10.0),
            lambda_conf=tuple(getattr(args, "lambda_conf", (0.2, 0.3, 0.5))),
            conf_threshold=getattr(args, "conf_threshold", 0.6),
            repair_ratio=getattr(args, "repair_ratio", 0.3),
            rollback_patience=getattr(args, "rollback_patience", 3),
            cproj_warmup_until=getattr(args, "cproj_warmup_until", 10000),
            cproj_pose_dist_thresh=getattr(args, "cproj_pose_dist_thresh", 0.75),
            rollback_eval_interval=getattr(args, "rollback_eval_interval", 500),
            profile_fix=getattr(args, "profile_fix", False),
            novel_pool_max_size=novel_pool_max_size,
            novel_pool_dedup_pose_dist=novel_pool_dedup_pose_dist,
            novel_loss_weight=getattr(args, "novel_loss_weight", 0.3),
            alpha_min=alpha_min,
        )


def _set_novel_pool_metadata(camera, iteration, confidence, forced_keep):
    """Attach pool-lifecycle metadata to a novel camera supervision sample."""
    camera.__novel_pool_stage__ = int(iteration)
    camera.__novel_pool_confidence__ = float(confidence)
    camera.__novel_pool_forced_keep__ = bool(forced_keep)


def finalize_novel_camera_acceptance(
    accepted_cams,
    rejected_count,
    best_cam,
    best_conf_mean,
    iteration,
    allow_forced_keep,
):
    """Keep the highest-confidence candidate when the novel camera pool is empty."""
    forced_keep = 0
    if best_cam is not None:
        _set_novel_pool_metadata(
            best_cam,
            iteration=iteration,
            confidence=best_conf_mean,
            forced_keep=False,
        )

    if not accepted_cams and best_cam is not None and allow_forced_keep:
        _set_novel_pool_metadata(
            best_cam,
            iteration=iteration,
            confidence=best_conf_mean,
            forced_keep=True,
        )
        accepted_cams.append(best_cam)
        rejected_count = max(rejected_count - 1, 0)
        forced_keep = 1

    return accepted_cams, rejected_count, forced_keep


def _novel_pool_eviction_key(camera, original_index):
    """Return the eviction priority key for novel camera pool entries."""
    forced_keep = bool(getattr(camera, "__novel_pool_forced_keep__", False))
    stage = int(getattr(camera, "__novel_pool_stage__", -1))
    return (0 if forced_keep else 1, stage, original_index)


def merge_novel_camera_pool(
    existing_cameras,
    accepted_cameras,
    interpolator,
    dedup_pose_distance,
    max_size,
):
    """Merge accepted novel cameras into a bounded, pose-deduplicated pool."""
    if dedup_pose_distance < 0:
        raise ValueError("dedup_pose_distance must be >= 0")
    if max_size < 1:
        raise ValueError("max_size must be >= 1")

    pool = list(existing_cameras or [])
    stats = {
        "added": 0,
        "dedup_replaced": 0,
        "forced_skipped": 0,
        "evicted": 0,
        "total": 0,
        "mean_confidence": 0.0,
    }

    for camera in accepted_cameras:
        forced_keep = bool(getattr(camera, "__novel_pool_forced_keep__", False))
        if forced_keep and pool:
            stats["forced_skipped"] += 1
            continue

        camera_pose = interpolator.get_poses([camera])[0]
        duplicate_index = None
        for index, pooled_camera in enumerate(pool):
            pooled_pose = interpolator.get_poses([pooled_camera])[0]
            pose_distance = interpolator.compute_pose_distance(camera_pose, pooled_pose)
            if pose_distance <= dedup_pose_distance:
                duplicate_index = index
                break

        if duplicate_index is None:
            pool.append(camera)
            stats["added"] += 1
        else:
            pool[duplicate_index] = camera
            stats["dedup_replaced"] += 1

    while len(pool) > max_size:
        eviction_index = min(
            range(len(pool)),
            key=lambda index: _novel_pool_eviction_key(pool[index], index),
        )
        pool.pop(eviction_index)
        stats["evicted"] += 1

    stats["total"] = len(pool)
    if pool:
        stats["mean_confidence"] = float(np.mean([
            getattr(camera, "__novel_pool_confidence__", 0.0)
            for camera in pool
        ]))
    return pool, stats


def refresh_novel_viewpoint_stack(
    novel_viewpoint_stack,
    novel_cameras,
    novel_tag,
    novel_pool_updated,
):
    """Refill exhausted novel-view sampling stacks from the current pool."""
    if not novel_tag or not novel_cameras:
        return None
    if novel_pool_updated or not novel_viewpoint_stack:
        return list(novel_cameras)
    return novel_viewpoint_stack


def sample_reliable_novel_poses(
    interpolator,
    args,
    iteration,
    train_poses,
    sorted_fix_iterations,
):
    fix_index = sorted_fix_iterations.index(iteration)
    return interpolator.sample_progressive_poses(
        train_poses,
        fix_index=fix_index,
        num_fix_stages=len(sorted_fix_iterations),
        n_views=args.n_fix_views,
        min_interp_alpha=args.min_interp_alpha,
        max_interp_alpha=args.max_interp_alpha,
        max_extrap_alpha=args.max_extrap_alpha,
        min_pose_dist=args.min_pose_dist,
        max_pose_dist=args.max_pose_dist,
    )


def run_dualdiff_repairs(image_fixer, image_pil, ref_pil, k_repair, width=1024, height=576, profile=False):
    repairs = []
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seeds = [random.randint(0, 999999) for _ in range(k_repair)]

    if profile:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_t = time.perf_counter()

    try:
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            output_pil = image_fixer.sample(
                image_pil,
                height=height,
                width=width,
                ref_image=ref_pil,
                prompt="remove degradation",
            )
            repairs.append(TO_TENSOR(output_pil).cuda().unsqueeze(0))
            del output_pil
    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    elapsed_ms = None
    memory = None
    if profile:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        memory = {
            "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved": torch.cuda.memory_reserved() / (1024 ** 2),
            "peak": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
    return torch.cat(repairs, dim=0), elapsed_ms, memory


@torch.no_grad()
def generate_reliable_novel_cameras(
    scene,
    gaussians,
    pipe,
    bg,
    render_func,
    image_fixer,
    interpolator,
    train_cams,
    train_poses,
    novel_poses,
    iteration,
    config,
    allow_forced_keep=True,
):
    ref_indices = interpolator.find_nearest_assignments(train_poses, novel_poses)
    novel_cams = [copy.deepcopy(train_cams[0]) for _ in range(len(novel_poses))]

    accepted_cams = []
    rejected_count = 0
    best_conf_mean = float("-inf")
    best_cam = None
    any_valid_projection = False

    for idx, cam in enumerate(novel_cams):
        apply_c2w_pose_to_camera(cam, novel_poses[idx])

        ref_cam_idx = ref_indices[idx]
        ref_cam = train_cams[ref_cam_idx]
        cam.ref_image = ref_cam.original_image

        ref_novel_pose_dist = interpolator.compute_pose_distance(train_poses[ref_cam_idx], novel_poses[idx])
        use_reprojection_conf = (
            iteration > config.cproj_warmup_until
            and ref_novel_pose_dist <= config.cproj_pose_dist_thresh
        )

        render_pkg = render_func(cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        I_low_bchw = render_pkg["render"].unsqueeze(0).clamp(0, 1).cuda()
        novel_depth = render_pkg["rendered_depth"]
        rendered_alpha = render_pkg.get("rendered_alpha", render_pkg.get("alpha"))
        rendered_depth_valid = render_pkg.get("rendered_depth_valid")
        any_valid_projection = any_valid_projection or has_valid_render_geometry(
            novel_depth, rendered_alpha, rendered_depth_valid, config.alpha_min
        )

        image_pil = TO_PIL(I_low_bchw.squeeze(0).cpu())
        ref_pil = TO_PIL(ref_cam.original_image.cpu())
        repairs_bchw, fix_ms, memory = run_dualdiff_repairs(
            image_fixer,
            image_pil,
            ref_pil,
            config.k_repair,
            profile=config.profile_fix,
        )

        I_ref_bchw = ref_cam.original_image.unsqueeze(0).cuda()
        I_rep_mean, M, C_var, C_diff, C_proj = _compute_confidences_from_repairs(
            I_low_bchw,
            I_ref_bchw,
            repairs_bchw,
            alpha=config.alpha_var,
            beta=config.beta_diff,
            gamma=config.gamma_proj,
            lambdas=config.lambda_conf,
            depth_hw=novel_depth,
            novel_cam=cam,
            ref_cam=ref_cam,
            rendered_alpha=rendered_alpha,
            rendered_depth_valid=rendered_depth_valid,
            alpha_min=config.alpha_min,
            debug_save_dir=(
                f"{scene.model_path}/novel/{iteration}/ReprojDebug"
                if config.profile_fix else None
            ),
            debug_idx=idx if config.profile_fix else None,
            use_reprojection_conf=use_reprojection_conf,
        )
        if use_reprojection_conf:
            any_valid_projection = any_valid_projection or bool(C_proj.any().item())

        if config.profile_fix:
            visualize_fix_results(
                I_low_bchw,
                I_ref_bchw,
                I_rep_mean,
                C_var,
                C_diff,
                C_proj,
                M,
                save_dir=f"{scene.model_path}/novel/{iteration}/Vis",
                idx=idx,
            )

        # Compute mean confidence over valid rasterized geometry pixels.
        geom_mask = torch.ones_like(M, dtype=torch.bool)
        depth_for_mask = novel_depth
        if depth_for_mask.ndim == 4:
            depth_for_mask = depth_for_mask[0, 0]
        elif depth_for_mask.ndim == 3:
            depth_for_mask = depth_for_mask[0]
        geom = torch.isfinite(depth_for_mask) & (depth_for_mask > 0)
        if rendered_alpha is not None:
            alpha_for_mask = rendered_alpha
            if alpha_for_mask.ndim == 4:
                alpha_for_mask = alpha_for_mask[0, 0]
            elif alpha_for_mask.ndim == 3:
                alpha_for_mask = alpha_for_mask[0]
            geom = geom & torch.isfinite(alpha_for_mask) & (alpha_for_mask >= config.alpha_min)
        if rendered_depth_valid is not None:
            valid_for_mask = rendered_depth_valid
            if valid_for_mask.ndim == 4:
                valid_for_mask = valid_for_mask[0, 0]
            elif valid_for_mask.ndim == 3:
                valid_for_mask = valid_for_mask[0]
            geom = geom & valid_for_mask.to(dtype=torch.bool)
        geom_mask = geom.unsqueeze(0).unsqueeze(0)
        conf_mean = float(M[geom_mask].mean().item()) if geom_mask.any() else 0.0
        if conf_mean > best_conf_mean:
            best_conf_mean = conf_mean
            cam.original_image = I_rep_mean.squeeze(0).clamp(0, 1)
            cam.__conf_mask__ = M.detach()
            cam.__I_low__ = I_low_bchw.detach()
            best_cam = cam

        if conf_mean >= config.conf_threshold:
            cam.original_image = I_rep_mean.squeeze(0).clamp(0, 1)
            cam.__conf_mask__ = M.detach()
            cam.__I_low__ = I_low_bchw.detach()
            _set_novel_pool_metadata(
                cam,
                iteration=iteration,
                confidence=conf_mean,
                forced_keep=False,
            )
            accepted_cams.append(cam)
        else:
            rejected_count += 1

    accepted_cams, rejected_count, forced_keep = finalize_novel_camera_acceptance(
        accepted_cams=accepted_cams,
        rejected_count=rejected_count,
        best_cam=best_cam,
        best_conf_mean=best_conf_mean,
        iteration=iteration,
        allow_forced_keep=allow_forced_keep and any_valid_projection,
    )

    stats = {
        "total": len(novel_cams),
        "accepted": len(accepted_cams),
        "rejected": rejected_count,
        "forced_keep": forced_keep,
    }
    return accepted_cams, stats


class RollbackMonitor:
    def __init__(self, patience, lpips_weight=0.7, psnr_weight=0.3):
        self.patience = patience
        self.lpips_weight = lpips_weight
        self.psnr_weight = psnr_weight
        self.best_val_score = float("inf")
        self.no_improve_count = 0
        self.best_model_state = None
        self.best_novel_cameras = None
        self.best_val_lpips = None
        self.best_val_psnr = None

    def update(self, val_lpips, val_psnr, gaussians, scene, opt):
        if val_psnr is None:
            return False, None

        # With LPIPS disabled, use PSNR-only rollback scoring.
        val_score = (
            self.lpips_weight * val_lpips - self.psnr_weight * (val_psnr / 100.0)
            if val_lpips is not None
            else -(val_psnr / 100.0)
        )
        if val_score < self.best_val_score:
            self.best_val_score = val_score
            self.best_val_lpips = val_lpips
            self.best_val_psnr = val_psnr
            self.best_model_state = copy.deepcopy(gaussians.capture())
            self.best_novel_cameras = copy.deepcopy(scene.novel_cameras)
            self.no_improve_count = 0
            return False, None

        self.no_improve_count += 1
        if self.no_improve_count < self.patience:
            return False, None

        if self.best_model_state is None:
            self.no_improve_count = 0
            return False, None

        gaussians.restore(self.best_model_state, opt)
        scene.novel_cameras = copy.deepcopy(self.best_novel_cameras)
        restored_novel_cameras = scene.getNovelCameras()
        novel_viewpoint_stack = restored_novel_cameras.copy() if restored_novel_cameras else None

        self.no_improve_count = 0
        gaussians.optimizer.zero_grad(set_to_none=True)
        return True, novel_viewpoint_stack
