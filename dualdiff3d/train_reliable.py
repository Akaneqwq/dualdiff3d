#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from utils.loss_utils import l1_loss, ssim
import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments.reliable import add_reliable_arguments, apply_reliable_preset
from utils.fix_utils import CameraPoseInterpolator
from utils.reliable_reconstruction import (
    ReliableReconstructionConfig,
    RollbackMonitor,
    generate_reliable_novel_cameras,
    merge_novel_camera_pool,
    refresh_novel_viewpoint_stack,
    sample_reliable_novel_poses,
)
import torch.nn.functional as F
from dualdiff.model import DualDiff
from dualdiff.hub import resolve_dualdiff_checkpoint
import numpy as np
import random
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def should_prune_low_opacity(fix_enabled, iteration, densify_until_iter, densification_interval):
    """Opacity-only pruning remains active after clone/split densification ends."""
    return bool(
        fix_enabled
        and iteration >= densify_until_iter
        and iteration % densification_interval == 0
    )


@torch.no_grad()
def evaluate_val_metrics(scene, gaussians, pipe, bg, iteration, lpips_loss_fn):
    val_cams = scene.getTestCameras().copy()
    if not val_cams:
        return None, None

    lpips_losses = []
    psnr_values = []

    for cam in val_cams:
        # Render the current validation view.
        render_pkg = render(cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image = render_pkg["render"]  # [C, H, W]
        gt = cam.original_image.cuda()  # [C, H, W]

        # === LPIPS (optional) ===
        if lpips_loss_fn is not None:
            lp = lpips_loss_fn(image.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            lpips_losses.append(lp)

        # === PSNR ===
        mse = F.mse_loss(image, gt).item()
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 10 * np.log10(1.0 / mse)
        psnr_values.append(psnr)

    avg_lpips = float(np.mean(lpips_losses)) if lpips_losses else None
    avg_psnr = float(np.mean(psnr_values))
    return avg_lpips, avg_psnr

def training(dataset, opt, pipe, args):
    # ----------------- Params -----------------
    reliable_cfg = ReliableReconstructionConfig.from_args(args)

    lpips_loss_fn = None if args.disable_lpips else lpips.LPIPS(net='vgg').cuda()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(
            args.start_checkpoint,
            weights_only=False,
        )
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    train_viewpoint_stack = None
    novel_viewpoint_stack = None
    novel_tag = False
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # ---------- Camera and repair setup ----------
    interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
    train_cams = scene.getTrainCameras().copy()
    train_poses = interpolator.get_poses(train_cams)

    if args.fix:
        fix_iteration_set = set(args.fix_iterations)

        dualdiff_ckpt = resolve_dualdiff_checkpoint(
            model_path=args.dualdiff_ckpt,
            model_id=args.dualdiff_model_id,
            filename=args.dualdiff_model_filename,
            revision=args.dualdiff_model_revision,
            cache_dir=args.dualdiff_cache_dir,
            local_files_only=args.dualdiff_local_files_only,
        )
        image_fixer = DualDiff(
            pretrained_path=dualdiff_ckpt,
            timestep=args.dualdiff_timestep,
        )
        image_fixer.set_eval()
        image_fixer.to("cuda")
        novel_pool_max_size = args.novel_pool_max_size or max(
            1, 2 * args.n_fix_views * len(train_cams)
        )
        sorted_fix_iterations = sorted(list(args.fix_iterations))
    else:
        fix_iteration_set = set()
        sorted_fix_iterations = []

    # ---------- Validation rollback monitor ----------
    rollback_monitor = RollbackMonitor(reliable_cfg.rollback_patience)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        novel_cam_update = False
        rolled_back = False

        if args.fix and 30000 <= iteration < 45000 and (iteration - 1) in fix_iteration_set:
            gaussians.apply_perturbation()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ========== FIX stage: generate and repair novel views ==========
        if args.fix and iteration in fix_iteration_set:
            novel_tag = True
            with torch.no_grad():
                # --- Sample novel camera poses ---
                novel_poses = sample_reliable_novel_poses(
                    interpolator,
                    args,
                    iteration,
                    train_poses,
                    sorted_fix_iterations,
                )

                if len(novel_poses) != 0:
                    accepted_cams, _ = generate_reliable_novel_cameras(
                        scene=scene,
                        gaussians=gaussians,
                        pipe=pipe,
                        bg=bg,
                        render_func=render,
                        image_fixer=image_fixer,
                        interpolator=interpolator,
                        train_cams=train_cams,
                        train_poses=train_poses,
                        novel_poses=novel_poses,
                        iteration=iteration,
                        config=reliable_cfg,
                        allow_forced_keep=not bool(scene.getNovelCameras()),
                    )

                    scene.novel_cameras, _ = merge_novel_camera_pool(
                        existing_cameras=scene.getNovelCameras(),
                        accepted_cameras=accepted_cams,
                        interpolator=interpolator,
                        dedup_pose_distance=reliable_cfg.novel_pool_dedup_pose_dist,
                        max_size=novel_pool_max_size,
                    )
                    novel_cam_update = True

        # ========== Build viewpoint sampling pools ==========
        if not train_viewpoint_stack or novel_cam_update:
            train_viewpoint_stack = scene.getTrainCameras().copy()
            novel_viewpoint_stack = refresh_novel_viewpoint_stack(
                novel_viewpoint_stack=novel_viewpoint_stack,
                novel_cameras=scene.getNovelCameras(),
                novel_tag=novel_tag,
                novel_pool_updated=novel_cam_update,
            )
        else:
            novel_viewpoint_stack = refresh_novel_viewpoint_stack(
                novel_viewpoint_stack=novel_viewpoint_stack,
                novel_cameras=scene.getNovelCameras(),
                novel_tag=novel_tag,
                novel_pool_updated=False,
            )

        if novel_tag and random.random() < reliable_cfg.repair_ratio and novel_viewpoint_stack:
            viewpoint_cam = random.choice(novel_viewpoint_stack)
            novel_viewpoint_stack.remove(viewpoint_cam)
            source_tag = 'novel'
        else:
            viewpoint_cam = random.choice(train_viewpoint_stack)
            train_viewpoint_stack.remove(viewpoint_cam)
            source_tag = 'train'

        gt_image = viewpoint_cam.original_image.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # ---------- Loss computation ----------
        if source_tag == "novel":
            # Confidence-masked pixel supervision for repaired novel views.
            M = getattr(viewpoint_cam, "__conf_mask__", torch.ones_like(image[None, :1, :, :]))
            per_pixel_l1 = (image - gt_image).abs().mean(dim=0, keepdim=True).unsqueeze(0)
            loss_pix = (M * per_pixel_l1).sum() / torch.clamp(M.sum(), min=1.0)
            loss = loss_pix

            # for logging only
            log_L1 = loss_pix.item()
            log_SSIM = None

        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)) if FUSED_SSIM_AVAILABLE else ssim(image, gt_image)
            loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim_value)

            # for logging only
            log_L1 = Ll1.item()
            log_SSIM = ssim_value.item()

        # Balance losses from training and novel-view samples.
        loss *= reliable_cfg.novel_loss_weight if source_tag == "novel" else 1.5

        loss.backward()
        iter_end.record()

        # ========== EMA-smoothed logging ==========
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                log_dict = {"Loss": f"{ema_loss_for_log:.7f}"}
                if log_L1 is not None:
                    log_dict["L1"] = f"{log_L1:.5f}"
                if log_SSIM is not None:
                    log_dict["SSIM"] = f"{log_SSIM:.5f}"
                log_dict["Type"] = source_tag
                progress_bar.set_postfix(log_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (pipe, background))

            # Densification
            if (not rolled_back) and iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if should_prune_low_opacity(
                args.fix,
                iteration,
                opt.densify_until_iter,
                opt.densification_interval,
            ):
                gaussians.prune_low_opacity(0.005)

            # Optimizer step
            if (not rolled_back) and iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Validate after applying this iteration's model updates. If rollback
            # triggers, checkpoints below save the restored best state.
            if iteration % reliable_cfg.rollback_eval_interval == 0:
                val_lpips, val_psnr = evaluate_val_metrics(
                    scene, gaussians, pipe, bg, iteration, lpips_loss_fn
                )
                rolled_back, restored_novel_stack = rollback_monitor.update(
                    val_lpips=val_lpips,
                    val_psnr=val_psnr,
                    gaussians=gaussians,
                    scene=scene,
                    opt=opt,
                )
                if rolled_back:
                    novel_viewpoint_stack = restored_novel_stack
                    novel_tag = bool(novel_viewpoint_stack)

            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    add_reliable_arguments(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = apply_reliable_preset(parser.parse_args(sys.argv[1:]))
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
