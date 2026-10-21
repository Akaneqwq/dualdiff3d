import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from dualdiff.hub import (
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_ID,
    resolve_dualdiff_checkpoint,
)
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.fix_utils import CameraPoseInterpolator
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    fused_ssim = None
    FUSED_SSIM_AVAILABLE = False


RENDER_MODES = ("3dgs", "dualdiff")


def metric_summary(values):
    available = [value for value in values if value is not None]
    return torch.tensor(available).mean().item() if available else None


def write_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key} : {value:>12.7f}\n" if value is not None else f"{key} : N/A\n")


def print_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key} : {value:>12.7f}" if value is not None else f"{key} : N/A")


def evaluate_pair(rendering, gt):
    rendering_batch = rendering.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    return {
        "PSNR": psnr(rendering_batch, gt_batch),
        "SSIM": (
            fused_ssim(rendering_batch, gt_batch)
            if FUSED_SSIM_AVAILABLE
            else ssim(rendering_batch, gt_batch)
        ),
    }


def load_fixer(mode, dualdiff_ckpt, dualdiff_timestep):
    if mode == "dualdiff":
        from dualdiff.model import DualDiff

        fixer = DualDiff(
            pretrained_path=dualdiff_ckpt,
            timestep=dualdiff_timestep,
        )
        fixer.set_eval()
        fixer.to("cuda")
        return fixer

    return None


def repair_image(mode, fixer, rendering_pil, ref_image_pil, width, height):
    if mode == "dualdiff":
        return fixer.sample(
            rendering_pil,
            height=height,
            width=width,
            ref_image=ref_image_pil,
            prompt="remove degradation",
        )

    raise ValueError(f"Unsupported repair mode: {mode}")


def render_plain_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    metrics = {"PSNR": [], "SSIM": []}

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        gt = view.original_image[0:3, :, :].cuda()
        rendering = render(view, gaussians, pipeline, background)["render"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

        values = evaluate_pair(rendering, gt)
        for key, value in values.items():
            metrics[key].append(value)

    summary = {key: metric_summary(value) for key, value in metrics.items()}
    print_metrics(summary)
    write_metrics(os.path.join(model_path, f"metrics_{iteration}.txt"), summary)


def render_repaired_test_set(
    mode,
    model_path,
    iteration,
    train_views,
    test_views,
    gaussians,
    pipeline,
    background,
    dualdiff_ckpt,
    dualdiff_timestep,
    repair_width,
    repair_height,
):
    render_path = os.path.join(model_path, "test", f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, "test", f"ours_{iteration}", "gt")
    fixed_path = os.path.join(model_path, "test", f"ours_{iteration}", "fixed")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(fixed_path, exist_ok=True)

    raw_metrics = {"PSNR": [], "SSIM": []}
    fixed_metrics = {"PSNR": [], "SSIM": []}

    fixer = load_fixer(mode, dualdiff_ckpt, dualdiff_timestep)
    interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
    ref_indices = interpolator.find_nearest_assignments(
        interpolator.get_poses(train_views),
        interpolator.get_poses(test_views),
    )
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for idx, view in enumerate(tqdm(test_views, desc=f"Rendering test with {mode} repair")):
        gt = view.original_image[0:3, :, :].cuda()
        rendering = render(view, gaussians, pipeline, background)["render"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

        ref_image = train_views[ref_indices[idx]].original_image.cuda()
        fixed_image = repair_image(
            mode,
            fixer,
            to_pil(rendering.cpu()),
            to_pil(ref_image.cpu()),
            repair_width,
            repair_height,
        )
        if fixed_image.size != (view.image_width, view.image_height):
            fixed_image = fixed_image.resize((view.image_width, view.image_height), resample=Image.LANCZOS)
        fixed_image.save(os.path.join(fixed_path, f"{idx:05d}.png"))

        raw_values = evaluate_pair(rendering, gt)
        fixed_tensor = to_tensor(fixed_image).cuda()
        fixed_values = evaluate_pair(fixed_tensor, gt)

        for key, value in raw_values.items():
            raw_metrics[key].append(value)
        for key, value in fixed_values.items():
            fixed_metrics[key].append(value)

    raw_summary = {key: metric_summary(value) for key, value in raw_metrics.items()}
    fixed_summary = {key: metric_summary(value) for key, value in fixed_metrics.items()}

    print("Raw render metrics:")
    print_metrics(raw_summary)
    print("DualDiff repair metrics:")
    print_metrics(fixed_summary)
    write_metrics(os.path.join(model_path, f"metrics_{iteration}.txt"), raw_summary)
    write_metrics(os.path.join(model_path, f"metrics_fixed_{iteration}.txt"), fixed_summary)


def render_sets(
    mode,
    dataset,
    iteration,
    pipeline,
    skip_train,
    skip_test,
    dualdiff_ckpt,
    dualdiff_timestep,
    repair_width,
    repair_height,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_plain_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if skip_test:
            return

        if mode == "3dgs":
            render_plain_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
        else:
            render_repaired_test_set(
                mode,
                dataset.model_path,
                scene.loaded_iter,
                scene.getTrainCameras(),
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dualdiff_ckpt,
                dualdiff_timestep,
                repair_width,
                repair_height,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Unified render/evaluation entry point.")
    parser.add_argument("mode", choices=RENDER_MODES, help="Render/evaluation mode.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--dualdiff_ckpt", type=str, default=None,
                        help="Local checkpoint path; overrides Hugging Face download.")
    parser.add_argument("--dualdiff_model_id", type=str, default=None,
                        help=f"Hugging Face model repository (default: {DEFAULT_MODEL_ID}).")
    parser.add_argument("--dualdiff_model_filename", type=str, default=None,
                        help=f"Checkpoint filename on the Hub (default: {DEFAULT_MODEL_FILENAME}).")
    parser.add_argument("--dualdiff_model_revision", type=str, default=None,
                        help="Optional Hugging Face branch, tag, or commit.")
    parser.add_argument("--dualdiff_cache_dir", type=str, default=None,
                        help="Optional Hugging Face cache directory.")
    parser.add_argument("--dualdiff_local_files_only", action="store_true",
                        help="Use only an already-cached Hugging Face checkpoint.")
    parser.add_argument("--dualdiff_timestep", type=int, default=199)
    parser.add_argument("--repair_width", type=int, default=1024)
    parser.add_argument("--repair_height", type=int, default=576)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print(f"Rendering {args.model_path} with mode={args.mode}")
    safe_state(args.quiet)
    dualdiff_ckpt = getattr(args, "dualdiff_ckpt", None)
    if args.mode == "dualdiff":
        dualdiff_ckpt = resolve_dualdiff_checkpoint(
            model_path=dualdiff_ckpt,
            model_id=getattr(args, "dualdiff_model_id", None),
            filename=getattr(args, "dualdiff_model_filename", None),
            revision=getattr(args, "dualdiff_model_revision", None),
            cache_dir=getattr(args, "dualdiff_cache_dir", None),
            local_files_only=getattr(args, "dualdiff_local_files_only", False),
        )
    render_sets(
        args.mode,
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        dualdiff_ckpt,
        args.dualdiff_timestep,
        args.repair_width,
        args.repair_height,
    )
