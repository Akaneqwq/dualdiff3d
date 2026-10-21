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

from argparse import ArgumentParser
from pathlib import Path


def read_metrics(path):
    metrics = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        if name in {"PSNR", "SSIM"} and value != "N/A":
            metrics[name] = float(value)
    if set(metrics) != {"PSNR", "SSIM"}:
        raise ValueError(f"Incomplete metric file: {path}")
    return metrics


def average_metrics(paths, label):
    values = [read_metrics(path) for path in paths]
    if not values:
        return None
    for path, metrics in zip(paths, values):
        print(f"{label} - {path.parent}: PSNR={metrics['PSNR']:.7f}")
    return {
        name: sum(item[name] for item in values) / len(values)
        for name in ("PSNR", "SSIM")
    }


parser = ArgumentParser("Average per-scene rendering metrics")
parser.add_argument("--path", "-s", required=True, type=Path)
parser.add_argument("--iteration", "-i", type=str, default="30000")
args = parser.parse_args()

direct_metric = args.path / f"metrics_{args.iteration}.txt"
scene_dirs = [args.path] if direct_metric.is_file() else sorted(
    path for path in args.path.iterdir() if path.is_dir()
)
raw_paths = [
    path / f"metrics_{args.iteration}.txt"
    for path in scene_dirs
    if (path / f"metrics_{args.iteration}.txt").is_file()
]
fixed_paths = [
    path / f"metrics_fixed_{args.iteration}.txt"
    for path in scene_dirs
    if (path / f"metrics_fixed_{args.iteration}.txt").is_file()
]

raw = average_metrics(raw_paths, "Original")
fixed = average_metrics(fixed_paths, "Fixed")
if raw is None and fixed is None:
    raise FileNotFoundError(
        f"No metrics for iteration {args.iteration} under {args.path}"
    )

output_path = args.path / "metrics_mean.txt"
with output_path.open("w", encoding="utf-8") as handle:
    for title, metrics in (
        (f"metrics_{args.iteration}", raw),
        (f"metrics_fixed_{args.iteration}", fixed),
    ):
        handle.write(f"{title}:\n")
        for name in ("PSNR", "SSIM"):
            value = metrics[name] if metrics is not None else "N/A"
            handle.write(f"{name} : {value}\n")
        handle.write("\n")

print("\n=== Final Results ===")
for title, metrics in (
    (f"metrics_{args.iteration}", raw),
    (f"metrics_fixed_{args.iteration}", fixed),
):
    print(f"{title}:")
    for name in ("PSNR", "SSIM"):
        value = metrics[name] if metrics is not None else "N/A"
        print(f"{name} : {value}")
