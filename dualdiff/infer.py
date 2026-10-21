import argparse
import random
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
if __package__:
    from .model import DualDiff
    from .hub import (
        DEFAULT_MODEL_FILENAME,
        DEFAULT_MODEL_ID,
        resolve_dualdiff_checkpoint,
    )
else:
    from model import DualDiff
    from hub import (
        DEFAULT_MODEL_FILENAME,
        DEFAULT_MODEL_ID,
        resolve_dualdiff_checkpoint,
    )


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images(path):
    path = Path(path)
    if path.is_dir():
        images = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in: {path}")
        return images
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, required=True, help='Path to the reference image or directory')
    parser.add_argument('--height', type=int, default=576, help='Height of the network input (match training)')
    parser.add_argument('--width', type=int, default=1024, help='Width of the network input (match training)')
    parser.add_argument('--prompt', type=str, default="remove degradation", help='The prompt to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Local DualDiff checkpoint; overrides Hugging Face download')
    parser.add_argument('--model_id', type=str, default=None, help=f'Hugging Face model repository (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--model_filename', type=str, default=None, help=f'Checkpoint filename in the Hugging Face repository (default: {DEFAULT_MODEL_FILENAME})')
    parser.add_argument('--model_revision', type=str, default=None, help='Optional Hugging Face branch, tag, or commit')
    parser.add_argument('--cache_dir', type=str, default=None, help='Optional Hugging Face cache directory')
    parser.add_argument('--local_files_only', action='store_true', help='Use only an already-cached Hugging Face checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='If the input is a video')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_path = resolve_dualdiff_checkpoint(
        model_path=args.model_path,
        model_id=args.model_id,
        filename=args.model_filename,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    model = DualDiff(
        pretrained_path=model_path,
        timestep=args.timestep,
    )
    model.set_eval()

    input_images = collect_images(args.input_image)
    ref_images = collect_images(args.ref_image)
    if len(input_images) != len(ref_images):
        raise ValueError(
            "Input and reference paths must contain the same number of images: "
            f"{len(input_images)} != {len(ref_images)}"
        )

    writer = None
    if args.video:
        writer = imageio.get_writer(output_dir / "output.mp4", fps=30)

    try:
        for input_path, ref_path in tqdm(
            zip(input_images, ref_images),
            total=len(input_images),
            desc="Repairing images",
        ):
            image = Image.open(input_path).convert("RGB")
            ref_image = Image.open(ref_path).convert("RGB")
            with torch.inference_mode():
                output_image = model.sample(
                    image,
                    height=args.height,
                    width=args.width,
                    ref_image=ref_image,
                    prompt=args.prompt,
                )
            if writer is not None:
                writer.append_data(np.asarray(output_image))
            else:
                output_image.save(output_dir / input_path.name)
    finally:
        if writer is not None:
            writer.close()