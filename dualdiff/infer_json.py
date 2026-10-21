import argparse
import json
import random
from pathlib import Path

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_json_path', type=str, required=True, help='Path to the dataset JSON with a "test" split')
    parser.add_argument('--save_path', type=str, default="./results/dualdiff_test", help='Directory to save the output')
    parser.add_argument('--height', type=int, default=576, help='Height of the network input (match training)')
    parser.add_argument('--width', type=int, default=1024, help='Width of the network input (match training)')
    parser.add_argument('--prompt', type=str, default="remove degradation", help='The prompt to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Local DualDiff checkpoint; overrides Hugging Face download')
    parser.add_argument('--model_id', type=str, default=None, help=f'Hugging Face model repository (default: {DEFAULT_MODEL_ID})')
    parser.add_argument('--model_filename', type=str, default=None, help=f'Checkpoint filename in the Hugging Face repository (default: {DEFAULT_MODEL_FILENAME})')
    parser.add_argument('--model_revision', type=str, default=None, help='Optional Hugging Face branch, tag, or commit')
    parser.add_argument('--cache_dir', type=str, default=None, help='Optional Hugging Face cache directory')
    parser.add_argument('--local_files_only', action='store_true', help='Use only an already-cached Hugging Face checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
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
    
    dataset_path = Path(args.testset_json_path).resolve()
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if "test" not in dataset or not isinstance(dataset["test"], dict):
        raise ValueError("Dataset JSON must contain a dictionary-valued 'test' split")
    data = dataset["test"]
    
    for img_id, sample in tqdm(data.items(), desc="Repairing test split"):
        lq_image_path = Path(sample["image"])
        ref_image_path = Path(sample["ref_image"])
        if not lq_image_path.is_absolute():
            lq_image_path = dataset_path.parent / lq_image_path
        if not ref_image_path.is_absolute():
            ref_image_path = dataset_path.parent / ref_image_path
        image = Image.open(lq_image_path).convert('RGB')
        ref_image = Image.open(ref_image_path).convert('RGB')
        with torch.inference_mode():
            output_image = model.sample(
                image,
                height=args.height,
                width=args.width,
                ref_image=ref_image,
                prompt=sample.get("prompt", args.prompt)
            )
        safe_id = str(img_id).replace("/", "_").replace("\\", "_")
        output_path = save_path / f"{safe_id}.png"
        output_image.save(output_path)
