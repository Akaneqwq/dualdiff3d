"""Build the DualDiff training/test JSON from sparse-view 3DGS renders.

Each entry pairs a degraded rendering with its ground truth and with the
nearest reference view, which is what the dual-branch model consumes:

    {"image": <rendered>, "target_image": <gt>, "ref_image": <nearest ref>,
     "prompt": "remove degradation"}

Scenes are split 90/10 into train/test with a fixed seed.

The released DualDiff model is trained on 24-view renders:

    python scripts/prepare_data.py -r <rendered_root> -g <gt_root> --view_counts 24
"""

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def collect_scene(rendered_scene_path, gt_scene_path, meta_info, data_type, data_id_counter):
    """Append one entry per rendered view of a single scene."""
    rendered_scene_path = Path(rendered_scene_path)
    gt_scene_path = Path(gt_scene_path)
    view_list = sorted(
        path for path in rendered_scene_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not view_list:
        raise ValueError(f"No rendered images found in {rendered_scene_path}")

    ref_indices = [i for i, view in enumerate(view_list) if "_ref" in view.stem]
    if not ref_indices:
        raise ValueError(
            f"No reference render marked with '_ref' found in {rendered_scene_path}"
        )

    for i, view in enumerate(view_list):
        # A reference view is its own reference; every other view takes the
        # closest one, breaking ties towards the lower index.
        if "_ref" in view.stem:
            ref_indice = i
        else:
            distances = [abs(i - indice) for indice in ref_indices]
            min_distance = min(distances)
            nearest_indices = [ref_indices[j] for j, d in enumerate(distances) if d == min_distance]
            ref_indice = min(nearest_indices)

        lq_img_path = view.resolve()
        ref_img_path = (gt_scene_path / view_list[ref_indice].name.replace("_ref", "")).resolve()
        tgt_img_path = (gt_scene_path / view.name.replace("_ref", "")).resolve()
        for role, path in (("target", tgt_img_path), ("reference", ref_img_path)):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {role} image for render {view}: {path}"
                )

        meta_info[f"{data_type}_{data_id_counter}"] = {
            "image": str(lq_img_path),
            "target_image": str(tgt_img_path),
            "ref_image": str(ref_img_path),
            "prompt": "remove degradation",
        }
        data_id_counter += 1

    return meta_info, data_id_counter


def main():
    default_output_dir = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-r", "--rendered_path", type=Path, required=True,
                        help="Root path of the rendered image directories")
    parser.add_argument("-g", "--gt_path", type=Path, required=True,
                        help="Root path of the gt image directories")
    parser.add_argument("-n", "--dataset_name", type=str, default="DL3DV",
                        help="Name of the dataset")
    parser.add_argument("-e", "--exp_name", type=str, default="enhance_nearest",
                        help="Name of the experiment")
    parser.add_argument("-v", "--view_counts", type=int, nargs="+", default=[24],
                        help="Input-view counts to include (default: 24)")
    parser.add_argument("-o", "--output_dir", type=Path, default=default_output_dir,
                        help=f"Directory to write the JSON into (default: {default_output_dir})")
    args = parser.parse_args()

    rendered_root = args.rendered_path.expanduser().resolve()
    gt_root = args.gt_path.expanduser().resolve()
    for label, path in (("rendered root", rendered_root), ("ground-truth root", gt_root)):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} does not exist or is not a directory: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = sorted(path.name for path in rendered_root.iterdir() if path.is_dir())
    if not scene_list:
        raise ValueError(f"No scene directories found in {rendered_root}")
    print(f"Total scenes found: {len(scene_list)}")

    random.seed(42)
    train_scene_list = random.sample(scene_list, k=max(0, round(len(scene_list) * 0.9)))
    test_scene_list = [scene for scene in scene_list if scene not in train_scene_list]
    print(f"Training scenes: {len(train_scene_list)}, Testing scenes: {len(test_scene_list)}")

    def process_scenes(scenes, data_type):
        meta_info = {}
        data_id_counter = 0

        for scene in scenes:
            gt_scene_path = gt_root / scene / "gaussian_splat/images_4"
            rendered_scene_paths = [
                rendered_root / scene / f"lr/{n}/train_{n}"
                for n in args.view_counts
            ]

            # A scene contributes only if every requested view count is present,
            # so that all splits cover the same set of scenes.
            missing = [p for p in [*rendered_scene_paths, gt_scene_path] if not p.is_dir()]
            if missing:
                for path in missing:
                    print(f"Missing directory {path}, skipping scene {scene}.")
                continue

            for rendered_scene_path in rendered_scene_paths:
                meta_info, data_id_counter = collect_scene(
                    rendered_scene_path, gt_scene_path, meta_info, data_type, data_id_counter
                )

        return meta_info

    print("Extracting meta information for training...")
    train_meta = process_scenes(train_scene_list, "train")
    print(f"Final train data count: {len(train_meta)}")

    print("Extracting meta information for testing...")
    test_meta = process_scenes(test_scene_list, "test")
    print(f"Final test data count: {len(test_meta)}")

    if not train_meta:
        raise RuntimeError(
            "No training samples were generated. Check the documented directory "
            "layout, requested view counts, and '_ref' filenames."
        )
    if not test_meta:
        print("Warning: no test samples were generated; validation will be unavailable.")

    output_data = {"train": train_meta, "test": test_meta}

    suffix = "meta" if args.view_counts == [24] else "meta_multiview"
    output_file = output_dir / f"{args.dataset_name}_{args.exp_name}_{suffix}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Meta information saved to {output_file}")


if __name__ == "__main__":
    main()
