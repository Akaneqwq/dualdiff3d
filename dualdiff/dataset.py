import json
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):
        for offset in range(len(self.img_ids)):
            current_idx = (idx + offset) % len(self.img_ids)
            img_id = self.img_ids[current_idx]

            input_path = self.data[img_id]["image"]
            output_path = self.data[img_id]["target_image"]
            ref_path = self.data[img_id].get("ref_image")
            caption = self.data[img_id]["prompt"]

            try:
                with Image.open(input_path) as image:
                    input_img = image.convert("RGB")
                with Image.open(output_path) as image:
                    output_img = image.convert("RGB")
                if ref_path is not None:
                    with Image.open(ref_path) as image:
                        ref_img = image.convert("RGB")
                else:
                    ref_img = None
            except (OSError, ValueError) as exc:
                print(
                    f"Skipping unreadable sample index={current_idx}, id={img_id}: "
                    f"image={input_path}, target_image={output_path}, "
                    f"ref_image={ref_path}; error={exc}"
                )
                continue
            break
        else:
            raise RuntimeError(
                f"No readable image samples found after checking {len(self.img_ids)} entries"
            )

        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if ref_img is not None:
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
            ref_t = ref_t.unsqueeze(0)          

        img_t = img_t.unsqueeze(0)
        output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "pixel_values": img_t,
            "ref_pixel_values": ref_t if ref_img is not None else None,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
